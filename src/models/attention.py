"""
Dynamic Decoupled Cross-Attention Module
=========================================
Reconstructed from HybridEditDif (Liu et al., Pattern Recognition 2026)

Architecture per paper Section 3.3:
  - Independent cross-attention layers for image (c_i) and text (c_t) features
  - Shared query Q from UNet feature map Z
  - Weighting parameters λ1, λ2 to balance contributions
  - Final: Z_new = λ1·Attn(Q,K,V) + λ2·Attn(Q,K',V')

Eq. (8):  Z'   = Softmax(QK^T / sqrt(d)) V        [image branch]
Eq. (9):  Z''  = Softmax(Q(K')^T / sqrt(d)) V'    [text branch]
Eq. (10): Z_new = λ1·Z' + λ2·Z''
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossAttentionBranch(nn.Module):
    """
    Single cross-attention branch (image OR text).
    Implements scaled dot-product attention:
        Z' = Softmax(QK^T / sqrt(d)) V
    where Q comes from UNet hidden state Z,
    K and V come from condition (c_i or c_t).
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/sqrt(d)

        # Linear projections — shared Q design (see paper: "same query Q")
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : UNet hidden state Z  [B, seq_q, query_dim]
            context : Condition c_i or c_t [B, seq_c, context_dim]
        Returns:
            out     : Attended output      [B, seq_q, query_dim]
        """
        B, seq_q, _ = x.shape
        h = self.heads

        Q = self.to_q(x)        # [B, seq_q, inner_dim]
        K = self.to_k(context)  # [B, seq_c, inner_dim]
        V = self.to_v(context)  # [B, seq_c, inner_dim]

        # Reshape to multi-head format
        def split_heads(t):
            # [B, seq, inner_dim] -> [B*h, seq, dim_head]
            t = t.reshape(B, -1, h, t.shape[-1] // h)
            return t.permute(0, 2, 1, 3).reshape(B * h, -1, t.shape[-1])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention: Softmax(QK^T / sqrt(d)) V
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B*h, seq_q, seq_c]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V)  # [B*h, seq_q, dim_head]

        # Merge heads back
        out = out.reshape(B, h, seq_q, -1).permute(0, 2, 1, 3)
        out = out.reshape(B, seq_q, -1)

        return self.to_out(out)


class DynamicDecoupledCrossAttention(nn.Module):
    """
    Dynamic Decoupled Cross-Attention (DDCA)
    ==========================================
    Core contribution of HybridEditDif.

    Per paper Eq. (10):
        Z_new = λ1 · Attn_img(Q, K_img, V_img)
              + λ2 · Attn_txt(Q, K_txt, V_txt)

    Key design choices (from paper):
    1. Shared query Q — same projection for both branches
    2. Independent K,V projections per modality
    3. Dynamic λ1, λ2 — adjustable at inference for modality control
       (λ2=0 → image-only, λ1=0 → text-only, both>0 → multimodal)

    This is injected into each of the 16 cross-attention layers of SD UNet.
    """

    def __init__(
        self,
        query_dim: int,
        image_context_dim: int,  # 1024 from CLIP ViT-H/14
        text_context_dim: int,   # 768 from CLIP text encoder
        heads: int = 8,
        dim_head: int = 64,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ):
        super().__init__()
        self.lambda1 = lambda1  # weight for image branch
        self.lambda2 = lambda2  # weight for text branch

        inner_dim = dim_head * heads

        # Shared Q projection (used by both branches, same Q per paper)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        # Image branch: independent K, V
        self.to_k_img = nn.Linear(image_context_dim, inner_dim, bias=False)
        self.to_v_img = nn.Linear(image_context_dim, inner_dim, bias=False)

        # Text branch: independent K', V'
        self.to_k_txt = nn.Linear(text_context_dim, inner_dim, bias=False)
        self.to_v_txt = nn.Linear(text_context_dim, inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

    def _attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Scaled dot-product attention for multi-head format."""
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, V)

    def _split_heads(self, t: torch.Tensor, B: int) -> torch.Tensor:
        """Reshape [B, seq, inner] -> [B*h, seq, dim_head]"""
        t = t.reshape(B, -1, self.heads, self.dim_head)
        return t.permute(0, 2, 1, 3).reshape(B * self.heads, -1, self.dim_head)

    def _merge_heads(self, t: torch.Tensor, B: int, seq: int) -> torch.Tensor:
        """Reshape [B*h, seq, dim_head] -> [B, seq, inner]"""
        t = t.reshape(B, self.heads, seq, self.dim_head)
        return t.permute(0, 2, 1, 3).reshape(B, seq, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,      # Z: UNet feature  [B, seq_q, query_dim]
        image_context: torch.Tensor,      # c_i: image embed  [B, seq_i, image_ctx_dim]
        text_context: torch.Tensor,       # c_t: text embed   [B, seq_t, text_ctx_dim]
        lambda1: Optional[float] = None,  # override at inference
        lambda2: Optional[float] = None,
    ) -> torch.Tensor:

        lam1 = lambda1 if lambda1 is not None else self.lambda1
        lam2 = lambda2 if lambda2 is not None else self.lambda2

        B, seq_q, _ = hidden_states.shape

        # Shared query projection
        Q = self.to_q(hidden_states)  # [B, seq_q, inner_dim]
        Q = self._split_heads(Q, B)   # [B*h, seq_q, dim_head]

        out = torch.zeros(B * self.heads, seq_q, self.dim_head,
                          device=hidden_states.device, dtype=hidden_states.dtype)

        # ── Image branch (Eq. 8) ─────────────────────────────────────────────
        if lam1 != 0.0 and image_context is not None:
            K_img = self._split_heads(self.to_k_img(image_context), B)
            V_img = self._split_heads(self.to_v_img(image_context), B)
            Z_img = self._attention(Q, K_img, V_img)   # [B*h, seq_q, dim_head]
            out = out + lam1 * Z_img

        # ── Text branch (Eq. 9) ──────────────────────────────────────────────
        if lam2 != 0.0 and text_context is not None:
            K_txt = self._split_heads(self.to_k_txt(text_context), B)
            V_txt = self._split_heads(self.to_v_txt(text_context), B)
            Z_txt = self._attention(Q, K_txt, V_txt)   # [B*h, seq_q, dim_head]
            out = out + lam2 * Z_txt

        # Merge heads and project out
        out = self._merge_heads(out, B, seq_q)
        return self.to_out(out)
