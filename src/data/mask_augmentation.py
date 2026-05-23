"""
Fourier-Based Mask Shape Augmentation
=======================================
Reconstructed from HybridEditDif Section 3.2 (Mask shape augmentation)

Paper equations:
    Eq.(1): γ(t) = Σ_{k=-∞}^{∞} (a_k·cos(2πkt) + b_k·sin(2πkt))
    Eq.(2): a'_k = a_k + Δa_k,  b'_k = b_k + Δb_k
    Eq.(3): γ'(t) = Σ_{k=-∞}^{∞} (a'_k·cos(2πkt) + b'_k·sin(2πkt))
    Eq.(4): m = Fill(γ'(t))
    Eq.(5): m̄ = 1 - U(m)   where U is random boundary distortion ±1 to ±5 px

Purpose: Rectangular masks fail to capture irregular target shapes.
This generates arbitrary-shape masks for more generalizable inpainting.
"""

import numpy as np
import cv2
import torch
from PIL import Image
import random
import math
from typing import Tuple, Optional


class FourierMaskGenerator:
    """
    Generates masks with arbitrary shapes using Fourier series decomposition.

    Strategy:
    1. Start from a base shape (circle, ellipse, or rectangle bounding box)
    2. Represent its boundary as a closed curve γ(t)
    3. Compute Fourier coefficients a_k, b_k
    4. Perturb coefficients: a'_k = a_k + Δa_k (Eq. 2)
    5. Reconstruct perturbed boundary γ'(t) (Eq. 3)
    6. Fill to binary mask (Eq. 4)
    7. Apply random boundary distortion U (Eq. 5)
    """

    def __init__(
        self,
        n_harmonics: int = 12,         # number of Fourier harmonics K
        perturbation_scale: float = 0.05,  # magnitude of Δa_k, Δb_k
        boundary_distortion_range: Tuple[int, int] = (1, 5),  # ±px per Eq.5
        n_points: int = 256,           # curve discretization resolution
    ):
        self.K = n_harmonics
        self.perturb_scale = perturbation_scale
        self.distort_min, self.distort_max = boundary_distortion_range
        self.n_points = n_points
        self.t = np.linspace(0, 1, n_points, endpoint=False)

    def _curve_to_fourier(
        self, x_curve: np.ndarray, y_curve: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose closed curve into Fourier coefficients.
        Returns a_k, b_k arrays of shape [2K+1]  (k from -K to K)
        """
        N = len(x_curve)
        k_vals = np.arange(-self.K, self.K + 1)

        a_k = np.zeros(len(k_vals))
        b_k = np.zeros(len(k_vals))

        for i, k in enumerate(k_vals):
            phase = 2 * np.pi * k * self.t
            a_k[i] = (2 / N) * np.sum(x_curve * np.cos(phase))
            b_k[i] = (2 / N) * np.sum(y_curve * np.sin(phase))

        return a_k, b_k

    def _fourier_to_curve(
        self, a_k: np.ndarray, b_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct curve from Fourier coefficients (Eq. 3).
        γ'(t) = Σ_{k} (a'_k·cos(2πkt) + b'_k·sin(2πkt))
        """
        k_vals = np.arange(-self.K, self.K + 1)
        x_out = np.zeros(self.n_points)
        y_out = np.zeros(self.n_points)

        for i, k in enumerate(k_vals):
            phase = 2 * np.pi * k * self.t
            x_out += a_k[i] * np.cos(phase)
            y_out += b_k[i] * np.sin(phase)

        return x_out, y_out

    def _perturb_coefficients(
        self, a_k: np.ndarray, b_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random perturbations (Eq. 2):
            a'_k = a_k + Δa_k
            b'_k = b_k + Δb_k
        """
        scale_a = np.abs(a_k).mean() * self.perturb_scale + 1e-6
        scale_b = np.abs(b_k).mean() * self.perturb_scale + 1e-6

        delta_a = np.random.uniform(-scale_a, scale_a, size=a_k.shape)
        delta_b = np.random.uniform(-scale_b, scale_b, size=b_k.shape)

        return a_k + delta_a, b_k + delta_b

    def _apply_boundary_distortion(self, mask: np.ndarray) -> np.ndarray:
        """
        Random boundary distortion U (Eq. 5): m̄ = 1 - U(m)
        Selects boundary points and shifts them by ±1 to ±5 pixels.
        """
        # Find boundary contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return mask

        # Apply random offset to each boundary point
        disturbed_contours = []
        for contour in contours:
            new_contour = contour.copy().astype(np.float32)
            for i in range(len(new_contour)):
                offset = random.randint(self.distort_min, self.distort_max)
                angle  = random.uniform(0, 2 * math.pi)
                new_contour[i, 0, 0] += offset * math.cos(angle)
                new_contour[i, 0, 1] += offset * math.sin(angle)
            disturbed_contours.append(new_contour.astype(np.int32))

        # Redraw mask with disturbed contour (Eq. 5: m̄ = 1 - U(m))
        h, w = mask.shape
        disturbed_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(disturbed_mask, disturbed_contours, -1, 1, thickness=cv2.FILLED)

        return disturbed_mask

    def generate_mask(
        self,
        image_size: Tuple[int, int],   # (H, W)
        bbox: Optional[Tuple[int, int, int, int]] = None,  # (x1,y1,x2,y2)
        base_shape: str = 'ellipse',   # 'ellipse', 'rect', 'random'
        apply_distortion: bool = True,
    ) -> np.ndarray:
        """
        Generate a Fourier-augmented mask.

        Args:
            image_size     : (H, W) of the target image
            bbox           : bounding box of target region; random if None
            base_shape     : starting shape before Fourier perturbation
            apply_distortion: apply Eq.5 boundary distortion

        Returns:
            mask : binary numpy array [H, W], 1 = masked region
        """
        H, W = image_size

        # If no bbox, generate a random one (covers 10–50% of image)
        if bbox is None:
            area_frac = random.uniform(0.1, 0.5)
            side = int(math.sqrt(area_frac * H * W))
            x1 = random.randint(0, W - side)
            y1 = random.randint(0, H - side)
            x2 = min(W, x1 + side)
            y2 = min(H, y1 + side)
            bbox = (x1, y1, x2, y2)

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = (x2 - x1) / 2.0
        ry = (y2 - y1) / 2.0

        # Generate base curve (Eq. 1 starting point)
        if base_shape == 'ellipse' or (base_shape == 'random' and random.random() > 0.5):
            x_curve = cx + rx * np.cos(2 * np.pi * self.t)
            y_curve = cy + ry * np.sin(2 * np.pi * self.t)
        else:  # rectangle
            # Parameterize rectangle boundary
            perimeter_t = self.t * 4  # 4 sides
            x_curve = np.where(
                perimeter_t < 1, cx - rx + perimeter_t * 2 * rx,
                np.where(perimeter_t < 2, cx + rx,
                np.where(perimeter_t < 3, cx + rx - (perimeter_t - 2) * 2 * rx,
                         cx - rx))
            )
            y_curve = np.where(
                perimeter_t < 1, cy - ry,
                np.where(perimeter_t < 2, cy - ry + (perimeter_t - 1) * 2 * ry,
                np.where(perimeter_t < 3, cy + ry,
                         cy + ry - (perimeter_t - 3) * 2 * ry))
            )

        # Fourier decomposition (Eq. 1)
        a_k, b_k = self._curve_to_fourier(x_curve, y_curve)

        # Perturb coefficients (Eq. 2)
        a_k_prime, b_k_prime = self._perturb_coefficients(a_k, b_k)

        # Reconstruct perturbed curve (Eq. 3)
        x_prime, y_prime = self._fourier_to_curve(a_k_prime, b_k_prime)

        # Clip to image bounds
        x_prime = np.clip(x_prime, 0, W - 1).astype(np.int32)
        y_prime = np.clip(y_prime, 0, H - 1).astype(np.int32)

        # Fill curve to binary mask (Eq. 4): m = Fill(γ'(t))
        contour = np.stack([x_prime, y_prime], axis=-1).reshape(-1, 1, 2)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)

        # Apply boundary distortion (Eq. 5): m̄ = 1 - U(m)
        if apply_distortion:
            mask = self._apply_boundary_distortion(mask)

        return mask

    def generate_batch(
        self,
        image_size: Tuple[int, int],
        batch_size: int,
        bboxes: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Generate batch of Fourier masks.
        Returns: [B, 1, H, W] float tensor
        """
        masks = []
        for i in range(batch_size):
            bbox = bboxes[i] if bboxes is not None else None
            m = self.generate_mask(image_size, bbox=bbox)
            masks.append(m)

        masks = np.stack(masks)  # [B, H, W]
        return torch.from_numpy(masks).unsqueeze(1).float()  # [B, 1, H, W]


class SelfSupervisedMaskGenerator:
    """
    Self-supervised mask strategy (Section 3.2, Self-supervised training):
        X_r = m · X_s    (reference = masked crop of source)
        Training data: (1-m)·X_s, X_r, T, m

    Uses bounding box of the target object as binary mask m.
    Combined with FourierMaskGenerator for shape augmentation.
    """

    def __init__(self, image_size: int = 512, use_fourier: bool = True):
        self.image_size = image_size
        self.fourier_gen = FourierMaskGenerator() if use_fourier else None

    def generate_training_sample(
        self,
        source_image: np.ndarray,    # [H, W, 3]
        bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            masked_source : (1-m) · X_s  [H, W, 3] — input to model
            reference_img : X_r = m · X_s [H, W, 3] — reference image
            mask          : binary mask    [H, W]
        """
        H, W = source_image.shape[:2]

        if self.fourier_gen is not None:
            # Use Fourier-augmented mask shape
            mask = self.fourier_gen.generate_mask(
                (H, W), bbox=bbox, base_shape='random'
            )
        else:
            # Simple bbox mask
            x1, y1, x2, y2 = bbox
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

        # X_r = m · X_s
        reference_img = source_image * mask[:, :, np.newaxis]

        # (1-m) · X_s
        masked_source = source_image * (1 - mask[:, :, np.newaxis])

        return masked_source, reference_img, mask
