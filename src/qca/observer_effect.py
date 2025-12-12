import torch
import numpy as np
import logging
from typing import Tuple, Dict, Optional, Union

class ObserverEffect:
    """
    Manages the 'Observer Effect' in Aetheria 5.0.

    Implements the distinction between:
    1. Superposition (Fog/Unobserved): Represented by low-resolution statistical moments (mu, sigma).
    2. Reality (Collapsed/Observed): Represented by high-resolution concrete quantum states.

    This system optimizes resources by only simulating the full physics in the observed viewport,
    while maintaining a coherent background evolution.
    """

    def __init__(self,
                 grid_size: int,
                 d_state: int,
                 device: torch.device,
                 downscale_factor: int = 8):
        """
        Args:
            grid_size: Full grid dimension (H, W).
            d_state: Number of state channels (dimensions).
            device: Computation device (CPU/CUDA).
            downscale_factor: Factor by which the background state is smaller than the full grid.
        """
        self.grid_size = grid_size
        self.d_state = d_state
        self.device = device
        self.downscale_factor = downscale_factor

        # Dimensions for the background "Fog"
        self.bg_size = grid_size // downscale_factor

        # Background State: Stores Mean (mu) and Variance (sigma) for each channel
        # Shape: [1, bg_size, bg_size, d_state * 2] (concatenated mu and sigma)
        # Initialized with some structure/noise
        self.background_stats = self._initialize_background()

        # Active State: The currently collapsed high-res reality
        # This is transient and depends on the viewport.
        # For simplicity in this v1, we might keep a full buffer but only update parts of it,
        # or actually allocate chunks. Let's use a full buffer for simplicity but track validity.
        self.collapsed_state = torch.zeros(1, grid_size, grid_size, d_state,
                                           dtype=torch.complex64, device=device)

        # Mask indicating which pixels are currently "Collapsed" (valid in high-res)
        self.collapse_mask = torch.zeros(1, grid_size, grid_size, dtype=torch.bool, device=device)

        self.last_viewport = None

    def _initialize_background(self) -> torch.Tensor:
        """Initializes the background statistical state."""
        # Random mean [-1, 1]
        mu = torch.randn(1, self.bg_size, self.bg_size, self.d_state, device=self.device)
        # Variance [0, 1]
        sigma = torch.rand(1, self.bg_size, self.bg_size, self.d_state, device=self.device)

        return torch.cat([mu, sigma], dim=-1)

    def get_viewport_state(self, viewport: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Retrieves the state for the given viewport, triggering a collapse if necessary.

        Args:
            viewport: (x_min, y_min, x_max, y_max)

        Returns:
            Tensor containing the high-res state for the viewport.
        """
        x_min, y_min, x_max, y_max = viewport

        # Define slice
        # Ensure bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(self.grid_size, x_max), min(self.grid_size, y_max)

        # Check which parts of the viewport are NOT collapsed yet
        current_mask_slice = self.collapse_mask[..., y_min:y_max, x_min:x_max]

        if not torch.all(current_mask_slice):
            self._collapse_region(x_min, y_min, x_max, y_max)

        # Return the slice
        return self.collapsed_state[..., y_min:y_max, x_min:x_max, :]

    def _collapse_region(self, x_min, y_min, x_max, y_max):
        """
        Performs the 'Wavefunction Collapse' (Sampling) from Background stats to High-Res state.
        Only applies to pixels that are not yet collapsed.
        """
        # Identify pixels needing collapse
        # (In a real optimized engine, we would do this chunk-based, not pixel-mask based)
        mask_slice = self.collapse_mask[..., y_min:y_max, x_min:x_max]
        needs_collapse = ~mask_slice

        if not torch.any(needs_collapse):
            return

        logging.info(f"👁️ Observer Effect: Collapsing region [{x_min}:{x_max}, {y_min}:{y_max}]")

        # 1. Upsample Background Stats to match High-Res region
        # We take the corresponding region in background
        bg_x_min, bg_y_min = x_min // self.downscale_factor, y_min // self.downscale_factor
        bg_x_max, bg_y_max = (x_max // self.downscale_factor) + 1, (y_max // self.downscale_factor) + 1

        # Clip
        bg_x_max = min(self.bg_size, bg_x_max)
        bg_y_max = min(self.bg_size, bg_y_max)

        # Extract stats
        stats_chunk = self.background_stats[..., bg_y_min:bg_y_max, bg_x_min:bg_x_max, :]

        # Upsample using bilinear interpolation
        # Need to reshape for grid_sample or interpolate: [B, C, H, W]
        stats_permuted = stats_chunk.permute(0, 3, 1, 2)

        target_h = y_max - y_min
        target_w = x_max - x_min

        upsampled_stats = torch.nn.functional.interpolate(
            stats_permuted,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

        # Split back into mu and sigma
        upsampled_stats = upsampled_stats.permute(0, 2, 3, 1) # [1, H, W, 2*d]
        mu = upsampled_stats[..., :self.d_state]
        sigma = upsampled_stats[..., self.d_state:]

        # 2. Sample from distribution: State = mu + sigma * Noise * Phase
        # We generate a coherent complex noise
        noise_real = torch.randn_like(mu)
        noise_imag = torch.randn_like(mu)

        # Construct the collapsed state
        # Real part = mu + sigma * noise
        # Imag part = sigma * noise (assuming mu is magnitude-like or real-centric)
        # OR better:
        # Magnitude ~ Normal(mu, sigma)
        # Phase ~ Uniform(0, 2pi) or correlated

        # Let's use simple additive noise for now
        real_part = mu + sigma * noise_real
        imag_part = sigma * noise_imag # Assuming mean imag is 0 for simplicity, or we could store mu_imag

        new_state_slice = torch.complex(real_part, imag_part)

        # 3. Write to Collapsed State (only where needed)
        # For simplicity, we overwrite the whole slice. In perfect implementation, we blend.
        current_slice = self.collapsed_state[..., y_min:y_max, x_min:x_max, :]

        # Update only uncollapsed pixels
        # Expand mask to dimensions
        needs_collapse_expanded = needs_collapse.unsqueeze(-1).expand_as(new_state_slice)

        current_slice[needs_collapse_expanded] = new_state_slice[needs_collapse_expanded]

        # Update Mask
        self.collapse_mask[..., y_min:y_max, x_min:x_max] = True

    def update_background(self):
        """
        Evolution of the 'Fog' (Unobserved Universe).
        Simple diffusion or cellular automata on the low-res stats.
        """
        # Example: Simple diffusion on Mu
        mu = self.background_stats[..., :self.d_state]
        sigma = self.background_stats[..., self.d_state:]

        # Simple laplacian diffusion approximation (very fast)
        # Roll shift
        mu_up = torch.roll(mu, 1, dims=1)
        mu_down = torch.roll(mu, -1, dims=1)
        mu_left = torch.roll(mu, 1, dims=2)
        mu_right = torch.roll(mu, -1, dims=2)

        laplacian = (mu_up + mu_down + mu_left + mu_right - 4*mu)

        # Diffuse
        mu_new = mu + 0.01 * laplacian

        # Update stats
        self.background_stats[..., :self.d_state] = mu_new

    def decay_observations(self, decay_rate=0.05):
        """
        Slowly returns unobserved collapsed regions back to the Fog.
        Use this if we track 'last_seen_time'.
        For now, we can just randomly un-collapse edges or similar.
        """
        pass

    def unobserve(self, roi: Tuple[int, int, int, int]):
        """
        Explicitly 'unobserves' a region, allowing it to return to the Fog (superposition).
        The current state in this region is integrated back into the background statistics
        before being cleared from the collapsed state.

        Args:
            roi: (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = roi

        # Clip to grid
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(self.grid_size, x_max), min(self.grid_size, y_max)

        # 1. Update Background Stats from Current Reality
        # We need to downsample the collapsed state to update the background
        # Calculate moments (mu, sigma) from the high-res data

        collapsed_slice = self.collapsed_state[..., y_min:y_max, x_min:x_max, :]
        mask_slice = self.collapse_mask[..., y_min:y_max, x_min:x_max]

        if not torch.any(mask_slice):
            return

        # We only care about valid pixels
        # Calculate local mean/std. Since downscale factor is 8, we have blocks of 8x8.
        # We can use avg_pool2d to get the mean for the background grid

        # We need to match the background grid alignment.
        # Strategy: Only unobserve fully aligned blocks.
        # Any partial blocks at the edges remain collapsed (part of reality) until the ROI shifts enough to clear them.

        # Calculate aligned boundaries (inclusive of start, exclusive of end)
        bg_x_min = (x_min + self.downscale_factor - 1) // self.downscale_factor
        bg_y_min = (y_min + self.downscale_factor - 1) // self.downscale_factor
        bg_x_max = x_max // self.downscale_factor
        bg_y_max = y_max // self.downscale_factor

        # Convert back to high-res coordinates to see what we are actually processing
        aligned_x_min = bg_x_min * self.downscale_factor
        aligned_y_min = bg_y_min * self.downscale_factor
        aligned_x_max = bg_x_max * self.downscale_factor
        aligned_y_max = bg_y_max * self.downscale_factor

        # Ensure we have a valid region
        if aligned_x_max > aligned_x_min and aligned_y_max > aligned_y_min:
            # Extract high-res region corresponding to these BG cells
            high_res_block = self.collapsed_state[..., aligned_y_min:aligned_y_max, aligned_x_min:aligned_x_max, :]

            # Calculate mean (downsample)
            # Permute to [B, C, H, W] for pooling
            block_tensor = high_res_block.real.permute(0, 3, 1, 2)

            new_mu = torch.nn.functional.avg_pool2d(
                block_tensor,
                kernel_size=self.downscale_factor,
                stride=self.downscale_factor
            )

            # Calculate variance (std)
            # Var = E[X^2] - (E[X])^2
            block_sq = block_tensor ** 2
            new_sq_mean = torch.nn.functional.avg_pool2d(
                block_sq,
                kernel_size=self.downscale_factor,
                stride=self.downscale_factor
            )
            new_sigma = torch.sqrt(torch.abs(new_sq_mean - new_mu**2) + 1e-6)

            # Permute back to [1, H_bg, W_bg, C]
            new_mu = new_mu.permute(0, 2, 3, 1)
            new_sigma = new_sigma.permute(0, 2, 3, 1)

            # Update Background Stats
            self.background_stats[..., bg_y_min:bg_y_max, bg_x_min:bg_x_max, :self.d_state] = new_mu
            self.background_stats[..., bg_y_min:bg_y_max, bg_x_min:bg_x_max, self.d_state:] = new_sigma

            # Clear Collapsed State ONLY for the processed region
            self.collapsed_state[..., aligned_y_min:aligned_y_max, aligned_x_min:aligned_x_max, :] = 0
            self.collapse_mask[..., aligned_y_min:aligned_y_max, aligned_x_min:aligned_x_max] = False

            logging.info(f"🌫️ Observer Effect: Unobserved aligned region [{aligned_x_min}:{aligned_x_max}, {aligned_y_min}:{aligned_y_max}] -> Returned to Fog.")
        else:
            logging.debug("Skipping unobserve: Region too small to cover full background blocks.")

        logging.info(f"🌫️ Observer Effect: Unobserved region [{x_min}:{x_max}, {y_min}:{y_max}] -> Returned to Fog.")
