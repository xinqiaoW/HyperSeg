import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelProj(nn.Module):
    """Channel Projection module for hyperspectral image processing.

    Each wavelength maps to a learnable vector. Input spectral features are
    projected to embedding space via matrix multiplication with wavelength-specific vectors.

    Input: (B, N, D) where D is the number of spectral channels (varies per input)
    Wavelengths: list of D wavelengths corresponding to input channels
    Output: (B, N, E) where E is the embedding dimension
    """

    def __init__(self, embed_dim: int, preset_wavelengths: list, nm_dis: float = 10.0):
        """Initialize ChannelProj module.

        Args:
            embed_dim: Output embedding dimension (E).
            preset_wavelengths: List of preset wavelengths to create learnable vectors for.
            nm_dis: Distance threshold (nm) for matching input wavelengths to preset wavelengths.
        """
        super(ChannelProj, self).__init__()

        self.embed_dim = embed_dim
        self.preset_wavelengths = np.array(preset_wavelengths)
        self.nm_dis = nm_dis
        self.num_wavelengths = len(preset_wavelengths)

        # Learnable vectors for each preset wavelength: (num_wavelengths, embed_dim)
        self.wavelength_vectors = nn.Parameter(torch.randn(self.num_wavelengths, embed_dim) * 0.02)

    def find_matching_indices(self, input_wavelengths: list) -> list:
        """Find indices in preset_wavelengths that match input wavelengths.

        Args:
            input_wavelengths: List of input wavelengths.

        Returns:
            List of indices into preset_wavelengths for each input wavelength.
        """
        input_wl = np.array(input_wavelengths)
        # Compute distance matrix: (D_input, num_preset)
        distances = np.abs(input_wl[:, None] - self.preset_wavelengths[None, :])
        # Find closest preset wavelength for each input wavelength
        closest_indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(input_wl)), closest_indices]

        # Check if within threshold
        if np.any(min_distances > self.nm_dis):
            # For wavelengths outside threshold, still use closest match but warn
            pass

        return closest_indices.tolist()

    def forward(self, x: torch.Tensor, input_wavelengths: list, test_mode: bool = False) -> torch.Tensor:
        """Forward pass through the channel projection layer.

        Args:
            x: Input tensor of shape (B, N, D) where D is number of spectral channels.
            input_wavelengths: List of D wavelengths corresponding to input channels.
            test_mode: Unused, kept for API compatibility.

        Returns:
        
            Projected features of shape (B, N, E).
        """
        B, N, D = x.shape

        # Find matching preset wavelength indices for each input wavelength
        matching_indices = self.find_matching_indices(input_wavelengths)

        # Gather wavelength vectors for the input wavelengths: (D, E)
        projection_matrix = self.wavelength_vectors[matching_indices]  # (D, E)

        # Matrix multiplication: (B, N, D) @ (D, E) -> (B, N, E)
        output = torch.matmul(x, projection_matrix)

        return output
