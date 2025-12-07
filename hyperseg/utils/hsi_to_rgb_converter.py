import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d
from typing import Union, Tuple, Optional


class HSIToRGBConverter:
    """
    Advanced HSI to RGB conversion with proper spectral response modeling.

    This converter uses realistic spectral response functions for RGB channels,
    incorporating human luminous efficiency and standard illuminant considerations.
    """

    def __init__(self, method: str = 'standard_rgb'):
        """
        Initialize the HSI to RGB converter.

        Args:
            method: Conversion method ('standard_rgb', 'cie_xyz', 'custom')
        """
        self.method = method
        self._setup_spectral_responses()

    def _setup_spectral_responses(self):
        """Setup spectral response functions for RGB channels."""
        # Standard RGB channel wavelength centers (nm)
        self.rgb_wavelengths = np.array([700, 546.1, 438.8])  # Red, Green, Blue

        # Create spectral response curves based on CIE color matching functions
        # These are simplified but more accurate than simple delta functions
        wavelengths_range = np.linspace(380, 780, 401)

        # RGB spectral response curves (normalized)
        self.red_response = self._create_gaussian_response(wavelengths_range, 700, 50)
        self.green_response = self._create_gaussian_response(wavelengths_range, 546.1, 60)
        self.blue_response = self._create_gaussian_response(wavelengths_range, 438.8, 40)

        # Apply photopic luminous efficiency (V(λ)) weighting
        self.luminous_efficiency = self._create_v_lambda(wavelengths_range)

        self.wavelengths_range = wavelengths_range

    def _create_gaussian_response(self, wavelengths: np.ndarray,
                                 center: float,
                                 sigma: float) -> np.ndarray:
        """Create Gaussian spectral response curve."""
        response = np.exp(-((wavelengths - center) ** 2) / (2 * sigma ** 2))
        return response / np.max(response)

    def _create_v_lambda(self, wavelengths: np.ndarray) -> np.ndarray:
        """Create photopic luminous efficiency function V(λ)."""
        # Simplified V(λ) approximation
        v_lambda = np.zeros_like(wavelengths)

        # Peak sensitivity at 555 nm
        peak_idx = np.argmin(np.abs(wavelengths - 555))

        # Create bell-shaped curve
        for i, wl in enumerate(wavelengths):
            if 380 <= wl <= 780:
                if wl < 555:
                    v_lambda[i] = np.exp(-((wl - 555) ** 2) / (2 * 70 ** 2))
                else:
                    v_lambda[i] = np.exp(-((wl - 555) ** 2) / (2 * 80 ** 2))

        return v_lambda / np.max(v_lambda)

    def convert_hsi_to_rgb(self,
                          hsi_data: torch.Tensor,
                          input_wavelengths: np.ndarray,
                          normalize: bool = True,
                          gamma_correction: bool = True,
                          white_balance: bool = True) -> torch.Tensor:
        """
        Convert HSI data to RGB using advanced spectral integration.

        Args:
            hsi_data: Input HSI data [C, H, W] or [B, C, H, W]
            input_wavelengths: Array of input wavelengths (nm)
            normalize: Whether to normalize output to [0, 255]
            gamma_correction: Whether to apply gamma correction
            white_balance: Whether to apply white balance

        Returns:
            RGB image tensor [3, H, W] or [B, 3, H, W]
        """
        if hsi_data.dim() == 3:
            hsi_data = hsi_data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, C, H, W = hsi_data.shape

        # Ensure wavelengths are within valid range
        valid_mask = (input_wavelengths >= 380) & (input_wavelengths <= 780)
        if not np.all(valid_mask):
            print(f"Warning: Some wavelengths outside visible range (380-780nm)")

        # Interpolate spectral responses to input wavelengths
        rgb_weights = self._compute_rgb_weights(input_wavelengths)

        # Apply weights to HSI data
        rgb_data = torch.einsum('rc,bchw->brhw', rgb_weights, hsi_data)

        # White balance
        if white_balance:
            rgb_data = self._apply_white_balance(rgb_data)

        # Gamma correction
        if gamma_correction:
            rgb_data = torch.pow(rgb_data, 1/2.2)

        # Normalize to [0, 255] if requested
        if normalize:
            rgb_data = torch.clamp(rgb_data, 0, 1)
            rgb_data = rgb_data * 255

        if squeeze_output:
            rgb_data = rgb_data.squeeze(0)

        return rgb_data

    def _compute_rgb_weights(self, input_wavelengths: np.ndarray) -> torch.Tensor:
        """Compute RGB channel weights for given input wavelengths."""
        # Clamp wavelengths to valid range
        input_wavelengths = np.clip(input_wavelengths, 380, 780)

        # Interpolate spectral responses
        interp_func = interp1d(self.wavelengths_range,
                              np.vstack([self.red_response,
                                       self.green_response,
                                       self.blue_response]),
                              axis=1,
                              bounds_error=False,
                              fill_value=0)

        rgb_weights = interp_func(input_wavelengths)

        # Apply luminous efficiency weighting
        luminous_weights = interp1d(self.wavelengths_range,
                                   self.luminous_efficiency,
                                   bounds_error=False,
                                   fill_value=0)(input_wavelengths)

        # Apply luminous efficiency to each channel
        for i in range(3):
            rgb_weights[i] *= luminous_weights

        # Normalize weights
        rgb_weights = rgb_weights / (rgb_weights.sum(axis=1, keepdims=True) + 1e-8)

        return torch.from_numpy(rgb_weights.astype(np.float32))

    def _apply_white_balance(self, rgb_data: torch.Tensor) -> torch.Tensor:
        """Apply white balance correction."""
        # Compute gray world assumption
        channel_means = rgb_data.mean(dim=[2, 3], keepdim=True)
        gray_mean = channel_means.mean(dim=1, keepdim=True)

        # Avoid division by zero
        channel_means = torch.clamp(channel_means, min=1e-8)

        # Apply white balance
        balance_factors = gray_mean / channel_means
        rgb_data = rgb_data * balance_factors

        return rgb_data


def interpolate_hyperspectral_image_transform_matrix_advanced(
    wave_lib: np.ndarray,
    target_wavelengths: Union[np.ndarray, torch.Tensor],
    method: str = 'cubic',
    use_spectral_responses: bool = True
) -> np.ndarray:
    """
    Advanced interpolation for hyperspectral to RGB conversion.

    Args:
        wave_lib: Input wavelengths
        target_wavelengths: Target RGB wavelengths [R, G, B] or batch [N, 3]
        method: Interpolation method ('linear', 'cubic', 'spectral')
        use_spectral_responses: Whether to use spectral response functions

    Returns:
        Transform matrix for conversion
    """
    if isinstance(target_wavelengths, torch.Tensor):
        target_wavelengths = target_wavelengths.cpu().numpy()

    if len(target_wavelengths.shape) == 1:
        target_wavelengths = target_wavelengths[None]

    # Initialize converter for spectral response method
    if use_spectral_responses:
        converter = HSIToRGBConverter()

    transform_matrices = []

    for target_set in target_wavelengths:
        transform_matrix = np.zeros((len(target_set), len(wave_lib)), dtype=np.float32)

        for i, target_wl in enumerate(target_set):
            if use_spectral_responses and method == 'spectral':
                # Use spectral response curves
                rgb_weights = converter._compute_rgb_weights(np.array([target_wl]))
                transform_matrix[i] = rgb_weights.numpy().flatten()
            else:
                # Standard interpolation
                if method == 'cubic':
                    f = interp1d(wave_lib, np.eye(len(wave_lib)),
                                axis=0, kind='cubic',
                                bounds_error=False, fill_value=0)
                else:  # linear
                    f = interp1d(wave_lib, np.eye(len(wave_lib)),
                                axis=0, kind='linear',
                                bounds_error=False, fill_value=0)

                # Find nearest wavelengths for interpolation
                if wave_lib[0] <= target_wl <= wave_lib[-1]:
                    transform_matrix[i] = f(target_wl)
                else:
                    # Handle out-of-bounds with nearest neighbor
                    nearest_idx = np.argmin(np.abs(wave_lib - target_wl))
                    transform_matrix[i, nearest_idx] = 1.0

        transform_matrices.append(transform_matrix)

    return np.stack(transform_matrices)


# Example usage and comparison function
def compare_hsi_to_rgb_methods(hsi_data: torch.Tensor,
                              wavelengths: np.ndarray) -> dict:
    """
    Compare different HSI to RGB conversion methods.

    Args:
        hsi_data: Input HSI data [C, H, W]
        wavelengths: Input wavelengths

    Returns:
        Dictionary with RGB images from different methods
    """
    results = {}

    # Method 1: Original simple interpolation
    from .spectral_process_utils import interpolate_hyperspectral_image_transform_matrix
    rgb_wavelengths = np.array([[700, 546.1, 438.8]])
    transform_matrix = interpolate_hyperspectral_image_transform_matrix(
        wavelengths, rgb_wavelengths
    )[0]
    rgb_original = torch.einsum('rc,chw->rhw',
                               torch.tensor(transform_matrix),
                               hsi_data)
    results['original'] = rgb_original

    # Method 2: Advanced spectral response
    converter = HSIToRGBConverter()
    rgb_advanced = converter.convert_hsi_to_rgb(hsi_data, wavelengths)
    results['advanced'] = rgb_advanced

    # Method 3: Advanced interpolation matrix
    transform_matrix_adv = interpolate_hyperspectral_image_transform_matrix_advanced(
        wavelengths, rgb_wavelengths,
        method='cubic', use_spectral_responses=True
    )[0]
    rgb_matrix = torch.einsum('rc,chw->rhw',
                             torch.tensor(transform_matrix_adv),
                             hsi_data)
    results['matrix'] = rgb_matrix

    return results