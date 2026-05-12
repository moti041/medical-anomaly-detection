import numpy as np


def _normalize_per_image(spectrum):
    min_value = spectrum.min()
    max_value = spectrum.max()
    denom = max_value - min_value
    if denom <= 1e-12:
        return np.zeros_like(spectrum, dtype=np.float32)
    return ((spectrum - min_value) / denom).astype(np.float32)


def compute_log_spectrum(image, spectral_mode='full_spectrum', cutoff_ratio=0.25):
    """
    Compute a normalized log magnitude spectrum from the 2D FFT.

    full_spectrum keeps every FFT coefficient:
        log_spectrum = log(1 + abs(FFT2(image)))

    high_freq removes the low-frequency center with a circular high-pass mask.
    This can help texture anomaly experiments because subtle pathology can alter
    local edge/detail patterns that are less dominant in the low-frequency image
    structure.

    Args:
        image (np.ndarray): Grayscale image of shape (H, W)
        spectral_mode (str): 'full_spectrum' or 'high_freq'
        cutoff_ratio (float): Keep centered frequencies with normalized radius
            greater than this value when spectral_mode is 'high_freq'.

    Returns:
        np.ndarray: Normalized log spectrum of shape (H, W)
    """
    if spectral_mode not in {'full_spectrum', 'high_freq'}:
        raise ValueError(f"Unsupported spectral_mode '{spectral_mode}'")

    fft_shifted = np.fft.fftshift(np.fft.fft2(image))

    if spectral_mode == 'high_freq':
        if not 0 <= cutoff_ratio <= 1:
            raise ValueError('cutoff_ratio must be between 0 and 1')
        height, width = image.shape
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx ** 2 + yy ** 2)
        fft_shifted = fft_shifted * (radius > cutoff_ratio)

    log_spectrum = np.log1p(np.abs(fft_shifted))
    return _normalize_per_image(log_spectrum)
