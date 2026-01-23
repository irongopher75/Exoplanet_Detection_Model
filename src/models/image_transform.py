"""
2D Phase Folding transformer for exoplanet light curve data.
Transforms 1D time-series into 2D images for CNN input.
"""
import numpy as np

def phase_fold_2d(time, flux, period, n_segments=10, image_height=10, image_width=100):
    phase = ((time % period) / period)
    idx = np.argsort(phase)
    phase_sorted = phase[idx]
    flux_sorted = flux[idx]
    segment_size = len(time) // n_segments
    image = np.zeros((image_height, image_width))
    for i in range(n_segments):
        seg_idx = idx[i*segment_size:(i+1)*segment_size]
        seg_phase = phase_sorted[i*segment_size:(i+1)*segment_size]
        seg_flux = flux_sorted[i*segment_size:(i+1)*segment_size]
        # Rescale to image width
        x = (seg_phase * (image_width-1)).astype(int)
        y = np.full_like(x, i)
        image[y, x] = seg_flux
    return image
