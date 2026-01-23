"""
Instrument-aware preprocessing utilities for exoplanet light curve data.
Handles outlier removal, noise normalization, and simulates sensor aging.
"""
import numpy as np

def remove_outliers(flux, threshold=6):
    std = np.nanstd(flux)
    mean = np.nanmean(flux)
    mask = np.abs(flux - mean) > threshold * std
    flux[mask] = mean
    return flux

def normalize_noise(flux, missing_value=np.nan):
    flux = np.array(flux)
    isnan = np.isnan(flux)
    for i in np.where(isnan)[0]:
        left = flux[i-1] if i > 0 else 0
        right = flux[i+1] if i < len(flux)-1 else 0
        flux[i] = (left + right) / 2
    return flux

def simulate_sensor_aging(flux, noise_level=0.1):
    noise = np.random.normal(0, noise_level * np.nanstd(flux), size=flux.shape)
    return flux + noise
