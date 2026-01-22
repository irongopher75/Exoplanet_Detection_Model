"""
Classical baseline detection methods for exoplanet transits.

This module implements non-ML detection methods that serve as:
1. Baselines for ML model comparison
2. Sanity checks for detection pipelines
3. Initial parameter estimates for ML models

Methods:
    - Box Least Squares (BLS): Detects periodic box-shaped transits
    - Lomb-Scargle Periodogram: Detects periodic signals in unevenly sampled data

Scientific Context:
    Before deep learning, exoplanet detection relied on these classical methods.
    BLS is the workhorse of transit surveys (Kepler, TESS pipelines).
    Lomb-Scargle is used for period finding in RV and photometric data.

These baselines are required for:
    - Reviewer comparisons
    - Understanding ML model improvements
    - Debugging detection failures
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import signal
from scipy.optimize import minimize_scalar
import warnings


def box_least_squares(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    period_min: float = 0.5,
    period_max: float = 100.0,
    n_periods: int = 10000,
    n_bins: int = 200
) -> Dict[str, Any]:
    """
    Box Least Squares (BLS) periodogram for transit detection.
    
    BLS searches for periodic box-shaped transits by fitting a box model
    at each trial period and selecting the best fit.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Flux measurements.
    flux_err : np.ndarray, optional
        Flux uncertainties.
    period_min : float
        Minimum period to search (days).
    period_max : float
        Maximum period to search (days).
    n_periods : int
        Number of trial periods.
    n_bins : int
        Number of phase bins for box model.
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary with:
        - 'period': Best-fit period (days)
        - 'power': BLS power (signal-to-noise)
        - 'depth': Transit depth
        - 't0': Time of mid-transit
        - 'duration': Transit duration (fraction of period)
        - 'periodogram': Full periodogram
    """
    # Normalize flux
    flux_mean = np.nanmean(flux)
    flux_norm = flux / flux_mean - 1.0  # Center around 0
    
    # Remove NaN/inf
    valid = np.isfinite(time) & np.isfinite(flux_norm)
    time = time[valid]
    flux_norm = flux_norm[valid]
    if flux_err is not None:
        flux_err = flux_err[valid]
    
    # Generate trial periods (log-spaced)
    periods = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)
    
    # Compute BLS periodogram
    best_power = -np.inf
    best_period = None
    best_t0 = None
    best_depth = None
    best_duration = None
    power_spectrum = np.zeros(n_periods)
    
    for i, period in enumerate(periods):
        # Phase fold
        phase = ((time / period) % 1.0)
        
        # Try different phase offsets
        n_offsets = min(50, int(period / np.median(np.diff(time))))
        max_power = -np.inf
        
        for offset in np.linspace(0, 1, n_offsets):
            phase_shifted = (phase + offset) % 1.0
            
            # Bin data
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            bin_indices = np.digitize(phase_shifted, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Compute mean flux in each bin
            bin_flux = np.array([
                np.nanmean(flux_norm[bin_indices == j]) 
                for j in range(n_bins)
            ])
            bin_counts = np.array([
                np.sum(bin_indices == j) 
                for j in range(n_bins)
            ])
            
            # Find transit (minimum flux bin)
            transit_bin = np.nanargmin(bin_flux)
            transit_depth = -bin_flux[transit_bin]
            
            # Compute BLS power (simplified)
            # Power = (depth^2 * n_in_transit) / variance
            in_transit = bin_indices == transit_bin
            n_in = np.sum(in_transit)
            
            if n_in > 0 and transit_depth > 0:
                # Simplified power calculation
                variance = np.nanvar(flux_norm)
                if variance > 0:
                    power = (transit_depth ** 2 * n_in) / variance
                else:
                    power = 0.0
                
                if power > max_power:
                    max_power = power
                    if power > best_power:
                        best_power = power
                        best_period = period
                        best_t0 = time[np.argmin(phase_shifted)] if len(time) > 0 else 0.0
                        best_depth = transit_depth * flux_mean
                        best_duration = (1.0 / n_bins)  # Simplified
        
        power_spectrum[i] = max_power
    
    return {
        'period': best_period,
        'power': best_power,
        'depth': best_depth,
        't0': best_t0,
        'duration': best_duration,
        'periodogram': {
            'periods': periods,
            'power': power_spectrum
        }
    }


def lomb_scargle_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    period_min: float = 0.5,
    period_max: float = 100.0,
    n_periods: int = 10000
) -> Dict[str, Any]:
    """
    Lomb-Scargle periodogram for periodic signal detection.
    
    The Lomb-Scargle periodogram is optimal for detecting sinusoidal signals
    in unevenly sampled data. It's widely used in astronomy for period finding.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Flux measurements.
    flux_err : np.ndarray, optional
        Flux uncertainties.
    period_min : float
        Minimum period to search (days).
    period_max : float
        Maximum period to search (days).
    n_periods : int
        Number of trial periods.
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary with:
        - 'period': Best-fit period (days)
        - 'power': Lomb-Scargle power
        - 'periodogram': Full periodogram
    """
    # Normalize flux
    flux_mean = np.nanmean(flux)
    flux_norm = flux / flux_mean - 1.0
    
    # Remove NaN/inf
    valid = np.isfinite(time) & np.isfinite(flux_norm)
    time = time[valid]
    flux_norm = flux_norm[valid]
    if flux_err is not None:
        flux_err = flux_err[valid]
        weights = 1.0 / (flux_err ** 2)
    else:
        weights = None
    
    # Generate trial frequencies (log-spaced)
    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_periods)
    
    # Compute Lomb-Scargle periodogram
    try:
        power = signal.lombscargle(time, flux_norm, frequencies)
    except Exception as e:
        warnings.warn(f"Lomb-Scargle computation failed: {e}")
        power = np.zeros(n_periods)
    
    # Find best period
    best_idx = np.argmax(power)
    best_period = 1.0 / frequencies[best_idx]
    best_power = power[best_idx]
    
    return {
        'period': best_period,
        'power': best_power,
        'periodogram': {
            'periods': 1.0 / frequencies,
            'power': power
        }
    }


def detect_transit_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect transits using BLS method.
    
    Convenience wrapper around box_least_squares with default parameters
    tuned for exoplanet detection.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    flux : np.ndarray
        Flux measurements.
    flux_err : np.ndarray, optional
        Flux uncertainties.
    **kwargs
        Additional arguments passed to box_least_squares.
    
    Returns
    -------
    Dict[str, Any]
        Detection results with period, depth, t0, etc.
    """
    # Default parameters for exoplanet detection
    defaults = {
        'period_min': 0.5,
        'period_max': min(100.0, (time.max() - time.min()) / 2.0),
        'n_periods': 10000,
        'n_bins': 200
    }
    defaults.update(kwargs)
    
    return box_least_squares(time, flux, flux_err, **defaults)


def compute_snr(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray],
    transit_model: np.ndarray
) -> float:
    """
    Compute signal-to-noise ratio for a transit detection.
    
    SNR = depth / (noise / sqrt(n_transit))
    
    Parameters
    ----------
    time : np.ndarray
        Observation times.
    flux : np.ndarray
        Observed flux.
    flux_err : np.ndarray, optional
        Flux uncertainties.
    transit_model : np.ndarray
        Model flux (1.0 = no transit, <1.0 = transit).
    
    Returns
    -------
    float
        Signal-to-noise ratio.
    """
    # Compute depth
    depth = 1.0 - np.min(transit_model)
    
    # Identify in-transit points
    in_transit = transit_model < 0.99  # Threshold for transit
    n_transit = np.sum(in_transit)
    
    if n_transit == 0:
        return 0.0
    
    # Compute noise
    if flux_err is not None:
        # Use provided errors
        noise = np.nanmean(flux_err[in_transit])
    else:
        # Estimate from out-of-transit scatter
        out_transit = ~in_transit
        if np.sum(out_transit) > 0:
            noise = np.nanstd(flux[out_transit])
        else:
            noise = np.nanstd(flux)
    
    if noise == 0:
        return np.inf if depth > 0 else 0.0
    
    # SNR
    snr = depth / (noise / np.sqrt(n_transit))
    
    return snr
