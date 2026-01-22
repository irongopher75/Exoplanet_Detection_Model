"""
Transit model physics for exoplanet light curves.

This module implements the analytical transit model based on Mandel & Agol (2002),
which describes the flux dimming during a planetary transit.

Scientific Context:
    When a planet passes in front of its host star, it blocks a fraction of
    the stellar light, causing a periodic dip in brightness. The depth and
    shape of this dip encode:
    - Planet-to-star radius ratio (Rp/Rs)
    - Orbital period (P)
    - Impact parameter (b)
    - Stellar limb darkening coefficients

The Mandel & Agol model is the gold standard for transit fitting and is
required for both classical detection methods and ML-based approaches.

References:
    Mandel & Agol (2002), ApJ, 580, L171
"""

import numpy as np
from typing import Optional, Tuple
from scipy.special import ellipkinc, ellipeinc
import warnings


def quadratic_limb_darkening(mu: np.ndarray, u1: float, u2: float) -> np.ndarray:
    """
    Quadratic limb darkening law.
    
    I(mu) / I(1) = 1 - u1*(1-mu) - u2*(1-mu)^2
    
    Parameters
    ----------
    mu : np.ndarray
        Cosine of angle from stellar center (0 to 1).
    u1 : float
        First limb darkening coefficient.
    u2 : float
        Second limb darkening coefficient.
    
    Returns
    -------
    np.ndarray
        Limb darkening factor.
    """
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


def compute_transit_depth(
    rp_rs: float,
    b: float,
    u1: float = 0.3,
    u2: float = 0.3
) -> float:
    """
    Compute transit depth (maximum flux decrease) for given parameters.
    
    Parameters
    ----------
    rp_rs : float
        Planet-to-star radius ratio.
    b : float
        Impact parameter (0 = central transit, 1 = grazing).
    u1 : float
        First limb darkening coefficient.
    u2 : float
        Second limb darkening coefficient.
    
    Returns
    -------
    float
        Transit depth (fractional flux decrease).
    """
    # For small planets and central transits, depth ≈ (Rp/Rs)^2
    # For limb-darkened transits, this is more complex
    # Simplified calculation
    if b + rp_rs >= 1.0:
        # Grazing transit
        return 0.0
    
    # Approximate depth (exact calculation requires full Mandel & Agol)
    depth_approx = rp_rs ** 2 * (1.0 - u1 / 3.0 - u2 / 6.0)
    
    return depth_approx


def mandel_agol_transit(
    time: np.ndarray,
    period: float,
    t0: float,
    rp_rs: float,
    a_rs: float,
    b: float,
    u1: float = 0.3,
    u2: float = 0.3,
    eccentricity: float = 0.0,
    omega: float = 90.0
) -> np.ndarray:
    """
    Compute Mandel & Agol (2002) transit model.
    
    This is a simplified version. For production use, consider using
    batman (Kreidberg 2015) or similar packages for full accuracy.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days).
    period : float
        Orbital period (days).
    t0 : float
        Time of mid-transit (days).
    rp_rs : float
        Planet-to-star radius ratio.
    a_rs : float
        Semi-major axis in stellar radii.
    b : float
        Impact parameter (0 = central, 1 = grazing).
    u1 : float
        First limb darkening coefficient.
    u2 : float
        Second limb darkening coefficient.
    eccentricity : float
        Orbital eccentricity.
    omega : float
        Argument of periastron (degrees).
    
    Returns
    -------
    np.ndarray
        Relative flux (1.0 = no transit, <1.0 = transit).
    """
    # Phase calculation
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # Center phase around 0
    
    # Orbital separation (simplified, assumes circular for now)
    if eccentricity > 0:
        # Eccentric orbit (simplified)
        true_anomaly = 2 * np.pi * phase
        r_sep = a_rs * (1.0 - eccentricity ** 2) / (1.0 + eccentricity * np.cos(true_anomaly))
    else:
        r_sep = a_rs
    
    # Projected separation
    z = np.sqrt((r_sep * np.sin(2 * np.pi * phase)) ** 2 + b ** 2)
    
    # Compute flux using simplified Mandel & Agol
    flux = np.ones_like(time)
    
    # Transit occurs when z < 1 + rp_rs
    in_transit = z < (1.0 + rp_rs)
    
    if np.any(in_transit):
        # Simplified calculation (full version requires elliptic integrals)
        # For production, use batman package
        z_transit = z[in_transit]
        
        # Cases: full occultation, partial occultation, no occultation
        for i, zi in enumerate(z_transit):
            idx = np.where(in_transit)[0][i]
            
            if zi <= 1.0 - rp_rs:
                # Planet fully inside star (full occultation)
                # Use small planet approximation with limb darkening
                flux[idx] = 1.0 - compute_transit_depth(rp_rs, b, u1, u2)
            elif zi < 1.0 + rp_rs:
                # Partial occultation (planet partially overlapping star)
                # Simplified calculation
                overlap = _compute_overlap(zi, rp_rs)
                flux[idx] = 1.0 - overlap * compute_transit_depth(rp_rs, b, u1, u2)
            # else: zi >= 1.0 + rp_rs, no transit (already handled by in_transit mask)
    
    return flux


def _compute_overlap(z: float, rp_rs: float) -> float:
    """
    Compute overlap fraction between planet and star.
    
    Simplified calculation for partial occultation.
    Full version requires elliptic integrals.
    
    Parameters
    ----------
    z : float
        Projected separation.
    rp_rs : float
        Planet-to-star radius ratio.
    
    Returns
    -------
    float
        Overlap fraction.
    """
    if z >= 1.0 + rp_rs:
        return 0.0
    elif z <= 1.0 - rp_rs:
        return 1.0
    else:
        # Partial overlap (simplified)
        # Full calculation requires elliptic integrals
        overlap = (rp_rs ** 2 - (z - 1.0) ** 2) / (rp_rs ** 2)
        return np.clip(overlap, 0.0, 1.0)


def compute_transit_duration(
    period: float,
    a_rs: float,
    rp_rs: float,
    b: float,
    eccentricity: float = 0.0
) -> float:
    """
    Compute transit duration (T14) from orbital parameters.
    
    T14 = (P / π) * arcsin(sqrt((1 + Rp/Rs)^2 - b^2) / (a/Rs))
    
    Parameters
    ----------
    period : float
        Orbital period (days).
    a_rs : float
        Semi-major axis in stellar radii.
    rp_rs : float
        Planet-to-star radius ratio.
    b : float
        Impact parameter.
    eccentricity : float
        Orbital eccentricity (simplified calculation).
    
    Returns
    -------
    float
        Transit duration (days).
    """
    # Simplified for circular orbit
    if eccentricity > 0:
        warnings.warn("Eccentricity not fully accounted for in duration calculation")
    
    # Argument of arcsin
    arg = np.sqrt((1.0 + rp_rs) ** 2 - b ** 2) / a_rs
    
    if arg >= 1.0:
        # Grazing transit or invalid parameters
        return 0.0
    
    duration = (period / np.pi) * np.arcsin(arg)
    
    return duration


def keplers_third_law(
    period: float,
    stellar_mass: float
) -> float:
    """
    Compute semi-major axis from period using Kepler's third law.
    
    a^3 / P^2 = G * M_star / (4 * π^2)
    
    Parameters
    ----------
    period : float
        Orbital period (days).
    stellar_mass : float
        Stellar mass (solar masses).
    
    Returns
    -------
    float
        Semi-major axis (AU).
    """
    # Constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M_sun = 1.989e30  # kg
    AU = 1.496e11  # m
    day = 86400  # seconds
    
    # Convert to SI
    P_si = period * day  # seconds
    M_si = stellar_mass * M_sun  # kg
    
    # Kepler's third law
    a_si = (G * M_si * P_si ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    
    # Convert to AU
    a_au = a_si / AU
    
    return a_au


def convert_a_au_to_a_rs(a_au: float, stellar_radius: float) -> float:
    """
    Convert semi-major axis from AU to stellar radii.
    
    Parameters
    ----------
    a_au : float
        Semi-major axis (AU).
    stellar_radius : float
        Stellar radius (solar radii).
    
    Returns
    -------
    float
        Semi-major axis in stellar radii.
    """
    R_sun = 6.957e8  # meters
    AU = 1.496e11  # meters
    
    a_rs = (a_au * AU) / (stellar_radius * R_sun)
    
    return a_rs
