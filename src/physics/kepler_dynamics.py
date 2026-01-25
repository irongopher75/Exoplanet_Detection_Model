"""
Keplerian orbital dynamics for exoplanet systems.

This module implements Kepler's laws and orbital mechanics for computing
planetary orbits and transit geometry.

Scientific Context:
    Exoplanet transits occur when planets pass in front of their host stars.
    The geometry depends on:
    - Orbital period (Kepler's third law)
    - Eccentricity and argument of periastron
    - Inclination and impact parameter
    - Stellar and planetary radii
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import fsolve


def keplers_third_law(
    period: float,
    stellar_mass: float,
    planet_mass: Optional[float] = None
) -> float:
    """
    Compute semi-major axis from period using Kepler's third law.
    
    a^3 / P^2 = G * (M_star + M_planet) / (4 * π^2)
    
    Parameters
    ----------
    period : float
        Orbital period (days).
    stellar_mass : float
        Stellar mass (solar masses).
    planet_mass : float, optional
        Planet mass (solar masses). If None, assumes M_planet << M_star.
    
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
    M_star_si = stellar_mass * M_sun  # kg
    
    if planet_mass is not None:
        M_planet_si = planet_mass * M_sun  # kg
        total_mass = M_star_si + M_planet_si
    else:
        total_mass = M_star_si  # Neglect planet mass
    
    # Kepler's third law
    a_si = (G * total_mass * P_si ** 2 / (4 * np.pi ** 2)) ** (1.0 / 3.0)
    
    # Convert to AU
    a_au = a_si / AU
    
    return a_au


def compute_impact_parameter(
    a_rs: float,
    inclination: float,
    eccentricity: float = 0.0,
    omega: float = 90.0
) -> float:
    """
    Compute impact parameter from orbital parameters.
    
    b = (a / R_star) * cos(i) * (1 - e^2) / (1 + e * sin(ω))
    
    Parameters
    ----------
    a_rs : float
        Semi-major axis in stellar radii.
    inclination : float
        Orbital inclination (degrees, 90 = edge-on).
    eccentricity : float
        Orbital eccentricity.
    omega : float
        Argument of periastron (degrees).
    
    Returns
    -------
    float
        Impact parameter (0 = central transit, 1 = grazing).
    """
    i_rad = np.radians(inclination)
    omega_rad = np.radians(omega)
    
    # For circular orbit
    if eccentricity == 0.0:
        b = a_rs * np.cos(i_rad)
    else:
        # Eccentric orbit
        b = a_rs * np.cos(i_rad) * (1 - eccentricity ** 2) / (1 + eccentricity * np.sin(omega_rad))
    
    return np.clip(b, 0.0, 1.0)


def compute_true_anomaly(
    mean_anomaly: float,
    eccentricity: float,
    tolerance: float = 1e-8
) -> float:
    """
    Solve Kepler's equation to find true anomaly.
    
    M = E - e * sin(E)
    tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    
    Parameters
    ----------
    mean_anomaly : float
        Mean anomaly (radians).
    eccentricity : float
        Orbital eccentricity.
    tolerance : float
        Convergence tolerance.
    
    Returns
    -------
    float
        True anomaly (radians).
    """
    if eccentricity == 0.0:
        # Circular orbit: true anomaly = mean anomaly
        return mean_anomaly
    
    # Solve Kepler's equation: M = E - e * sin(E)
    def kepler_equation(E):
        return E - eccentricity * np.sin(E) - mean_anomaly
    
    # Initial guess
    E0 = mean_anomaly if eccentricity < 0.8 else np.pi
    
    # Solve
    E = fsolve(kepler_equation, E0, xtol=tolerance)[0]
    
    # Convert to true anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(E / 2),
        np.sqrt(1 - eccentricity) * np.cos(E / 2)
    )
    
    return nu


def compute_orbital_separation(
    a_rs: float,
    true_anomaly: float,
    eccentricity: float = 0.0
) -> float:
    """
    Compute orbital separation at given true anomaly.
    
    r = a * (1 - e^2) / (1 + e * cos(ν))
    
    Parameters
    ----------
    a_rs : float
        Semi-major axis in stellar radii.
    true_anomaly : float
        True anomaly (radians).
    eccentricity : float
        Orbital eccentricity.
    
    Returns
    -------
    float
        Orbital separation in stellar radii.
    """
    if eccentricity == 0.0:
        return a_rs
    
    r = a_rs * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(true_anomaly))
    return r


def compute_transit_probability(
    a_rs: float,
    rp_rs: float
) -> float:
    """
    Compute geometric transit probability.
    
    P_transit = (R_star + R_planet) / a
    
    Parameters
    ----------
    a_rs : float
        Semi-major axis in stellar radii.
    rp_rs : float
        Planet-to-star radius ratio.
    
    Returns
    -------
    float
        Transit probability (0 to 1).
    """
    return (1.0 + rp_rs) / a_rs


def compute_orbital_velocity(
    period: float,
    a_rs: float,
    stellar_radius: float
) -> float:
    """
    Compute orbital velocity at semi-major axis.
    
    v = 2 * π * a / P
    
    Parameters
    ----------
    period : float
        Orbital period (days).
    a_rs : float
        Semi-major axis in stellar radii.
    stellar_radius : float
        Stellar radius (solar radii).
    
    Returns
    -------
    float
        Orbital velocity (km/s).
    """
    R_sun = 6.957e8  # meters
    day = 86400  # seconds
    
    # Convert to meters
    a_m = a_rs * stellar_radius * R_sun
    
    # Velocity
    v_m_s = 2 * np.pi * a_m / (period * day)
    
    # Convert to km/s
    v_km_s = v_m_s / 1000.0
    
    return v_km_s
