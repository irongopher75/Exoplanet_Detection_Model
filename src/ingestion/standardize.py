"""
Data standardization module for unified light curve format.

This module converts mission-specific light curve formats into a single
canonical representation that ML models can consume without knowing
the source mission.

Scientific Context:
    Different missions use different:
    - Time units (BJD, BTJD, JD)
    - Cadences (30 min, 2 min, daily)
    - Quality flag conventions
    - Flux normalization

For ML models (especially PINNs), we need a unified format where:
    - Time is in a consistent unit (days)
    - Flux is normalized appropriately
    - Quality flags are standardized
    - Metadata is preserved but normalized
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class StandardizedLightCurve:
    """
    Canonical light curve representation.
    
    This is the unified format used throughout the pipeline.
    All missions are converted to this format before ML processing.
    """
    time: np.ndarray  # Days, relative to first observation
    flux: np.ndarray  # Normalized flux (typically median=1.0)
    flux_err: Optional[np.ndarray] = None  # Flux uncertainties
    quality: Optional[np.ndarray] = None  # Quality flags (0=good, non-zero=bad)
    mission: str = "unknown"  # Source mission
    cadence: Optional[float] = None  # Median cadence in days
    metadata: Dict[str, Any] = field(default_factory=dict)  # Preserved metadata
    
    def __post_init__(self):
        """Validate standardized format."""
        assert len(self.time) == len(self.flux), "Time and flux must have same length"
        if self.flux_err is not None:
            assert len(self.flux_err) == len(self.time), "Flux errors must match time length"
        if self.quality is not None:
            assert len(self.quality) == len(self.time), "Quality flags must match time length"


class LightCurveStandardizer:
    """
    Converts mission-specific light curves to standardized format.
    
    Handles time unit conversion, flux normalization, and quality flag
    standardization across Kepler, TESS, and other missions.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize standardizer.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def standardize(
        self,
        light_curve: Dict[str, Any],
        mission: Optional[str] = None,
        normalize_flux: bool = True,
        remove_outliers: bool = False
    ) -> StandardizedLightCurve:
        """
        Convert mission-specific light curve to standardized format.
        
        Parameters
        ----------
        light_curve : Dict[str, Any]
            Mission-specific light curve with keys: time, flux, flux_err, quality, metadata
        mission : str, optional
            Mission name. If None, inferred from metadata.
        normalize_flux : bool
            Whether to normalize flux to median=1.0
        remove_outliers : bool
            Whether to remove extreme outliers (optional preprocessing)
        
        Returns
        -------
        StandardizedLightCurve
            Standardized light curve in canonical format.
        """
        # Extract arrays
        time = np.asarray(light_curve['time'])
        flux = np.asarray(light_curve['flux'])
        flux_err = light_curve.get('flux_err')
        quality = light_curve.get('quality')
        metadata = light_curve.get('metadata', {})
        
        # Determine mission
        if mission is None:
            mission = metadata.get('mission', 'unknown')
        
        # Step 1: Standardize time (convert to days, relative to first observation)
        time_standardized = self._standardize_time(time, mission, metadata)
        
        # Step 2: Standardize quality flags
        quality_standardized = self._standardize_quality(quality, mission, len(time))
        
        # Step 3: Apply quality mask to data
        if quality_standardized is not None:
            good_mask = (quality_standardized == 0)
            time_standardized = time_standardized[good_mask]
            flux = flux[good_mask]
            if flux_err is not None:
                flux_err = flux_err[good_mask]
            quality_standardized = quality_standardized[good_mask]
        
        # Step 4: Remove NaN/inf values
        valid_mask = np.isfinite(flux) & np.isfinite(time_standardized)
        if flux_err is not None:
            valid_mask = valid_mask & np.isfinite(flux_err)
        
        time_standardized = time_standardized[valid_mask]
        flux = flux[valid_mask]
        if flux_err is not None:
            flux_err = flux_err[valid_mask]
        if quality_standardized is not None:
            quality_standardized = quality_standardized[valid_mask]
        
        # Step 5: Remove outliers (optional)
        if remove_outliers:
            outlier_mask = self._detect_outliers(flux)
            time_standardized = time_standardized[~outlier_mask]
            flux = flux[~outlier_mask]
            if flux_err is not None:
                flux_err = flux_err[~outlier_mask]
            if quality_standardized is not None:
                quality_standardized = quality_standardized[~outlier_mask]
        
        # Step 6: Normalize flux
        if normalize_flux:
            flux, flux_err = self._normalize_flux(flux, flux_err)
        
        # Step 7: Calculate cadence
        if len(time_standardized) > 1:
            cadence = np.median(np.diff(time_standardized))
        else:
            cadence = None
        
        # Step 8: Create standardized light curve
        standardized = StandardizedLightCurve(
            time=time_standardized,
            flux=flux,
            flux_err=flux_err,
            quality=quality_standardized,
            mission=mission,
            cadence=cadence,
            metadata=metadata
        )
        
        return standardized
    
    def _standardize_time(
        self,
        time: np.ndarray,
        mission: str,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Convert time to standardized format (days, relative to first observation).
        
        Parameters
        ----------
        time : np.ndarray
            Original time array.
        mission : str
            Mission name.
        metadata : Dict[str, Any]
            Metadata dictionary.
        
        Returns
        -------
        np.ndarray
            Standardized time array (days, relative to first observation).
        """
        time = np.asarray(time)
        
        # Time is already in days for Kepler and TESS
        # Just make it relative to first observation
        time_standardized = time - time[0]
        
        return time_standardized
    
    def _standardize_quality(
        self,
        quality: Optional[np.ndarray],
        mission: str,
        length: int
    ) -> Optional[np.ndarray]:
        """
        Standardize quality flags (0 = good, non-zero = bad).
        
        Parameters
        ----------
        quality : np.ndarray, optional
            Original quality flags.
        mission : str
            Mission name.
        length : int
            Expected length of quality array.
        
        Returns
        -------
        np.ndarray, optional
            Standardized quality flags (0=good, non-zero=bad).
        """
        if quality is None:
            return None
        
        quality = np.asarray(quality)
        
        # Both Kepler and TESS use 0 = good data
        # So no conversion needed, but ensure it's the right shape
        if len(quality) != length:
            self.logger.warning(
                f"Quality array length ({len(quality)}) doesn't match data length ({length})"
            )
            return None
        
        return quality
    
    def _normalize_flux(
        self,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Normalize flux to median=1.0.
        
        Parameters
        ----------
        flux : np.ndarray
            Original flux array.
        flux_err : np.ndarray, optional
            Original flux errors.
        
        Returns
        -------
        tuple
            (normalized_flux, normalized_flux_err)
        """
        # Use median for robustness
        flux_median = np.nanmedian(flux)
        
        if flux_median == 0 or not np.isfinite(flux_median):
            self.logger.warning("Cannot normalize flux: median is zero or invalid")
            return flux, flux_err
        
        # Normalize flux
        flux_normalized = flux / flux_median
        
        # Normalize errors proportionally
        flux_err_normalized = None
        if flux_err is not None:
            flux_err_normalized = flux_err / flux_median
        
        return flux_normalized, flux_err_normalized
    
    def _detect_outliers(self, flux: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
        """
        Detect extreme outliers using robust statistics.
        
        Parameters
        ----------
        flux : np.ndarray
            Flux array.
        n_sigma : float
            Number of standard deviations for outlier threshold.
        
        Returns
        -------
        np.ndarray
            Boolean mask: True for outliers.
        """
        # Use median and MAD (median absolute deviation) for robustness
        flux_median = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - flux_median))
        
        # Convert MAD to approximate standard deviation
        sigma = 1.4826 * mad  # Factor for normal distribution
        
        # Identify outliers
        outlier_mask = np.abs(flux - flux_median) > n_sigma * sigma
        
        return outlier_mask


def load_and_standardize(file_path: Path, **kwargs) -> StandardizedLightCurve:
    """
    Load a light curve file and standardize it.
    
    Parameters
    ----------
    file_path : Path
        Path to .npz file containing light curve.
    **kwargs
        Additional arguments passed to standardize().
    
    Returns
    -------
    StandardizedLightCurve
        Standardized light curve.
    """
    # Load data
    data = np.load(file_path, allow_pickle=True)
    
    light_curve = {
        'time': data['time'],
        'flux': data['flux'],
        'flux_err': data.get('flux_err'),
        'quality': data.get('quality'),
        'metadata': {}
    }
    
    # Load metadata if available
    metadata_path = file_path.with_suffix('.json')
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            light_curve['metadata'] = json.load(f)
    
    # Standardize
    standardizer = LightCurveStandardizer()
    return standardizer.standardize(light_curve, **kwargs)

