"""
Data validation module for light curve quality assurance.

This module provides scientific validation checks to ensure downloaded
data is ready for analysis. It catches archive issues, data corruption,
and format inconsistencies before they propagate to ML models.

Scientific Context:
    Real astronomical data has many failure modes:
    - Time arrays with gaps or non-monotonic values
    - Flux arrays with all-NaN segments
    - Quality flags that don't match actual data quality
    - Inconsistent time units between missions
    - Cadence mismatches

These issues must be caught early to prevent garbage-in → garbage-out.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class ValidationResult:
    """Result of data validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class LightCurveValidator:
    """
    Validator for light curve data quality.
    
    Performs scientific checks on time, flux, and quality arrays
    to ensure data integrity before analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize validator.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for validation messages.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate(self, light_curve: Dict[str, Any], mission: str = "unknown") -> ValidationResult:
        """
        Perform comprehensive validation on a light curve.
        
        Parameters
        ----------
        light_curve : Dict[str, Any]
            Light curve dictionary with keys: time, flux, flux_err, quality, metadata
        mission : str
            Mission name (Kepler, TESS, etc.) for mission-specific checks
        
        Returns
        -------
        ValidationResult
            Validation result with errors, warnings, and metadata.
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metadata={})
        
        # Extract arrays
        time = light_curve.get('time')
        flux = light_curve.get('flux')
        flux_err = light_curve.get('flux_err')
        quality = light_curve.get('quality')
        metadata = light_curve.get('metadata', {})
        
        # Check 1: Arrays exist and are numpy arrays
        if time is None:
            result.errors.append("Time array is missing")
            result.is_valid = False
            return result
        
        if flux is None:
            result.errors.append("Flux array is missing")
            result.is_valid = False
            return result
        
        time = np.asarray(time)
        flux = np.asarray(flux)
        
        # Check 2: Arrays have same length
        if len(time) != len(flux):
            result.errors.append(
                f"Time and flux arrays have different lengths: {len(time)} vs {len(flux)}"
            )
            result.is_valid = False
            return result
        
        # Check 3: Time array is strictly increasing
        if not self._check_time_monotonic(time, result):
            result.is_valid = False
            return result
        
        # Check 4: No all-NaN segments in flux
        if not self._check_flux_validity(flux, result):
            result.is_valid = False
            return result
        
        # Check 5: Quality flags are valid
        if quality is not None:
            self._check_quality_flags(quality, len(time), mission, result)
        
        # Check 6: Flux errors are positive (if provided)
        if flux_err is not None:
            self._check_flux_errors(flux_err, len(time), result)
        
        # Check 7: Cadence matches metadata (if available)
        if metadata:
            self._check_cadence_consistency(time, metadata, result)
        
        # Check 8: Mission-specific time unit checks
        self._check_time_units(time, mission, metadata, result)
        
        # Collect metadata
        result.metadata = {
            'n_points': len(time),
            'time_range': [float(time.min()), float(time.max())],
            'time_span_days': float(time.max() - time.min()),
            'flux_mean': float(np.nanmean(flux)),
            'flux_std': float(np.nanstd(flux)),
            'fraction_valid': float(np.sum(~np.isnan(flux)) / len(flux)),
            'mission': mission,
        }
        
        if flux_err is not None:
            result.metadata['flux_err_mean'] = float(np.nanmean(flux_err))
        
        return result
    
    def _check_time_monotonic(self, time: np.ndarray, result: ValidationResult) -> bool:
        """Check that time array is strictly increasing."""
        if len(time) < 2:
            result.warnings.append("Time array has fewer than 2 points")
            return True
        
        # Check for strictly increasing
        time_diff = np.diff(time)
        
        if np.any(time_diff <= 0):
            non_increasing = np.sum(time_diff <= 0)
            result.errors.append(
                f"Time array is not strictly increasing: {non_increasing} non-positive differences"
            )
            return False
        
        # Check for large gaps (potential data loss)
        median_diff = np.median(time_diff)
        large_gaps = np.sum(time_diff > 10 * median_diff)
        
        if large_gaps > 0:
            result.warnings.append(
                f"Found {large_gaps} large time gaps (>10x median cadence)"
            )
        
        return True
    
    def _check_flux_validity(self, flux: np.ndarray, result: ValidationResult) -> bool:
        """Check that flux array has valid data."""
        n_total = len(flux)
        n_valid = np.sum(~np.isnan(flux))
        n_finite = np.sum(np.isfinite(flux))
        
        if n_valid == 0:
            result.errors.append("Flux array contains no valid (non-NaN) data")
            return False
        
        if n_finite == 0:
            result.errors.append("Flux array contains no finite data")
            return False
        
        fraction_valid = n_valid / n_total
        if fraction_valid < 0.5:
            result.warnings.append(
                f"Only {fraction_valid:.1%} of flux values are valid"
            )
        
        # Check for all-NaN segments (consecutive NaNs)
        if n_valid < n_total:
            nan_mask = np.isnan(flux)
            # Find consecutive NaN segments
            diff_nan = np.diff(nan_mask.astype(int))
            segment_starts = np.where(diff_nan == 1)[0] + 1
            segment_ends = np.where(diff_nan == -1)[0] + 1
            
            if len(segment_starts) > 0 or len(segment_ends) > 0:
                max_segment_length = 0
                if len(segment_starts) > 0 and len(segment_ends) > 0:
                    segment_lengths = segment_ends - segment_starts
                    if len(segment_lengths) > 0:
                        max_segment_length = np.max(segment_lengths)
                
                if max_segment_length > 100:
                    result.warnings.append(
                        f"Found NaN segment of length {max_segment_length} points"
                    )
        
        # Check for extreme outliers (likely bad data)
        if n_finite > 10:
            flux_median = np.nanmedian(flux)
            flux_mad = np.nanmedian(np.abs(flux - flux_median))
            outliers = np.sum(np.abs(flux - flux_median) > 10 * flux_mad)
            
            if outliers > 0:
                result.warnings.append(
                    f"Found {outliers} extreme flux outliers (>10 MAD from median)"
                )
        
        return True
    
    def _check_quality_flags(
        self, 
        quality: np.ndarray, 
        expected_length: int,
        mission: str,
        result: ValidationResult
    ) -> None:
        """Check quality flags for consistency."""
        quality = np.asarray(quality)
        
        if len(quality) != expected_length:
            result.errors.append(
                f"Quality array length ({len(quality)}) doesn't match time array ({expected_length})"
            )
            return
        
        # Mission-specific quality flag checks
        if mission.lower() == 'kepler':
            # Kepler: 0 = good data, non-zero = bad data
            bad_quality = np.sum(quality != 0)
            if bad_quality > 0:
                result.metadata['fraction_bad_quality'] = float(bad_quality / len(quality))
        elif mission.lower() == 'tess':
            # TESS: 0 = good data, non-zero = bad data
            bad_quality = np.sum(quality != 0)
            if bad_quality > 0:
                result.metadata['fraction_bad_quality'] = float(bad_quality / len(quality))
    
    def _check_flux_errors(
        self,
        flux_err: np.ndarray,
        expected_length: int,
        result: ValidationResult
    ) -> None:
        """Check flux error array for validity."""
        flux_err = np.asarray(flux_err)
        
        if len(flux_err) != expected_length:
            result.errors.append(
                f"Flux error array length ({len(flux_err)}) doesn't match time array ({expected_length})"
            )
            return
        
        # Check for negative or zero errors
        negative_errors = np.sum(flux_err <= 0)
        if negative_errors > 0:
            result.warnings.append(
                f"Found {negative_errors} non-positive flux errors"
            )
        
        # Check for extremely large errors (likely bad data)
        if np.sum(np.isfinite(flux_err)) > 0:
            median_err = np.nanmedian(flux_err)
            large_errors = np.sum(flux_err > 100 * median_err)
            if large_errors > 0:
                result.warnings.append(
                    f"Found {large_errors} extremely large flux errors (>100x median)"
                )
    
    def _check_cadence_consistency(
        self,
        time: np.ndarray,
        metadata: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check that cadence matches metadata."""
        if len(time) < 2:
            return
        
        cadence_metadata = metadata.get('cadence')
        if cadence_metadata is None:
            return
        
        # Calculate actual cadence
        time_diff = np.diff(time)
        median_cadence = np.median(time_diff)
        
        # Expected cadences (in days)
        expected_cadences = {
            'short': 0.0204,  # ~30 minutes for Kepler
            'long': 1.0,      # Daily cadence
            'fast': 0.0014,   # ~2 minutes for TESS
        }
        
        # Check if cadence string matches
        if isinstance(cadence_metadata, str):
            expected = expected_cadences.get(cadence_metadata.lower())
            if expected is not None:
                if abs(median_cadence - expected) / expected > 0.5:
                    result.warnings.append(
                        f"Median cadence ({median_cadence:.4f} days) doesn't match "
                        f"metadata cadence ({cadence_metadata})"
                    )
    
    def _check_time_units(
        self,
        time: np.ndarray,
        mission: str,
        metadata: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check time units are reasonable for the mission."""
        if len(time) == 0:
            return
        
        # Expected time ranges (in days since mission start)
        # These are rough bounds
        time_min = time.min()
        time_max = time.max()
        
        if mission.lower() == 'kepler':
            # Kepler: BJD - 2454833, should be roughly 0-1600 days
            if time_min < -100 or time_max > 2000:
                result.warnings.append(
                    f"Kepler time values seem out of range: [{time_min:.2f}, {time_max:.2f}]"
                )
        elif mission.lower() == 'tess':
            # TESS: BTJD, should be roughly 1300-2000+ days
            if time_min < 1000 or time_max > 3000:
                result.warnings.append(
                    f"TESS time values seem out of range: [{time_min:.2f}, {time_max:.2f}]"
                )


def validate_light_curve_file(file_path: Path, mission: str = "unknown") -> ValidationResult:
    """
    Validate a light curve from a saved file.
    
    Parameters
    ----------
    file_path : Path
        Path to .npz file containing light curve data.
    mission : str
        Mission name for mission-specific checks.
    
    Returns
    -------
    ValidationResult
        Validation result.
    """
    import numpy as np
    
    if not file_path.exists():
        return ValidationResult(
            is_valid=False,
            errors=[f"File does not exist: {file_path}"],
            warnings=[],
            metadata={}
        )
    
    try:
        data = np.load(file_path, allow_pickle=True)
        light_curve = {
            'time': data.get('time'),
            'flux': data.get('flux'),
            'flux_err': data.get('flux_err'),
            'quality': data.get('quality'),
            'metadata': {}
        }
        
        # Try to load metadata JSON if it exists
        metadata_path = file_path.with_suffix('.json')
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                light_curve['metadata'] = json.load(f)
        
        validator = LightCurveValidator()
        return validator.validate(light_curve, mission)
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"Failed to load/validate file: {e}"],
            warnings=[],
            metadata={}
        )

