"""
Transit injection-recovery framework for detection validation.

This module implements the gold-standard validation method for exoplanet
detection pipelines: inject synthetic transits into real noise and measure
recovery efficiency.

Scientific Context:
    Injection-recovery is the standard method for:
    - Validating detection pipelines
    - Measuring detection efficiency vs SNR
    - Calibrating false positive rates
    - Comparing ML vs classical methods

The framework:
    1. Takes real light curves (with no known planets)
    2. Injects synthetic transits with known parameters
    3. Runs detection algorithms (BLS, ML models)
    4. Measures recovery rate and parameter accuracy
    5. Builds detection efficiency curves

This is non-optional for any exoplanet detection pipeline.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

from ..physics.transit_model import mandel_agol_transit
from .baselines import detect_transit_bls, compute_snr


@dataclass
class InjectionParameters:
    """Parameters for a synthetic transit injection."""
    period: float  # Days
    t0: float  # Time of mid-transit (days)
    rp_rs: float  # Planet-to-star radius ratio
    a_rs: float  # Semi-major axis in stellar radii
    b: float  # Impact parameter (0 = central, 1 = grazing)
    u1: float = 0.3  # Limb darkening coefficient 1
    u2: float = 0.3  # Limb darkening coefficient 2
    eccentricity: float = 0.0  # Orbital eccentricity
    omega: float = 90.0  # Argument of periastron (degrees)


@dataclass
class RecoveryResult:
    """Result of a detection attempt on injected transit."""
    recovered: bool  # Whether transit was detected
    period_estimate: Optional[float] = None  # Estimated period
    period_error: Optional[float] = None  # Period error (fractional)
    depth_estimate: Optional[float] = None  # Estimated depth
    depth_error: Optional[float] = None  # Depth error (fractional)
    t0_estimate: Optional[float] = None  # Estimated t0
    t0_error: Optional[float] = None  # t0 error (days)
    snr: Optional[float] = None  # Signal-to-noise ratio
    method: str = "unknown"  # Detection method used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info


@dataclass
class InjectionRecoveryResult:
    """Complete result of injection-recovery test."""
    injection_params: InjectionParameters
    recovery_result: RecoveryResult
    true_snr: float
    n_transits: int  # Number of transits in light curve
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransitInjector:
    """
    Injects synthetic transits into real light curves.
    
    This class handles the physics of transit injection, ensuring
    realistic transit shapes and timing.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize injector.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def inject_transit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        params: InjectionParameters,
        add_noise: bool = True,
        flux_err: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject synthetic transit into light curve.
        
        Parameters
        ----------
        time : np.ndarray
            Observation times (days).
        flux : np.ndarray
            Original flux (will be modified).
        params : InjectionParameters
            Transit parameters.
        add_noise : bool
            Whether to add realistic noise (uses flux_err if available).
        flux_err : np.ndarray, optional
            Flux uncertainties for noise model.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (flux_with_transit, flux_err_updated)
        """
        # Generate transit model
        transit_model = mandel_agol_transit(
            time=time,
            period=params.period,
            t0=params.t0,
            rp_rs=params.rp_rs,
            a_rs=params.a_rs,
            b=params.b,
            u1=params.u1,
            u2=params.u2,
            eccentricity=params.eccentricity,
            omega=params.omega
        )
        
        # Apply transit to flux
        flux_with_transit = flux * transit_model
        
        # Add realistic noise if requested
        if add_noise and flux_err is not None:
            noise = np.random.normal(0, flux_err)
            flux_with_transit = flux_with_transit + noise
        
        return flux_with_transit, flux_err if flux_err is not None else np.ones_like(flux) * np.nanstd(flux)
    
    def compute_true_snr(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        params: InjectionParameters
    ) -> float:
        """
        Compute true SNR of injected transit.
        
        Parameters
        ----------
        time : np.ndarray
            Observation times.
        flux : np.ndarray
            Original flux.
        flux_err : np.ndarray
            Flux uncertainties.
        params : InjectionParameters
            Transit parameters.
        
        Returns
        -------
        float
            True signal-to-noise ratio.
        """
        # Generate transit model
        transit_model = mandel_agol_transit(
            time=time,
            period=params.period,
            t0=params.t0,
            rp_rs=params.rp_rs,
            a_rs=params.a_rs,
            b=params.b,
            u1=params.u1,
            u2=params.u2
        )
        
        return compute_snr(time, flux, flux_err, transit_model)
    
    def count_transits(
        self,
        time: np.ndarray,
        period: float,
        t0: float
    ) -> int:
        """
        Count number of transits in light curve.
        
        Parameters
        ----------
        time : np.ndarray
            Observation times.
        period : float
            Orbital period.
        t0 : float
            Time of first transit.
        
        Returns
        -------
        int
            Number of transits.
        """
        if len(time) == 0:
            return 0
        
        time_span = time.max() - time.min()
        n_transits = int(time_span / period) + 1
        
        return n_transits


class InjectionRecoveryFramework:
    """
    Complete framework for injection-recovery testing.
    
    This class orchestrates the full injection-recovery pipeline:
    - Parameter space sampling
    - Transit injection
    - Detection attempts
    - Recovery statistics
    """
    
    def __init__(
        self,
        detector: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize framework.
        
        Parameters
        ----------
        detector : Callable, optional
            Detection function: (time, flux, flux_err) -> RecoveryResult
            If None, uses BLS baseline.
        logger : logging.Logger, optional
            Logger instance.
        """
        self.detector = detector or self._default_bls_detector
        self.injector = TransitInjector(logger)
        self.logger = logger or logging.getLogger(__name__)
    
    def run_single_injection(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
        params: InjectionParameters,
        add_noise: bool = True
    ) -> InjectionRecoveryResult:
        """
        Run single injection-recovery test.
        
        Parameters
        ----------
        time : np.ndarray
            Observation times.
        flux : np.ndarray
            Original flux (no transits).
        flux_err : np.ndarray, optional
            Flux uncertainties.
        params : InjectionParameters
            Transit parameters to inject.
        add_noise : bool
            Whether to add noise during injection.
        
        Returns
        -------
        InjectionRecoveryResult
            Complete injection-recovery result.
        """
        # Compute true SNR
        true_snr = self.injector.compute_true_snr(time, flux, flux_err, params)
        
        # Count transits
        n_transits = self.injector.count_transits(time, params.period, params.t0)
        
        # Inject transit
        flux_injected, flux_err_used = self.injector.inject_transit(
            time, flux, params, add_noise=add_noise, flux_err=flux_err
        )
        
        # Attempt detection
        recovery_result = self.detector(time, flux_injected, flux_err_used)
        
        # Compute recovery metrics
        if recovery_result.recovered:
            recovery_result.period_error = abs(
                (recovery_result.period_estimate - params.period) / params.period
            )
            recovery_result.depth_error = abs(
                (recovery_result.depth_estimate - (1.0 - np.min(mandel_agol_transit(
                    time, params.period, params.t0, params.rp_rs, params.a_rs, params.b
                )))) / (1.0 - np.min(mandel_agol_transit(
                    time, params.period, params.t0, params.rp_rs, params.a_rs, params.b
                )))
            ) if recovery_result.depth_estimate is not None else None
            recovery_result.t0_error = abs(
                recovery_result.t0_estimate - params.t0
            ) if recovery_result.t0_estimate is not None else None
        
        return InjectionRecoveryResult(
            injection_params=params,
            recovery_result=recovery_result,
            true_snr=true_snr,
            n_transits=n_transits,
            metadata={}
        )
    
    def run_parameter_sweep(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
        period_range: Tuple[float, float],
        rp_rs_range: Tuple[float, float],
        n_injections: int = 100,
        **kwargs
    ) -> List[InjectionRecoveryResult]:
        """
        Run injection-recovery over parameter space.
        
        Parameters
        ----------
        time : np.ndarray
            Observation times.
        flux : np.ndarray
            Original flux.
        flux_err : np.ndarray, optional
            Flux uncertainties.
        period_range : Tuple[float, float]
            (min_period, max_period) in days.
        rp_rs_range : Tuple[float, float]
            (min_rp_rs, max_rp_rs).
        n_injections : int
            Number of injections to perform.
        **kwargs
            Additional parameters for injection.
        
        Returns
        -------
        List[InjectionRecoveryResult]
            List of injection-recovery results.
        """
        results = []
        
        # Sample parameter space
        periods = np.random.uniform(period_range[0], period_range[1], n_injections)
        rp_rs_values = np.random.uniform(rp_rs_range[0], rp_rs_range[1], n_injections)
        
        # Fixed parameters (can be made variable)
        a_rs = kwargs.get('a_rs', 10.0)  # Default semi-major axis
        b = kwargs.get('b', 0.0)  # Central transit
        u1 = kwargs.get('u1', 0.3)
        u2 = kwargs.get('u2', 0.3)
        
        for i, (period, rp_rs) in enumerate(zip(periods, rp_rs_values)):
            # Random t0 within first period
            t0 = time.min() + np.random.uniform(0, period)
            
            params = InjectionParameters(
                period=period,
                t0=t0,
                rp_rs=rp_rs,
                a_rs=a_rs,
                b=b,
                u1=u1,
                u2=u2
            )
            
            try:
                result = self.run_single_injection(
                    time, flux, flux_err, params, add_noise=True
                )
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{n_injections} injections")
            
            except Exception as e:
                self.logger.warning(f"Injection {i + 1} failed: {e}")
                continue
        
        return results
    
    def compute_detection_efficiency(
        self,
        results: List[InjectionRecoveryResult],
        snr_bins: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute detection efficiency vs SNR.
        
        Parameters
        ----------
        results : List[InjectionRecoveryResult]
            Injection-recovery results.
        snr_bins : np.ndarray, optional
            SNR bins for efficiency curve. If None, auto-generates.
        
        Returns
        -------
        Dict[str, Any]
            Efficiency curve with:
            - 'snr_bins': SNR bin centers
            - 'efficiency': Recovery fraction in each bin
            - 'n_injections': Number of injections per bin
            - 'n_recovered': Number recovered per bin
        """
        if len(results) == 0:
            return {'snr_bins': np.array([]), 'efficiency': np.array([])}
        
        snr_values = np.array([r.true_snr for r in results])
        recovered = np.array([r.recovery_result.recovered for r in results])
        
        if snr_bins is None:
            snr_min, snr_max = snr_values.min(), snr_values.max()
            snr_bins = np.linspace(snr_min, snr_max, 10)
        
        efficiency = []
        n_injections = []
        n_recovered = []
        
        for i in range(len(snr_bins) - 1):
            bin_mask = (snr_values >= snr_bins[i]) & (snr_values < snr_bins[i + 1])
            n_in_bin = np.sum(bin_mask)
            
            if n_in_bin > 0:
                eff = np.sum(recovered[bin_mask]) / n_in_bin
                efficiency.append(eff)
                n_injections.append(n_in_bin)
                n_recovered.append(np.sum(recovered[bin_mask]))
            else:
                efficiency.append(0.0)
                n_injections.append(0)
                n_recovered.append(0)
        
        bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2.0
        
        return {
            'snr_bins': bin_centers,
            'efficiency': np.array(efficiency),
            'n_injections': np.array(n_injections),
            'n_recovered': np.array(n_recovered)
        }
    
    def _default_bls_detector(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray]
    ) -> RecoveryResult:
        """Default BLS detector for baseline comparison."""
        try:
            bls_result = detect_transit_bls(time, flux, flux_err)
            
            # Determine if detection is successful
            # (simplified: check if power is above threshold)
            recovered = (bls_result['power'] > 10.0 and 
                        bls_result['period'] is not None)
            
            return RecoveryResult(
                recovered=recovered,
                period_estimate=bls_result.get('period'),
                depth_estimate=bls_result.get('depth'),
                t0_estimate=bls_result.get('t0'),
                snr=bls_result.get('power', 0.0),
                method='BLS',
                metadata=bls_result
            )
        except Exception as e:
            return RecoveryResult(
                recovered=False,
                method='BLS',
                metadata={'error': str(e)}
            )
