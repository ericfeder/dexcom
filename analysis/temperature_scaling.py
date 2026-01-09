"""
Temperature Scaling for TCN Distribution Models

Post-hoc calibration technique that learns a scalar temperature T for each
prediction horizon. Temperature scaling divides logits by T before applying
softmax, which adjusts the confidence of predictions without changing the
ranking.

For well-calibrated models, T ≈ 1.0
For overconfident models, T > 1.0 (softens the distribution)
For underconfident models, T < 1.0 (sharpens the distribution)

Calibration modes:
- "nll": Minimize negative log-likelihood (default, existing behavior)
- "coverage": Find T that achieves target interval coverage (e.g., 80%)
- "nll_then_coverage": Fit via NLL first, then adjust to hit target coverage
- "coverage_bucketed": Fit separate T per horizon AND volatility bucket

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, Union, List

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# Default temperature bounds
DEFAULT_T_MIN = 0.2
DEFAULT_T_MAX = 5.0

# Calibration modes
CALIBRATION_MODE_NLL = "nll"
CALIBRATION_MODE_COVERAGE = "coverage"
CALIBRATION_MODE_NLL_THEN_COVERAGE = "nll_then_coverage"
CALIBRATION_MODE_COVERAGE_BUCKETED = "coverage_bucketed"

# Default coverage target
DEFAULT_COVERAGE_TARGET = 0.80
DEFAULT_COVERAGE_TOLERANCE = 0.005  # ±0.5%

# Volatility thresholds for bucketed mode (mg/dL per min)
VOL_THRESH_LOW = 0.5
VOL_THRESH_HIGH = 1.5
VOL_BUCKETS = ["low", "med", "high"]


def compute_nll(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    """
    Compute negative log-likelihood with temperature scaling.
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        y_true: Shape (N,) - true bin indices (integers)
        temperature: Scalar temperature value
    
    Returns:
        Mean NLL across all samples
    """
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Compute cross-entropy loss (numerically stable via TensorFlow)
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(y_true, tf.int32),
        logits=scaled_logits
    )
    
    return float(tf.reduce_mean(nll).numpy())


def compute_quantile_from_probs(probs: np.ndarray, bin_centers: np.ndarray, q: float) -> np.ndarray:
    """
    Compute quantile from PMF probabilities.
    
    Args:
        probs: Shape (N, n_bins) - probability mass function
        bin_centers: Shape (n_bins,) - center values for each bin
        q: Quantile to compute (e.g., 0.10, 0.50, 0.90)
    
    Returns:
        Shape (N,) - quantile values for each sample
    """
    cdf = np.cumsum(probs, axis=1)
    indices = np.argmax(cdf >= q, axis=1)
    return bin_centers[indices]


def compute_coverage_and_width(
    logits: np.ndarray,
    y_deltas: np.ndarray,
    bin_centers: np.ndarray,
    temperature: float,
    q_lo: float = 0.10,
    q_hi: float = 0.90
) -> Tuple[float, float]:
    """
    Compute interval coverage and mean width for given temperature.
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        y_deltas: Shape (N,) - true delta values (continuous)
        bin_centers: Shape (n_bins,) - center values for each bin
        temperature: Temperature to apply
        q_lo: Lower quantile (default 0.10)
        q_hi: Upper quantile (default 0.90)
    
    Returns:
        Tuple of (coverage_fraction, mean_width)
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    probs = tf.nn.softmax(scaled_logits, axis=-1).numpy()
    
    # Compute quantiles
    q_lo_vals = compute_quantile_from_probs(probs, bin_centers, q_lo)
    q_hi_vals = compute_quantile_from_probs(probs, bin_centers, q_hi)
    
    # Coverage: fraction of true values within [q_lo, q_hi]
    in_interval = (y_deltas >= q_lo_vals) & (y_deltas <= q_hi_vals)
    coverage = np.mean(in_interval)
    
    # Mean width
    width = np.mean(q_hi_vals - q_lo_vals)
    
    return float(coverage), float(width)


def fit_temperature_for_coverage(
    logits: np.ndarray,
    y_deltas: np.ndarray,
    bin_centers: np.ndarray,
    target_coverage: float = DEFAULT_COVERAGE_TARGET,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    max_iterations: int = 40,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Fit temperature to achieve target interval coverage using binary search.
    
    Coverage is approximately monotonic in T:
    - Increasing T softens PMF => wider intervals => higher coverage
    - Decreasing T sharpens PMF => narrower intervals => lower coverage
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        y_deltas: Shape (N,) - true delta values (continuous)
        bin_centers: Shape (n_bins,) - center values for each bin
        target_coverage: Target coverage fraction (default 0.80)
        q_lo: Lower quantile (default 0.10)
        q_hi: Upper quantile (default 0.90)
        t_min: Minimum temperature to search
        t_max: Maximum temperature to search
        tolerance: Stop when |coverage - target| < tolerance
        max_iterations: Maximum binary search iterations
        verbose: Print search progress
    
    Returns:
        Tuple of (fitted_temperature, info_dict)
        info_dict contains target, achieved_coverage, mean_width, iterations
    """
    # Compute coverage at boundaries to ensure target is achievable
    cov_min, width_min = compute_coverage_and_width(logits, y_deltas, bin_centers, t_min, q_lo, q_hi)
    cov_max, width_max = compute_coverage_and_width(logits, y_deltas, bin_centers, t_max, q_lo, q_hi)
    
    if verbose:
        print(f"    T={t_min:.2f}: coverage={cov_min:.3f}, width={width_min:.1f}")
        print(f"    T={t_max:.2f}: coverage={cov_max:.3f}, width={width_max:.1f}")
        print(f"    Target coverage: {target_coverage:.3f}")
    
    # Check if target is achievable
    if target_coverage < cov_min:
        if verbose:
            print(f"    Warning: Target {target_coverage:.3f} < min achievable {cov_min:.3f}, using T={t_min}")
        cov, width = cov_min, width_min
        return float(t_min), {
            'target_coverage': target_coverage,
            'achieved_coverage': cov,
            'mean_width': width,
            'iterations': 0,
            'method': 'coverage',
            'note': 'target below minimum'
        }
    
    if target_coverage > cov_max:
        if verbose:
            print(f"    Warning: Target {target_coverage:.3f} > max achievable {cov_max:.3f}, using T={t_max}")
        cov, width = cov_max, width_max
        return float(t_max), {
            'target_coverage': target_coverage,
            'achieved_coverage': cov,
            'mean_width': width,
            'iterations': 0,
            'method': 'coverage',
            'note': 'target above maximum'
        }
    
    # Binary search
    lo, hi = t_min, t_max
    best_T = (lo + hi) / 2
    best_diff = float('inf')
    iterations = 0
    
    for i in range(max_iterations):
        mid = (lo + hi) / 2
        cov, width = compute_coverage_and_width(logits, y_deltas, bin_centers, mid, q_lo, q_hi)
        diff = abs(cov - target_coverage)
        
        if diff < best_diff:
            best_T = mid
            best_diff = diff
            best_cov = cov
            best_width = width
        
        if verbose and i % 10 == 0:
            print(f"    Iter {i}: T={mid:.4f}, coverage={cov:.4f}, width={width:.1f}")
        
        if diff < tolerance:
            iterations = i + 1
            break
        
        # Coverage increases with T, so:
        if cov < target_coverage:
            lo = mid  # Need higher T for more coverage
        else:
            hi = mid  # Need lower T for less coverage
        
        iterations = i + 1
    
    if verbose:
        print(f"    Final: T={best_T:.4f}, coverage={best_cov:.4f}, width={best_width:.1f}")
    
    return float(best_T), {
        'target_coverage': target_coverage,
        'achieved_coverage': best_cov,
        'mean_width': best_width,
        'iterations': iterations,
        'method': 'coverage'
    }


def fit_temperature_nll_then_coverage(
    logits: np.ndarray,
    y_bins: np.ndarray,
    y_deltas: np.ndarray,
    bin_centers: np.ndarray,
    target_coverage: float = DEFAULT_COVERAGE_TARGET,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Hybrid calibration: fit via NLL first, then adjust to hit target coverage.
    
    Minimizes |log(T) - log(T_nll)| subject to coverage within tolerance.
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        y_bins: Shape (N,) - true bin indices (for NLL)
        y_deltas: Shape (N,) - true delta values (for coverage)
        bin_centers: Shape (n_bins,) - center values for each bin
        target_coverage: Target coverage fraction
        q_lo, q_hi: Quantile bounds
        t_min, t_max: Temperature bounds
        coverage_tolerance: Acceptable deviation from target
        verbose: Print progress
    
    Returns:
        Tuple of (fitted_temperature, info_dict)
    """
    # Step 1: Fit NLL-optimal temperature
    T_nll, nll_info = fit_temperature(logits, y_bins, t_min=t_min, t_max=t_max, verbose=False)
    
    if verbose:
        print(f"    NLL-optimal T: {T_nll:.4f}")
    
    # Step 2: Check coverage at T_nll
    cov_nll, width_nll = compute_coverage_and_width(logits, y_deltas, bin_centers, T_nll, q_lo, q_hi)
    
    if verbose:
        print(f"    Coverage at T_nll: {cov_nll:.4f}, width: {width_nll:.1f}")
    
    # If already within tolerance, use NLL-optimal T
    if abs(cov_nll - target_coverage) < coverage_tolerance:
        if verbose:
            print(f"    NLL-optimal T already meets coverage target!")
        return T_nll, {
            'method': 'nll_then_coverage',
            'T_nll': T_nll,
            'T_final': T_nll,
            'target_coverage': target_coverage,
            'achieved_coverage': cov_nll,
            'mean_width': width_nll,
            'nll_before': nll_info['nll_before'],
            'nll_after': nll_info['nll_after']
        }
    
    # Step 3: Search near T_nll to find T that hits coverage target
    # Search in range [T_nll * 0.5, T_nll * 2.0] first, then expand if needed
    search_lo = max(t_min, T_nll * 0.5)
    search_hi = min(t_max, T_nll * 2.0)
    
    T_cov, cov_info = fit_temperature_for_coverage(
        logits, y_deltas, bin_centers,
        target_coverage=target_coverage,
        q_lo=q_lo, q_hi=q_hi,
        t_min=search_lo, t_max=search_hi,
        tolerance=coverage_tolerance,
        verbose=verbose
    )
    
    if verbose:
        print(f"    Final T: {T_cov:.4f} (was NLL-optimal {T_nll:.4f})")
    
    return T_cov, {
        'method': 'nll_then_coverage',
        'T_nll': T_nll,
        'T_final': T_cov,
        'target_coverage': target_coverage,
        'achieved_coverage': cov_info['achieved_coverage'],
        'mean_width': cov_info['mean_width'],
        'nll_before': nll_info['nll_before'],
        'nll_after': nll_info['nll_after']
    }


def get_volatility_bucket(volatility: np.ndarray) -> np.ndarray:
    """
    Assign volatility bucket to each sample.
    
    Args:
        volatility: Shape (N,) - volatility values (mg/dL per min)
    
    Returns:
        Shape (N,) - bucket labels ("low", "med", "high")
    """
    buckets = np.empty(len(volatility), dtype=object)
    buckets[:] = "med"
    buckets[volatility < VOL_THRESH_LOW] = "low"
    buckets[volatility >= VOL_THRESH_HIGH] = "high"
    return buckets


def fit_temperatures_coverage_bucketed(
    logits: np.ndarray,
    y_deltas: np.ndarray,
    volatility: np.ndarray,
    bin_centers: np.ndarray,
    target_coverage: float = DEFAULT_COVERAGE_TARGET,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    verbose: bool = False
) -> Tuple[Dict[str, float], Dict]:
    """
    Fit separate temperatures for each volatility bucket.
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        y_deltas: Shape (N,) - true delta values
        volatility: Shape (N,) - volatility values
        bin_centers: Shape (n_bins,) - center values for each bin
        target_coverage: Target coverage fraction
        q_lo, q_hi: Quantile bounds
        t_min, t_max: Temperature bounds
        tolerance: Coverage tolerance
        verbose: Print progress
    
    Returns:
        Tuple of (temps_dict, info_dict)
        temps_dict maps bucket name -> temperature
    """
    buckets = get_volatility_bucket(volatility)
    temps = {}
    infos = {}
    
    for bucket in VOL_BUCKETS:
        mask = buckets == bucket
        n_samples = np.sum(mask)
        
        if n_samples < 100:
            if verbose:
                print(f"    {bucket}: only {n_samples} samples, using default T=1.0")
            temps[bucket] = 1.0
            infos[bucket] = {'note': 'insufficient samples', 'n_samples': int(n_samples)}
            continue
        
        if verbose:
            print(f"    {bucket} bucket (n={n_samples})...")
        
        T, info = fit_temperature_for_coverage(
            logits[mask],
            y_deltas[mask],
            bin_centers,
            target_coverage=target_coverage,
            q_lo=q_lo, q_hi=q_hi,
            t_min=t_min, t_max=t_max,
            tolerance=tolerance,
            verbose=verbose
        )
        
        temps[bucket] = T
        infos[bucket] = info
        infos[bucket]['n_samples'] = int(n_samples)
        
        if verbose:
            print(f"      T={T:.4f}, coverage={info['achieved_coverage']:.4f}, width={info['mean_width']:.1f}")
    
    return temps, {'method': 'coverage_bucketed', 'buckets': infos}


def apply_temperature_bucketed(
    logits: np.ndarray,
    volatility: np.ndarray,
    temps: Dict[str, float]
) -> np.ndarray:
    """
    Apply bucket-specific temperatures to logits.
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        volatility: Shape (N,) - volatility values
        temps: Dict mapping bucket name -> temperature
    
    Returns:
        Shape (N, n_bins) - calibrated probabilities
    """
    buckets = get_volatility_bucket(volatility)
    probs = np.zeros_like(logits)
    
    for bucket in VOL_BUCKETS:
        mask = buckets == bucket
        if np.sum(mask) > 0:
            T = temps.get(bucket, 1.0)
            scaled = logits[mask] / T
            probs[mask] = tf.nn.softmax(scaled, axis=-1).numpy()
    
    return probs


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    learning_rate: float = 0.01,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """
    Fit a scalar temperature parameter to minimize NLL on calibration data.
    
    Uses gradient descent to optimize log(T), which ensures T > 0.
    Temperature is then clipped to [t_min, t_max].
    
    Args:
        logits: Shape (N, n_bins) - raw model outputs
        y_true: Shape (N,) - true bin indices
        t_min: Minimum allowed temperature
        t_max: Maximum allowed temperature
        learning_rate: Learning rate for Adam optimizer
        max_iterations: Maximum optimization steps
        tolerance: Stop if improvement is less than this
        verbose: Print optimization progress
    
    Returns:
        Tuple of (fitted_temperature, info_dict)
        info_dict contains 'nll_before', 'nll_after', 'iterations'
    """
    # Convert to tensors
    logits_tf = tf.constant(logits, dtype=tf.float32)
    y_true_tf = tf.constant(y_true, dtype=tf.int32)
    
    # Initialize log_T = 0 (T = 1)
    log_T = tf.Variable(0.0, dtype=tf.float32)
    
    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # Compute NLL before optimization (T=1)
    nll_before = compute_nll(logits, y_true, 1.0)
    
    # Optimization loop
    best_nll = float('inf')
    best_log_T = 0.0
    prev_nll = float('inf')
    iterations_used = 0
    
    for iteration in range(max_iterations):
        with tf.GradientTape() as tape:
            T = tf.exp(log_T)
            # Clip T during forward pass for stability
            T_clipped = tf.clip_by_value(T, t_min, t_max)
            scaled_logits = logits_tf / T_clipped
            
            # NLL loss
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_true_tf,
                    logits=scaled_logits
                )
            )
        
        # Compute gradient and update
        gradients = tape.gradient(loss, [log_T])
        optimizer.apply_gradients(zip(gradients, [log_T]))
        
        current_nll = float(loss.numpy())
        iterations_used = iteration + 1
        
        # Track best
        if current_nll < best_nll:
            best_nll = current_nll
            best_log_T = float(log_T.numpy())
        
        # Check convergence
        improvement = prev_nll - current_nll
        if abs(improvement) < tolerance and iteration > 10:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
        
        prev_nll = current_nll
        
        if verbose and iteration % 100 == 0:
            T_current = float(tf.exp(log_T).numpy())
            print(f"  Iter {iteration}: T={T_current:.4f}, NLL={current_nll:.6f}")
    
    # Get final temperature from best point, clipped to bounds
    fitted_T = np.clip(np.exp(best_log_T), t_min, t_max)
    nll_after = compute_nll(logits, y_true, fitted_T)
    
    info = {
        'nll_before': nll_before,
        'nll_after': nll_after,
        'iterations': iterations_used,
        'nll_improvement': nll_before - nll_after
    }
    
    return float(fitted_T), info


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits and return calibrated probabilities.
    
    Args:
        logits: Shape (N, n_bins) or (n_bins,) - raw model outputs
        temperature: Scalar temperature value
    
    Returns:
        Calibrated probabilities with same shape as logits, summing to 1
    """
    scaled_logits = logits / temperature
    probs = tf.nn.softmax(scaled_logits, axis=-1).numpy()
    return probs


def apply_temperatures_dict(
    logits_dict: Dict[str, np.ndarray],
    temperatures: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    Apply per-head temperature scaling to a dict of logits.
    
    Designed for multi-horizon TCN models where logits_dict has keys like
    "h5", "h10", etc. and temperatures has matching keys.
    
    Args:
        logits_dict: Dict mapping head_name -> (N, n_bins) logits
        temperatures: Dict mapping head_name -> temperature value
                     Can also accept horizon (int) keys like {5: 1.2, 10: 1.1, ...}
    
    Returns:
        Dict mapping head_name -> (N, n_bins) calibrated probabilities
    """
    probs_dict = {}
    
    for head_name, logits in logits_dict.items():
        # Try to find temperature by head_name or by horizon int
        if head_name in temperatures:
            T = temperatures[head_name]
        else:
            # Try extracting horizon from head_name (e.g., "h5" -> 5)
            try:
                horizon = int(head_name[1:])
                T = temperatures.get(horizon, 1.0)
            except (ValueError, KeyError):
                T = 1.0  # Default to no scaling
        
        probs_dict[head_name] = apply_temperature(logits, T)
    
    return probs_dict


def fit_temperatures_all_horizons(
    horizon_data: Dict[int, Dict],
    bin_centers: np.ndarray = None,
    mode: str = CALIBRATION_MODE_NLL,
    target_coverage: float = DEFAULT_COVERAGE_TARGET,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    verbose: bool = True
) -> Tuple[Dict[int, Union[float, Dict[str, float]]], Dict]:
    """
    Fit temperature for each prediction horizon.
    
    Args:
        horizon_data: Dict mapping horizon (int) to:
            - 'logits': np.ndarray
            - 'y_bins': np.ndarray (for NLL modes)
            - 'y_deltas': np.ndarray (for coverage modes)
            - 'volatility': np.ndarray (for bucketed mode)
        bin_centers: Shape (n_bins,) - required for coverage modes
        mode: Calibration mode - "nll", "coverage", "nll_then_coverage", "coverage_bucketed"
        target_coverage: Target coverage for coverage modes (default 0.80)
        t_min: Minimum allowed temperature
        t_max: Maximum allowed temperature
        verbose: Print fitting progress
    
    Returns:
        Tuple of (temperatures_dict, metadata_dict)
        For non-bucketed: temps[horizon] = float
        For bucketed: temps[horizon] = {bucket: float}
    """
    temperatures = {}
    all_info = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"TEMPERATURE SCALING - MODE: {mode.upper()}")
        print("=" * 70)
    
    for horizon in sorted(horizon_data.keys()):
        data = horizon_data[horizon]
        logits = data['logits']
        
        if verbose:
            print(f"\n+{horizon} min horizon (n={len(logits):,})...")
        
        if mode == CALIBRATION_MODE_NLL:
            y_bins = data['y_bins']
            fitted_T, info = fit_temperature(
                logits, y_bins,
                t_min=t_min, t_max=t_max,
                verbose=False
            )
            temperatures[horizon] = fitted_T
            all_info[horizon] = info
            
            if verbose:
                print(f"  Temperature: {fitted_T:.4f}")
                print(f"  NLL: {info['nll_before']:.4f} → {info['nll_after']:.4f} "
                      f"(Δ = {info['nll_improvement']:+.4f})")
        
        elif mode == CALIBRATION_MODE_COVERAGE:
            y_deltas = data['y_deltas']
            assert bin_centers is not None, "bin_centers required for coverage mode"
            
            fitted_T, info = fit_temperature_for_coverage(
                logits, y_deltas, bin_centers,
                target_coverage=target_coverage,
                t_min=t_min, t_max=t_max,
                verbose=verbose
            )
            temperatures[horizon] = fitted_T
            all_info[horizon] = info
            
            if verbose:
                print(f"  Temperature: {fitted_T:.4f}")
                print(f"  Coverage: {info['achieved_coverage']*100:.1f}% (target: {target_coverage*100:.0f}%)")
                print(f"  Width: {info['mean_width']:.1f}")
        
        elif mode == CALIBRATION_MODE_NLL_THEN_COVERAGE:
            y_bins = data['y_bins']
            y_deltas = data['y_deltas']
            assert bin_centers is not None, "bin_centers required for coverage mode"
            
            fitted_T, info = fit_temperature_nll_then_coverage(
                logits, y_bins, y_deltas, bin_centers,
                target_coverage=target_coverage,
                t_min=t_min, t_max=t_max,
                verbose=verbose
            )
            temperatures[horizon] = fitted_T
            all_info[horizon] = info
            
            if verbose:
                print(f"  Temperature: {fitted_T:.4f} (NLL-optimal was {info['T_nll']:.4f})")
                print(f"  Coverage: {info['achieved_coverage']*100:.1f}%")
                print(f"  Width: {info['mean_width']:.1f}")
        
        elif mode == CALIBRATION_MODE_COVERAGE_BUCKETED:
            y_deltas = data['y_deltas']
            volatility = data['volatility']
            assert bin_centers is not None, "bin_centers required for coverage mode"
            
            temps_bucketed, info = fit_temperatures_coverage_bucketed(
                logits, y_deltas, volatility, bin_centers,
                target_coverage=target_coverage,
                t_min=t_min, t_max=t_max,
                verbose=verbose
            )
            temperatures[horizon] = temps_bucketed
            all_info[horizon] = info
            
            if verbose:
                print(f"  Temperatures: low={temps_bucketed['low']:.4f}, "
                      f"med={temps_bucketed['med']:.4f}, high={temps_bucketed['high']:.4f}")
        
        else:
            raise ValueError(f"Unknown calibration mode: {mode}")
    
    metadata = {
        'mode': mode,
        'target_coverage': target_coverage if mode != CALIBRATION_MODE_NLL else None,
        't_min': t_min,
        't_max': t_max,
        'per_horizon_info': all_info
    }
    
    return temperatures, metadata


def save_temperatures(
    temperatures: Dict[int, Union[float, Dict[str, float]]],
    path: str,
    metadata: Dict = None
):
    """
    Save temperature dict to JSON file.
    
    Args:
        temperatures: Dict mapping horizon -> temp (or horizon -> {bucket: temp})
        path: Output file path
        metadata: Optional metadata dict to include
    """
    # Convert int keys to strings for JSON
    data = {}
    for k, v in temperatures.items():
        if isinstance(v, dict):
            # Bucketed mode
            data[str(k)] = v
        else:
            data[str(k)] = float(v)
    
    # Add metadata
    data['_metadata'] = metadata or {
        't_min': DEFAULT_T_MIN,
        't_max': DEFAULT_T_MAX,
        'description': 'Per-horizon temperature scaling factors for TCN distribution model'
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Temperatures saved to {path}")


def load_temperatures(path: str) -> Tuple[Dict[int, Union[float, Dict[str, float]]], Dict]:
    """
    Load temperature dict from JSON file.
    
    Returns:
        Tuple of (temperatures_dict, metadata_dict)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    metadata = data.pop('_metadata', {})
    
    # Convert string keys back to ints
    temps = {}
    for k, v in data.items():
        if not k.startswith('_'):
            temps[int(k)] = v
    
    return temps, metadata


def apply_temperatures_dict_with_mode(
    logits_dict: Dict[str, np.ndarray],
    temperatures: Dict[int, Union[float, Dict[str, float]]],
    volatility: np.ndarray = None,
    mode: str = CALIBRATION_MODE_NLL
) -> Dict[str, np.ndarray]:
    """
    Apply per-head temperature scaling, supporting bucketed mode.
    
    Args:
        logits_dict: Dict mapping head_name -> (N, n_bins) logits
        temperatures: Dict mapping horizon -> temp (or horizon -> {bucket: temp})
        volatility: Shape (N,) - required for bucketed mode
        mode: Calibration mode
    
    Returns:
        Dict mapping head_name -> (N, n_bins) calibrated probabilities
    """
    probs_dict = {}
    
    for head_name, logits in logits_dict.items():
        # Extract horizon from head_name (e.g., "h5" -> 5)
        try:
            horizon = int(head_name[1:])
        except (ValueError, KeyError):
            horizon = None
        
        # Get temperature(s) for this horizon
        temp_val = temperatures.get(horizon, temperatures.get(head_name, 1.0))
        
        if isinstance(temp_val, dict) and mode == CALIBRATION_MODE_COVERAGE_BUCKETED:
            # Bucketed mode - apply different T per bucket
            assert volatility is not None, "volatility required for bucketed mode"
            probs_dict[head_name] = apply_temperature_bucketed(logits, volatility, temp_val)
        else:
            # Simple mode - single T
            T = temp_val if isinstance(temp_val, (int, float)) else 1.0
            probs_dict[head_name] = apply_temperature(logits, T)
    
    return probs_dict


# ============================================================================
# Unit Tests
# ============================================================================
def test_uniform_logits():
    """If logits are uniform, any T gives same NLL. Fitted T should be stable."""
    print("Testing uniform logits...", end=" ")
    
    n_samples = 1000
    n_bins = 100
    
    # Uniform logits (all zeros)
    logits = np.zeros((n_samples, n_bins))
    y_true = np.random.randint(0, n_bins, size=n_samples)
    
    fitted_T, info = fit_temperature(logits, y_true, verbose=False)
    
    # T should be finite and positive
    assert np.isfinite(fitted_T), f"T is not finite: {fitted_T}"
    assert fitted_T > 0, f"T is not positive: {fitted_T}"
    
    # Probabilities should sum to 1
    probs = apply_temperature(logits, fitted_T)
    assert np.allclose(probs.sum(axis=1), 1.0), "Probs don't sum to 1"
    
    print("PASSED")


def test_scaled_logits():
    """Test that temperature scaling compensates for logit scaling."""
    print("Testing scaled logits...", end=" ")
    
    n_samples = 1000
    n_bins = 100
    
    # Create well-calibrated logits by simulating from the true distribution
    np.random.seed(42)
    y_true = np.random.randint(10, n_bins - 10, size=n_samples)
    
    # Create logits that peak at the true class with some noise
    # Using log-probabilities as logits (so T=1 is calibrated)
    logits = np.random.randn(n_samples, n_bins) * 0.5
    for i in range(n_samples):
        logits[i, y_true[i]] += 3.0  # Make true class most likely
    
    # Fit T to the original logits - should be close to 1
    fitted_T_base, _ = fit_temperature(logits, y_true, verbose=False)
    
    # Scale logits by factor 2 (makes predictions more confident)
    scale_factor = 2.0
    scaled_logits = logits * scale_factor
    
    # Fit temperature to scaled logits
    fitted_T_scaled, _ = fit_temperature(scaled_logits, y_true, verbose=False)
    
    # The ratio of fitted temperatures should approximately equal the scale factor
    # (T_scaled / T_base ≈ scale_factor) if base was close to 1
    # Or more robustly: fitted_T_scaled should be larger than fitted_T_base
    assert fitted_T_scaled > fitted_T_base, \
        f"Scaled T ({fitted_T_scaled:.2f}) should be larger than base T ({fitted_T_base:.2f})"
    
    # Also verify both are finite and positive
    assert np.isfinite(fitted_T_base) and fitted_T_base > 0
    assert np.isfinite(fitted_T_scaled) and fitted_T_scaled > 0
    
    print("PASSED")


def test_probability_normalization():
    """Ensure apply_temperature produces valid probabilities."""
    print("Testing probability normalization...", end=" ")
    
    n_samples = 100
    n_bins = 50
    
    # Random logits
    np.random.seed(42)
    logits = np.random.randn(n_samples, n_bins) * 3
    
    for T in [0.5, 1.0, 2.0, 5.0]:
        probs = apply_temperature(logits, T)
        
        # Check shape
        assert probs.shape == logits.shape
        
        # Check probabilities sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), f"Probs don't sum to 1 for T={T}"
        
        # Check all probabilities are in [0, 1]
        assert np.all(probs >= 0), f"Negative probabilities for T={T}"
        assert np.all(probs <= 1), f"Probabilities > 1 for T={T}"
        
        # Check all finite
        assert np.all(np.isfinite(probs)), f"Non-finite probabilities for T={T}"
    
    print("PASSED")


def test_nll_computation():
    """Test that NLL computation is correct and numerically stable."""
    print("Testing NLL computation...", end=" ")
    
    n_samples = 100
    n_bins = 50
    
    # Create simple logits where class 0 is most likely
    logits = np.zeros((n_samples, n_bins))
    logits[:, 0] = 10.0  # High logit for class 0
    y_true = np.zeros(n_samples, dtype=np.int32)  # All true labels are 0
    
    nll = compute_nll(logits, y_true, temperature=1.0)
    
    # NLL should be close to 0 (since model is confident and correct)
    assert nll < 1.0, f"NLL too high: {nll}"
    assert np.isfinite(nll), f"NLL is not finite: {nll}"
    
    # Now test with wrong labels - NLL should be high
    y_wrong = np.ones(n_samples, dtype=np.int32) * 25  # Wrong class
    nll_wrong = compute_nll(logits, y_wrong, temperature=1.0)
    assert nll_wrong > nll, "NLL should be higher for wrong predictions"
    
    print("PASSED")


def test_temperature_sharpens_coverage():
    """Test that T<1 sharpens PMF (reduces interval width and coverage)."""
    print("Testing T<1 sharpens coverage...", end=" ")
    
    np.random.seed(42)
    n_samples = 1000
    n_bins = 101  # Bins from -50 to +50
    bin_centers = np.arange(-50, 51)
    
    # Create synthetic logits with moderate confidence centered at 0
    # Gaussian-like logits centered at bin index 50 (value = 0)
    logits = np.zeros((n_samples, n_bins))
    for i in range(n_samples):
        center = 50 + np.random.randint(-10, 11)  # Slight variation in center
        for j in range(n_bins):
            logits[i, j] = -0.1 * (j - center) ** 2 + np.random.randn() * 0.5
    
    # Create actual values near center
    y_deltas = np.random.randn(n_samples) * 8  # Moderate spread
    
    # Compute coverage and width at T=1.0
    cov_base, width_base = compute_coverage_and_width(
        logits, y_deltas, bin_centers, temperature=1.0, q_lo=0.10, q_hi=0.90
    )
    
    # Compute coverage and width at T=0.5 (sharper)
    cov_sharp, width_sharp = compute_coverage_and_width(
        logits, y_deltas, bin_centers, temperature=0.5, q_lo=0.10, q_hi=0.90
    )
    
    # T<1 should reduce width (sharper distribution)
    assert width_sharp < width_base, \
        f"T<1 should reduce width: {width_sharp:.1f} < {width_base:.1f}"
    
    # Coverage should also decrease (narrower intervals miss more actuals)
    assert cov_sharp < cov_base, \
        f"T<1 should reduce coverage: {cov_sharp:.3f} < {cov_base:.3f}"
    
    print("PASSED")


def test_temperature_softens_coverage():
    """Test that T>1 softens PMF (increases interval width and coverage)."""
    print("Testing T>1 softens coverage...", end=" ")
    
    np.random.seed(42)
    n_samples = 1000
    n_bins = 101  # Bins from -50 to +50
    bin_centers = np.arange(-50, 51)
    
    # Create synthetic logits with moderate confidence
    logits = np.zeros((n_samples, n_bins))
    for i in range(n_samples):
        center = 50 + np.random.randint(-10, 11)
        for j in range(n_bins):
            logits[i, j] = -0.1 * (j - center) ** 2 + np.random.randn() * 0.5
    
    # Create actual values near center
    y_deltas = np.random.randn(n_samples) * 8
    
    # Compute coverage and width at T=1.0
    cov_base, width_base = compute_coverage_and_width(
        logits, y_deltas, bin_centers, temperature=1.0, q_lo=0.10, q_hi=0.90
    )
    
    # Compute coverage and width at T=2.0 (softer)
    cov_soft, width_soft = compute_coverage_and_width(
        logits, y_deltas, bin_centers, temperature=2.0, q_lo=0.10, q_hi=0.90
    )
    
    # T>1 should increase width (softer distribution)
    assert width_soft > width_base, \
        f"T>1 should increase width: {width_soft:.1f} > {width_base:.1f}"
    
    # Coverage should also increase (wider intervals capture more actuals)
    assert cov_soft > cov_base, \
        f"T>1 should increase coverage: {cov_soft:.3f} > {cov_base:.3f}"
    
    print("PASSED")


def test_coverage_binary_search_converges():
    """Test that coverage binary search finds the target when achievable."""
    print("Testing coverage binary search convergence...", end=" ")
    
    np.random.seed(42)
    n_samples = 1000
    n_bins = 101
    bin_centers = np.arange(-50, 51)
    
    # Create well-calibrated logits where predictions should match actuals well
    # Each prediction is a Gaussian centered at a random location, and the
    # actual value is sampled from a similar distribution
    logits = np.zeros((n_samples, n_bins))
    y_deltas = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Random predicted center
        pred_center = np.random.uniform(-15, 15)
        pred_center_idx = int(pred_center + 50)  # Map to bin index
        
        # Create logits peaked at pred_center (narrower spread = sharper predictions)
        for j in range(n_bins):
            logits[i, j] = -0.2 * (j - pred_center_idx) ** 2
        
        # Actual value is near predicted center with moderate noise
        # Std=3 means 95% of actuals are within ~6 units of prediction
        y_deltas[i] = pred_center + np.random.randn() * 3
    
    # Check coverage at T=1.0 first (should be high with well-matched predictions)
    cov_t1, _ = compute_coverage_and_width(logits, y_deltas, bin_centers, 1.0)
    
    # If T=1 coverage is already high (>85%), try to fit down to 80%
    # If T=1 coverage is low (<75%), try to fit up to 80%
    target_coverage = 0.80
    
    fitted_T, info = fit_temperature_for_coverage(
        logits, y_deltas, bin_centers,
        target_coverage=target_coverage,
        t_min=0.2, t_max=5.0,
        tolerance=0.02,
        verbose=False
    )
    
    achieved = info['achieved_coverage']
    
    # The key test: binary search should converge to either:
    # 1. The target (if achievable within bounds)
    # 2. The boundary coverage (if target not achievable)
    # Check that the algorithm returns a valid result
    assert 0.2 <= fitted_T <= 5.0, f"T={fitted_T} out of bounds"
    assert achieved >= 0 and achieved <= 1.0, f"Invalid coverage {achieved}"
    
    # The method should be monotonic: coverage at fitted_T should be
    # closer to target than either boundary
    cov_lo, _ = compute_coverage_and_width(logits, y_deltas, bin_centers, 0.2)
    cov_hi, _ = compute_coverage_and_width(logits, y_deltas, bin_centers, 5.0)
    
    # If target is achievable (between cov_lo and cov_hi), verify convergence
    if cov_lo <= target_coverage <= cov_hi or cov_hi <= target_coverage <= cov_lo:
        assert abs(achieved - target_coverage) <= 0.03, \
            f"Coverage {achieved:.3f} should be close to achievable target {target_coverage:.3f}"
    else:
        # Target not achievable, should return boundary
        assert 'note' in info, "Should note when target not achievable"
    
    print("PASSED")


def test_temperature_extremes():
    """Test behavior at temperature boundaries."""
    print("Testing temperature extremes...", end=" ")
    
    np.random.seed(42)
    n_samples = 200
    n_bins = 51
    bin_centers = np.arange(-25, 26)
    
    # Create peaked logits (concentrated)
    logits = np.zeros((n_samples, n_bins))
    logits[:, 25] = 10.0  # Strong peak at center
    
    y_deltas = np.random.randn(n_samples) * 5
    
    # Very low T (0.2) - should make extremely sharp
    cov_low, width_low = compute_coverage_and_width(
        logits, y_deltas, bin_centers, temperature=0.2, q_lo=0.10, q_hi=0.90
    )
    
    # Very high T (5.0) - should make very flat
    cov_high, width_high = compute_coverage_and_width(
        logits, y_deltas, bin_centers, temperature=5.0, q_lo=0.10, q_hi=0.90
    )
    
    # High T should give much wider intervals than low T
    assert width_high > width_low * 2, \
        f"High T width ({width_high:.1f}) should be much larger than low T ({width_low:.1f})"
    
    print("PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING - UNIT TESTS")
    print("=" * 70)
    
    # Basic tests
    test_uniform_logits()
    test_scaled_logits()
    test_probability_normalization()
    test_nll_computation()
    
    # Coverage-related tests (T<1 and T>1 behavior)
    test_temperature_sharpens_coverage()
    test_temperature_softens_coverage()
    test_coverage_binary_search_converges()
    test_temperature_extremes()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()

