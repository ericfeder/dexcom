"""
Temperature Scaling for TCN Distribution Models

Post-hoc calibration technique that learns a scalar temperature T for each
prediction horizon. Temperature scaling divides logits by T before applying
softmax, which adjusts the confidence of predictions without changing the
ranking.

For well-calibrated models, T ≈ 1.0
For overconfident models, T > 1.0 (softens the distribution)
For underconfident models, T < 1.0 (sharpens the distribution)

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# Default temperature bounds
DEFAULT_T_MIN = 0.2
DEFAULT_T_MAX = 5.0


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


def fit_temperatures_all_horizons(
    horizon_data: Dict[int, Dict],
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Fit temperature for each prediction horizon.
    
    Args:
        horizon_data: Dict mapping horizon (int) to {'logits': np.ndarray, 'y_bins': np.ndarray}
        t_min: Minimum allowed temperature
        t_max: Maximum allowed temperature
        verbose: Print fitting progress
    
    Returns:
        Dict mapping horizon (int) to fitted temperature (float)
    """
    temperatures = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print("TEMPERATURE SCALING - FITTING PER-HORIZON TEMPERATURES")
        print("=" * 70)
    
    for horizon in sorted(horizon_data.keys()):
        data = horizon_data[horizon]
        logits = data['logits']
        y_bins = data['y_bins']
        
        if verbose:
            print(f"\n+{horizon} min horizon (n={len(logits):,})...")
        
        fitted_T, info = fit_temperature(
            logits, y_bins,
            t_min=t_min, t_max=t_max,
            verbose=False
        )
        
        temperatures[horizon] = fitted_T
        
        if verbose:
            print(f"  Temperature: {fitted_T:.4f}")
            print(f"  NLL: {info['nll_before']:.4f} → {info['nll_after']:.4f} "
                  f"(Δ = {info['nll_improvement']:+.4f})")
    
    return temperatures


def save_temperatures(temperatures: Dict[int, float], path: str):
    """Save temperature dict to JSON file."""
    # Convert int keys to strings for JSON
    data = {str(k): v for k, v in temperatures.items()}
    data['_metadata'] = {
        't_min': DEFAULT_T_MIN,
        't_max': DEFAULT_T_MAX,
        'description': 'Per-horizon temperature scaling factors for TCN distribution model'
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Temperatures saved to {path}")


def load_temperatures(path: str) -> Dict[int, float]:
    """Load temperature dict from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Convert string keys back to ints, skip metadata
    return {int(k): v for k, v in data.items() if not k.startswith('_')}


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


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING - UNIT TESTS")
    print("=" * 70)
    
    test_uniform_logits()
    test_scaled_logits()
    test_probability_normalization()
    test_nll_computation()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()

