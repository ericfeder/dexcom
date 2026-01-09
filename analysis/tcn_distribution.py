"""
TCN Distribution Model for Glucose Delta Prediction

Outputs a full predictive distribution (horizon-specific PMF) for glucose delta prediction.
Uses soft targets and smoothness regularization to avoid jagged distributions.
Includes post-hoc temperature scaling for improved calibration.
Compares against GBM quantile regression baseline.

Usage:
    python tcn_distribution.py              # Train 15-min model only
    python tcn_distribution.py --all        # Train all horizons (6 separate models)
    python tcn_distribution.py --multihead  # Train ONE multi-horizon model with 6 heads
    python tcn_distribution.py --calibrate  # Apply temperature scaling to existing models
    python tcn_distribution.py --test       # Run unit tests only

Multi-Horizon Model (--multihead):
    Trains a single shared-backbone TCN with 6 output heads (one per horizon).
    Includes cross-horizon coherence regularization:
    - Curve smoothness penalty: reduces jumpiness in predicted means across horizons
    - Variance monotonicity penalty: ensures variance increases with horizon
    
    Config flags (top of file):
    - USE_SHARED_BINS: If True, all heads output 341 bins (default: True)
    - LAMBDA_CURVE: Weight for curve smoothness penalty (default: 1e-3)
    - LAMBDA_VARMONO: Weight for variance monotonicity penalty (default: 1e-3)

Temperature Scaling:
    After training with --all or --multihead, per-horizon temperatures are fitted on
    the validation set and saved to ../data/tcn_temps.json (or tcn_multihead_temps.json).
    These temperatures are applied at inference time by computing softmax(logits / T)
    to improve calibration.
"""
import os
import sys

# Set TensorFlow env vars BEFORE importing tf to avoid graph compilation issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import warnings
from zoneinfo import ZoneInfo

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add, ZeroPadding1D,
    GlobalAveragePooling1D, Softmax, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Disable eager execution for faster training (set to False at module load)
tf.config.run_functions_eagerly(False)

# Import temperature scaling module
from temperature_scaling import (
    fit_temperature, apply_temperature, apply_temperatures_dict,
    fit_temperatures_all_horizons, save_temperatures, load_temperatures, compute_nll
)

# ============================================================================
# Configuration
# ============================================================================
# Horizon-specific binning configuration (99.9% coverage of actual deltas)
# Each horizon has different bin ranges based on empirical delta distributions
BIN_CONFIG = {
    5:  {'min': -60,  'max': 70},   # 131 bins
    10: {'min': -80,  'max': 90},   # 171 bins
    15: {'min': -100, 'max': 120},  # 221 bins
    20: {'min': -120, 'max': 150},  # 271 bins
    25: {'min': -140, 'max': 170},  # 311 bins
    30: {'min': -150, 'max': 190},  # 341 bins
}

def get_bin_config(horizon):
    """Get binning configuration for a specific horizon."""
    config = BIN_CONFIG[horizon]
    delta_min = config['min']
    delta_max = config['max']
    n_bins = delta_max - delta_min + 1
    bin_edges = np.linspace(delta_min - 0.5, delta_max + 0.5, n_bins + 1)
    bin_centers = np.arange(delta_min, delta_max + 1)
    return {
        'n_bins': n_bins,
        'delta_min': delta_min,
        'delta_max': delta_max,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers
    }

# Legacy globals for backward compatibility (default to 15-min horizon)
_default_config = get_bin_config(15)
N_BINS = _default_config['n_bins']
DELTA_MIN = _default_config['delta_min']
DELTA_MAX = _default_config['delta_max']
BIN_WIDTH = 1.0   # mg/dL (always 1)
BIN_EDGES = _default_config['bin_edges']
BIN_CENTERS = _default_config['bin_centers']

# Model configuration
PREDICTION_HORIZONS = [5, 10, 15, 20, 25, 30]  # All horizons to train (minutes)
PREDICTION_HORIZON = 15  # Default horizon for single-model mode (minutes ahead)
LOOKBACK_MINUTES = 40    # minutes of history (configurable)
SAMPLE_INTERVAL = 5      # minutes between readings
SEQ_LEN = LOOKBACK_MINUTES // SAMPLE_INTERVAL  # 8 samples for 40 min

# Feature configuration
INCLUDE_FIRST_DIFF = True  # Include velocity (first difference) feature

# Training hyperparameters
SIGMA_BINS = 4.0          # Soft target Gaussian smoothing (in bins = mg/dL)
LAMBDA_SMOOTH = 1e-4      # Smoothness regularizer weight
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE_EARLY_STOP = 10
PATIENCE_LR_REDUCE = 5
LR_REDUCE_FACTOR = 0.5
DROPOUT_RATE = 0.1

# TCN architecture
TCN_FILTERS = 32
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]

# Data configuration
DATA_MONTHS = 6  # Use last N months of data
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20  # Identical to GBM test set

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# Multi-Horizon Configuration
# ============================================================================
USE_SHARED_BINS = True     # If True, all heads output 341 bins (range [-150, +190])
LAMBDA_CURVE = 0           # Curve smoothness OFF - backbone already provides smoothness
LAMBDA_VARMONO = 1e-3      # Variance monotonicity penalty weight (guardrail only)

# Curve penalty type: "l2" (sum of squared 2nd diff), "l1" (sum of abs), "huber"
CURVE_PENALTY_TYPE = "l2"
HUBER_DELTA = 2.0          # Delta parameter for Huber loss

# Two-stage training: first train without curve penalty, then fine-tune with it
TWO_STAGE_TRAINING = False
STAGE_B_EPOCHS = 10        # Number of fine-tuning epochs in stage B
STAGE_B_LR = 1e-4          # Learning rate for stage B
STAGE_B_LAMBDA_CURVE = 1e-4  # Curve penalty for stage B (if TWO_STAGE_TRAINING)

# Per-head adapter layers (reduces negative transfer)
USE_HEAD_ADAPTER = True    # Enable adapters to allow horizon-specific specialization
HEAD_ADAPTER_DIM = 64      # Hidden dim of adapter layer

# Volatility-aware sample weighting (targets high-volatility MAE gap)
VOLATILITY_WEIGHTING_ENABLED = True
VOL_WEIGHT_LOW = 1.0       # Weight for low volatility samples
VOL_WEIGHT_MED = 1.3       # Weight for medium volatility samples
VOL_WEIGHT_HIGH = 2.0      # Weight for high volatility samples
VOL_THRESH_LOW = 0.5       # mg/dL per min threshold for low->med
VOL_THRESH_HIGH = 1.5      # mg/dL per min threshold for med->high

# Auxiliary point loss (volatility-weighted MAE/Huber on predicted mean)
# This allows volatility weighting to improve MAE without inflating PMF uncertainty
POINT_LOSS_ENABLED = True
POINT_LOSS_WEIGHT = 0.1    # Weight for point loss vs PMF loss (try 0.05, 0.1, 0.2)
POINT_LOSS_TYPE = "huber"  # "huber" or "mae"
POINT_LOSS_HUBER_DELTA = 2.0  # Delta parameter for Huber loss

# Tail mass thresholds for diagnostics (mg/dL per horizon)
TAIL_MASS_THRESHOLDS = {5: 20, 10: 35, 15: 50, 20: 65, 25: 80, 30: 95}

# Head names for multi-horizon model
HEAD_NAMES = [f"h{h}" for h in PREDICTION_HORIZONS]  # ["h5", "h10", ..., "h30"]

# Sweep configuration for experiments
LAMBDA_CURVE_SWEEP_VALUES = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

def get_shared_bin_config():
    """Get shared bin configuration for multi-horizon model (uses 30-min range)."""
    delta_min = -150
    delta_max = 190
    n_bins = delta_max - delta_min + 1  # 341 bins
    bin_edges = np.linspace(delta_min - 0.5, delta_max + 0.5, n_bins + 1)
    bin_centers = np.arange(delta_min, delta_max + 1)
    return {
        'n_bins': n_bins,
        'delta_min': delta_min,
        'delta_max': delta_max,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers
    }

SHARED_BIN_CONFIG = get_shared_bin_config()

print(f"Configuration:")
print(f"  Prediction horizon: {PREDICTION_HORIZON} min")
print(f"  Lookback window: {LOOKBACK_MINUTES} min ({SEQ_LEN} samples)")
print(f"  Bins: {N_BINS} bins from {DELTA_MIN} to {DELTA_MAX} mg/dL")
print(f"  Soft target sigma: {SIGMA_BINS} bins")
print(f"  Smoothness lambda: {LAMBDA_SMOOTH}")
print(f"  Include velocity: {INCLUDE_FIRST_DIFF}")
print(f"  Random seed: {RANDOM_SEED}")


# ============================================================================
# Binning Utilities
# ============================================================================
def delta_to_bin(delta, horizon=15):
    """
    Convert delta (mg/dL) to bin index for a specific horizon.
    Clamps to [0, n_bins-1] for out-of-range values.
    
    Args:
        delta: Scalar or array of delta values in mg/dL
        horizon: Prediction horizon in minutes (5, 10, 15, 20, 25, 30)
    
    Returns:
        Bin index where bin 0 corresponds to delta_min
    """
    config = get_bin_config(horizon)
    delta = np.asarray(delta)
    clamped = np.clip(delta, config['delta_min'], config['delta_max'])
    bin_idx = np.round(clamped - config['delta_min']).astype(int)
    return bin_idx


def bin_to_delta(bin_idx, horizon=15):
    """
    Convert bin index to delta value (bin center in mg/dL).
    
    Args:
        bin_idx: Scalar or array of bin indices
        horizon: Prediction horizon in minutes
    
    Returns:
        Delta value at bin center in mg/dL
    """
    config = get_bin_config(horizon)
    bin_idx = np.asarray(bin_idx)
    return config['bin_centers'][bin_idx]


# ============================================================================
# PMF Diagnostics: Tail Mass and HDI Width
# ============================================================================
def compute_tail_mass(probs, bin_centers, threshold_K):
    """
    Compute P(|delta| > K) from PMF - measures tail probability.
    
    Higher tail mass indicates the model is hedging by putting probability
    mass far from the predicted mean.
    
    Args:
        probs: (n_samples, n_bins) probability distributions
        bin_centers: (n_bins,) center values for each bin
        threshold_K: Threshold in mg/dL - bins with |center| > K are "tails"
    
    Returns:
        Mean tail mass across all samples (scalar)
    """
    probs = np.asarray(probs)
    bin_centers = np.asarray(bin_centers)
    tail_mask = np.abs(bin_centers) > threshold_K
    tail_probs = probs[:, tail_mask]
    return np.mean(np.sum(tail_probs, axis=1))


def compute_hdi_width(probs, bin_centers, credible_mass=0.80):
    """
    Compute HDI (Highest Density Interval) - the shortest contiguous
    interval containing the specified probability mass.
    
    HDI is a better measure of "useful" interval width than P90-P10 because
    it finds the densest region regardless of distribution shape.
    
    Args:
        probs: (n_samples, n_bins) probability distributions
        bin_centers: (n_bins,) center values for each bin
        credible_mass: Target probability mass (default 0.80 for 80% interval)
    
    Returns:
        Mean HDI width across all samples (scalar)
    """
    probs = np.asarray(probs)
    bin_centers = np.asarray(bin_centers)
    n_samples, n_bins = probs.shape
    
    hdi_widths = []
    for p in probs:
        # Cumulative sum for sliding window
        cumsum = np.cumsum(p)
        
        # Find shortest interval containing credible_mass
        best_width = bin_centers[-1] - bin_centers[0]  # Max possible
        
        for start_idx in range(n_bins):
            # Find end index where cumsum reaches target
            target_cumsum = (cumsum[start_idx - 1] if start_idx > 0 else 0) + credible_mass
            end_idx = np.searchsorted(cumsum, target_cumsum)
            
            if end_idx < n_bins:
                width = bin_centers[end_idx] - bin_centers[start_idx]
                if width < best_width:
                    best_width = width
        
        hdi_widths.append(best_width)
    
    return np.mean(hdi_widths)


def compute_pmf_diagnostics(probs, bin_centers, horizon, y_deltas=None):
    """
    Compute comprehensive PMF diagnostics for a single horizon.
    
    Args:
        probs: (n_samples, n_bins) probability distributions
        bin_centers: (n_bins,) bin center values
        horizon: Prediction horizon in minutes (for threshold lookup)
        y_deltas: Optional (n_samples,) actual delta values for MAE
    
    Returns:
        Dict with: tail_mass, hdi_width, p90_p10_width, coverage (if y_deltas), mae (if y_deltas)
    """
    probs = np.asarray(probs)
    bin_centers = np.asarray(bin_centers)
    
    # Tail mass
    threshold_K = TAIL_MASS_THRESHOLDS.get(horizon, 50)
    tail_mass = compute_tail_mass(probs, bin_centers, threshold_K)
    
    # HDI width
    hdi_width = compute_hdi_width(probs, bin_centers, credible_mass=0.80)
    
    # P90-P10 width
    cumsum = np.cumsum(probs, axis=1)
    q10_idx = np.argmax(cumsum >= 0.10, axis=1)
    q90_idx = np.argmax(cumsum >= 0.90, axis=1)
    q10 = bin_centers[q10_idx]
    q90 = bin_centers[q90_idx]
    p90_p10_width = np.mean(q90 - q10)
    
    result = {
        'tail_mass': tail_mass,
        'hdi_width': hdi_width,
        'p90_p10_width': p90_p10_width,
    }
    
    # If we have true values, compute coverage and MAE
    if y_deltas is not None:
        y_deltas = np.asarray(y_deltas)
        coverage = np.mean((y_deltas >= q10) & (y_deltas <= q90))
        result['coverage'] = coverage
        
        # Predicted mean (q50 proxy)
        pred_mean = np.sum(probs * bin_centers, axis=1)
        mae = np.mean(np.abs(pred_mean - y_deltas))
        result['mae'] = mae
    
    return result


def make_soft_target(true_bin, n_bins, sigma_bins=SIGMA_BINS):
    """
    Create a soft target distribution (Gaussian around true bin).
    
    Args:
        true_bin: Integer bin index for the true delta value
        n_bins: Number of bins for this horizon
        sigma_bins: Standard deviation of Gaussian in bins (= mg/dL since bin_width=1)
    
    Returns:
        Normalized probability vector of length n_bins
    """
    bin_indices = np.arange(n_bins)
    weights = np.exp(-0.5 * ((bin_indices - true_bin) / sigma_bins) ** 2)
    return weights / weights.sum()


def make_soft_targets_batch(true_bins, n_bins, sigma_bins=SIGMA_BINS):
    """
    Create soft targets for a batch of true bin indices.
    
    Args:
        true_bins: Array of shape (batch,) with integer bin indices
        n_bins: Number of bins for this horizon
        sigma_bins: Standard deviation of Gaussian
    
    Returns:
        Array of shape (batch, n_bins) with normalized probabilities
    """
    bin_indices = np.arange(n_bins)[np.newaxis, :]  # (1, n_bins)
    true_bins = np.asarray(true_bins)[:, np.newaxis]  # (batch, 1)
    weights = np.exp(-0.5 * ((bin_indices - true_bins) / sigma_bins) ** 2)
    return weights / weights.sum(axis=1, keepdims=True)


def quantile_from_pmf(probs, q, bin_edges):
    """
    Compute quantile from PMF using CDF and linear interpolation.
    
    Args:
        probs: Array of shape (N_BINS,) with probabilities summing to 1
        q: Quantile value in [0, 1] (e.g., 0.5 for median)
        bin_edges: Array of shape (N_BINS+1,) with bin edge values
    
    Returns:
        Quantile value in mg/dL
    """
    probs = np.asarray(probs)
    cdf = np.cumsum(probs)
    
    # Find smallest k where CDF >= q
    k = np.searchsorted(cdf, q)
    
    # Handle edge cases
    if k == 0:
        return bin_edges[0]
    if k >= len(probs):
        return bin_edges[-1]
    
    # Linear interpolation within bin k
    cdf_prev = cdf[k - 1]
    alpha = (q - cdf_prev) / max(probs[k], 1e-9)
    alpha = np.clip(alpha, 0, 1)  # Guard against numerical issues
    
    return bin_edges[k] + alpha * (bin_edges[k + 1] - bin_edges[k])


def quantiles_from_pmf_batch(probs_batch, quantiles, bin_edges):
    """
    Compute multiple quantiles from a batch of PMFs.
    
    Args:
        probs_batch: Array of shape (batch, n_bins)
        quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9])
        bin_edges: Array of shape (n_bins+1,) with bin edge values
    
    Returns:
        Dict mapping quantile -> array of shape (batch,)
    """
    results = {}
    for q in quantiles:
        q_values = np.array([quantile_from_pmf(p, q, bin_edges) for p in probs_batch])
        results[q] = q_values
    return results


def pmf_mean(probs, bin_centers):
    """
    Compute expected value (mean) from PMF.
    
    Args:
        probs: Array of shape (n_bins,) or (batch, n_bins)
        bin_centers: Array of shape (n_bins,) with bin center values
    
    Returns:
        Mean value(s) in mg/dL
    """
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return np.sum(probs * bin_centers)
    else:
        return np.sum(probs * bin_centers[np.newaxis, :], axis=1)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def parse_to_local_time(ts_series):
    """Parse timestamps to Eastern local time."""
    eastern = ZoneInfo('America/New_York')
    parsed = pd.to_datetime(ts_series, format='mixed', utc=False)
    result = []
    for ts in parsed:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=eastern)
        else:
            ts = ts.astimezone(eastern)
        result.append(ts)
    return pd.to_datetime(result)


def load_glucose_data():
    """Load and preprocess glucose readings."""
    print("\nLoading glucose data...", flush=True)
    df = pd.read_csv('../data/readings.csv')
    df['displayTime'] = parse_to_local_time(df['displayTime'])
    df = df.sort_values('displayTime').reset_index(drop=True)
    print(f"Loaded {len(df):,} glucose readings", flush=True)
    print(f"Date range: {df['displayTime'].min().strftime('%Y-%m-%d')} to {df['displayTime'].max().strftime('%Y-%m-%d')}")
    return df


def create_sequences(df, seq_len=SEQ_LEN, horizon_minutes=PREDICTION_HORIZON,
                     include_velocity=INCLUDE_FIRST_DIFF):
    """
    Create sliding window sequences for TCN input.
    
    Features:
        x0: (G[t-k] - G[t]) / 50  - normalized relative glucose
        x1: (G[t-k] - G[t-k-1]) / 10  - velocity (optional)
    
    Target:
        bin index for delta = G[t+horizon] - G[t]
    
    Args:
        df: DataFrame with 'value' column (glucose readings)
        seq_len: Number of past samples in window
        horizon_minutes: Prediction horizon in minutes (5, 10, 15, 20, 25, 30)
        include_velocity: Whether to include first-difference velocity feature
    
    Returns:
        X: Array of shape (n_samples, seq_len, n_channels)
        y_bins: Array of shape (n_samples,) with bin indices
        y_deltas: Array of shape (n_samples,) with raw delta values
        valid_indices: Indices into original DataFrame
    """
    values = df['value'].values
    n = len(values)
    horizon_steps = horizon_minutes // SAMPLE_INTERVAL
    
    # Need: seq_len samples before (plus 1 more for velocity), and horizon_steps after
    # For velocity we need values[idx - seq_len] to exist, so start_idx >= seq_len
    start_idx = seq_len  # This ensures we have seq_len+1 past values for velocity
    end_idx = n - 1 - horizon_steps
    
    if end_idx <= start_idx:
        raise ValueError("Not enough data for the given sequence length and horizon")
    
    n_samples = end_idx - start_idx + 1
    n_channels = 2 if include_velocity else 1
    
    X = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    y_deltas = np.zeros(n_samples, dtype=np.float32)
    valid_indices = np.arange(start_idx, end_idx + 1)
    
    for i, idx in enumerate(valid_indices):
        # Current glucose value at time t
        g_current = values[idx]
        
        # Future glucose value at t + horizon
        g_future = values[idx + horizon_steps]
        
        # Delta target
        delta = g_future - g_current
        y_deltas[i] = delta
        
        # Past glucose values: [t-seq_len+1, ..., t-1, t]
        past_values = values[idx - seq_len + 1:idx + 1]  # seq_len values
        
        # Feature x0: normalized relative glucose (G[t-k] - G[t]) / 50
        # This creates a sequence where the last value is 0 (current - current)
        x0 = (past_values - g_current) / 50.0
        X[i, :, 0] = x0
        
        # Feature x1: velocity (first difference) (G[t-k] - G[t-k-1]) / 10
        if include_velocity:
            # Need one more past value for velocity at the first step
            # extended_past has seq_len + 1 values: [t-seq_len, t-seq_len+1, ..., t]
            extended_past = values[idx - seq_len:idx + 1]  # seq_len + 1 values
            velocity = np.diff(extended_past) / 10.0  # seq_len values
            X[i, :, 1] = velocity
    
    # Convert deltas to bin indices (horizon-specific bins)
    y_bins = delta_to_bin(y_deltas, horizon=horizon_minutes)
    
    return X, y_bins, y_deltas, valid_indices


def filter_valid_samples(df, X, y_bins, y_deltas, valid_indices, horizon_minutes=None):
    """
    Filter samples to ensure valid glucose values (40-400 mg/dL range).
    
    Args:
        df: Original DataFrame
        X, y_bins, y_deltas, valid_indices: Output from create_sequences
        horizon_minutes: Prediction horizon in minutes (if None, uses default)
    
    Returns:
        Filtered versions of X, y_bins, y_deltas, valid_indices
    """
    values = df['value'].values
    
    # Check current and future values are in valid range
    if horizon_minutes is None:
        horizon_minutes = PREDICTION_HORIZON
    horizon_steps = horizon_minutes // SAMPLE_INTERVAL
    
    valid_mask = np.ones(len(valid_indices), dtype=bool)
    
    for i, idx in enumerate(valid_indices):
        # Current value
        g_current = values[idx]
        g_future = values[idx + horizon_steps]
        
        # Check sensor limits
        if g_current < 40 or g_current > 400:
            valid_mask[i] = False
        if g_future < 40 or g_future > 400:
            valid_mask[i] = False
        
        # Check all past values in the window
        past_values = values[idx - SEQ_LEN + 1:idx + 1]
        if np.any(past_values < 40) or np.any(past_values > 400):
            valid_mask[i] = False
    
    return (X[valid_mask], y_bins[valid_mask], y_deltas[valid_mask], 
            valid_indices[valid_mask])


def prepare_data_splits(df, horizon_minutes=PREDICTION_HORIZON, verbose=True):
    """
    Prepare train/val/test splits with 70/10/20 ratio.
    Test set is identical to GBM (last 20%).
    
    Args:
        df: DataFrame with glucose readings
        horizon_minutes: Prediction horizon in minutes
        verbose: Whether to print progress
    
    Returns:
        Dictionary with train/val/test data
    """
    # Filter to last N months of data
    latest_date = df['displayTime'].max()
    cutoff_date = latest_date - pd.DateOffset(months=DATA_MONTHS)
    df_filtered = df[df['displayTime'] >= cutoff_date].copy().reset_index(drop=True)
    if verbose:
        print(f"Using last {DATA_MONTHS} months: {cutoff_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        print(f"Filtered samples: {len(df_filtered):,}")
    
    # Create sequences with specified horizon
    X, y_bins, y_deltas, valid_indices = create_sequences(
        df_filtered, 
        seq_len=SEQ_LEN, 
        horizon_minutes=horizon_minutes,
        include_velocity=INCLUDE_FIRST_DIFF
    )
    if verbose:
        print(f"Created {len(X):,} sequences")
    
    # Filter invalid samples
    X, y_bins, y_deltas, valid_indices = filter_valid_samples(
        df_filtered, X, y_bins, y_deltas, valid_indices, horizon_minutes=horizon_minutes
    )
    if verbose:
        print(f"After filtering: {len(X):,} valid sequences")
    
    # Time-based split: 70% train, 10% val, 20% test
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    splits = {
        'train': {
            'X': X[:train_end],
            'y_bins': y_bins[:train_end],
            'y_deltas': y_deltas[:train_end],
            'indices': valid_indices[:train_end]
        },
        'val': {
            'X': X[train_end:val_end],
            'y_bins': y_bins[train_end:val_end],
            'y_deltas': y_deltas[train_end:val_end],
            'indices': valid_indices[train_end:val_end]
        },
        'test': {
            'X': X[val_end:],
            'y_bins': y_bins[val_end:],
            'y_deltas': y_deltas[val_end:],
            'indices': valid_indices[val_end:]
        },
        'df': df_filtered  # Keep reference to filtered DataFrame
    }
    
    if verbose:
        print(f"\nData splits:")
        print(f"  Train: {len(splits['train']['X']):,} samples ({100*TRAIN_RATIO:.0f}%)")
        print(f"  Val:   {len(splits['val']['X']):,} samples ({100*VAL_RATIO:.0f}%)")
        print(f"  Test:  {len(splits['test']['X']):,} samples ({100*TEST_RATIO:.0f}%)")
    
    return splits


# ============================================================================
# Multi-Horizon Data Pipeline
# ============================================================================
def create_sequences_multihorizon(df, seq_len=SEQ_LEN, include_velocity=INCLUDE_FIRST_DIFF):
    """
    Create sliding window sequences for multi-horizon TCN input.
    
    For each anchor time t, builds ONE input X and SIX labels (one per horizon).
    Drops samples where ANY horizon's future value is missing/invalid.
    
    Features:
        x0: (G[t-k] - G[t]) / 50  - normalized relative glucose
        x1: (G[t-k] - G[t-k-1]) / 10  - velocity (optional)
    
    Targets:
        bin index for delta = G[t+horizon] - G[t] for each horizon
    
    Args:
        df: DataFrame with 'value' column (glucose readings)
        seq_len: Number of past samples in window
        include_velocity: Whether to include first-difference velocity feature
    
    Returns:
        X: Array of shape (n_samples, seq_len, n_channels)
        y_bins: Dict mapping head name -> (n_samples,) int bin indices
        y_deltas: Dict mapping head name -> (n_samples,) float deltas
        valid_indices: Indices into original DataFrame
    """
    values = df['value'].values
    n = len(values)
    
    # Use longest horizon to determine end index
    max_horizon_steps = max(PREDICTION_HORIZONS) // SAMPLE_INTERVAL
    
    # Need: seq_len samples before (plus 1 more for velocity), and max_horizon_steps after
    start_idx = seq_len
    end_idx = n - 1 - max_horizon_steps
    
    if end_idx <= start_idx:
        raise ValueError("Not enough data for the given sequence length and horizons")
    
    n_samples = end_idx - start_idx + 1
    n_channels = 2 if include_velocity else 1
    
    X = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    y_deltas = {f"h{h}": np.zeros(n_samples, dtype=np.float32) for h in PREDICTION_HORIZONS}
    valid_indices = np.arange(start_idx, end_idx + 1)
    
    for i, idx in enumerate(valid_indices):
        # Current glucose value at time t
        g_current = values[idx]
        
        # Compute deltas for all horizons
        for horizon in PREDICTION_HORIZONS:
            horizon_steps = horizon // SAMPLE_INTERVAL
            g_future = values[idx + horizon_steps]
            delta = g_future - g_current
            y_deltas[f"h{horizon}"][i] = delta
        
        # Past glucose values: [t-seq_len+1, ..., t-1, t]
        past_values = values[idx - seq_len + 1:idx + 1]
        
        # Feature x0: normalized relative glucose
        x0 = (past_values - g_current) / 50.0
        X[i, :, 0] = x0
        
        # Feature x1: velocity (first difference)
        if include_velocity:
            extended_past = values[idx - seq_len:idx + 1]
            velocity = np.diff(extended_past) / 10.0
            X[i, :, 1] = velocity
    
    # Convert deltas to bin indices
    y_bins = {}
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        if USE_SHARED_BINS:
            # Use shared bin config for all horizons
            deltas = y_deltas[head_name]
            clamped = np.clip(deltas, SHARED_BIN_CONFIG['delta_min'], SHARED_BIN_CONFIG['delta_max'])
            y_bins[head_name] = np.round(clamped - SHARED_BIN_CONFIG['delta_min']).astype(int)
        else:
            # Use horizon-specific bin config
            y_bins[head_name] = delta_to_bin(y_deltas[head_name], horizon=horizon)
    
    # Compute per-sample volatility from last velocity value
    # Velocity feature is (G[t] - G[t-1]) / 10, actual rate = velocity * 10 / 5min = velocity * 2
    if include_velocity:
        volatility = np.abs(X[:, -1, 1]) * 2.0  # mg/dL per minute
    else:
        # Fallback: compute from raw glucose values
        volatility = np.abs(values[valid_indices] - values[valid_indices - 1]) / 5.0
    
    return X, y_bins, y_deltas, valid_indices, volatility


def filter_valid_samples_multihorizon(df, X, y_bins, y_deltas, valid_indices, volatility):
    """
    Filter samples to ensure valid glucose values for ALL horizons.
    
    Drops samples where ANY horizon's future value is outside valid range (40-400 mg/dL).
    
    Args:
        df: Original DataFrame
        X, y_bins, y_deltas, valid_indices, volatility: Output from create_sequences_multihorizon
    
    Returns:
        Filtered versions of X, y_bins, y_deltas, valid_indices, volatility
    """
    values = df['value'].values
    valid_mask = np.ones(len(valid_indices), dtype=bool)
    
    for i, idx in enumerate(valid_indices):
        # Check current value
        g_current = values[idx]
        if g_current < 40 or g_current > 400:
            valid_mask[i] = False
            continue
        
        # Check all past values in the window
        past_values = values[idx - SEQ_LEN + 1:idx + 1]
        if np.any(past_values < 40) or np.any(past_values > 400):
            valid_mask[i] = False
            continue
        
        # Check future values for ALL horizons
        for horizon in PREDICTION_HORIZONS:
            horizon_steps = horizon // SAMPLE_INTERVAL
            g_future = values[idx + horizon_steps]
            if g_future < 40 or g_future > 400:
                valid_mask[i] = False
                break
    
    # Apply mask
    X_filtered = X[valid_mask]
    valid_indices_filtered = valid_indices[valid_mask]
    volatility_filtered = volatility[valid_mask]
    
    y_bins_filtered = {}
    y_deltas_filtered = {}
    for head_name in y_bins:
        y_bins_filtered[head_name] = y_bins[head_name][valid_mask]
        y_deltas_filtered[head_name] = y_deltas[head_name][valid_mask]
    
    return X_filtered, y_bins_filtered, y_deltas_filtered, valid_indices_filtered, volatility_filtered


def prepare_data_splits_multihorizon(df, verbose=True):
    """
    Prepare train/val/test splits for multi-horizon model with 70/10/20 ratio.
    
    Args:
        df: DataFrame with glucose readings
        verbose: Whether to print progress
    
    Returns:
        Dictionary with train/val/test data, each containing:
            X: input features
            y_bins: dict of bin indices per head
            y_deltas: dict of delta values per head
            indices: indices into df
    """
    # Filter to last N months of data
    latest_date = df['displayTime'].max()
    cutoff_date = latest_date - pd.DateOffset(months=DATA_MONTHS)
    df_filtered = df[df['displayTime'] >= cutoff_date].copy().reset_index(drop=True)
    if verbose:
        print(f"Using last {DATA_MONTHS} months: {cutoff_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        print(f"Filtered samples: {len(df_filtered):,}")
    
    # Create sequences for all horizons at once
    X, y_bins, y_deltas, valid_indices, volatility = create_sequences_multihorizon(
        df_filtered, 
        seq_len=SEQ_LEN,
        include_velocity=INCLUDE_FIRST_DIFF
    )
    if verbose:
        print(f"Created {len(X):,} sequences")
    
    # Filter invalid samples (drops if ANY horizon is invalid)
    X, y_bins, y_deltas, valid_indices, volatility = filter_valid_samples_multihorizon(
        df_filtered, X, y_bins, y_deltas, valid_indices, volatility
    )
    if verbose:
        print(f"After filtering: {len(X):,} valid sequences")
    
    # Time-based split: 70% train, 10% val, 20% test
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    def split_dict(d, start, end):
        return {k: v[start:end] for k, v in d.items()}
    
    splits = {
        'train': {
            'X': X[:train_end],
            'y_bins': split_dict(y_bins, 0, train_end),
            'y_deltas': split_dict(y_deltas, 0, train_end),
            'indices': valid_indices[:train_end],
            'volatility': volatility[:train_end]
        },
        'val': {
            'X': X[train_end:val_end],
            'y_bins': split_dict(y_bins, train_end, val_end),
            'y_deltas': split_dict(y_deltas, train_end, val_end),
            'indices': valid_indices[train_end:val_end],
            'volatility': volatility[train_end:val_end]
        },
        'test': {
            'X': X[val_end:],
            'y_bins': split_dict(y_bins, val_end, n),
            'y_deltas': split_dict(y_deltas, val_end, n),
            'indices': valid_indices[val_end:],
            'volatility': volatility[val_end:]
        },
        'df': df_filtered
    }
    
    if verbose:
        print(f"\nData splits:")
        print(f"  Train: {len(splits['train']['X']):,} samples ({100*TRAIN_RATIO:.0f}%)")
        print(f"  Val:   {len(splits['val']['X']):,} samples ({100*VAL_RATIO:.0f}%)")
        print(f"  Test:  {len(splits['test']['X']):,} samples ({100*TEST_RATIO:.0f}%)")
        
        # Show bin config info
        if USE_SHARED_BINS:
            print(f"\n  Using shared bins: {SHARED_BIN_CONFIG['n_bins']} bins [{SHARED_BIN_CONFIG['delta_min']}, {SHARED_BIN_CONFIG['delta_max']}]")
        else:
            print(f"\n  Using horizon-specific bins")
    
    return splits


def compute_volatility_weights(volatility, 
                               thresh_low=None, 
                               thresh_high=None,
                               weight_low=None,
                               weight_med=None,
                               weight_high=None):
    """
    Compute sample weights based on volatility buckets.
    
    Higher volatility samples get higher weights to address the MAE gap
    observed in high-volatility periods.
    
    Args:
        volatility: Array of volatility values (mg/dL per minute)
        thresh_low: Threshold for low->medium (default: VOL_THRESH_LOW)
        thresh_high: Threshold for medium->high (default: VOL_THRESH_HIGH)
        weight_low: Weight for low volatility (default: VOL_WEIGHT_LOW)
        weight_med: Weight for medium volatility (default: VOL_WEIGHT_MED)
        weight_high: Weight for high volatility (default: VOL_WEIGHT_HIGH)
    
    Returns:
        Array of sample weights (same shape as volatility)
    """
    # Use defaults from config if not specified
    if thresh_low is None:
        thresh_low = VOL_THRESH_LOW
    if thresh_high is None:
        thresh_high = VOL_THRESH_HIGH
    if weight_low is None:
        weight_low = VOL_WEIGHT_LOW
    if weight_med is None:
        weight_med = VOL_WEIGHT_MED
    if weight_high is None:
        weight_high = VOL_WEIGHT_HIGH
    
    weights = np.full_like(volatility, weight_low, dtype=np.float32)
    weights[(volatility >= thresh_low) & (volatility < thresh_high)] = weight_med
    weights[volatility >= thresh_high] = weight_high
    
    return weights


# ============================================================================
# TCN Model Architecture
# ============================================================================
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate, name_prefix):
    """
    Residual block with causal dilated convolution.
    
    Uses causal padding (pad left only) to prevent future leakage.
    
    Args:
        x: Input tensor of shape (batch, seq_len, channels)
        filters: Number of convolutional filters
        kernel_size: Kernel size for Conv1D
        dilation_rate: Dilation rate for Conv1D
        dropout_rate: Dropout probability
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor of shape (batch, seq_len, filters)
    """
    # Calculate causal padding size
    # For causal convolution: pad (kernel_size - 1) * dilation_rate on the left
    pad_size = (kernel_size - 1) * dilation_rate
    
    # Residual connection (1x1 conv if channel mismatch)
    if x.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding='same', 
                         name=f'{name_prefix}_residual_conv')(x)
    else:
        residual = x
    
    # Causal padding: pad on left only
    x_padded = ZeroPadding1D(padding=(pad_size, 0), 
                              name=f'{name_prefix}_pad')(x)
    
    # First conv + activation
    conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding='valid', activation='relu',
                   name=f'{name_prefix}_conv1')(x_padded)
    conv1 = LayerNormalization(name=f'{name_prefix}_ln1')(conv1)
    conv1 = Dropout(dropout_rate, name=f'{name_prefix}_drop1')(conv1)
    
    # Second causal padding
    conv1_padded = ZeroPadding1D(padding=(pad_size, 0),
                                  name=f'{name_prefix}_pad2')(conv1)
    
    # Second conv + activation
    conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding='valid', activation='relu',
                   name=f'{name_prefix}_conv2')(conv1_padded)
    conv2 = LayerNormalization(name=f'{name_prefix}_ln2')(conv2)
    conv2 = Dropout(dropout_rate, name=f'{name_prefix}_drop2')(conv2)
    
    # Residual connection
    out = Add(name=f'{name_prefix}_add')([residual, conv2])
    
    return out


def build_tcn(seq_len=SEQ_LEN, n_channels=2, n_bins=N_BINS,
              filters=TCN_FILTERS, kernel_size=TCN_KERNEL_SIZE,
              dilations=TCN_DILATIONS, dropout_rate=DROPOUT_RATE):
    """
    Build TCN model for distribution prediction.
    
    Args:
        seq_len: Length of input sequence
        n_channels: Number of input channels (features)
        n_bins: Number of output bins
        filters: Number of filters per conv layer
        kernel_size: Kernel size for conv layers
        dilations: List of dilation rates for residual blocks
        dropout_rate: Dropout probability
    
    Returns:
        Keras Model with inputs and single logits output
    """
    inputs = Input(shape=(seq_len, n_channels), name='input')
    
    # Initial projection to filter dimension
    x = Conv1D(filters, 1, padding='same', name='input_proj')(inputs)
    
    # Stack of residual blocks with increasing dilation
    for i, d in enumerate(dilations):
        x = residual_block(x, filters, kernel_size, d, dropout_rate,
                          name_prefix=f'block_{i}_d{d}')
    
    # Global pooling over sequence dimension
    x = GlobalAveragePooling1D(name='global_pool')(x)
    
    # Output head
    x = Dense(64, activation='relu', name='fc')(x)
    x = Dropout(dropout_rate, name='fc_drop')(x)
    logits = Dense(n_bins, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=logits, name='TCN_Distribution')
    
    return model


def count_params(model):
    """Count trainable parameters in a model."""
    return sum(np.prod(w.shape) for w in model.trainable_weights)


def build_tcn_multihead(seq_len=SEQ_LEN, n_channels=2,
                        filters=TCN_FILTERS, kernel_size=TCN_KERNEL_SIZE,
                        dilations=TCN_DILATIONS, dropout_rate=DROPOUT_RATE,
                        use_head_adapter=USE_HEAD_ADAPTER,
                        head_adapter_dim=HEAD_ADAPTER_DIM):
    """
    Build multi-head TCN model for multi-horizon distribution prediction.
    
    Same backbone as single-horizon TCN, but branches into 6 heads after FC layer.
    Each head outputs logits for its horizon's bin count.
    
    Args:
        seq_len: Length of input sequence
        n_channels: Number of input channels (features)
        filters: Number of filters per conv layer
        kernel_size: Kernel size for conv layers
        dilations: List of dilation rates for residual blocks
        dropout_rate: Dropout probability
        use_head_adapter: If True, add per-head adapter layer before logits
        head_adapter_dim: Hidden dimension for adapter layer
    
    Returns:
        Keras Model with inputs and dict outputs: {"h5": logits_5, ..., "h30": logits_30}
    """
    inputs = Input(shape=(seq_len, n_channels), name='input')
    
    # Shared backbone (same as single-horizon TCN)
    # Initial projection to filter dimension
    x = Conv1D(filters, 1, padding='same', name='input_proj')(inputs)
    
    # Stack of residual blocks with increasing dilation
    for i, d in enumerate(dilations):
        x = residual_block(x, filters, kernel_size, d, dropout_rate,
                          name_prefix=f'block_{i}_d{d}')
    
    # Global pooling over sequence dimension
    x = GlobalAveragePooling1D(name='global_pool')(x)
    
    # Shared fully connected layer
    x = Dense(64, activation='relu', name='fc')(x)
    x = Dropout(dropout_rate, name='fc_drop')(x)
    
    # Multi-head outputs (one per horizon)
    outputs = {}
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        
        if USE_SHARED_BINS:
            # All heads output same number of bins
            n_bins = SHARED_BIN_CONFIG['n_bins']
        else:
            # Each head outputs horizon-specific bin count
            n_bins = get_bin_config(horizon)['n_bins']
        
        if use_head_adapter:
            # Per-head adapter: Dense(adapter_dim, relu) -> Dense(n_bins)
            head_hidden = Dense(head_adapter_dim, activation='relu', 
                               name=f'{head_name}_adapter')(x)
            logits = Dense(n_bins, name=f'{head_name}_logits')(head_hidden)
        else:
            logits = Dense(n_bins, name=f'{head_name}_logits')(x)
        
        outputs[head_name] = logits
    
    model = Model(inputs=inputs, outputs=outputs, name='TCN_MultiHead')
    
    return model


# ============================================================================
# Custom Loss Function
# ============================================================================
def distribution_loss_fn(y_true, logits, sigma_bins=SIGMA_BINS, lambda_smooth=LAMBDA_SMOOTH):
    """
    Loss function for distribution prediction with soft targets and smoothness regularization.
    
    Components:
    1. Cross-entropy between soft Gaussian targets and predicted distribution
    2. Smoothness penalty on logits (second difference) to avoid jagged PMFs
    
    Args:
        y_true: Tensor of shape (batch,) with integer bin indices
        logits: Tensor of shape (batch, n_bins) with raw logits
        sigma_bins: Standard deviation of Gaussian soft targets
        lambda_smooth: Weight for smoothness regularizer
    
    Returns:
        Scalar loss value
    """
    # Get n_bins dynamically from logits shape (supports different bin counts per horizon)
    n_bins = tf.shape(logits)[1]
    
    # Generate bin indices as tensor
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    
    # Generate soft Gaussian targets
    y_true_float = tf.cast(y_true, tf.float32)[:, tf.newaxis]  # (batch, 1)
    bin_indices_expanded = bin_indices[tf.newaxis, :]  # (1, n_bins)
    
    # Gaussian weights
    weights = tf.exp(-0.5 * tf.square((bin_indices_expanded - y_true_float) / sigma_bins))
    soft_targets = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    
    # Cross-entropy with soft targets
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    ce_loss = -tf.reduce_sum(soft_targets * log_probs, axis=-1)
    
    # Smoothness regularizer on logits (second difference penalty)
    z = logits
    second_diff = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
    smooth_penalty = tf.reduce_sum(tf.square(second_diff), axis=-1)
    
    # Total loss
    total_loss = ce_loss + lambda_smooth * smooth_penalty
    
    return tf.reduce_mean(total_loss)


def create_distribution_loss(sigma_bins=SIGMA_BINS, lambda_smooth=LAMBDA_SMOOTH):
    """Factory function for distribution loss."""
    def loss(y_true, logits):
        return distribution_loss_fn(y_true, logits, sigma_bins, lambda_smooth)
    return loss


# ============================================================================
# Multi-Horizon Model with Coherence Regularization
# ============================================================================
class MultiHorizonTCN(Model):
    """
    Multi-horizon TCN with custom train_step for cross-horizon coherence regularization.
    
    Wraps a multi-head TCN backbone and adds:
    1. Per-head soft-target cross-entropy + smoothness penalty
    2. Cross-horizon curve smoothness penalty on predicted means
    3. Variance monotonicity penalty (variance should increase with horizon)
    """
    
    def __init__(self, backbone, sigma_bins=SIGMA_BINS, lambda_smooth=LAMBDA_SMOOTH,
                 lambda_curve=LAMBDA_CURVE, lambda_varmono=LAMBDA_VARMONO,
                 curve_penalty_type=CURVE_PENALTY_TYPE, huber_delta=HUBER_DELTA,
                 point_loss_enabled=POINT_LOSS_ENABLED, point_loss_weight=POINT_LOSS_WEIGHT,
                 point_loss_type=POINT_LOSS_TYPE, point_loss_huber_delta=POINT_LOSS_HUBER_DELTA,
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.sigma_bins = sigma_bins
        self.lambda_smooth = lambda_smooth
        self.lambda_curve = lambda_curve
        self.lambda_varmono = lambda_varmono
        self.curve_penalty_type = curve_penalty_type
        self.huber_delta = huber_delta
        
        # Point loss parameters (volatility-weighted MAE/Huber on predicted mean)
        self.point_loss_enabled = point_loss_enabled
        self.point_loss_weight = point_loss_weight
        self.point_loss_type = point_loss_type
        self.point_loss_huber_delta = point_loss_huber_delta
        
        # Pre-compute bin centers as tensors for each head
        self._bin_centers = {}
        for horizon in PREDICTION_HORIZONS:
            head_name = f"h{horizon}"
            if USE_SHARED_BINS:
                centers = SHARED_BIN_CONFIG['bin_centers']
            else:
                centers = get_bin_config(horizon)['bin_centers']
            self._bin_centers[head_name] = tf.constant(centers, dtype=tf.float32)
    
    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'sigma_bins': self.sigma_bins,
            'lambda_smooth': self.lambda_smooth,
            'lambda_curve': self.lambda_curve,
            'lambda_varmono': self.lambda_varmono,
            'curve_penalty_type': self.curve_penalty_type,
            'huber_delta': self.huber_delta,
            'point_loss_enabled': self.point_loss_enabled,
            'point_loss_weight': self.point_loss_weight,
            'point_loss_type': self.point_loss_type,
            'point_loss_huber_delta': self.point_loss_huber_delta,
        })
        return config
    
    def call(self, inputs, training=None):
        """Forward pass through backbone."""
        return self.backbone(inputs, training=training)
    
    def _compute_head_loss(self, y_true, logits, sample_weight=None):
        """
        Compute per-head loss: soft-target CE + smoothness penalty.
        
        Args:
            y_true: (batch,) bin indices
            logits: (batch, n_bins) raw logits
            sample_weight: Optional (batch,) weights for each sample
        
        Returns:
            Scalar mean loss for this head
        """
        n_bins = tf.shape(logits)[1]
        bin_indices = tf.cast(tf.range(n_bins), tf.float32)
        
        # Soft Gaussian targets
        y_true_float = tf.cast(y_true, tf.float32)[:, tf.newaxis]
        bin_indices_expanded = bin_indices[tf.newaxis, :]
        weights = tf.exp(-0.5 * tf.square((bin_indices_expanded - y_true_float) / self.sigma_bins))
        soft_targets = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
        
        # Cross-entropy
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        ce_loss = -tf.reduce_sum(soft_targets * log_probs, axis=-1)
        
        # Smoothness penalty on logits
        second_diff = logits[:, 2:] - 2 * logits[:, 1:-1] + logits[:, :-2]
        smooth_penalty = tf.reduce_sum(tf.square(second_diff), axis=-1)
        
        total = ce_loss + self.lambda_smooth * smooth_penalty
        
        # Apply sample weights if provided
        if sample_weight is not None:
            total = total * sample_weight
            # Weighted mean: sum(w * loss) / sum(w)
            return tf.reduce_sum(total) / tf.reduce_sum(sample_weight)
        
        return tf.reduce_mean(total)
    
    def _compute_point_loss(self, y_delta, logits, head_name, sample_weight=None):
        """
        Compute point loss on predicted mean delta (volatility-weighted).
        
        This loss targets MAE improvement in high-volatility windows without
        inflating the PMF uncertainty (which would widen prediction intervals).
        
        Args:
            y_delta: (batch,) actual delta values (float)
            logits: (batch, n_bins) raw logits
            head_name: Name of the head (for bin centers lookup)
            sample_weight: Optional (batch,) volatility weights
        
        Returns:
            Scalar mean loss for this head
        """
        # Compute predicted mean from PMF
        probs = tf.nn.softmax(logits, axis=-1)  # (batch, n_bins)
        centers = self._bin_centers[head_name]  # (n_bins,)
        pred_mean = tf.reduce_sum(probs * centers[tf.newaxis, :], axis=-1)  # (batch,)
        
        # Compute error
        y_delta_float = tf.cast(y_delta, tf.float32)
        error = pred_mean - y_delta_float
        
        # Apply Huber or MAE loss
        if self.point_loss_type == "huber":
            abs_error = tf.abs(error)
            loss = tf.where(
                abs_error <= self.point_loss_huber_delta,
                0.5 * tf.square(error),
                self.point_loss_huber_delta * (abs_error - 0.5 * self.point_loss_huber_delta)
            )
        else:  # MAE
            loss = tf.abs(error)
        
        # Apply volatility weights HERE (not to PMF loss)
        if sample_weight is not None:
            loss = loss * sample_weight
            return tf.reduce_sum(loss) / tf.reduce_sum(sample_weight)
        
        return tf.reduce_mean(loss)
    
    def _compute_coherence_losses(self, logits_dict):
        """
        Compute cross-horizon coherence penalties.
        
        Args:
            logits_dict: Dict mapping head_name -> (batch, n_bins) logits
        
        Returns:
            Tuple of (curve_loss, varmono_loss) scalars
        """
        # Compute predicted mean and variance for each horizon
        mus = []  # (n_horizons, batch)
        variances = []  # (n_horizons, batch)
        
        for horizon in PREDICTION_HORIZONS:
            head_name = f"h{horizon}"
            logits = logits_dict[head_name]
            probs = tf.nn.softmax(logits, axis=-1)  # (batch, n_bins)
            centers = self._bin_centers[head_name]  # (n_bins,)
            
            # Mean: E[delta] = sum(p * centers)
            mu = tf.reduce_sum(probs * centers[tf.newaxis, :], axis=-1)  # (batch,)
            mus.append(mu)
            
            # Variance: Var[delta] = sum(p * (centers - mu)^2)
            centered = centers[tf.newaxis, :] - mu[:, tf.newaxis]  # (batch, n_bins)
            var = tf.reduce_sum(probs * tf.square(centered), axis=-1)  # (batch,)
            variances.append(var)
        
        # Stack: (n_horizons, batch)
        mus = tf.stack(mus, axis=0)
        variances = tf.stack(variances, axis=0)
        
        # Curve smoothness: second difference on means across horizons
        # For horizons [h0, h1, h2, h3, h4, h5], compute (h2 - 2*h1 + h0), (h3 - 2*h2 + h1), etc.
        # mus shape: (6, batch)
        second_diff_mu = mus[2:, :] - 2 * mus[1:-1, :] + mus[:-2, :]  # (4, batch)
        
        # Apply curve penalty based on type
        if self.curve_penalty_type == "l1":
            # L1: sum of absolute values (less sensitive to outliers)
            curve_penalty = tf.reduce_sum(tf.abs(second_diff_mu), axis=0)
        elif self.curve_penalty_type == "huber":
            # Huber: smooth L1 loss (quadratic near 0, linear far from 0)
            curve_penalty = tf.reduce_sum(
                tf.where(
                    tf.abs(second_diff_mu) <= self.huber_delta,
                    0.5 * tf.square(second_diff_mu),
                    self.huber_delta * (tf.abs(second_diff_mu) - 0.5 * self.huber_delta)
                ),
                axis=0
            )
        else:  # "l2" (default)
            curve_penalty = tf.reduce_sum(tf.square(second_diff_mu), axis=0)
        
        curve_loss = tf.reduce_mean(curve_penalty)
        
        # Variance monotonicity: penalize var_h > var_{h+1} (variance should increase with horizon)
        # variances shape: (6, batch)
        var_diff = variances[:-1, :] - variances[1:, :]  # (5, batch) - positive if var decreases
        varmono_penalty = tf.nn.relu(var_diff)  # Only penalize if var decreases
        varmono_loss = tf.reduce_mean(tf.reduce_sum(tf.square(varmono_penalty), axis=0))
        
        return curve_loss, varmono_loss
    
    def train_step(self, data):
        """
        Custom train step with multi-head losses and coherence regularization.
        
        New loss structure (when point_loss_enabled):
        - PMF loss (soft-target CE + smoothness): UNWEIGHTED
        - Point loss (Huber/MAE on predicted mean): VOLATILITY WEIGHTED
        
        This separates concerns: PMF learns distribution shape without hedging,
        while point loss focuses high-vol samples on mean accuracy.
        
        Args:
            data: Tuple of (X, y_dict) or (X, y_dict, sample_weight)
                  y_dict maps either:
                  - head_name -> bin_indices (legacy format)
                  - {head_name}_bins -> bin_indices, {head_name}_deltas -> delta values (new format)
        
        Returns:
            Dict of metrics
        """
        # Unpack data - can be (x, y) or (x, y, sample_weight)
        if len(data) == 3:
            x, y_dict, sample_weight = data
            # When using multi-output, Keras passes sample_weight as a dict
            # where each key has the same weights - extract just one
            if isinstance(sample_weight, dict):
                sample_weight = list(sample_weight.values())[0]
        else:
            x, y_dict = data
            sample_weight = None
        
        # Detect y_dict format (legacy vs new with deltas)
        # New format has keys like "h5_bins", "h5_deltas"
        # Legacy format has keys like "h5" directly
        has_deltas = f"{HEAD_NAMES[0]}_bins" in y_dict
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits_dict = self(x, training=True)
            
            # Per-head PMF losses - UNWEIGHTED (no sample_weight!)
            head_losses = {}
            total_pmf_loss = 0.0
            for head_name in HEAD_NAMES:
                if has_deltas:
                    y_bins = y_dict[f'{head_name}_bins']
                else:
                    y_bins = y_dict[head_name]
                logits = logits_dict[head_name]
                # PMF loss is UNWEIGHTED - critical for not inflating uncertainty
                pmf_loss = self._compute_head_loss(y_bins, logits, sample_weight=None)
                head_losses[head_name] = pmf_loss
                total_pmf_loss += pmf_loss
            
            # Point loss - VOLATILITY WEIGHTED (only if enabled and deltas available)
            total_point_loss = tf.constant(0.0)
            if self.point_loss_enabled and self.point_loss_weight > 0 and has_deltas:
                for head_name in HEAD_NAMES:
                    y_delta = y_dict[f'{head_name}_deltas']
                    logits = logits_dict[head_name]
                    point_loss = self._compute_point_loss(y_delta, logits, head_name, sample_weight)
                    total_point_loss += point_loss
            
            # Coherence losses
            curve_loss, varmono_loss = self._compute_coherence_losses(logits_dict)
            
            # Total loss
            total_loss = (total_pmf_loss 
                         + self.point_loss_weight * total_point_loss
                         + self.lambda_curve * curve_loss 
                         + self.lambda_varmono * varmono_loss)
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return metrics
        metrics = {
            'loss': total_loss,
            'pmf_loss': total_pmf_loss,
            'point_loss': total_point_loss,
            'curve_loss': curve_loss,
            'varmono_loss': varmono_loss,
        }
        return metrics
    
    def test_step(self, data):
        """
        Custom test step (validation) with same loss computation.
        Supports both legacy format (head_name -> bins) and new format (head_name_bins, head_name_deltas).
        """
        x, y_dict = data
        
        # Detect y_dict format
        has_deltas = f"{HEAD_NAMES[0]}_bins" in y_dict
        
        # Forward pass
        logits_dict = self(x, training=False)
        
        # Per-head PMF losses (unweighted)
        total_pmf_loss = 0.0
        for head_name in HEAD_NAMES:
            if has_deltas:
                y_bins = y_dict[f'{head_name}_bins']
            else:
                y_bins = y_dict[head_name]
            logits = logits_dict[head_name]
            loss = self._compute_head_loss(y_bins, logits)
            total_pmf_loss += loss
        
        # Point loss (unweighted on val since no sample_weight)
        total_point_loss = tf.constant(0.0)
        if self.point_loss_enabled and self.point_loss_weight > 0 and has_deltas:
            for head_name in HEAD_NAMES:
                y_delta = y_dict[f'{head_name}_deltas']
                logits = logits_dict[head_name]
                point_loss = self._compute_point_loss(y_delta, logits, head_name)
                total_point_loss += point_loss
        
        # Coherence losses
        curve_loss, varmono_loss = self._compute_coherence_losses(logits_dict)
        
        # Total loss
        total_loss = (total_pmf_loss 
                     + self.point_loss_weight * total_point_loss
                     + self.lambda_curve * curve_loss 
                     + self.lambda_varmono * varmono_loss)
        
        return {
            'loss': total_loss,
            'pmf_loss': total_pmf_loss,
            'point_loss': total_point_loss,
            'curve_loss': curve_loss,
            'varmono_loss': varmono_loss,
        }
    
    def predict_logits(self, X, batch_size=BATCH_SIZE):
        """
        Get raw logits for all heads.
        
        Args:
            X: Input features (n_samples, seq_len, n_channels)
        
        Returns:
            Dict mapping head_name -> (n_samples, n_bins) logits
        """
        # Use manual loop for horizon-specific bins (model.predict hangs with tf.data)
        if not USE_SHARED_BINS:
            n_samples = len(X)
            all_logits = {head_name: [] for head_name in HEAD_NAMES}
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_output = self(X[start:end], training=False)
                for head_name in HEAD_NAMES:
                    all_logits[head_name].append(batch_output[head_name].numpy())
            
            return {head_name: np.concatenate(chunks, axis=0) 
                    for head_name, chunks in all_logits.items()}
        else:
            return self.predict(X, batch_size=batch_size, verbose=0)
    
    def predict_probs(self, X, batch_size=BATCH_SIZE, temperatures=None):
        """
        Get probability distributions for all heads.
        
        Args:
            X: Input features (n_samples, seq_len, n_channels)
            temperatures: Optional dict mapping head_name -> temperature for scaling
        
        Returns:
            Dict mapping head_name -> (n_samples, n_bins) probabilities
        """
        logits_dict = self.predict_logits(X, batch_size=batch_size)
        
        probs_dict = {}
        for head_name, logits in logits_dict.items():
            if temperatures is not None and head_name in temperatures:
                T = temperatures[head_name]
                probs = tf.nn.softmax(logits / T, axis=-1).numpy()
            else:
                probs = tf.nn.softmax(logits, axis=-1).numpy()
            probs_dict[head_name] = probs
        
        return probs_dict


# ============================================================================
# Training
# ============================================================================
class TCNTrainer:
    """
    Trainer class for TCN distribution model.
    
    Handles model building, training, and prediction.
    """
    
    def __init__(self, seq_len=SEQ_LEN, n_channels=2, n_bins=N_BINS):
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build and compile the TCN model."""
        self.model = build_tcn(
            seq_len=self.seq_len,
            n_channels=self.n_channels,
            n_bins=self.n_bins
        )
        
        # Custom loss
        loss = create_distribution_loss()
        
        # Compile with custom loss (jit_compile=False to avoid XLA issues)
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss,
            jit_compile=False,
        )
        
        n_params = count_params(self.model)
        print(f"\nModel built: {n_params:,} trainable parameters")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model.
        
        Args:
            X_train: Training features (batch, seq_len, n_channels)
            y_train: Training bin indices (batch,)
            X_val: Validation features
            y_val: Validation bin indices
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE_EARLY_STOP,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=PATIENCE_LR_REDUCE,
                factor=LR_REDUCE_FACTOR,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"\nTraining TCN model...")
        print(f"  Epochs: {MAX_EPOCHS} (early stopping patience: {PATIENCE_EARLY_STOP})")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Learning rate: {LEARNING_RATE}")
        
        # Train (verbose=2 for simpler output without progress bar)
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2
        )
        
        return self.history
    
    def predict(self, X):
        """
        Get predictions from the model.
        
        Args:
            X: Input features (batch, seq_len, n_channels)
        
        Returns:
            logits: Raw logits (batch, n_bins)
            probs: Softmax probabilities (batch, n_bins)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logits = self.model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        # Apply softmax to get probabilities
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        return logits, probs
    
    def save(self, path):
        """Save model to file."""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from file."""
        # Custom loss function needs to be provided for loading
        custom_objects = {'loss': create_distribution_loss()}
        self.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(f"Model loaded from {path}")


# ============================================================================
# Evaluation Metrics
# ============================================================================
def compute_calibration_metrics(probs, y_deltas, bin_edges):
    """
    Compute calibration metrics from predicted PMFs.
    
    Args:
        probs: Array of shape (n, n_bins) with predicted probabilities
        y_deltas: Array of shape (n,) with actual delta values
        bin_edges: Array of shape (n_bins+1,) with bin edge values
    
    Returns:
        Dictionary with calibration metrics
    """
    n = len(y_deltas)
    
    # Compute quantiles from PMFs
    quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.5, 0.9], bin_edges)
    q10 = quantiles[0.1]
    q50 = quantiles[0.5]
    q90 = quantiles[0.9]
    
    # Coverage: fraction where actual falls within Q10-Q90
    coverage = np.mean((y_deltas >= q10) & (y_deltas <= q90)) * 100
    
    # Below Q10 and above Q90
    below_q10 = np.mean(y_deltas < q10) * 100
    above_q90 = np.mean(y_deltas > q90) * 100
    
    # Interval width
    interval_width = q90 - q10
    mean_width = np.mean(interval_width)
    
    return {
        'coverage_80': coverage,
        'below_q10': below_q10,
        'above_q90': above_q90,
        'mean_interval_width': mean_width,
        'q10': q10,
        'q50': q50,
        'q90': q90
    }


def compute_point_metrics(probs, y_deltas, bin_edges, bin_centers):
    """
    Compute point prediction metrics.
    
    Args:
        probs: Array of shape (n, n_bins) with predicted probabilities
        y_deltas: Array of shape (n,) with actual delta values
        bin_edges: Array of shape (n_bins+1,) with bin edge values
        bin_centers: Array of shape (n_bins,) with bin center values
    
    Returns:
        Dictionary with point metrics
    """
    # Predicted median (Q50)
    quantiles = quantiles_from_pmf_batch(probs, [0.5], bin_edges)
    pred_median = quantiles[0.5]
    
    # Predicted mean from PMF
    pred_mean = pmf_mean(probs, bin_centers)
    
    # MAE using median
    mae_median = np.mean(np.abs(y_deltas - pred_median))
    
    # MAE using mean
    mae_mean = np.mean(np.abs(y_deltas - pred_mean))
    
    return {
        'mae_median': mae_median,
        'mae_mean': mae_mean,
        'pred_median': pred_median,
        'pred_mean': pred_mean
    }


def compute_event_probabilities(probs, y_deltas, bin_centers):
    """
    Compute event probabilities and Brier scores.
    
    Events:
    - P(Δ > +10): Rising
    - P(Δ < -10): Falling
    - P(|Δ| <= 5): Steady
    
    Args:
        probs: Array of shape (n, n_bins) with predicted probabilities
        y_deltas: Array of shape (n,) with actual delta values
        bin_centers: Array of shape (n_bins,) with bin center values
    
    Returns:
        Dictionary with event probabilities and Brier scores
    """
    n = len(y_deltas)
    
    # Dynamically find bin indices for events based on bin_centers
    rising_bins = np.where(bin_centers > 10)[0]
    falling_bins = np.where(bin_centers < -10)[0]
    steady_bins = np.where(np.abs(bin_centers) <= 5)[0]
    
    # Predicted probabilities for events
    p_rising = probs[:, rising_bins].sum(axis=1) if len(rising_bins) > 0 else np.zeros(n)
    p_falling = probs[:, falling_bins].sum(axis=1) if len(falling_bins) > 0 else np.zeros(n)
    p_steady = probs[:, steady_bins].sum(axis=1) if len(steady_bins) > 0 else np.zeros(n)
    
    # Actual outcomes (binary)
    y_rising = (y_deltas > 10).astype(float)
    y_falling = (y_deltas < -10).astype(float)
    y_steady = (np.abs(y_deltas) <= 5).astype(float)
    
    # Brier scores
    brier_rising = np.mean((p_rising - y_rising) ** 2)
    brier_falling = np.mean((p_falling - y_falling) ** 2)
    brier_steady = np.mean((p_steady - y_steady) ** 2)
    
    # Empirical frequencies
    freq_rising = np.mean(y_rising) * 100
    freq_falling = np.mean(y_falling) * 100
    freq_steady = np.mean(y_steady) * 100
    
    return {
        'brier_rising': brier_rising,
        'brier_falling': brier_falling,
        'brier_steady': brier_steady,
        'freq_rising': freq_rising,
        'freq_falling': freq_falling,
        'freq_steady': freq_steady,
        'p_rising': p_rising,
        'p_falling': p_falling,
        'p_steady': p_steady,
        'y_rising': y_rising,
        'y_falling': y_falling,
        'y_steady': y_steady
    }


def compute_reliability_table(p_pred, y_actual, n_bins=10):
    """
    Compute reliability table (calibration curve).
    
    Bins predictions into deciles and compares mean predicted probability
    to empirical frequency.
    
    Args:
        p_pred: Array of predicted probabilities
        y_actual: Array of actual binary outcomes
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with calibration data
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(p_pred, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    mean_pred = []
    mean_actual = []
    counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred.append(p_pred[mask].mean())
            mean_actual.append(y_actual[mask].mean())
            counts.append(mask.sum())
        else:
            mean_pred.append(np.nan)
            mean_actual.append(np.nan)
            counts.append(0)
    
    return {
        'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2,
        'mean_predicted': np.array(mean_pred),
        'mean_actual': np.array(mean_actual),
        'counts': np.array(counts)
    }


def print_reliability_table(event_name, p_pred, y_actual, n_bins=10):
    """Print reliability table for an event."""
    rel = compute_reliability_table(p_pred, y_actual, n_bins)
    
    print(f"\n  Reliability Table: {event_name}")
    print(f"  {'Pred Range':>12s} {'Mean Pred':>10s} {'Actual':>10s} {'Count':>8s}")
    print(f"  {'-'*42}")
    
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mp = rel['mean_predicted'][i]
        ma = rel['mean_actual'][i]
        c = rel['counts'][i]
        
        if c > 0:
            print(f"  {low:.1f}-{high:.1f}      {mp:>10.3f} {ma:>10.3f} {c:>8d}")
        else:
            print(f"  {low:.1f}-{high:.1f}      {'--':>10s} {'--':>10s} {c:>8d}")


def evaluate_model(trainer, splits, set_name='test', horizon=15):
    """
    Comprehensive evaluation of TCN model.
    
    Args:
        trainer: Trained TCNTrainer instance
        splits: Data splits dictionary
        set_name: Which split to evaluate ('val' or 'test')
        horizon: Prediction horizon in minutes (for bin configuration)
    
    Returns:
        Dictionary with all metrics
    """
    X = splits[set_name]['X']
    y_deltas = splits[set_name]['y_deltas']
    
    # Get horizon-specific bin configuration
    bin_config = get_bin_config(horizon)
    bin_edges = bin_config['bin_edges']
    bin_centers = bin_config['bin_centers']
    
    print(f"\n{'='*70}")
    print(f"TCN EVALUATION ({set_name.upper()} SET)")
    print(f"{'='*70}")
    print(f"Samples: {len(X):,}")
    
    # Get predictions
    _, probs = trainer.predict(X)
    
    # Calibration metrics
    cal = compute_calibration_metrics(probs, y_deltas, bin_edges)
    print(f"\n--- Calibration ---")
    print(f"Coverage Q10-Q90 (target 80%): {cal['coverage_80']:.1f}%")
    print(f"Below Q10 (target 10%):        {cal['below_q10']:.1f}%")
    print(f"Above Q90 (target 10%):        {cal['above_q90']:.1f}%")
    print(f"Mean interval width:           {cal['mean_interval_width']:.1f} mg/dL")
    
    # Point metrics
    point = compute_point_metrics(probs, y_deltas, bin_edges, bin_centers)
    print(f"\n--- Point Predictions ---")
    print(f"MAE (median):  {point['mae_median']:.2f} mg/dL")
    print(f"MAE (mean):    {point['mae_mean']:.2f} mg/dL")
    
    # Event probabilities
    events = compute_event_probabilities(probs, y_deltas, bin_centers)
    print(f"\n--- Event Probabilities ---")
    print(f"{'Event':20s} {'Brier':>10s} {'Empirical':>12s}")
    print(f"{'-'*44}")
    print(f"{'Rising (Δ > +10)':20s} {events['brier_rising']:>10.4f} {events['freq_rising']:>11.1f}%")
    print(f"{'Falling (Δ < -10)':20s} {events['brier_falling']:>10.4f} {events['freq_falling']:>11.1f}%")
    print(f"{'Steady (|Δ| <= 5)':20s} {events['brier_steady']:>10.4f} {events['freq_steady']:>11.1f}%")
    
    # Reliability tables
    print_reliability_table("Rising (Δ > +10)", events['p_rising'], events['y_rising'])
    print_reliability_table("Falling (Δ < -10)", events['p_falling'], events['y_falling'])
    print_reliability_table("Steady (|Δ| <= 5)", events['p_steady'], events['y_steady'])
    
    return {
        'calibration': cal,
        'point': point,
        'events': events
    }


def compare_with_gbm(tcn_metrics, splits):
    """
    Compare TCN results with GBM baseline.
    
    Trains GBM quantile models on the same data for comparison.
    """
    from lightgbm import LGBMRegressor
    
    print(f"\n{'='*70}")
    print("GBM BASELINE COMPARISON")
    print(f"{'='*70}")
    
    # Prepare GBM features (same as quantile_regression.py)
    # Use past deltas as features
    X_train = splits['train']['X']
    y_train = splits['train']['y_deltas']
    X_test = splits['test']['X']
    y_test = splits['test']['y_deltas']
    
    # Flatten the TCN input to match GBM feature format
    # Use the first channel (normalized glucose) at all timesteps
    # Plus optionally the velocity channel
    n_train = len(X_train)
    n_test = len(X_test)
    
    # Feature: past deltas (x0 * 50 gives back the raw deltas from current)
    X_train_gbm = X_train[:, :, 0] * 50  # De-normalize
    X_test_gbm = X_test[:, :, 0] * 50
    
    # GBM parameters
    gbm_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': RANDOM_SEED
    }
    
    # Train Q10, Q50, Q90 models
    print("\nTraining GBM quantile models...")
    
    q10_model = LGBMRegressor(**gbm_params, objective='quantile', alpha=0.1)
    q50_model = LGBMRegressor(**gbm_params, objective='quantile', alpha=0.5)
    q90_model = LGBMRegressor(**gbm_params, objective='quantile', alpha=0.9)
    
    q10_model.fit(X_train_gbm, y_train)
    q50_model.fit(X_train_gbm, y_train)
    q90_model.fit(X_train_gbm, y_train)
    
    # Predict
    gbm_q10 = q10_model.predict(X_test_gbm)
    gbm_q50 = q50_model.predict(X_test_gbm)
    gbm_q90 = q90_model.predict(X_test_gbm)
    
    # GBM metrics
    gbm_coverage = np.mean((y_test >= gbm_q10) & (y_test <= gbm_q90)) * 100
    gbm_below_q10 = np.mean(y_test < gbm_q10) * 100
    gbm_above_q90 = np.mean(y_test > gbm_q90) * 100
    gbm_width = np.mean(gbm_q90 - gbm_q10)
    gbm_mae = np.mean(np.abs(y_test - gbm_q50))
    
    # TCN metrics
    tcn_coverage = tcn_metrics['calibration']['coverage_80']
    tcn_below_q10 = tcn_metrics['calibration']['below_q10']
    tcn_above_q90 = tcn_metrics['calibration']['above_q90']
    tcn_width = tcn_metrics['calibration']['mean_interval_width']
    tcn_mae = tcn_metrics['point']['mae_median']
    
    # Comparison table
    print(f"\n{'Metric':30s} {'TCN':>12s} {'GBM':>12s} {'Δ':>12s}")
    print(f"{'-'*68}")
    print(f"{'Coverage Q10-Q90 (%)':30s} {tcn_coverage:>12.1f} {gbm_coverage:>12.1f} {tcn_coverage - gbm_coverage:>+12.1f}")
    print(f"{'Below Q10 (%)':30s} {tcn_below_q10:>12.1f} {gbm_below_q10:>12.1f} {tcn_below_q10 - gbm_below_q10:>+12.1f}")
    print(f"{'Above Q90 (%)':30s} {tcn_above_q90:>12.1f} {gbm_above_q90:>12.1f} {tcn_above_q90 - gbm_above_q90:>+12.1f}")
    print(f"{'Mean Interval Width (mg/dL)':30s} {tcn_width:>12.1f} {gbm_width:>12.1f} {tcn_width - gbm_width:>+12.1f}")
    print(f"{'MAE Q50 (mg/dL)':30s} {tcn_mae:>12.2f} {gbm_mae:>12.2f} {tcn_mae - gbm_mae:>+12.2f}")
    
    return {
        'gbm': {
            'coverage_80': gbm_coverage,
            'below_q10': gbm_below_q10,
            'above_q90': gbm_above_q90,
            'mean_interval_width': gbm_width,
            'mae_median': gbm_mae,
            'q10': gbm_q10,
            'q50': gbm_q50,
            'q90': gbm_q90
        }
    }


# ============================================================================
# Unit Tests
# ============================================================================
def test_binning():
    """Test delta_to_bin and bin_to_delta functions."""
    print("Testing binning functions...", end=" ")
    
    # Test with 15-min horizon (range: -100 to 120, 221 bins)
    horizon = 15
    config = get_bin_config(horizon)
    
    # Test exact bin centers
    assert delta_to_bin(-100, horizon) == 0, f"Expected 0, got {delta_to_bin(-100, horizon)}"
    assert delta_to_bin(0, horizon) == 100, f"Expected 100, got {delta_to_bin(0, horizon)}"
    assert delta_to_bin(100, horizon) == 200, f"Expected 200, got {delta_to_bin(100, horizon)}"
    assert delta_to_bin(120, horizon) == 220, f"Expected 220, got {delta_to_bin(120, horizon)}"
    
    # Test clamping
    assert delta_to_bin(-150, horizon) == 0, f"Expected 0 (clamped), got {delta_to_bin(-150, horizon)}"
    assert delta_to_bin(150, horizon) == 220, f"Expected 220 (clamped), got {delta_to_bin(150, horizon)}"
    
    # Test intermediate values
    assert delta_to_bin(50, horizon) == 150, f"Expected 150, got {delta_to_bin(50, horizon)}"
    assert delta_to_bin(-50, horizon) == 50, f"Expected 50, got {delta_to_bin(-50, horizon)}"
    
    # Test bin_to_delta inverse
    assert bin_to_delta(0, horizon) == -100
    assert bin_to_delta(100, horizon) == 0
    assert bin_to_delta(200, horizon) == 100
    
    # Test array input
    deltas = np.array([-100, -50, 0, 50, 100])
    bins = delta_to_bin(deltas, horizon)
    assert np.array_equal(bins, [0, 50, 100, 150, 200])
    
    # Test with 5-min horizon (range: -60 to 70, 131 bins)
    horizon5 = 5
    assert delta_to_bin(-60, horizon5) == 0
    assert delta_to_bin(0, horizon5) == 60
    assert delta_to_bin(70, horizon5) == 130
    
    print("PASSED")


def test_soft_target():
    """Test soft target generation."""
    print("Testing soft target generation...", end=" ")
    
    # Using 15-min horizon with 221 bins
    n_bins = 221
    
    # Test that soft target sums to 1
    t = make_soft_target(100, n_bins=n_bins, sigma_bins=4.0)
    assert abs(t.sum() - 1.0) < 1e-6, f"Soft target sum = {t.sum()}, expected 1.0"
    
    # Test that peak is at the correct bin
    assert t[100] > t[90], "Peak should be at center bin"
    assert t[100] > t[110], "Peak should be at center bin"
    assert np.argmax(t) == 100, f"Peak at wrong bin: {np.argmax(t)}"
    
    # Test edge bins
    t_low = make_soft_target(0, n_bins=n_bins, sigma_bins=4.0)
    assert abs(t_low.sum() - 1.0) < 1e-6
    assert np.argmax(t_low) == 0
    
    t_high = make_soft_target(220, n_bins=n_bins, sigma_bins=4.0)  # max bin is 220 for 221 bins
    assert abs(t_high.sum() - 1.0) < 1e-6
    assert np.argmax(t_high) == 220
    
    # Test batch version
    batch = make_soft_targets_batch([0, 100, 220], n_bins=n_bins, sigma_bins=4.0)
    assert batch.shape == (3, 221)
    assert np.allclose(batch.sum(axis=1), 1.0)
    
    print("PASSED")


def test_quantile_from_pmf():
    """Test quantile computation from PMF."""
    print("Testing quantile_from_pmf...", end=" ")
    
    # Use 15-min horizon configuration
    config = get_bin_config(15)
    n_bins = config['n_bins']
    bin_edges = config['bin_edges']
    
    # Test uniform PMF - median should be near center of range
    p_uniform = np.ones(n_bins) / n_bins
    q50_uniform = quantile_from_pmf(p_uniform, 0.5, bin_edges)
    expected_median = (config['delta_min'] + config['delta_max']) / 2  # 10 for -100 to 120
    assert abs(q50_uniform - expected_median) < 2.0, f"Uniform median = {q50_uniform}, expected ~{expected_median}"
    
    # Test point mass at bin 150 (delta = +50)
    p_point = np.zeros(n_bins)
    p_point[150] = 1.0
    q50_point = quantile_from_pmf(p_point, 0.5, bin_edges)
    assert abs(q50_point - 50.0) < 1.0, f"Point mass median = {q50_point}, expected ~50"
    
    # Test point mass at bin 50 (delta = -50)
    p_point2 = np.zeros(n_bins)
    p_point2[50] = 1.0
    q50_point2 = quantile_from_pmf(p_point2, 0.5, bin_edges)
    assert abs(q50_point2 - (-50.0)) < 1.0, f"Point mass median = {q50_point2}, expected ~-50"
    
    # Test Q10 and Q90 on uniform
    q10_uniform = quantile_from_pmf(p_uniform, 0.1, bin_edges)
    q90_uniform = quantile_from_pmf(p_uniform, 0.9, bin_edges)
    assert q10_uniform < q50_uniform < q90_uniform
    
    # Test Gaussian-like PMF (soft target centered at bin 100 = delta 0)
    p_gauss = make_soft_target(100, n_bins=n_bins, sigma_bins=10)
    q50_gauss = quantile_from_pmf(p_gauss, 0.5, bin_edges)
    assert abs(q50_gauss - 0.0) < 2.0, f"Gaussian median = {q50_gauss}, expected ~0"
    
    print("PASSED")


def test_pmf_sums_to_one():
    """Test that model outputs valid PMFs that sum to 1."""
    print("Testing PMF normalization...", end=" ")
    
    # Use 15-min horizon configuration
    n_bins = get_bin_config(15)['n_bins']
    
    # Build a small model
    model = build_tcn(seq_len=8, n_channels=2, n_bins=n_bins)
    
    # Random input
    x = np.random.randn(5, 8, 2).astype(np.float32)
    
    # Get logits and apply softmax
    logits = model(x)
    probs = tf.nn.softmax(logits, axis=-1)
    probs_np = probs.numpy()
    
    # Check sums
    sums = probs_np.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), f"PMF sums: {sums}"
    
    # Check non-negative
    assert np.all(probs_np >= 0), "PMF has negative values"
    
    print("PASSED")


def test_soft_target_loss():
    """Test the custom distribution loss."""
    print("Testing distribution loss...", end=" ")
    
    loss_fn = create_distribution_loss(sigma_bins=4.0, lambda_smooth=1e-4)
    
    # Create dummy data (using 15-min horizon with 221 bins)
    n_bins = 221
    batch_size = 16
    y_true = tf.constant(np.random.randint(0, n_bins, size=batch_size), dtype=tf.int32)
    logits = tf.constant(np.random.randn(batch_size, n_bins).astype(np.float32))
    
    # Compute loss
    loss_value = loss_fn(y_true, logits)
    
    # Should be a scalar
    assert loss_value.shape == (), f"Loss shape: {loss_value.shape}"
    
    # Should be finite and positive
    assert np.isfinite(loss_value.numpy()), f"Loss is not finite: {loss_value.numpy()}"
    assert loss_value.numpy() > 0, f"Loss should be positive: {loss_value.numpy()}"
    
    print("PASSED")


def test_multihead_output_shapes():
    """Test that multi-head model outputs correct shapes for each head."""
    print("Testing multi-head output shapes...", end=" ")
    
    # Build model
    model = build_tcn_multihead(seq_len=8, n_channels=2)
    
    # Random input
    x = np.random.randn(5, 8, 2).astype(np.float32)
    
    # Get outputs
    outputs = model(x)
    
    # Check output is a dict with correct keys
    assert isinstance(outputs, dict), "Output should be a dict"
    assert set(outputs.keys()) == set(HEAD_NAMES), f"Keys should be {HEAD_NAMES}"
    
    # Check each head has correct shape
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        logits = outputs[head_name]
        
        if USE_SHARED_BINS:
            expected_bins = SHARED_BIN_CONFIG['n_bins']
        else:
            expected_bins = get_bin_config(horizon)['n_bins']
        
        assert logits.shape == (5, expected_bins), \
            f"Head {head_name}: expected (5, {expected_bins}), got {logits.shape}"
    
    print("PASSED")


def test_coherence_gradient_flow():
    """Test that coherence losses have non-zero gradients."""
    print("Testing coherence gradient flow...", end=" ")
    
    # Build wrapped model
    backbone = build_tcn_multihead(seq_len=8, n_channels=2)
    model = MultiHorizonTCN(backbone, lambda_curve=1.0, lambda_varmono=1.0)
    
    # Build by calling once
    x = np.random.randn(4, 8, 2).astype(np.float32)
    _ = model(x)
    
    # Create dummy labels
    y_dict = {}
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        if USE_SHARED_BINS:
            n_bins = SHARED_BIN_CONFIG['n_bins']
        else:
            n_bins = get_bin_config(horizon)['n_bins']
        y_dict[head_name] = np.random.randint(0, n_bins, size=4)
    
    # Compute gradients
    x_tf = tf.constant(x)
    y_dict_tf = {k: tf.constant(v) for k, v in y_dict.items()}
    
    with tf.GradientTape() as tape:
        logits_dict = model(x_tf, training=True)
        curve_loss, varmono_loss = model._compute_coherence_losses(logits_dict)
        total = curve_loss + varmono_loss
    
    gradients = tape.gradient(total, model.trainable_variables)
    
    # Check at least some gradients are non-zero
    non_zero_grads = sum(1 for g in gradients if g is not None and tf.reduce_sum(tf.abs(g)) > 0)
    assert non_zero_grads > 0, "No non-zero gradients from coherence loss"
    
    print("PASSED")


def test_jumpiness_computation():
    """Test curve jumpiness metric computation."""
    print("Testing jumpiness computation...", end=" ")
    
    # Test with perfectly linear quantiles (should have zero jumpiness)
    n_samples = 100
    linear_q = {}
    for i, horizon in enumerate(PREDICTION_HORIZONS):
        # Linear increase: 0, 1, 2, 3, 4, 5 across horizons
        linear_q[horizon] = np.ones(n_samples) * i
    
    jumpiness = compute_curve_jumpiness(linear_q)
    assert jumpiness < 1e-6, f"Linear curve should have ~0 jumpiness, got {jumpiness}"
    
    # Test with jumpy quantiles (should have non-zero jumpiness)
    jumpy_q = {}
    for i, horizon in enumerate(PREDICTION_HORIZONS):
        # Alternating: 0, 10, 0, 10, 0, 10
        jumpy_q[horizon] = np.ones(n_samples) * (10 if i % 2 else 0)
    
    jumpiness_jumpy = compute_curve_jumpiness(jumpy_q)
    assert jumpiness_jumpy > 0, f"Jumpy curve should have non-zero jumpiness, got {jumpiness_jumpy}"
    
    # Jumpy should be much higher than linear
    assert jumpiness_jumpy > jumpiness + 1, "Jumpy curve should have higher jumpiness"
    
    print("PASSED")


def test_multihead_quantile_extraction():
    """Test quantile extraction from multi-head model outputs."""
    print("Testing multi-head quantile extraction...", end=" ")
    
    # Build model
    backbone = build_tcn_multihead(seq_len=8, n_channels=2)
    model = MultiHorizonTCN(backbone)
    
    # Build by calling
    x = np.random.randn(10, 8, 2).astype(np.float32)
    _ = model(x)
    
    # Get probabilities
    probs_dict = model.predict_probs(x)
    
    # Check each head
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        probs = probs_dict[head_name]
        
        # Probs should sum to 1
        sums = probs.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), f"PMF sums for {head_name}: {sums}"
        
        # Compute quantiles
        if USE_SHARED_BINS:
            bin_edges = SHARED_BIN_CONFIG['bin_edges']
        else:
            bin_edges = get_bin_config(horizon)['bin_edges']
        
        quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.5, 0.9], bin_edges)
        
        # Check order: q10 < q50 < q90
        assert np.all(quantiles[0.1] <= quantiles[0.5]), f"Q10 > Q50 for {head_name}"
        assert np.all(quantiles[0.5] <= quantiles[0.9]), f"Q50 > Q90 for {head_name}"
    
    print("PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS")
    print("="*70 + "\n")
    
    test_binning()
    test_soft_target()
    test_quantile_from_pmf()
    test_pmf_sums_to_one()
    test_soft_target_loss()
    
    # Multi-horizon tests
    test_multihead_output_shapes()
    test_coherence_gradient_flow()
    test_jumpiness_computation()
    test_multihead_quantile_extraction()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)


# ============================================================================
# Main Execution
# ============================================================================
def save_predictions(trainer, splits, output_path='../data/tcn_predictions_15min.csv'):
    """Save test set predictions to CSV."""
    X_test = splits['test']['X']
    y_deltas = splits['test']['y_deltas']
    indices = splits['test']['indices']
    df = splits['df']
    
    # Get predictions
    _, probs = trainer.predict(X_test)
    
    # Compute quantiles
    quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Compute mean
    pred_mean = pmf_mean(probs)
    
    # Build output DataFrame
    output_df = pd.DataFrame()
    # Format timestamps as ISO strings with timezone to avoid UTC conversion issues
    output_df['displayTime'] = df.iloc[indices]['displayTime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    output_df['value'] = df.iloc[indices]['value'].values
    output_df['actual_delta'] = y_deltas
    output_df['pred_q10'] = quantiles[0.1]
    output_df['pred_q25'] = quantiles[0.25]
    output_df['pred_q50'] = quantiles[0.5]
    output_df['pred_q75'] = quantiles[0.75]
    output_df['pred_q90'] = quantiles[0.9]
    output_df['pred_mean'] = pred_mean
    
    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    return output_df


def main():
    """Main training and evaluation pipeline (single horizon)."""
    print("\n" + "="*70)
    print("TCN DISTRIBUTION MODEL FOR GLUCOSE PREDICTION")
    print("="*70)
    
    # Load data
    df = load_glucose_data()
    
    # Prepare splits for default horizon
    horizon = PREDICTION_HORIZON
    splits = prepare_data_splits(df, horizon_minutes=horizon)
    
    # Get horizon-specific bin configuration
    bin_config = get_bin_config(horizon)
    n_bins = bin_config['n_bins']
    print(f"Bin range: [{bin_config['delta_min']}, {bin_config['delta_max']}] ({n_bins} bins)")
    
    # Determine number of channels
    n_channels = 2 if INCLUDE_FIRST_DIFF else 1
    
    # Create trainer with horizon-specific output size
    trainer = TCNTrainer(seq_len=SEQ_LEN, n_channels=n_channels, n_bins=n_bins)
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train(
        splits['train']['X'], splits['train']['y_bins'],
        splits['val']['X'], splits['val']['y_bins']
    )
    
    # Evaluate on validation set
    val_metrics = evaluate_model(trainer, splits, set_name='val', horizon=horizon)
    
    # Evaluate on test set
    test_metrics = evaluate_model(trainer, splits, set_name='test', horizon=horizon)
    
    # Compare with GBM
    gbm_comparison = compare_with_gbm(test_metrics, splits)
    
    # Save model
    trainer.save('../data/tcn_15min.keras')
    
    # Save predictions
    save_predictions(trainer, splits)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


# ============================================================================
# Temperature Scaling Integration
# ============================================================================
def evaluate_calibration_comparison(
    logits: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    y_bins: np.ndarray,
    y_deltas: np.ndarray,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    temperature: float,
    horizon: int,
    set_name: str = "Calibration"
):
    """
    Evaluate and print calibration metrics before and after temperature scaling.
    
    Args:
        logits: Raw model outputs (N, n_bins)
        probs_before: Probabilities before scaling (N, n_bins)
        probs_after: Probabilities after temperature scaling (N, n_bins)
        y_bins: True bin indices (N,)
        y_deltas: True delta values (N,)
        bin_edges: Bin edge values (n_bins+1,)
        bin_centers: Bin center values (n_bins,)
        temperature: Fitted temperature
        horizon: Prediction horizon in minutes
        set_name: Name of the dataset for display
    """
    print(f"\n{'='*70}")
    print(f"+{horizon} MIN: BEFORE vs AFTER TEMPERATURE SCALING (T={temperature:.4f})")
    print(f"{'='*70}")
    print(f"Dataset: {set_name} ({len(logits):,} samples)")
    
    # NLL comparison
    nll_before = compute_nll(logits, y_bins, temperature=1.0)
    nll_after = compute_nll(logits, y_bins, temperature=temperature)
    
    print(f"\n--- Negative Log-Likelihood ---")
    print(f"{'':20s} {'Before':>12s} {'After':>12s} {'Change':>12s}")
    print(f"{'-'*56}")
    print(f"{'NLL':20s} {nll_before:>12.4f} {nll_after:>12.4f} {nll_after-nll_before:>+12.4f}")
    
    # Calibration metrics (coverage)
    cal_before = compute_calibration_metrics(probs_before, y_deltas, bin_edges)
    cal_after = compute_calibration_metrics(probs_after, y_deltas, bin_edges)
    
    print(f"\n--- Calibration (Coverage) ---")
    print(f"{'Metric':20s} {'Before':>12s} {'After':>12s} {'Target':>12s}")
    print(f"{'-'*56}")
    print(f"{'<Q10':20s} {cal_before['below_q10']:>11.1f}% {cal_after['below_q10']:>11.1f}% {'10%':>12s}")
    print(f"{'Q10-Q90 Coverage':20s} {cal_before['coverage_80']:>11.1f}% {cal_after['coverage_80']:>11.1f}% {'80%':>12s}")
    print(f"{'> Q90':20s} {cal_before['above_q90']:>11.1f}% {cal_after['above_q90']:>11.1f}% {'10%':>12s}")
    print(f"{'Interval Width':20s} {cal_before['mean_interval_width']:>10.1f}  {cal_after['mean_interval_width']:>10.1f}  {'—':>12s}")
    
    # Point metrics
    point_before = compute_point_metrics(probs_before, y_deltas, bin_edges, bin_centers)
    point_after = compute_point_metrics(probs_after, y_deltas, bin_edges, bin_centers)
    
    print(f"\n--- Point Predictions ---")
    print(f"{'Metric':20s} {'Before':>12s} {'After':>12s} {'Change':>12s}")
    print(f"{'-'*56}")
    print(f"{'MAE (Q50)':20s} {point_before['mae_median']:>11.2f}  {point_after['mae_median']:>11.2f}  {point_after['mae_median']-point_before['mae_median']:>+11.2f} ")
    print(f"{'MAE (mean)':20s} {point_before['mae_mean']:>11.2f}  {point_after['mae_mean']:>11.2f}  {point_after['mae_mean']-point_before['mae_mean']:>+11.2f} ")
    
    # Event probabilities (Brier scores)
    events_before = compute_event_probabilities(probs_before, y_deltas, bin_centers)
    events_after = compute_event_probabilities(probs_after, y_deltas, bin_centers)
    
    print(f"\n--- Event Probabilities (Brier Score, lower is better) ---")
    print(f"{'Event':20s} {'Before':>12s} {'After':>12s} {'Change':>12s}")
    print(f"{'-'*56}")
    print(f"{'Rising (Δ > +10)':20s} {events_before['brier_rising']:>12.4f} {events_after['brier_rising']:>12.4f} {events_after['brier_rising']-events_before['brier_rising']:>+12.4f}")
    print(f"{'Falling (Δ < -10)':20s} {events_before['brier_falling']:>12.4f} {events_after['brier_falling']:>12.4f} {events_after['brier_falling']-events_before['brier_falling']:>+12.4f}")
    print(f"{'Steady (|Δ| <= 5)':20s} {events_before['brier_steady']:>12.4f} {events_after['brier_steady']:>12.4f} {events_after['brier_steady']-events_before['brier_steady']:>+12.4f}")
    
    return {
        'before': {
            'nll': nll_before,
            'calibration': cal_before,
            'point': point_before,
            'events': events_before
        },
        'after': {
            'nll': nll_after,
            'calibration': cal_after,
            'point': point_after,
            'events': events_after
        },
        'temperature': temperature
    }


def train_all_horizons():
    """
    Train TCN models for all prediction horizons (5, 10, 15, 20, 25, 30 min).
    Saves models and predictions for each horizon.
    Also saves full PMF data for webapp visualization.
    After training, fits per-horizon temperature scaling on validation set.
    """
    print("\n" + "="*70)
    print("TCN DISTRIBUTION MODEL - TRAINING ALL HORIZONS")
    print("="*70)
    print(f"Horizons to train: {PREDICTION_HORIZONS}")
    
    # Load data once
    df = load_glucose_data()
    
    # Determine number of channels
    n_channels = 2 if INCLUDE_FIRST_DIFF else 1
    
    # Store results for each horizon
    all_results = {}
    all_pmfs = {}  # Store full PMF data for each horizon
    all_trainers = {}  # Store trainers for temperature scaling
    all_splits = {}  # Store splits for temperature scaling
    
    for horizon in PREDICTION_HORIZONS:
        print("\n" + "="*70)
        print(f"TRAINING {horizon}-MINUTE HORIZON MODEL")
        print("="*70)
        
        # Prepare splits for this horizon
        splits = prepare_data_splits(df, horizon_minutes=horizon, verbose=(horizon == PREDICTION_HORIZONS[0]))
        all_splits[horizon] = splits
        
        # Get horizon-specific bin configuration
        bin_config = get_bin_config(horizon)
        n_bins = bin_config['n_bins']
        print(f"  Bin range: [{bin_config['delta_min']}, {bin_config['delta_max']}] ({n_bins} bins)")
        
        # Create and train model with horizon-specific output size
        trainer = TCNTrainer(seq_len=SEQ_LEN, n_channels=n_channels, n_bins=n_bins)
        trainer.build_model()
        
        trainer.train(
            splits['train']['X'], splits['train']['y_bins'],
            splits['val']['X'], splits['val']['y_bins']
        )
        
        all_trainers[horizon] = trainer
        
        # Evaluate on test set (before temperature scaling)
        test_metrics = evaluate_model(trainer, splits, set_name='test', horizon=horizon)
        
        # Save model
        model_path = f'../data/tcn_{horizon}min.keras'
        trainer.save(model_path)
        
        # Get test predictions (quantiles and full PMF)
        X_test = splits['test']['X']
        y_deltas = splits['test']['y_deltas']
        indices = splits['test']['indices']
        
        logits, probs = trainer.predict(X_test)
        
        # Compute quantiles using horizon-specific bin edges
        quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.25, 0.5, 0.75, 0.9], bin_config['bin_edges'])
        pred_mean = pmf_mean(probs, bin_config['bin_centers'])
        
        # Store results with bin configuration
        all_results[horizon] = {
            'indices': indices,
            'y_deltas': y_deltas,
            'q10': quantiles[0.1],
            'q25': quantiles[0.25],
            'q50': quantiles[0.5],
            'q75': quantiles[0.75],
            'q90': quantiles[0.9],
            'mean': pred_mean,
            'metrics': test_metrics,
            'bin_config': bin_config,
            'logits': logits,  # Store for temperature scaling
            'probs': probs
        }
        
        # Store full PMF for this horizon
        all_pmfs[horizon] = probs
        
        print(f"  MAE (Q50): {test_metrics['point']['mae_median']:.2f} mg/dL")
        
        # Store df_filtered for saving (same for all horizons)
        if horizon == PREDICTION_HORIZONS[0]:
            df_filtered = splits['df']
    
    # =========================================================================
    # TEMPERATURE SCALING
    # =========================================================================
    print("\n" + "="*70)
    print("TEMPERATURE SCALING - FITTING ON VALIDATION SET")
    print("="*70)
    
    # Collect logits and labels on validation set for each horizon
    temperature_fitting_data = {}
    for horizon in PREDICTION_HORIZONS:
        splits = all_splits[horizon]
        trainer = all_trainers[horizon]
        X_val = splits['val']['X']
        y_bins_val = splits['val']['y_bins']
        
        # Get logits on validation set
        logits_val, _ = trainer.predict(X_val)
        
        temperature_fitting_data[horizon] = {
            'logits': logits_val,
            'y_bins': y_bins_val
        }
    
    # Fit per-horizon temperatures
    temperatures = fit_temperatures_all_horizons(temperature_fitting_data, verbose=True)
    
    # Save temperatures
    temps_path = '../data/tcn_temps.json'
    save_temperatures(temperatures, temps_path)
    
    # =========================================================================
    # EVALUATE BEFORE vs AFTER TEMPERATURE SCALING (on test set)
    # =========================================================================
    print("\n" + "="*70)
    print("CALIBRATION COMPARISON: BEFORE vs AFTER TEMPERATURE SCALING")
    print("="*70)
    
    temp_scaling_results = {}
    for horizon in PREDICTION_HORIZONS:
        splits = all_splits[horizon]
        bin_config = all_results[horizon]['bin_config']
        
        # Get test set data
        X_test = splits['test']['X']
        y_bins_test = splits['test']['y_bins']
        y_deltas_test = splits['test']['y_deltas']
        
        # Get logits and probs (before)
        logits_test = all_results[horizon]['logits']
        probs_before = all_results[horizon]['probs']
        
        # Apply temperature scaling (after)
        T = temperatures[horizon]
        probs_after = apply_temperature(logits_test, T)
        
        # Evaluate comparison
        comparison = evaluate_calibration_comparison(
            logits=logits_test,
            probs_before=probs_before,
            probs_after=probs_after,
            y_bins=y_bins_test,
            y_deltas=y_deltas_test,
            bin_edges=bin_config['bin_edges'],
            bin_centers=bin_config['bin_centers'],
            temperature=T,
            horizon=horizon,
            set_name="Test"
        )
        
        temp_scaling_results[horizon] = comparison
        
        # Update all_results with temperature-scaled quantiles
        quantiles_after = quantiles_from_pmf_batch(
            probs_after, [0.1, 0.25, 0.5, 0.75, 0.9], bin_config['bin_edges']
        )
        all_results[horizon]['q10_scaled'] = quantiles_after[0.1]
        all_results[horizon]['q25_scaled'] = quantiles_after[0.25]
        all_results[horizon]['q50_scaled'] = quantiles_after[0.5]
        all_results[horizon]['q75_scaled'] = quantiles_after[0.75]
        all_results[horizon]['q90_scaled'] = quantiles_after[0.9]
        all_results[horizon]['probs_scaled'] = probs_after
        all_results[horizon]['temperature'] = T
    
    # Update PMFs with temperature-scaled versions
    for horizon in PREDICTION_HORIZONS:
        all_pmfs[horizon] = all_results[horizon]['probs_scaled']
    
    # Save combined predictions CSV (use df_filtered from first horizon since it's the same base data)
    save_all_predictions(all_results, df_filtered, '../data/tcn_predictions_all.csv')
    
    # Save full PMF data for webapp visualization (using temperature-scaled probabilities)
    save_pmf_data(all_results, all_pmfs, df_filtered, '../data/tcn_pmf_data.json')
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING SUMMARY (BEFORE TEMPERATURE SCALING)")
    print("="*70)
    print(f"{'Horizon':>10s} {'MAE Q50':>12s} {'Coverage':>12s} {'Width':>12s}")
    print("-" * 50)
    for horizon in PREDICTION_HORIZONS:
        m = all_results[horizon]['metrics']
        print(f"{horizon:>7d} min {m['point']['mae_median']:>10.2f} {m['calibration']['coverage_80']:>10.1f}% {m['calibration']['mean_interval_width']:>10.1f}")
    
    print("\n" + "="*70)
    print("TEMPERATURE SCALING SUMMARY")
    print("="*70)
    print(f"{'Horizon':>10s} {'Temp':>8s} {'NLL Before':>12s} {'NLL After':>12s} {'Δ NLL':>10s}")
    print("-" * 55)
    for horizon in PREDICTION_HORIZONS:
        T = temperatures[horizon]
        nll_before = temp_scaling_results[horizon]['before']['nll']
        nll_after = temp_scaling_results[horizon]['after']['nll']
        print(f"{horizon:>7d} min {T:>8.4f} {nll_before:>12.4f} {nll_after:>12.4f} {nll_after-nll_before:>+10.4f}")
    
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY (AFTER TEMPERATURE SCALING)")
    print("="*70)
    print(f"{'Horizon':>10s} {'<Q10':>8s} {'Cov 80%':>10s} {'>Q90':>8s} {'Width':>10s}")
    print("-" * 50)
    for horizon in PREDICTION_HORIZONS:
        cal = temp_scaling_results[horizon]['after']['calibration']
        print(f"{horizon:>7d} min {cal['below_q10']:>7.1f}% {cal['coverage_80']:>9.1f}% {cal['above_q90']:>7.1f}% {cal['mean_interval_width']:>9.1f}")
    
    print("\n" + "="*70)
    print("ALL HORIZONS COMPLETE!")
    print("="*70)
    print(f"\nTemperatures saved to: {temps_path}")
    
    return all_results, temperatures


def save_all_predictions(all_results, df, output_path):
    """
    Save predictions for all horizons to a single CSV.
    
    Different horizons may have different sample counts (longer horizons have fewer
    valid samples). We use the longest horizon's indices as the base, then merge
    shorter horizons by matching indices.
    
    Args:
        all_results: Dict mapping horizon -> results dict
        df: Filtered DataFrame with glucose readings
        output_path: Path to save CSV
    """
    # Use longest horizon (fewest samples) as base - all others should include these
    longest_horizon = max(PREDICTION_HORIZONS)
    base_indices = all_results[longest_horizon]['indices']
    
    # Build output DataFrame
    output_df = pd.DataFrame()
    output_df['idx'] = base_indices
    # Format timestamps as ISO strings with timezone to avoid UTC conversion issues
    # Use .values to avoid pandas index alignment issues
    output_df['displayTime'] = df.iloc[base_indices]['displayTime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z').values
    output_df['value'] = df.iloc[base_indices]['value'].values
    
    # Add predictions for each horizon
    for horizon in PREDICTION_HORIZONS:
        results = all_results[horizon]
        horizon_indices = results['indices']
        
        # Create a mapping from index to position in results
        idx_to_pos = {idx: pos for pos, idx in enumerate(horizon_indices)}
        
        # Map base indices to results - unscaled predictions
        columns_to_save = [
            (f'actual_delta_{horizon}', results['y_deltas']),
            (f'pred_q10_{horizon}', results['q10']),
            (f'pred_q25_{horizon}', results['q25']),
            (f'pred_q50_{horizon}', results['q50']),
            (f'pred_q75_{horizon}', results['q75']),
            (f'pred_q90_{horizon}', results['q90']),
            (f'pred_mean_{horizon}', results['mean']),
        ]
        
        # Add temperature-scaled predictions if available
        if 'q50_scaled' in results:
            columns_to_save.extend([
                (f'pred_q10_scaled_{horizon}', results['q10_scaled']),
                (f'pred_q25_scaled_{horizon}', results['q25_scaled']),
                (f'pred_q50_scaled_{horizon}', results['q50_scaled']),
                (f'pred_q75_scaled_{horizon}', results['q75_scaled']),
                (f'pred_q90_scaled_{horizon}', results['q90_scaled']),
            ])
        
        for col_suffix, data in columns_to_save:
            values = []
            for base_idx in base_indices:
                if base_idx in idx_to_pos:
                    values.append(data[idx_to_pos[base_idx]])
                else:
                    values.append(np.nan)
            output_df[col_suffix] = values
    
    # Drop the idx column
    output_df = output_df.drop(columns=['idx'])
    
    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\nAll predictions saved to {output_path}")
    
    return output_df


def save_pmf_data(all_results, all_pmfs, df, output_path):
    """
    Save full PMF data for webapp visualization.
    
    Uses the longest horizon's indices as base (consistent with save_all_predictions).
    PMFs from shorter horizons are aligned by matching indices.
    
    Args:
        all_results: Dict mapping horizon -> results dict
        all_pmfs: Dict mapping horizon -> PMF array (n_samples, 201)
        df: Filtered DataFrame with glucose readings
        output_path: Path to save JSON
    """
    import json
    
    # Use longest horizon (fewest samples) as base for consistency
    longest_horizon = max(PREDICTION_HORIZONS)
    base_indices = all_results[longest_horizon]['indices']
    
    # Create timestamp index mapping - convert to local timezone first to match CSV
    display_times = df.iloc[base_indices]['displayTime']
    # Convert to local time if timezone-aware, then format
    if display_times.dt.tz is not None:
        display_times = display_times.dt.tz_convert('America/New_York')
    timestamps = display_times.dt.strftime('%Y-%m-%d %H:%M').tolist()
    
    # Create index lookup
    timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    
    # Store PMF data - convert to list for JSON
    # Note: Each horizon has different bin centers, so we store them per-horizon
    pmf_data = {
        'timestamps': timestamps,
        'horizons': PREDICTION_HORIZONS,
        'index': timestamp_to_idx
    }
    
    # Add bin centers for each horizon
    for horizon in PREDICTION_HORIZONS:
        bin_config = all_results[horizon]['bin_config']
        pmf_data[f'bin_centers_{horizon}'] = bin_config['bin_centers'].tolist()
    
    # Add PMF arrays for each horizon (as nested lists)
    # Align to base indices, with null for missing samples
    for horizon in PREDICTION_HORIZONS:
        horizon_indices = all_results[horizon]['indices']
        horizon_pmfs = all_pmfs[horizon]
        bin_config = all_results[horizon]['bin_config']
        n_bins = bin_config['n_bins']
        
        # Create a mapping from index to position in PMF array
        idx_to_pos = {idx: pos for pos, idx in enumerate(horizon_indices)}
        
        # Build aligned PMF list
        aligned_pmfs = []
        null_pmf = [0.0] * n_bins  # Placeholder for missing (horizon-specific size)
        
        for base_idx in base_indices:
            if base_idx in idx_to_pos:
                pos = idx_to_pos[base_idx]
                aligned_pmfs.append(np.round(horizon_pmfs[pos], 6).tolist())
            else:
                aligned_pmfs.append(null_pmf)
        
        pmf_data[f'pmf_{horizon}'] = aligned_pmfs
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(pmf_data, f)
    
    # Get file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"PMF data saved to {output_path} ({size_mb:.1f} MB)")


def calibrate_existing_models(max_samples=5000):
    """
    Apply temperature scaling to already-trained models.
    Useful for recalibrating without retraining.
    
    Loads existing .keras models, fits temperatures on validation set,
    and saves temperatures to JSON.
    
    Args:
        max_samples: Maximum validation samples per horizon for fitting (for speed)
    """
    print("\n" + "="*70)
    print("TEMPERATURE SCALING - CALIBRATING EXISTING MODELS")
    print("="*70)
    
    # Disable eager execution for faster inference during calibration
    tf.config.run_functions_eagerly(False)
    
    # Load data
    print("Loading glucose data...")
    df = load_glucose_data()
    
    # Determine number of channels
    n_channels = 2 if INCLUDE_FIRST_DIFF else 1
    
    # Collect data for temperature fitting
    temperature_fitting_data = {}
    all_splits = {}
    all_trainers = {}
    
    for horizon in PREDICTION_HORIZONS:
        model_path = f'../data/tcn_{horizon}min.keras'
        if not os.path.exists(model_path):
            print(f"  Skipping {horizon}min - model not found at {model_path}")
            continue
        
        print(f"\n[{horizon}min] Loading model...", end=" ", flush=True)
        
        # Prepare splits for this horizon
        splits = prepare_data_splits(df, horizon_minutes=horizon, verbose=False)
        all_splits[horizon] = splits
        
        # Get horizon-specific bin configuration
        bin_config = get_bin_config(horizon)
        n_bins = bin_config['n_bins']
        
        # Create trainer and load model
        trainer = TCNTrainer(seq_len=SEQ_LEN, n_channels=n_channels, n_bins=n_bins)
        trainer.load(model_path)
        all_trainers[horizon] = trainer
        
        # Get logits on validation set (subsample for speed if needed)
        X_val = splits['val']['X']
        y_bins_val = splits['val']['y_bins']
        
        if len(X_val) > max_samples:
            # Subsample for faster temperature fitting
            np.random.seed(42)
            indices = np.random.choice(len(X_val), max_samples, replace=False)
            X_val_sub = X_val[indices]
            y_bins_sub = y_bins_val[indices]
            print(f"subsampled {max_samples}/{len(X_val)}...", end=" ", flush=True)
        else:
            X_val_sub = X_val
            y_bins_sub = y_bins_val
        
        print("predicting...", end=" ", flush=True)
        logits_val, _ = trainer.predict(X_val_sub)
        
        temperature_fitting_data[horizon] = {
            'logits': logits_val,
            'y_bins': y_bins_sub
        }
        
        print(f"done ({len(X_val_sub):,} samples)")
    
    if not temperature_fitting_data:
        print("\nNo models found to calibrate!")
        return
    
    # Fit per-horizon temperatures
    temperatures = fit_temperatures_all_horizons(temperature_fitting_data, verbose=True)
    
    # Save temperatures
    temps_path = '../data/tcn_temps.json'
    save_temperatures(temperatures, temps_path)
    
    # Evaluate before/after on test set
    print("\n" + "="*70)
    print("CALIBRATION COMPARISON: BEFORE vs AFTER TEMPERATURE SCALING")
    print("="*70)
    
    for horizon in sorted(temperatures.keys()):
        splits = all_splits[horizon]
        trainer = all_trainers[horizon]
        bin_config = get_bin_config(horizon)
        
        X_test = splits['test']['X']
        y_bins_test = splits['test']['y_bins']
        y_deltas_test = splits['test']['y_deltas']
        
        # Subsample test set for faster evaluation
        if len(X_test) > max_samples:
            np.random.seed(123)  # Different seed for test
            test_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test = X_test[test_indices]
            y_bins_test = y_bins_test[test_indices]
            y_deltas_test = y_deltas_test[test_indices]
        
        print(f"\n[{horizon}min] Evaluating on {len(X_test):,} test samples...")
        logits_test, probs_before = trainer.predict(X_test)
        
        T = temperatures[horizon]
        probs_after = apply_temperature(logits_test, T)
        
        evaluate_calibration_comparison(
            logits=logits_test,
            probs_before=probs_before,
            probs_after=probs_after,
            y_bins=y_bins_test,
            y_deltas=y_deltas_test,
            bin_edges=bin_config['bin_edges'],
            bin_centers=bin_config['bin_centers'],
            temperature=T,
            horizon=horizon,
            set_name="Test"
        )
    
    # Summary table
    print("\n" + "="*70)
    print("TEMPERATURE SCALING SUMMARY")
    print("="*70)
    print(f"{'Horizon':>10s} {'Temperature':>12s}")
    print("-" * 25)
    for horizon in sorted(temperatures.keys()):
        print(f"{horizon:>7d} min {temperatures[horizon]:>12.4f}")
    
    print(f"\nTemperatures saved to: {temps_path}")
    
    # Re-enable eager execution for training
    tf.config.run_functions_eagerly(True)
    
    return temperatures


# ============================================================================
# Multi-Horizon Training and Evaluation
# ============================================================================
def compute_curve_jumpiness(quantiles_dict, horizons=PREDICTION_HORIZONS):
    """
    Compute curve jumpiness metric across horizons using second differences.
    
    Given quantile values across horizons for each sample, compute the
    curvature (sum of squared second differences).
    
    Args:
        quantiles_dict: Dict mapping horizon -> (n_samples,) quantile values
        horizons: List of horizon values in order
    
    Returns:
        Float: mean jumpiness over all samples
    """
    # Stack quantiles: (n_horizons, n_samples)
    q_values = np.stack([quantiles_dict[h] for h in horizons], axis=0)
    
    # Second difference across horizons: (n_horizons - 2, n_samples)
    second_diff = q_values[2:, :] - 2 * q_values[1:-1, :] + q_values[:-2, :]
    
    # Sum of squared second differences per sample, then mean
    jumpiness_per_sample = np.sum(second_diff ** 2, axis=0)
    return float(np.mean(jumpiness_per_sample))


def evaluate_multihead_model(model, splits, temperatures=None, set_name='test'):
    """
    Comprehensive evaluation of multi-head TCN model.
    
    Prints per-horizon metrics and cross-horizon jumpiness metrics.
    
    Args:
        model: Trained MultiHorizonTCN instance
        splits: Data splits dictionary from prepare_data_splits_multihorizon
        temperatures: Optional dict mapping head_name -> temperature
        set_name: Which split to evaluate ('val' or 'test')
    
    Returns:
        Dictionary with all metrics
    """
    X = splits[set_name]['X']
    y_deltas = splits[set_name]['y_deltas']
    y_bins = splits[set_name]['y_bins']
    
    print(f"\n{'='*70}")
    print(f"MULTI-HORIZON TCN EVALUATION ({set_name.upper()} SET)")
    print(f"{'='*70}")
    print(f"Samples: {len(X):,}")
    
    # Get predictions
    probs_dict = model.predict_probs(X, temperatures=temperatures)
    
    # Get bin config (shared or per-horizon)
    if USE_SHARED_BINS:
        bin_edges = SHARED_BIN_CONFIG['bin_edges']
        bin_centers = SHARED_BIN_CONFIG['bin_centers']
    
    # Compute per-horizon metrics
    all_metrics = {}
    q50_per_horizon = {}
    q10_per_horizon = {}
    q90_per_horizon = {}
    
    print(f"\n{'Horizon':>8s} {'MAE Q50':>10s} {'Coverage':>10s} {'<Q10':>8s} {'>Q90':>8s} {'Width':>10s}")
    print("-" * 60)
    
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        probs = probs_dict[head_name]
        deltas = y_deltas[head_name]
        
        if not USE_SHARED_BINS:
            config = get_bin_config(horizon)
            bin_edges = config['bin_edges']
            bin_centers = config['bin_centers']
        
        # Compute quantiles
        quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.5, 0.9], bin_edges)
        q10 = quantiles[0.1]
        q50 = quantiles[0.5]
        q90 = quantiles[0.9]
        
        # Store for jumpiness computation
        q10_per_horizon[horizon] = q10
        q50_per_horizon[horizon] = q50
        q90_per_horizon[horizon] = q90
        
        # Coverage
        coverage = np.mean((deltas >= q10) & (deltas <= q90)) * 100
        below_q10 = np.mean(deltas < q10) * 100
        above_q90 = np.mean(deltas > q90) * 100
        width = np.mean(q90 - q10)
        
        # MAE
        mae = np.mean(np.abs(deltas - q50))
        
        all_metrics[horizon] = {
            'mae_q50': mae,
            'coverage_80': coverage,
            'below_q10': below_q10,
            'above_q90': above_q90,
            'width': width,
            'q10': q10,
            'q50': q50,
            'q90': q90
        }
        
        print(f"{horizon:>5d} min {mae:>10.2f} {coverage:>9.1f}% {below_q10:>7.1f}% {above_q90:>7.1f}% {width:>10.1f}")
    
    # Compute jumpiness metrics
    jumpiness_q10 = compute_curve_jumpiness(q10_per_horizon)
    jumpiness_q50 = compute_curve_jumpiness(q50_per_horizon)
    jumpiness_q90 = compute_curve_jumpiness(q90_per_horizon)
    
    print(f"\n--- Cross-Horizon Curve Jumpiness (lower is better) ---")
    print(f"Q10 jumpiness: {jumpiness_q10:.4f}")
    print(f"Q50 jumpiness: {jumpiness_q50:.4f}")
    print(f"Q90 jumpiness: {jumpiness_q90:.4f}")
    
    all_metrics['jumpiness'] = {
        'q10': jumpiness_q10,
        'q50': jumpiness_q50,
        'q90': jumpiness_q90
    }
    
    # =========================================================================
    # NEW DIAGNOSTICS: Q50 Delta Variance and MAE by Volatility
    # =========================================================================
    
    # Q50 Delta Standard Deviation per horizon
    # This shows how much variance the model's predictions have
    print(f"\n--- Q50 Prediction Variance (higher = less conservative) ---")
    print(f"{'Horizon':>8s} {'Q50 Std':>10s} {'Actual Std':>12s} {'Ratio':>10s}")
    print("-" * 45)
    
    q50_delta_stds = {}
    for horizon in PREDICTION_HORIZONS:
        q50 = q50_per_horizon[horizon]
        actual = y_deltas[f"h{horizon}"]
        q50_std = np.std(q50)
        actual_std = np.std(actual)
        ratio = q50_std / actual_std if actual_std > 0 else 0
        q50_delta_stds[horizon] = {'q50_std': q50_std, 'actual_std': actual_std, 'ratio': ratio}
        print(f"{horizon:>5d} min {q50_std:>10.2f} {actual_std:>12.2f} {ratio:>10.1%}")
    
    all_metrics['q50_delta_stds'] = q50_delta_stds
    
    # =========================================================================
    # PMF DIAGNOSTICS: Tail Mass and HDI Width
    # =========================================================================
    print(f"\n--- PMF Tail Mass and HDI Width (diagnoses over-hedging) ---")
    print(f"{'Horizon':>8s} {'Tail K':>8s} {'Tail Mass':>10s} {'HDI 80%':>10s} {'P90-P10':>10s}")
    print("-" * 50)
    
    tail_hdi_metrics = {}
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        probs = probs_dict[head_name]
        
        if USE_SHARED_BINS:
            centers = SHARED_BIN_CONFIG['bin_centers']
        else:
            centers = get_bin_config(horizon)['bin_centers']
        
        # Compute diagnostics
        threshold_K = TAIL_MASS_THRESHOLDS.get(horizon, 50)
        tail_mass = compute_tail_mass(probs, centers, threshold_K)
        hdi_width = compute_hdi_width(probs, centers, credible_mass=0.80)
        p90_p10_width = all_metrics[horizon]['width']
        
        tail_hdi_metrics[horizon] = {
            'tail_threshold': threshold_K,
            'tail_mass': tail_mass,
            'hdi_width': hdi_width,
            'p90_p10_width': p90_p10_width
        }
        
        print(f"{horizon:>5d} min {threshold_K:>8d} {tail_mass*100:>9.1f}% {hdi_width:>10.1f} {p90_p10_width:>10.1f}")
    
    # Averages
    avg_tail = np.mean([m['tail_mass'] for m in tail_hdi_metrics.values()])
    avg_hdi = np.mean([m['hdi_width'] for m in tail_hdi_metrics.values()])
    avg_p90_p10 = np.mean([m['p90_p10_width'] for m in tail_hdi_metrics.values()])
    print("-" * 50)
    print(f"{'Average':>8s} {'':>8s} {avg_tail*100:>9.1f}% {avg_hdi:>10.1f} {avg_p90_p10:>10.1f}")
    
    all_metrics['tail_hdi'] = tail_hdi_metrics
    all_metrics['tail_hdi_avg'] = {
        'tail_mass': avg_tail,
        'hdi_width': avg_hdi,
        'p90_p10_width': avg_p90_p10
    }
    
    # MAE by Volatility Bucket
    if 'volatility' in splits[set_name]:
        volatility = splits[set_name]['volatility']
        
        # Define buckets
        low_mask = volatility < VOL_THRESH_LOW
        med_mask = (volatility >= VOL_THRESH_LOW) & (volatility < VOL_THRESH_HIGH)
        high_mask = volatility >= VOL_THRESH_HIGH
        
        print(f"\n--- MAE by Volatility Bucket ---")
        print(f"Low (<{VOL_THRESH_LOW}): {np.sum(low_mask):,} samples")
        print(f"Med ({VOL_THRESH_LOW}-{VOL_THRESH_HIGH}): {np.sum(med_mask):,} samples")
        print(f"High (>{VOL_THRESH_HIGH}): {np.sum(high_mask):,} samples")
        print()
        print(f"{'Horizon':>8s} {'Low MAE':>10s} {'Med MAE':>10s} {'High MAE':>10s}")
        print("-" * 45)
        
        mae_by_vol = {}
        for horizon in PREDICTION_HORIZONS:
            q50 = q50_per_horizon[horizon]
            actual = y_deltas[f"h{horizon}"]
            
            mae_low = np.mean(np.abs(actual[low_mask] - q50[low_mask])) if np.any(low_mask) else np.nan
            mae_med = np.mean(np.abs(actual[med_mask] - q50[med_mask])) if np.any(med_mask) else np.nan
            mae_high = np.mean(np.abs(actual[high_mask] - q50[high_mask])) if np.any(high_mask) else np.nan
            
            mae_by_vol[horizon] = {'low': mae_low, 'med': mae_med, 'high': mae_high}
            print(f"{horizon:>5d} min {mae_low:>10.2f} {mae_med:>10.2f} {mae_high:>10.2f}")
        
        # Average across horizons
        avg_low = np.nanmean([mae_by_vol[h]['low'] for h in PREDICTION_HORIZONS])
        avg_med = np.nanmean([mae_by_vol[h]['med'] for h in PREDICTION_HORIZONS])
        avg_high = np.nanmean([mae_by_vol[h]['high'] for h in PREDICTION_HORIZONS])
        print("-" * 45)
        print(f"{'Average':>8s} {avg_low:>10.2f} {avg_med:>10.2f} {avg_high:>10.2f}")
        
        all_metrics['mae_by_volatility'] = mae_by_vol
        all_metrics['avg_mae_by_volatility'] = {'low': avg_low, 'med': avg_med, 'high': avg_high}
    
    return all_metrics


def _custom_training_loop(model, X_train, y_train, X_val, y_val, 
                          train_weights, epochs, batch_size, patience):
    """
    Custom training loop for horizon-specific bins.
    
    model.fit() hangs with tf.data when using dictionary outputs with 
    different shapes per key. This manual loop bypasses that issue.
    """
    import sys
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    
    n_samples = len(X_train)
    n_batches = n_samples // batch_size
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    lr_patience = 5
    lr_patience_counter = 0
    current_lr = float(model.optimizer.learning_rate)
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        
        epoch_losses = []
        
        # Training loop
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]
            
            x_batch = X_train[batch_indices]
            y_batch = {k: v[batch_indices] for k, v in y_train.items()}
            
            # Get sample weights for this batch
            if train_weights is not None:
                # Extract weights for one key (all have same weights)
                weight_key = list(train_weights.keys())[0]
                sw_batch = train_weights[weight_key][batch_indices]
            else:
                sw_batch = None
            
            result = model.train_step((x_batch, y_batch, sw_batch))
            epoch_losses.append(float(result['loss']))
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"\rEpoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{n_batches} - loss: {np.mean(epoch_losses):.4f}", end='', flush=True)
        
        train_loss = np.mean(epoch_losses)
        
        # Validation
        val_losses = []
        n_val = len(X_val)
        n_val_batches = max(1, n_val // batch_size)
        for batch_idx in range(n_val_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_val)
            x_batch = X_val[start:end]
            y_batch = {k: v[start:end] for k, v in y_val.items()}
            result = model.test_step((x_batch, y_batch))
            val_losses.append(float(result['loss']))
        
        val_loss = np.mean(val_losses)
        
        print(f"\rEpoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}", flush=True)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
            lr_patience_counter = 0
        else:
            patience_counter += 1
            lr_patience_counter += 1
            
            # Reduce LR on plateau
            if lr_patience_counter >= lr_patience and current_lr > 1e-5:
                current_lr = current_lr * 0.5
                model.optimizer.learning_rate.assign(current_lr)
                print(f"Reducing learning rate to {current_lr:.6f}")
                lr_patience_counter = 0
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)
        print(f"Restored best weights (val_loss: {best_val_loss:.4f})")
    
    return {'loss': [train_loss], 'val_loss': [val_loss]}


def train_multihead():
    """
    Train multi-horizon TCN model with cross-horizon coherence regularization.
    
    Produces a single model with 6 heads (one per horizon).
    """
    print("\n" + "="*70)
    print("MULTI-HORIZON TCN TRAINING")
    print("="*70)
    print(f"Horizons: {PREDICTION_HORIZONS}")
    print(f"Shared bins: {USE_SHARED_BINS}")
    print(f"Lambda curve: {LAMBDA_CURVE}")
    print(f"Lambda varmono: {LAMBDA_VARMONO}")
    print(f"Head adapter: {USE_HEAD_ADAPTER} (dim={HEAD_ADAPTER_DIM})")
    print(f"Point loss: {POINT_LOSS_ENABLED} (weight={POINT_LOSS_WEIGHT}, type={POINT_LOSS_TYPE})")
    print(f"Volatility weighting: {VOLATILITY_WEIGHTING_ENABLED} (applied to point loss only)")
    
    # Disable eager execution for faster training
    tf.config.run_functions_eagerly(False)
    
    # Load data
    df = load_glucose_data()
    
    # Prepare multi-horizon splits
    splits = prepare_data_splits_multihorizon(df, verbose=True)
    
    # Determine number of channels
    n_channels = 2 if INCLUDE_FIRST_DIFF else 1
    
    # Build multi-head backbone
    print("\nBuilding multi-head model...")
    backbone = build_tcn_multihead(
        seq_len=SEQ_LEN,
        n_channels=n_channels
    )
    
    # Wrap in MultiHorizonTCN for custom training
    model = MultiHorizonTCN(
        backbone,
        sigma_bins=SIGMA_BINS,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_curve=LAMBDA_CURVE,
        lambda_varmono=LAMBDA_VARMONO
    )
    
    # Compile with optimizer (loss computed in train_step)
    # Note: When USE_SHARED_BINS=False, we must use eager mode because TensorFlow's
    # graph tracing hangs with variable output sizes per head
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        run_eagerly=(not USE_SHARED_BINS)  # Eager mode for horizon-specific bins
    )
    
    # Count parameters
    # Build model by calling it once
    dummy_input = np.zeros((1, SEQ_LEN, n_channels), dtype=np.float32)
    _ = model(dummy_input)
    n_params = count_params(model)
    print(f"Model parameters: {n_params:,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=PATIENCE_LR_REDUCE,
            factor=LR_REDUCE_FACTOR,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Prepare training data with BOTH bins and deltas for point loss
    X_train = splits['train']['X']
    X_val = splits['val']['X']
    
    # Create combined y_dict with both bins and deltas per head
    # Format: {h5_bins: ..., h5_deltas: ..., h10_bins: ..., h10_deltas: ..., ...}
    y_train = {}
    y_val = {}
    for head_name in HEAD_NAMES:
        y_train[f'{head_name}_bins'] = splits['train']['y_bins'][head_name]
        y_train[f'{head_name}_deltas'] = splits['train']['y_deltas'][head_name]
        y_val[f'{head_name}_bins'] = splits['val']['y_bins'][head_name]
        y_val[f'{head_name}_deltas'] = splits['val']['y_deltas'][head_name]
    
    # Compute sample weights based on volatility
    if VOLATILITY_WEIGHTING_ENABLED:
        train_weights_array = compute_volatility_weights(splits['train']['volatility'])
        # For multi-output models, Keras expects sample_weight as a dict matching y_dict keys
        # We provide weights for both _bins and _deltas keys (same weights for each sample)
        train_weights = {}
        for head_name in HEAD_NAMES:
            train_weights[f'{head_name}_bins'] = train_weights_array
            train_weights[f'{head_name}_deltas'] = train_weights_array
        n_low = np.sum(train_weights_array == VOL_WEIGHT_LOW)
        n_med = np.sum(train_weights_array == VOL_WEIGHT_MED)
        n_high = np.sum(train_weights_array == VOL_WEIGHT_HIGH)
        print(f"\nVolatility sample weights (applied to point loss only):")
        print(f"  Low (<{VOL_THRESH_LOW}): {n_low:,} samples (weight={VOL_WEIGHT_LOW})")
        print(f"  Med ({VOL_THRESH_LOW}-{VOL_THRESH_HIGH}): {n_med:,} samples (weight={VOL_WEIGHT_MED})")
        print(f"  High (>{VOL_THRESH_HIGH}): {n_high:,} samples (weight={VOL_WEIGHT_HIGH})")
    else:
        train_weights = None
    
    print(f"\nTraining multi-head model...")
    print(f"  Epochs: {MAX_EPOCHS} (early stopping patience: {PATIENCE_EARLY_STOP})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Use custom training loop for horizon-specific bins (model.fit hangs with tf.data)
    if not USE_SHARED_BINS:
        print(f"  Using custom training loop (horizon-specific bins)")
        history = _custom_training_loop(
            model, X_train, y_train, X_val, y_val,
            train_weights, MAX_EPOCHS, BATCH_SIZE, PATIENCE_EARLY_STOP
        )
    else:
        # Train with model.fit (works fine with shared bins)
        history = model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    
    # =========================================================================
    # TEMPERATURE SCALING
    # =========================================================================
    print("\n" + "="*70)
    print("TEMPERATURE SCALING - FITTING ON VALIDATION SET")
    print("="*70)
    
    # Get logits on validation set for each head
    logits_dict = model.predict_logits(X_val)
    
    # Fit temperature for each head
    # Note: y_val uses new format with keys like h5_bins, h5_deltas
    temperature_fitting_data = {}
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        temperature_fitting_data[horizon] = {
            'logits': logits_dict[head_name],
            'y_bins': y_val[f'{head_name}_bins']
        }
    
    temperatures, temp_metadata = fit_temperatures_all_horizons(temperature_fitting_data, verbose=True)
    
    # Convert to head_name keys for model.predict_probs
    temperatures_by_head = {f"h{h}": T for h, T in temperatures.items()}
    
    # Save temperatures
    temps_path = '../data/tcn_multihead_temps.json'
    save_temperatures(temperatures, temps_path, metadata=temp_metadata)
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("EVALUATION BEFORE TEMPERATURE SCALING")
    print("="*70)
    
    metrics_before = evaluate_multihead_model(model, splits, temperatures=None, set_name='test')
    
    print("\n" + "="*70)
    print("EVALUATION AFTER TEMPERATURE SCALING")
    print("="*70)
    
    metrics_after = evaluate_multihead_model(model, splits, temperatures=temperatures_by_head, set_name='test')
    
    # =========================================================================
    # SAVE MODEL
    # =========================================================================
    model_path = '../data/tcn_multihead.keras'
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\n{'Horizon':>8s} {'Temp':>8s} {'MAE (before)':>12s} {'MAE (after)':>12s}")
    print("-" * 45)
    for horizon in PREDICTION_HORIZONS:
        T = temperatures[horizon]
        mae_before = metrics_before[horizon]['mae_q50']
        mae_after = metrics_after[horizon]['mae_q50']
        print(f"{horizon:>5d} min {T:>8.4f} {mae_before:>12.2f} {mae_after:>12.2f}")
    
    print(f"\nJumpiness (Q50):")
    print(f"  Before temp scaling: {metrics_before['jumpiness']['q50']:.4f}")
    print(f"  After temp scaling:  {metrics_after['jumpiness']['q50']:.4f}")
    
    print("\n" + "="*70)
    print("MULTI-HORIZON TRAINING COMPLETE!")
    print("="*70)
    
    return model, temperatures, metrics_after


# ============================================================================
# Configurable Multi-Horizon Training with Sweep Support
# ============================================================================
def train_multihead_configurable(
    splits,
    lambda_curve=LAMBDA_CURVE,
    lambda_varmono=LAMBDA_VARMONO,
    curve_penalty_type=CURVE_PENALTY_TYPE,
    huber_delta=HUBER_DELTA,
    use_head_adapter=USE_HEAD_ADAPTER,
    head_adapter_dim=HEAD_ADAPTER_DIM,
    two_stage_training=TWO_STAGE_TRAINING,
    stage_b_epochs=STAGE_B_EPOCHS,
    stage_b_lr=STAGE_B_LR,
    stage_b_lambda_curve=STAGE_B_LAMBDA_CURVE,
    verbose=True
):
    """
    Train multi-horizon TCN with configurable hyperparameters.
    
    Args:
        splits: Data splits from prepare_data_splits_multihorizon()
        lambda_curve: Curve smoothness penalty weight
        lambda_varmono: Variance monotonicity penalty weight
        curve_penalty_type: "l2", "l1", or "huber"
        huber_delta: Delta for Huber loss
        use_head_adapter: Whether to use per-head adapter layers
        head_adapter_dim: Hidden dim for adapter layers
        two_stage_training: If True, train without curve penalty first, then fine-tune
        stage_b_epochs: Number of fine-tuning epochs
        stage_b_lr: Learning rate for fine-tuning
        stage_b_lambda_curve: Curve penalty for fine-tuning stage
        verbose: Whether to print progress
    
    Returns:
        Tuple of (model, temperatures, metrics_dict)
    """
    # Disable eager execution for faster training
    tf.config.run_functions_eagerly(False)
    
    n_channels = 2 if INCLUDE_FIRST_DIFF else 1
    
    # Build model
    backbone = build_tcn_multihead(
        seq_len=SEQ_LEN,
        n_channels=n_channels,
        use_head_adapter=use_head_adapter,
        head_adapter_dim=head_adapter_dim
    )
    
    # Prepare training data with BOTH bins and deltas for point loss
    X_train = splits['train']['X']
    X_val = splits['val']['X']
    
    # Create combined y_dict with both bins and deltas per head
    y_train = {}
    y_val = {}
    for head_name in HEAD_NAMES:
        y_train[f'{head_name}_bins'] = splits['train']['y_bins'][head_name]
        y_train[f'{head_name}_deltas'] = splits['train']['y_deltas'][head_name]
        y_val[f'{head_name}_bins'] = splits['val']['y_bins'][head_name]
        y_val[f'{head_name}_deltas'] = splits['val']['y_deltas'][head_name]
    
    # Compute sample weights based on volatility
    if VOLATILITY_WEIGHTING_ENABLED and 'volatility' in splits['train']:
        train_weights_array = compute_volatility_weights(splits['train']['volatility'])
        # For multi-output models, Keras expects sample_weight as a dict matching y_dict keys
        train_weights = {}
        for head_name in HEAD_NAMES:
            train_weights[f'{head_name}_bins'] = train_weights_array
            train_weights[f'{head_name}_deltas'] = train_weights_array
        if verbose:
            n_low = np.sum(train_weights_array == VOL_WEIGHT_LOW)
            n_med = np.sum(train_weights_array == VOL_WEIGHT_MED)
            n_high = np.sum(train_weights_array == VOL_WEIGHT_HIGH)
            print(f"\nVolatility weighting (point loss only): low={n_low}, med={n_med}, high={n_high}")
    else:
        train_weights = None
    
    if two_stage_training:
        # =====================================================================
        # STAGE A: Train without curve penalty
        # =====================================================================
        if verbose:
            print("\n--- STAGE A: Training without curve penalty ---")
        
        model = MultiHorizonTCN(
            backbone,
            sigma_bins=SIGMA_BINS,
            lambda_smooth=LAMBDA_SMOOTH,
            lambda_curve=0.0,  # No curve penalty in stage A
            lambda_varmono=lambda_varmono,
            curve_penalty_type=curve_penalty_type,
            huber_delta=huber_delta
        )
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            run_eagerly=(not USE_SHARED_BINS)
        )
        
        # Build model
        dummy_input = np.zeros((1, SEQ_LEN, n_channels), dtype=np.float32)
        _ = model(dummy_input)
        
        callbacks_a = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE_EARLY_STOP,
                restore_best_weights=True,
                verbose=1 if verbose else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=PATIENCE_LR_REDUCE,
                factor=LR_REDUCE_FACTOR,
                min_lr=1e-6,
                verbose=1 if verbose else 0
            )
        ]
        
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_a,
            verbose=2 if verbose else 0
        )
        
        # Save stage A weights
        stage_a_weights = model.get_weights()
        
        # =====================================================================
        # STAGE B: Fine-tune with curve penalty
        # =====================================================================
        if verbose:
            print(f"\n--- STAGE B: Fine-tuning with lambda_curve={stage_b_lambda_curve} ---")
        
        # Update lambda_curve for stage B
        model.lambda_curve = stage_b_lambda_curve
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=stage_b_lr),
            run_eagerly=(not USE_SHARED_BINS)
        )
        
        callbacks_b = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # Shorter patience for fine-tuning
                restore_best_weights=True,
                verbose=1 if verbose else 0
            )
        ]
        
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            validation_data=(X_val, y_val),
            epochs=stage_b_epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_b,
            verbose=2 if verbose else 0
        )
        
    else:
        # =====================================================================
        # Single-stage training (original approach)
        # =====================================================================
        model = MultiHorizonTCN(
            backbone,
            sigma_bins=SIGMA_BINS,
            lambda_smooth=LAMBDA_SMOOTH,
            lambda_curve=lambda_curve,
            lambda_varmono=lambda_varmono,
            curve_penalty_type=curve_penalty_type,
            huber_delta=huber_delta
        )
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            run_eagerly=(not USE_SHARED_BINS)
        )
        
        # Build model
        dummy_input = np.zeros((1, SEQ_LEN, n_channels), dtype=np.float32)
        _ = model(dummy_input)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE_EARLY_STOP,
                restore_best_weights=True,
                verbose=1 if verbose else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=PATIENCE_LR_REDUCE,
                factor=LR_REDUCE_FACTOR,
                min_lr=1e-6,
                verbose=1 if verbose else 0
            )
        ]
        
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2 if verbose else 0
        )
    
    # =========================================================================
    # TEMPERATURE SCALING
    # =========================================================================
    if verbose:
        print("\nFitting temperature scaling...")
    
    logits_dict = model.predict_logits(X_val)
    
    temperature_fitting_data = {}
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        temperature_fitting_data[horizon] = {
            'logits': logits_dict[head_name],
            'y_bins': y_val[head_name]
        }
    
    temperatures, _ = fit_temperatures_all_horizons(temperature_fitting_data, verbose=False)
    temperatures_by_head = {f"h{h}": T for h, T in temperatures.items()}
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    metrics = evaluate_multihead_model(
        model, splits, temperatures=temperatures_by_head, set_name='test'
    ) if verbose else _evaluate_multihead_quiet(model, splits, temperatures_by_head)
    
    return model, temperatures, metrics


def _evaluate_multihead_quiet(model, splits, temperatures_by_head):
    """Evaluate model without printing (for sweep runs)."""
    X_test = splits['test']['X']
    y_deltas = splits['test']['y_deltas']
    
    probs_dict = model.predict_probs(X_test, temperatures=temperatures_by_head)
    
    if USE_SHARED_BINS:
        bin_edges = SHARED_BIN_CONFIG['bin_edges']
    
    metrics = {}
    q50_per_horizon = {}
    q10_per_horizon = {}
    q90_per_horizon = {}
    
    for horizon in PREDICTION_HORIZONS:
        head_name = f"h{horizon}"
        probs = probs_dict[head_name]
        deltas = y_deltas[head_name]
        
        if not USE_SHARED_BINS:
            bin_edges = get_bin_config(horizon)['bin_edges']
        
        quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.5, 0.9], bin_edges)
        q10 = quantiles[0.1]
        q50 = quantiles[0.5]
        q90 = quantiles[0.9]
        
        q10_per_horizon[horizon] = q10
        q50_per_horizon[horizon] = q50
        q90_per_horizon[horizon] = q90
        
        coverage = np.mean((deltas >= q10) & (deltas <= q90)) * 100
        mae = np.mean(np.abs(deltas - q50))
        width = np.mean(q90 - q10)
        
        metrics[horizon] = {
            'mae_q50': mae,
            'coverage_80': coverage,
            'width': width
        }
    
    # Jumpiness
    metrics['jumpiness'] = {
        'q10': compute_curve_jumpiness(q10_per_horizon),
        'q50': compute_curve_jumpiness(q50_per_horizon),
        'q90': compute_curve_jumpiness(q90_per_horizon)
    }
    
    return metrics


def run_lambda_curve_sweep(save_path='../data/sweep_results.json'):
    """
    Run ablation/sweep over LAMBDA_CURVE values.
    
    Trains the multi-horizon model with each lambda value and saves results.
    """
    import json
    
    print("\n" + "="*70)
    print("LAMBDA_CURVE SWEEP EXPERIMENT")
    print("="*70)
    
    # Disable eager execution
    tf.config.run_functions_eagerly(False)
    
    # Load data once
    df = load_glucose_data()
    splits = prepare_data_splits_multihorizon(df, verbose=True)
    
    # Configurations to test
    configs = []
    
    # Basic sweep over lambda values
    for lc in LAMBDA_CURVE_SWEEP_VALUES:
        configs.append({
            'name': f'lambda_{lc}',
            'lambda_curve': lc,
            'curve_penalty_type': 'l2',
            'use_head_adapter': False,
            'two_stage_training': False
        })
    
    # Add Huber variant at a few lambda values
    for lc in [1e-4, 3e-4]:
        configs.append({
            'name': f'huber_lambda_{lc}',
            'lambda_curve': lc,
            'curve_penalty_type': 'huber',
            'use_head_adapter': False,
            'two_stage_training': False
        })
    
    # Add L1 variant
    for lc in [1e-4, 3e-4]:
        configs.append({
            'name': f'l1_lambda_{lc}',
            'lambda_curve': lc,
            'curve_penalty_type': 'l1',
            'use_head_adapter': False,
            'two_stage_training': False
        })
    
    # Add head adapter variant
    configs.append({
        'name': 'adapter_lambda_0',
        'lambda_curve': 0,
        'curve_penalty_type': 'l2',
        'use_head_adapter': True,
        'two_stage_training': False
    })
    configs.append({
        'name': 'adapter_lambda_1e-4',
        'lambda_curve': 1e-4,
        'curve_penalty_type': 'l2',
        'use_head_adapter': True,
        'two_stage_training': False
    })
    
    # Add two-stage variant
    configs.append({
        'name': 'two_stage_1e-4',
        'lambda_curve': 0,  # Stage A has no penalty
        'curve_penalty_type': 'l2',
        'use_head_adapter': False,
        'two_stage_training': True,
        'stage_b_lambda_curve': 1e-4
    })
    
    results = []
    
    print(f"\nRunning {len(configs)} configurations...")
    print("-" * 70)
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config['name']}")
        print(f"  lambda_curve={config['lambda_curve']}, penalty={config['curve_penalty_type']}, "
              f"adapter={config['use_head_adapter']}, two_stage={config['two_stage_training']}")
        
        try:
            _, temps, metrics = train_multihead_configurable(
                splits,
                lambda_curve=config['lambda_curve'],
                curve_penalty_type=config['curve_penalty_type'],
                use_head_adapter=config['use_head_adapter'],
                two_stage_training=config['two_stage_training'],
                stage_b_lambda_curve=config.get('stage_b_lambda_curve', STAGE_B_LAMBDA_CURVE),
                verbose=False
            )
            
            # Compile result row
            row = {
                'config_name': config['name'],
                'lambda_curve': config['lambda_curve'],
                'penalty_type': config['curve_penalty_type'],
                'use_head_adapter': config['use_head_adapter'],
                'two_stage': config['two_stage_training'],
            }
            
            # Per-horizon metrics
            for horizon in PREDICTION_HORIZONS:
                row[f'mae_h{horizon}'] = metrics[horizon]['mae_q50']
                row[f'cov_h{horizon}'] = metrics[horizon]['coverage_80']
                row[f'width_h{horizon}'] = metrics[horizon]['width']
            
            # Aggregate metrics
            row['mae_avg'] = np.mean([metrics[h]['mae_q50'] for h in PREDICTION_HORIZONS])
            row['cov_avg'] = np.mean([metrics[h]['coverage_80'] for h in PREDICTION_HORIZONS])
            
            # Jumpiness
            row['jump_q10'] = metrics['jumpiness']['q10']
            row['jump_q50'] = metrics['jumpiness']['q50']
            row['jump_q90'] = metrics['jumpiness']['q90']
            
            # Temperatures
            for horizon in PREDICTION_HORIZONS:
                row[f'temp_h{horizon}'] = temps[horizon]
            
            results.append(row)
            
            print(f"  -> MAE avg: {row['mae_avg']:.2f}, Cov avg: {row['cov_avg']:.1f}%, "
                  f"Jump Q50: {row['jump_q50']:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # =========================================================================
    # SAVE AND PRINT RESULTS
    # =========================================================================
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    
    # Also save as CSV
    csv_path = save_path.replace('.json', '.csv')
    import csv
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SWEEP RESULTS SUMMARY")
    print("="*70)
    print(f"{'Config':<25s} {'MAE avg':>8s} {'Cov avg':>8s} {'Jump Q50':>10s}")
    print("-" * 55)
    
    # Sort by jump_q50 to show smoothest first
    sorted_results = sorted(results, key=lambda x: x['jump_q50'])
    
    for r in sorted_results:
        print(f"{r['config_name']:<25s} {r['mae_avg']:>8.2f} {r['cov_avg']:>7.1f}% {r['jump_q50']:>10.4f}")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find baseline (lambda=0)
    baseline = next((r for r in results if r['lambda_curve'] == 0 and not r['two_stage']), None)
    
    if baseline:
        print(f"\nBaseline (lambda=0): MAE={baseline['mae_avg']:.2f}, Jump Q50={baseline['jump_q50']:.4f}")
        
        # Find configs with >=10x jumpiness reduction and minimal MAE increase
        good_candidates = []
        for r in sorted_results:
            if r['config_name'] == baseline['config_name']:
                continue
            jump_reduction = baseline['jump_q50'] / max(r['jump_q50'], 1e-6)
            mae_increase = r['mae_avg'] - baseline['mae_avg']
            
            if jump_reduction >= 10:
                good_candidates.append({
                    **r,
                    'jump_reduction': jump_reduction,
                    'mae_increase': mae_increase
                })
        
        if good_candidates:
            # Sort by MAE increase (least first)
            good_candidates.sort(key=lambda x: x['mae_increase'])
            
            print("\nTop candidates (>=10x jumpiness reduction, sorted by MAE increase):")
            for i, c in enumerate(good_candidates[:3]):
                print(f"  {i+1}. {c['config_name']}: "
                      f"MAE={c['mae_avg']:.2f} (+{c['mae_increase']:.2f}), "
                      f"Jump={c['jump_q50']:.4f} ({c['jump_reduction']:.0f}x better)")
        else:
            print("\nNo configs achieved >=10x jumpiness reduction.")
    
    return results


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_all_tests()
    elif '--sweep' in sys.argv:
        # Run lambda_curve sweep experiment
        run_lambda_curve_sweep()
    elif '--multihead' in sys.argv:
        # Train multi-horizon model
        train_multihead()
    elif '--all' in sys.argv:
        # Train all horizons (legacy single-horizon models)
        train_all_horizons()
    elif '--calibrate' in sys.argv:
        # Apply temperature scaling to existing models
        calibrate_existing_models()
    else:
        main()


