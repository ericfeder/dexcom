"""
Compare TCN model performance with different history lookback windows.

Tests whether P50 MAE, prediction interval width (P10-P90), or prediction 
interval calibration changes when switching from 40 minutes to 60 minutes of history.

Usage:
    python test_lookback_comparison.py
"""
import os
import sys

# Set TensorFlow env vars BEFORE importing tf
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

# Use eager execution to avoid graph compilation issues with dynamic loss function
tf.config.run_functions_eagerly(True)

# ============================================================================
# Configuration
# ============================================================================
# Lookback windows to compare
LOOKBACK_WINDOWS_TO_TEST = [40, 60]  # minutes

# Model configuration
SAMPLE_INTERVAL = 5      # minutes between readings
PREDICTION_HORIZONS = [15]  # Test on 15-min horizon for speed

# Horizon-specific binning configuration
BIN_CONFIG = {
    5:  {'min': -60,  'max': 70},
    10: {'min': -80,  'max': 90},
    15: {'min': -100, 'max': 120},
    20: {'min': -120, 'max': 150},
    25: {'min': -140, 'max': 170},
    30: {'min': -150, 'max': 190},
}

# Training hyperparameters
SIGMA_BINS = 4.0
LAMBDA_SMOOTH = 1e-4
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 20  # Reduced for faster comparison
PATIENCE_EARLY_STOP = 4  # Reduced for faster comparison
PATIENCE_LR_REDUCE = 2
LR_REDUCE_FACTOR = 0.5
DROPOUT_RATE = 0.1

# Data subsampling for faster testing (set to None for full data)
MAX_TRAIN_SAMPLES = 10000  # Subsample training data for speed
MAX_VAL_SAMPLES = 2000
MAX_TEST_SAMPLES = 3000

# TCN architecture
TCN_FILTERS = 32
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]

# Data configuration
JUNE_START = pd.Timestamp('2025-06-01', tz='America/New_York')
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# Binning Utilities
# ============================================================================
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


def delta_to_bin(delta, horizon=15):
    """Convert delta (mg/dL) to bin index for a specific horizon."""
    config = get_bin_config(horizon)
    delta = np.asarray(delta)
    clamped = np.clip(delta, config['delta_min'], config['delta_max'])
    bin_idx = np.round(clamped - config['delta_min']).astype(int)
    return bin_idx


def quantile_from_pmf(probs, q, bin_edges):
    """Compute quantile from PMF using CDF and linear interpolation."""
    probs = np.asarray(probs)
    cdf = np.cumsum(probs)
    k = np.searchsorted(cdf, q)
    if k == 0:
        return bin_edges[0]
    if k >= len(probs):
        return bin_edges[-1]
    cdf_prev = cdf[k - 1]
    alpha = (q - cdf_prev) / max(probs[k], 1e-9)
    alpha = np.clip(alpha, 0, 1)
    return bin_edges[k] + alpha * (bin_edges[k + 1] - bin_edges[k])


def quantiles_from_pmf_batch(probs_batch, quantiles, bin_edges):
    """Compute multiple quantiles from a batch of PMFs."""
    results = {}
    for q in quantiles:
        q_values = np.array([quantile_from_pmf(p, q, bin_edges) for p in probs_batch])
        results[q] = q_values
    return results


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
    print(f"Loaded {len(df):,} glucose readings")
    return df


def create_sequences(df, seq_len, horizon_minutes, include_velocity=True):
    """
    Create sliding window sequences for TCN input.
    
    Args:
        df: DataFrame with 'value' column
        seq_len: Number of past samples in window
        horizon_minutes: Prediction horizon in minutes
        include_velocity: Whether to include first-difference velocity feature
    
    Returns:
        X, y_bins, y_deltas, valid_indices
    """
    values = df['value'].values
    n = len(values)
    horizon_steps = horizon_minutes // SAMPLE_INTERVAL
    
    start_idx = seq_len  # Need seq_len+1 past values for velocity
    end_idx = n - 1 - horizon_steps
    
    if end_idx <= start_idx:
        raise ValueError("Not enough data for the given sequence length and horizon")
    
    n_samples = end_idx - start_idx + 1
    n_channels = 2 if include_velocity else 1
    
    X = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    y_deltas = np.zeros(n_samples, dtype=np.float32)
    valid_indices = np.arange(start_idx, end_idx + 1)
    
    for i, idx in enumerate(valid_indices):
        g_current = values[idx]
        g_future = values[idx + horizon_steps]
        delta = g_future - g_current
        y_deltas[i] = delta
        
        past_values = values[idx - seq_len + 1:idx + 1]
        x0 = (past_values - g_current) / 50.0
        X[i, :, 0] = x0
        
        if include_velocity:
            extended_past = values[idx - seq_len:idx + 1]
            velocity = np.diff(extended_past) / 10.0
            X[i, :, 1] = velocity
    
    y_bins = delta_to_bin(y_deltas, horizon=horizon_minutes)
    return X, y_bins, y_deltas, valid_indices


def filter_valid_samples(df, X, y_bins, y_deltas, valid_indices, seq_len, horizon_minutes):
    """Filter samples to ensure valid glucose values (40-400 mg/dL range)."""
    values = df['value'].values
    horizon_steps = horizon_minutes // SAMPLE_INTERVAL
    
    valid_mask = np.ones(len(valid_indices), dtype=bool)
    
    for i, idx in enumerate(valid_indices):
        g_current = values[idx]
        g_future = values[idx + horizon_steps]
        
        if g_current < 40 or g_current > 400:
            valid_mask[i] = False
        if g_future < 40 or g_future > 400:
            valid_mask[i] = False
        
        past_values = values[idx - seq_len + 1:idx + 1]
        if np.any(past_values < 40) or np.any(past_values > 400):
            valid_mask[i] = False
    
    return (X[valid_mask], y_bins[valid_mask], y_deltas[valid_mask], 
            valid_indices[valid_mask])


def prepare_data_splits(df, seq_len, horizon_minutes, verbose=True):
    """Prepare train/val/test splits with 70/10/20 ratio."""
    df_filtered = df[df['displayTime'] >= JUNE_START].copy().reset_index(drop=True)
    if verbose:
        print(f"Filtered to from {JUNE_START.strftime('%Y-%m-%d')}: {len(df_filtered):,} samples")
    
    X, y_bins, y_deltas, valid_indices = create_sequences(
        df_filtered, seq_len=seq_len, horizon_minutes=horizon_minutes, include_velocity=True
    )
    if verbose:
        print(f"Created {len(X):,} sequences (seq_len={seq_len})")
    
    X, y_bins, y_deltas, valid_indices = filter_valid_samples(
        df_filtered, X, y_bins, y_deltas, valid_indices, seq_len, horizon_minutes
    )
    if verbose:
        print(f"After filtering: {len(X):,} valid sequences")
    
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    splits = {
        'train': {
            'X': X[:train_end],
            'y_bins': y_bins[:train_end],
            'y_deltas': y_deltas[:train_end],
        },
        'val': {
            'X': X[train_end:val_end],
            'y_bins': y_bins[train_end:val_end],
            'y_deltas': y_deltas[train_end:val_end],
        },
        'test': {
            'X': X[val_end:],
            'y_bins': y_bins[val_end:],
            'y_deltas': y_deltas[val_end:],
        },
        'df': df_filtered
    }
    
    # Subsample for faster testing
    if MAX_TRAIN_SAMPLES and len(splits['train']['X']) > MAX_TRAIN_SAMPLES:
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(len(splits['train']['X']), MAX_TRAIN_SAMPLES, replace=False)
        splits['train'] = {k: v[idx] if isinstance(v, np.ndarray) else v for k, v in splits['train'].items()}
    
    if MAX_VAL_SAMPLES and len(splits['val']['X']) > MAX_VAL_SAMPLES:
        np.random.seed(RANDOM_SEED + 1)
        idx = np.random.choice(len(splits['val']['X']), MAX_VAL_SAMPLES, replace=False)
        splits['val'] = {k: v[idx] if isinstance(v, np.ndarray) else v for k, v in splits['val'].items()}
    
    if MAX_TEST_SAMPLES and len(splits['test']['X']) > MAX_TEST_SAMPLES:
        np.random.seed(RANDOM_SEED + 2)
        idx = np.random.choice(len(splits['test']['X']), MAX_TEST_SAMPLES, replace=False)
        splits['test'] = {k: v[idx] if isinstance(v, np.ndarray) else v for k, v in splits['test'].items()}
    
    if verbose:
        print(f"  Train: {len(splits['train']['X']):,}, Val: {len(splits['val']['X']):,}, Test: {len(splits['test']['X']):,}")
    
    return splits


# ============================================================================
# TCN Model Architecture
# ============================================================================
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate, name_prefix):
    """Residual block with causal dilated convolution."""
    pad_size = (kernel_size - 1) * dilation_rate
    
    if x.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding='same', 
                         name=f'{name_prefix}_residual_conv')(x)
    else:
        residual = x
    
    x_padded = ZeroPadding1D(padding=(pad_size, 0), name=f'{name_prefix}_pad')(x)
    conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding='valid', activation='relu', name=f'{name_prefix}_conv1')(x_padded)
    conv1 = LayerNormalization(name=f'{name_prefix}_ln1')(conv1)
    conv1 = Dropout(dropout_rate, name=f'{name_prefix}_drop1')(conv1)
    
    conv1_padded = ZeroPadding1D(padding=(pad_size, 0), name=f'{name_prefix}_pad2')(conv1)
    conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding='valid', activation='relu', name=f'{name_prefix}_conv2')(conv1_padded)
    conv2 = LayerNormalization(name=f'{name_prefix}_ln2')(conv2)
    conv2 = Dropout(dropout_rate, name=f'{name_prefix}_drop2')(conv2)
    
    out = Add(name=f'{name_prefix}_add')([residual, conv2])
    return out


def build_tcn(seq_len, n_channels, n_bins, filters=TCN_FILTERS, kernel_size=TCN_KERNEL_SIZE,
              dilations=TCN_DILATIONS, dropout_rate=DROPOUT_RATE):
    """Build TCN model for distribution prediction."""
    inputs = Input(shape=(seq_len, n_channels), name='input')
    x = Conv1D(filters, 1, padding='same', name='input_proj')(inputs)
    
    for i, d in enumerate(dilations):
        x = residual_block(x, filters, kernel_size, d, dropout_rate, name_prefix=f'block_{i}_d{d}')
    
    x = GlobalAveragePooling1D(name='global_pool')(x)
    x = Dense(64, activation='relu', name='fc')(x)
    x = Dropout(dropout_rate, name='fc_drop')(x)
    logits = Dense(n_bins, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=logits, name='TCN_Distribution')
    return model


# ============================================================================
# Loss Function
# ============================================================================
def distribution_loss_fn(y_true, logits, sigma_bins=SIGMA_BINS, lambda_smooth=LAMBDA_SMOOTH):
    """Loss function for distribution prediction."""
    n_bins = tf.shape(logits)[1]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    y_true_float = tf.cast(y_true, tf.float32)[:, tf.newaxis]
    bin_indices_expanded = bin_indices[tf.newaxis, :]
    
    weights = tf.exp(-0.5 * tf.square((bin_indices_expanded - y_true_float) / sigma_bins))
    soft_targets = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    ce_loss = -tf.reduce_sum(soft_targets * log_probs, axis=-1)
    
    z = logits
    second_diff = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
    smooth_penalty = tf.reduce_sum(tf.square(second_diff), axis=-1)
    
    total_loss = ce_loss + lambda_smooth * smooth_penalty
    return tf.reduce_mean(total_loss)


def create_distribution_loss():
    """Factory function for distribution loss."""
    def loss(y_true, logits):
        return distribution_loss_fn(y_true, logits)
    return loss


# ============================================================================
# Training
# ============================================================================
class TCNTrainer:
    """Trainer class for TCN distribution model."""
    
    def __init__(self, seq_len, n_channels=2, n_bins=221):
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.model = None
    
    def build_model(self):
        """Build and compile the TCN model."""
        self.model = build_tcn(
            seq_len=self.seq_len,
            n_channels=self.n_channels,
            n_bins=self.n_bins
        )
        
        loss = create_distribution_loss()
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=loss,
            jit_compile=False,
        )
        
        n_params = sum(np.prod(w.shape) for w in self.model.trainable_weights)
        print(f"Model built: {n_params:,} trainable parameters (seq_len={self.seq_len})")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, verbose=2):
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY_STOP,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=PATIENCE_LR_REDUCE,
                             factor=LR_REDUCE_FACTOR, min_lr=1e-6, verbose=1)
        ]
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def predict(self, X):
        """Get predictions from the model."""
        logits = self.model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        return logits, probs


# ============================================================================
# Evaluation
# ============================================================================
def compute_metrics(probs, y_deltas, bin_edges):
    """Compute key metrics: MAE (Q50), interval width, calibration."""
    quantiles = quantiles_from_pmf_batch(probs, [0.1, 0.5, 0.9], bin_edges)
    q10 = quantiles[0.1]
    q50 = quantiles[0.5]
    q90 = quantiles[0.9]
    
    # P50 MAE
    mae_q50 = np.mean(np.abs(y_deltas - q50))
    
    # Prediction interval width
    interval_width = q90 - q10
    mean_width = np.mean(interval_width)
    
    # Calibration: coverage of Q10-Q90 interval (target: 80%)
    coverage_80 = np.mean((y_deltas >= q10) & (y_deltas <= q90)) * 100
    
    # Also check tail calibration
    below_q10 = np.mean(y_deltas < q10) * 100  # target: 10%
    above_q90 = np.mean(y_deltas > q90) * 100  # target: 10%
    
    return {
        'mae_q50': mae_q50,
        'mean_interval_width': mean_width,
        'coverage_80': coverage_80,
        'below_q10': below_q10,
        'above_q90': above_q90,
    }


# ============================================================================
# Main Comparison
# ============================================================================
def run_comparison():
    """Run comparison between different lookback windows."""
    print("\n" + "="*80)
    print("LOOKBACK WINDOW COMPARISON: 40 min vs 60 min")
    print("="*80)
    
    # Load data
    df = load_glucose_data()
    
    # Store results
    results = {}
    
    for lookback_minutes in LOOKBACK_WINDOWS_TO_TEST:
        seq_len = lookback_minutes // SAMPLE_INTERVAL
        print(f"\n{'='*80}")
        print(f"LOOKBACK WINDOW: {lookback_minutes} minutes ({seq_len} samples)")
        print("="*80)
        
        results[lookback_minutes] = {}
        
        for horizon in PREDICTION_HORIZONS:
            print(f"\n--- Training for {horizon}-min horizon ---")
            
            # Prepare data splits
            splits = prepare_data_splits(df, seq_len=seq_len, horizon_minutes=horizon, verbose=True)
            
            # Get bin configuration
            bin_config = get_bin_config(horizon)
            n_bins = bin_config['n_bins']
            print(f"Bins: [{bin_config['delta_min']}, {bin_config['delta_max']}] ({n_bins} bins)")
            
            # Create and train model
            trainer = TCNTrainer(seq_len=seq_len, n_channels=2, n_bins=n_bins)
            trainer.build_model()
            
            print("Training... (this may take a few minutes)")
            trainer.train(
                splits['train']['X'], splits['train']['y_bins'],
                splits['val']['X'], splits['val']['y_bins'],
                verbose=2  # Show one line per epoch
            )
            print("Training complete.")
            
            # Evaluate on test set
            X_test = splits['test']['X']
            y_deltas_test = splits['test']['y_deltas']
            
            _, probs = trainer.predict(X_test)
            metrics = compute_metrics(probs, y_deltas_test, bin_config['bin_edges'])
            
            results[lookback_minutes][horizon] = metrics
            
            print(f"\n  Results for {lookback_minutes}min lookback, {horizon}min horizon:")
            print(f"    MAE (Q50):           {metrics['mae_q50']:.2f} mg/dL")
            print(f"    Interval Width:      {metrics['mean_interval_width']:.1f} mg/dL")
            print(f"    Coverage (80% PI):   {metrics['coverage_80']:.1f}%")
            print(f"    Below Q10:           {metrics['below_q10']:.1f}% (target: 10%)")
            print(f"    Above Q90:           {metrics['above_q90']:.1f}% (target: 10%)")
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for horizon in PREDICTION_HORIZONS:
        print(f"\n{horizon}-minute prediction horizon:")
        print(f"{'Metric':<25s} {'40 min':<15s} {'60 min':<15s} {'Δ (60-40)':<15s}")
        print("-" * 70)
        
        r40 = results[40][horizon]
        r60 = results[60][horizon]
        
        # P50 MAE
        delta_mae = r60['mae_q50'] - r40['mae_q50']
        pct_mae = (delta_mae / r40['mae_q50']) * 100
        print(f"{'MAE (Q50)':<25s} {r40['mae_q50']:<15.2f} {r60['mae_q50']:<15.2f} {delta_mae:+.2f} ({pct_mae:+.1f}%)")
        
        # Interval Width
        delta_width = r60['mean_interval_width'] - r40['mean_interval_width']
        pct_width = (delta_width / r40['mean_interval_width']) * 100
        print(f"{'Interval Width (P10-P90)':<25s} {r40['mean_interval_width']:<15.1f} {r60['mean_interval_width']:<15.1f} {delta_width:+.1f} ({pct_width:+.1f}%)")
        
        # Coverage
        delta_cov = r60['coverage_80'] - r40['coverage_80']
        print(f"{'Coverage (target: 80%)':<25s} {r40['coverage_80']:<15.1f} {r60['coverage_80']:<15.1f} {delta_cov:+.1f}pp")
        
        # Tail calibration
        delta_below = r60['below_q10'] - r40['below_q10']
        print(f"{'Below Q10 (target: 10%)':<25s} {r40['below_q10']:<15.1f} {r60['below_q10']:<15.1f} {delta_below:+.1f}pp")
        
        delta_above = r60['above_q90'] - r40['above_q90']
        print(f"{'Above Q90 (target: 10%)':<25s} {r40['above_q90']:<15.1f} {r60['above_q90']:<15.1f} {delta_above:+.1f}pp")
        
        # Calibration error (deviation from target)
        cal_error_40 = abs(r40['coverage_80'] - 80) + abs(r40['below_q10'] - 10) + abs(r40['above_q90'] - 10)
        cal_error_60 = abs(r60['coverage_80'] - 80) + abs(r60['below_q10'] - 10) + abs(r60['above_q90'] - 10)
        print(f"{'Calibration Error (sum)':<25s} {cal_error_40:<15.1f} {cal_error_60:<15.1f} {cal_error_60 - cal_error_40:+.1f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Key metrics to consider:
1. MAE (Q50): Lower is better - measures point prediction accuracy
2. Interval Width: Narrower is better IF calibration is maintained
3. Coverage: Should be close to 80% - measures prediction interval accuracy
4. Below Q10 / Above Q90: Should each be close to 10% for well-calibrated intervals

If 60 min improves MAE without hurting calibration, it's a net win.
If calibration degrades, consider whether the MAE improvement is worth it.
""")
    
    return results


if __name__ == '__main__':
    run_comparison()

