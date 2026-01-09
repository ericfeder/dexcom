"""
Export TCN predictions for webapp visualization.

Compares:
- Baseline: 6 separate single-horizon TCN models
- E2: Multi-horizon TCN with horizon-specific bins

Applies per-horizon temperature scaling to BOTH models for proper calibration.

Only exports predictions for the test set timeframe.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Required for E2 with horizon-specific bins

import tcn_distribution
tcn_distribution.USE_SHARED_BINS = False  # E2 uses horizon-specific bins

from tcn_distribution import (
    load_glucose_data, prepare_data_splits_multihorizon,
    build_tcn_multihead, MultiHorizonTCN,
    get_bin_config, PREDICTION_HORIZONS, HEAD_NAMES, DATA_MONTHS
)
from tensorflow.keras.models import load_model

# Temperature scaling paths
E2_TEMPS_PATH = '../models/temps_e2_calibrated.json'
E2_TEMPS_FALLBACK_PATH = '../data/tcn_multihead_temps_coverage.json'
BASELINE_TEMPS_PATH = '../data/tcn_temps.json'


def load_temperatures(paths, model_name="model"):
    """
    Load per-horizon temperatures from JSON file(s).
    
    Args:
        paths: List of paths to try (uses first existing file)
        model_name: Name for logging
    
    Returns:
        Dict[int, float] mapping horizon -> temperature, or None if not found
    """
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Extract temperatures (skip metadata)
            temps = {}
            for k, v in data.items():
                if not k.startswith('_'):
                    temps[int(k)] = float(v)
            
            print(f"  Loaded {model_name} temperatures from {path}")
            for h in PREDICTION_HORIZONS:
                T = temps.get(h, 1.0)
                print(f"    +{h}min: T = {T:.4f}")
            
            return temps
    
    print(f"  Warning: No {model_name} temperature file found, using T=1.0 for all horizons")
    return None


def load_e2_temperatures():
    """Load per-horizon temperatures for E2 model."""
    return load_temperatures([E2_TEMPS_PATH, E2_TEMPS_FALLBACK_PATH], "E2")


def load_baseline_temperatures():
    """Load per-horizon temperatures for baseline single-horizon models."""
    return load_temperatures([BASELINE_TEMPS_PATH], "baseline")


def export_predictions():
    """Export predictions from baseline and E2 models for webapp comparison."""
    
    print("="*70)
    print("EXPORTING TCN PREDICTIONS FOR WEBAPP")
    print("Baseline (6 single-horizon) vs E2 (multi-horizon, horizon-specific bins)")
    print("="*70, flush=True)
    
    # Load data
    df = load_glucose_data()
    splits = prepare_data_splits_multihorizon(df, verbose=True)
    
    X_test = splits['test']['X']
    y_test_deltas = splits['test']['y_deltas']
    test_indices = splits['test']['indices']
    n_test = len(X_test)
    
    # Get the filtered dataframe
    latest_date = df['displayTime'].max()
    cutoff_date = latest_date - pd.DateOffset(months=DATA_MONTHS)
    df_filtered = df[df['displayTime'] >= cutoff_date].copy().reset_index(drop=True)
    
    print(f"\nTest set: {n_test:,} samples")
    print(f"Test range: {df_filtered.iloc[test_indices[0]]['displayTime']} to {df_filtered.iloc[test_indices[-1]]['displayTime']}", flush=True)
    
    # =========================================================================
    # BASELINE: 6 separate single-horizon models
    # =========================================================================
    print("\n" + "-"*70)
    print("BASELINE (single-horizon) predictions...", flush=True)
    
    # Load baseline temperatures
    print("\n  Loading temperature scaling factors...", flush=True)
    baseline_temps = load_baseline_temperatures()
    
    baseline_preds = {}
    
    for horizon in PREDICTION_HORIZONS:
        model_path = f'../data/tcn_{horizon}min.keras'
        if not os.path.exists(model_path):
            print(f"  {horizon}min: NOT FOUND - skipping", flush=True)
            continue
        
        print(f"  {horizon}min: loading...", end='', flush=True)
        model = load_model(model_path, compile=False)
        print(" predicting...", end='', flush=True)
        
        # Manual batch loop to avoid tf.data hanging
        all_logits = []
        for start in range(0, len(X_test), 256):
            end = min(start + 256, len(X_test))
            batch_logits = model(X_test[start:end], training=False)
            all_logits.append(batch_logits.numpy())
        logits = np.concatenate(all_logits, axis=0)
        
        # Apply temperature scaling if available
        T = baseline_temps.get(horizon, 1.0) if baseline_temps else 1.0
        if T != 1.0:
            scaled_logits = logits / T
        else:
            scaled_logits = logits
        
        probs = tf.nn.softmax(scaled_logits).numpy()
        
        config = get_bin_config(horizon)
        bin_centers = np.linspace(config['delta_min'], config['delta_max'], config['n_bins'])
        
        # Calculate quantiles
        cdf = np.cumsum(probs, axis=1)
        q10_idx = np.argmax(cdf >= 0.10, axis=1)
        q50_idx = np.argmax(cdf >= 0.50, axis=1)
        q90_idx = np.argmax(cdf >= 0.90, axis=1)
        
        baseline_preds[horizon] = {
            'q10': bin_centers[q10_idx],
            'q50': bin_centers[q50_idx],
            'q90': bin_centers[q90_idx]
        }
        print(f"  {horizon}min: done (T={T:.4f})", flush=True)
    
    # =========================================================================
    # E2: Multi-horizon with horizon-specific bins
    # =========================================================================
    print("\n" + "-"*70)
    print("E2 (multi-horizon, horizon-specific bins) predictions...", flush=True)
    
    # Build model architecture
    backbone = build_tcn_multihead(
        seq_len=8, n_channels=2,
        use_head_adapter=True, head_adapter_dim=64
    )
    e2_model = MultiHorizonTCN(
        backbone,
        lambda_curve=0,
        lambda_varmono=0.001,
        point_loss_enabled=True,
        point_loss_weight=0.1
    )
    
    # Build by calling once
    dummy = np.zeros((1, 8, 2), dtype=np.float32)
    _ = e2_model(dummy)
    
    # Load weights
    e2_model.load_weights('../data/tcn_multihead.keras')
    print("  Weights loaded", flush=True)
    
    # Load per-horizon temperatures for calibration
    print("\n  Loading E2 temperature scaling factors...", flush=True)
    temps = load_e2_temperatures()
    
    # Get predictions using manual loop (required for horizon-specific bins)
    e2_logits = e2_model.predict_logits(X_test)
    
    e2_preds = {}
    for horizon in PREDICTION_HORIZONS:
        h_name = f'h{horizon}'
        logits = e2_logits[h_name]
        
        # Apply temperature scaling if available
        T = temps.get(horizon, 1.0) if temps else 1.0
        if T != 1.0:
            scaled_logits = logits / T
        else:
            scaled_logits = logits
        
        probs = tf.nn.softmax(scaled_logits).numpy()
        
        config = get_bin_config(horizon)
        bin_centers = np.linspace(config['delta_min'], config['delta_max'], config['n_bins'])
        
        # Calculate quantiles
        cdf = np.cumsum(probs, axis=1)
        q10_idx = np.argmax(cdf >= 0.10, axis=1)
        q50_idx = np.argmax(cdf >= 0.50, axis=1)
        q90_idx = np.argmax(cdf >= 0.90, axis=1)
        
        e2_preds[horizon] = {
            'q10': bin_centers[q10_idx],
            'q50': bin_centers[q50_idx],
            'q90': bin_centers[q90_idx]
        }
        print(f"  {horizon}min: done (T={T:.4f})", flush=True)
    
    # =========================================================================
    # BUILD OUTPUT JSON
    # =========================================================================
    print("\n" + "-"*70)
    print("Building JSON...", flush=True)
    
    readings = []
    for i, idx in enumerate(test_indices):
        row = df_filtered.iloc[idx]
        val = int(row['value'])
        
        reading = {
            'displayTime': row['displayTime'].isoformat(),
            'value': val,
            'trend': row.get('trend', 'flat'),
            'trendRate': float(row['trendRate']) if pd.notna(row.get('trendRate')) else None
        }
        
        # Baseline predictions (absolute glucose values)
        for h in PREDICTION_HORIZONS:
            if h in baseline_preds:
                reading[f'tcn_single_q10_{h}'] = round(float(val + baseline_preds[h]['q10'][i]), 1)
                reading[f'tcn_single_q50_{h}'] = round(float(val + baseline_preds[h]['q50'][i]), 1)
                reading[f'tcn_single_q90_{h}'] = round(float(val + baseline_preds[h]['q90'][i]), 1)
        
        # E2 predictions (absolute glucose values)
        for h in PREDICTION_HORIZONS:
            if h in e2_preds:
                reading[f'tcn_multi_q10_{h}'] = round(float(val + e2_preds[h]['q10'][i]), 1)
                reading[f'tcn_multi_q50_{h}'] = round(float(val + e2_preds[h]['q50'][i]), 1)
                reading[f'tcn_multi_q90_{h}'] = round(float(val + e2_preds[h]['q90'][i]), 1)
        
        readings.append(reading)
        
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1:,}/{n_test:,} readings", flush=True)
    
    output = {
        'metadata': {
            'description': 'TCN model comparison: Baseline (6 single-horizon) vs E2 (multi-horizon). Both use per-horizon temperature scaling.',
            'test_start': df_filtered.iloc[test_indices[0]]['displayTime'].isoformat(),
            'test_end': df_filtered.iloc[test_indices[-1]]['displayTime'].isoformat(),
            'n_samples': len(readings),
            'models': {
                'tcn_single': {
                    'type': '6 separate single-horizon models',
                    'description': 'Baseline - each horizon trained independently + temperature scaling',
                    'config': {
                        'temperatures': {str(h): baseline_temps.get(h, 1.0) for h in PREDICTION_HORIZONS} if baseline_temps else None
                    }
                },
                'tcn_multi': {
                    'type': 'E2 multi-horizon',
                    'description': 'Shared backbone + head adapters + horizon-specific bins + temperature scaling',
                    'config': {
                        'USE_SHARED_BINS': False,
                        'USE_HEAD_ADAPTER': True,
                        'HEAD_ADAPTER_DIM': 64,
                        'LAMBDA_CURVE': 0,
                        'POINT_LOSS_WEIGHT': 0.1,
                        'temperatures': {str(h): temps.get(h, 1.0) for h in PREDICTION_HORIZONS} if temps else None
                    }
                }
            }
        },
        'readings': readings
    }
    
    # Save
    output_path = '../data/quantile_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(output, f)
    
    print(f"\n{'='*70}")
    print(f"✓ Saved {len(readings):,} readings to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB", flush=True)


if __name__ == '__main__':
    export_predictions()
