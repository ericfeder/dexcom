"""
Export LightGBM Quantile Regression Models to ONNX Format

Trains exclusion-aware GBM models for glucose prediction and exports
them to ONNX format for on-device inference in the GlucoDataHandler app.

Uses a rolling 6-month training window from the latest available data.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo
from lightgbm import LGBMRegressor
import onnxmltools
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
PREDICTION_HORIZONS = [5, 10, 15, 20, 25, 30]
LOOKBACK_WINDOWS = [5, 10, 15, 20, 25, 30, 35, 40]
QUANTILES = [0.1, 0.5, 0.9]  # Q10, Q50 (median), Q90
TRAINING_WINDOW_MONTHS = 6  # Use latest 6 months of data
EXCLUSION_BEFORE_WINDOW = 30  # minutes before T to check for bolus/low
LOW_GLUCOSE_THRESHOLD = 80  # mg/dL threshold for low glucose exclusion

# Output directories
OUTPUT_DIR = '../data/onnx_models'
# Android assets dir - will be copied there separately
ANDROID_ASSETS_DIR = None  # Set to path if you want to copy directly

def get_exclusion_after(horizon_min):
    """
    Get the 'after' component of exclusion window for a given horizon.
    Insulin takes ~15 min to affect glucose, so we exclude boluses
    that would affect values up to horizon - 15 min, plus 10 min buffer.
    """
    return max(0, horizon_min - 15) + 10

# ============================================================================
# Data Loading
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

print("=" * 70)
print("ONNX MODEL EXPORT FOR GLUCODATAHANDLER")
print("=" * 70)

print("\nLoading glucose data...", flush=True)
df = pd.read_csv('../data/readings.csv')
df['displayTime'] = parse_to_local_time(df['displayTime'])
df = df.sort_values('displayTime').reset_index(drop=True)
print(f"Loaded {len(df):,} glucose readings", flush=True)
print(f"Date range: {df['displayTime'].min().strftime('%Y-%m-%d')} to {df['displayTime'].max().strftime('%Y-%m-%d')}")

# ============================================================================
# Load Bolus Data
# ============================================================================
print("\nLoading bolus data...", flush=True)
bolus_df = pd.read_csv('../glooko/bolus_data.csv')
bolus_df = bolus_df[bolus_df['series'] == 'deliveredBolus'].copy()
eastern = ZoneInfo('America/New_York')
bolus_df['timestamp'] = pd.to_datetime(
    bolus_df['timestamp'].str.replace('Z', ''), 
    format='%Y-%m-%dT%H:%M:%S.%f'
).apply(lambda x: x.replace(tzinfo=eastern))
bolus_df = bolus_df.sort_values('timestamp').reset_index(drop=True)
bolus_times = bolus_df['timestamp'].sort_values().reset_index(drop=True)
print(f"Loaded {len(bolus_df):,} bolus events", flush=True)

# ============================================================================
# Feature Engineering
# ============================================================================
print("\nCreating features and targets...", flush=True)

# Create future delta targets for all horizons
for mins in PREDICTION_HORIZONS:
    shift_rows = mins // 5
    df[f'actual_{mins}min'] = df['value'].shift(-shift_rows)
    df[f'delta_{mins}min'] = df[f'actual_{mins}min'] - df['value']

# Create lookback features (past deltas)
for mins in LOOKBACK_WINDOWS:
    shift_rows = mins // 5
    df[f'value_minus_{mins}min'] = df['value'].shift(shift_rows)
    df[f'past_delta_{mins}min'] = df['value'] - df[f'value_minus_{mins}min']

# Create interval-based features (timestamp-to-timestamp changes)
# e.g., change from 10min ago to 5min ago, 15min to 10min, etc.
INTERVAL_WINDOWS = list(zip(LOOKBACK_WINDOWS[1:], LOOKBACK_WINDOWS[:-1]))  # [(10,5), (15,10), ...]
for older, newer in INTERVAL_WINDOWS:
    df[f'interval_{older}_to_{newer}'] = df[f'value_minus_{newer}min'] - df[f'value_minus_{older}min']

# ============================================================================
# Feature Column Definitions
# ============================================================================
feature_cols = [f'past_delta_{m}min' for m in LOOKBACK_WINDOWS]
feature_cols_interval = [f'interval_{older}_to_{newer}' for older, newer in INTERVAL_WINDOWS]
feature_cols_combined = feature_cols + feature_cols_interval

# ============================================================================
# Exclusion Flags: Bolus + Low Glucose (Per Horizon) - KEPT FOR REFERENCE ONLY
# ============================================================================
print("\nCalculating exclusion flags...", flush=True)

def has_bolus_in_window(reading_time, bolus_times_series, before_min, after_min):
    """Check if any bolus occurred in [T - before_min, T + after_min]."""
    window_start = reading_time - pd.Timedelta(minutes=before_min)
    window_end = reading_time + pd.Timedelta(minutes=after_min)
    return ((bolus_times_series >= window_start) & (bolus_times_series <= window_end)).any()

def has_low_in_window(idx, df_full, before_min, after_min):
    """Check if any reading < LOW_GLUCOSE_THRESHOLD in [T - before_min, T + after_min]."""
    current_time = df_full.loc[idx, 'displayTime']
    window_start = current_time - pd.Timedelta(minutes=before_min)
    window_end = current_time + pd.Timedelta(minutes=after_min)
    window_mask = (df_full['displayTime'] >= window_start) & (df_full['displayTime'] <= window_end)
    return (df_full.loc[window_mask, 'value'] < LOW_GLUCOSE_THRESHOLD).any()

# Create per-horizon exclusion flags
for mins in PREDICTION_HORIZONS:
    after = get_exclusion_after(mins)
    bolus_col = f'bolus_nearby_{mins}min'
    low_col = f'low_nearby_{mins}min'
    exclude_col = f'exclude_{mins}min'
    
    print(f"  {mins:2d}-min horizon: [T-{EXCLUSION_BEFORE_WINDOW}, T+{after}]...", flush=True)
    
    df[bolus_col] = df['displayTime'].apply(
        lambda t: has_bolus_in_window(t, bolus_times, EXCLUSION_BEFORE_WINDOW, after)
    )
    df[low_col] = df.index.to_series().apply(
        lambda idx: has_low_in_window(idx, df, EXCLUSION_BEFORE_WINDOW, after)
    )
    df[exclude_col] = df[bolus_col] | df[low_col]

# ============================================================================
# Valid Rows (need all lookback features)
# ============================================================================
valid_mask = pd.Series(True, index=df.index)

for mins in LOOKBACK_WINDOWS:
    valid_mask &= df[f'value_minus_{mins}min'].notna()

valid_mask &= (df['value'] >= 40) & (df['value'] <= 400)
for mins in LOOKBACK_WINDOWS:
    valid_mask &= (df[f'value_minus_{mins}min'] >= 40) & (df[f'value_minus_{mins}min'] <= 400)

train_valid_mask = valid_mask.copy()
for mins in PREDICTION_HORIZONS:
    train_valid_mask &= df[f'actual_{mins}min'].notna()
    train_valid_mask &= (df[f'actual_{mins}min'] >= 40) & (df[f'actual_{mins}min'] <= 400)

df_for_training = df[train_valid_mask].copy().reset_index(drop=True)

# ============================================================================
# Rolling 6-Month Training Window
# ============================================================================
print("\nApplying rolling 6-month training window...", flush=True)
max_date = df_for_training['displayTime'].max()
training_start = max_date - relativedelta(months=TRAINING_WINDOW_MONTHS)

# Make timezone-aware if needed
if training_start.tzinfo is None:
    training_start = training_start.replace(tzinfo=ZoneInfo('America/New_York'))

df_for_training = df_for_training[df_for_training['displayTime'] >= training_start].copy().reset_index(drop=True)

print(f"Training window: {training_start.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
print(f"Training samples: {len(df_for_training):,}")

# ============================================================================
# Train Exclusion-Aware Models and Export to ONNX
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING AND EXPORTING MODELS")
print("=" * 70)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
if ANDROID_ASSETS_DIR:
    os.makedirs(ANDROID_ASSETS_DIR, exist_ok=True)
    
output_dirs = [OUTPUT_DIR]
if ANDROID_ASSETS_DIR:
    output_dirs.append(ANDROID_ASSETS_DIR)

# Common parameters
BASE_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

# Use combined features (cumulative + interval) - 15 features total
n_features = len(feature_cols_combined)
print(f"\nUsing {n_features} combined features:")
for i, feat in enumerate(feature_cols_combined):
    print(f"  [{i}] {feat}")

quantile_names = {0.1: 'q10', 0.5: 'q50', 0.9: 'q90'}
models_exported = []

for mins in PREDICTION_HORIZONS:
    print(f"\n--- {mins}-min horizon ---", flush=True)
    
    # Use ALL training data (Original model, not exclusion-aware)
    X_train = df_for_training[feature_cols_combined].values.astype(np.float32)
    y_train = df_for_training[f'delta_{mins}min'].values.astype(np.float32)
    
    print(f"  Training samples (all data): {len(df_for_training):,}")
    
    for q in QUANTILES:
        q_name = quantile_names[q]
        model_name = f'model_{q_name}_{mins}min'
        
        # Train model
        params = {**BASE_PARAMS, 'objective': 'quantile', 'alpha': q}
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Convert to ONNX
        initial_type = [('input', FloatTensorType([None, n_features]))]
        onnx_model = onnxmltools.convert_lightgbm(
            model, 
            initial_types=initial_type,
            target_opset=11
        )
        
        # Save to output locations
        for output_dir in output_dirs:
            onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
        
        models_exported.append({
            'name': model_name,
            'horizon': mins,
            'quantile': q,
            'quantile_name': q_name
        })
        
        print(f"  ✓ Exported {model_name}.onnx")

# ============================================================================
# Create Metadata JSON
# ============================================================================
print("\n" + "=" * 70)
print("CREATING METADATA")
print("=" * 70)

metadata = {
    'version': '2.0',
    'model_type': 'original_combined',  # Trained on all data with combined features
    'created_at': datetime.now().isoformat(),
    'training_start': training_start.strftime('%Y-%m-%d'),
    'training_end': max_date.strftime('%Y-%m-%d'),
    'training_samples': len(df_for_training),
    'training_window_months': TRAINING_WINDOW_MONTHS,
    'feature_names': feature_cols_combined,
    'num_features': n_features,
    'prediction_horizons': PREDICTION_HORIZONS,
    'quantiles': {
        'q10': 0.1,
        'q50': 0.5,
        'q90': 0.9
    },
    'interval_windows': [[older, newer] for older, newer in INTERVAL_WINDOWS],
    'model_params': BASE_PARAMS,
    'models': models_exported
}

# Save metadata to output locations
for output_dir in output_dirs:
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)
print(f"\nModels exported: {len(models_exported)}")
print(f"Output directories:")
for d in output_dirs:
    print(f"  - {d}")
print(f"\nTraining window: {TRAINING_WINDOW_MONTHS} months")
print(f"  From: {training_start.strftime('%Y-%m-%d')}")
print(f"  To:   {max_date.strftime('%Y-%m-%d')}")
print(f"\nFeatures ({n_features}):")
for i, feat in enumerate(feature_cols_combined):
    print(f"  [{i}] {feat}")
print("\n✓ Done!")

