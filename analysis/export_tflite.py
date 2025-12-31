"""
Export TCN Keras Models to TensorFlow Lite Format

Converts trained TCN distribution models to TFLite format for on-device
inference in the GlucoDataHandler Android app.

Usage:
    python export_tflite.py
"""
import os
import json
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# ============================================================================
# Configuration (must match tcn_distribution.py)
# ============================================================================
PREDICTION_HORIZONS = [5, 10, 15, 20, 25, 30]

# Horizon-specific binning configuration
# n_bins must match actual model output size (no overflow bins)
BIN_CONFIG = {
    5:  {'min': -60,  'max': 70,  'n_bins': 131},
    10: {'min': -80,  'max': 90,  'n_bins': 171},
    15: {'min': -100, 'max': 120, 'n_bins': 221},
    20: {'min': -120, 'max': 150, 'n_bins': 271},
    25: {'min': -140, 'max': 170, 'n_bins': 311},
    30: {'min': -150, 'max': 190, 'n_bins': 341},
}

# Model input configuration
SEQ_LEN = 8  # 40 minutes / 5 minutes per sample
N_CHANNELS = 2  # relative glucose + velocity

# Output directories
DATA_DIR = '../data'
OUTPUT_DIR = '../data/tflite_models'
ANDROID_ASSETS_DIR = '../../StudioProjects/GlucoDataHandler/common/src/main/assets/tcn_models'


def get_bin_centers(horizon):
    """Get bin centers for a specific horizon (no overflow bins, matches model output)."""
    cfg = BIN_CONFIG[horizon]
    # Regular bins only - must match model output size
    return np.arange(cfg['min'], cfg['max'] + 1).tolist()


def convert_model_to_tflite(keras_path, tflite_path, horizon):
    """
    Convert a Keras model to TFLite format.
    
    Args:
        keras_path: Path to the .keras model file
        tflite_path: Path to save the .tflite file
        horizon: Prediction horizon in minutes (for loading custom loss)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load model with custom loss function
        n_bins = BIN_CONFIG[horizon]['n_bins']
        
        # Define a dummy loss that matches the training loss signature
        def distribution_loss(y_true, logits):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(y_true, tf.int32), logits=logits))
        
        custom_objects = {'loss': distribution_loss}
        model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
        
        print(f"  Loaded model: input shape {model.input_shape}, output shape {model.output_shape}")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file size
        size_kb = os.path.getsize(tflite_path) / 1024
        print(f"  Saved: {tflite_path} ({size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def load_temperatures():
    """Load per-horizon temperature scaling factors from tcn_temps.json."""
    temps_path = os.path.join(DATA_DIR, 'tcn_temps.json')
    
    if not os.path.exists(temps_path):
        print(f"  WARNING: Temperature file not found: {temps_path}")
        print(f"  Using default temperature of 1.0 for all horizons")
        return {str(h): 1.0 for h in PREDICTION_HORIZONS}
    
    with open(temps_path, 'r') as f:
        temps_data = json.load(f)
    
    # Extract temperatures, ignoring metadata keys
    temperatures = {}
    for key, value in temps_data.items():
        if not key.startswith('_'):
            temperatures[key] = float(value)
    
    print(f"  Loaded temperatures: {temperatures}")
    return temperatures


def create_metadata(output_dirs):
    """Create metadata JSON file with bin configuration for each horizon."""
    
    # Load temperature scaling factors
    temperatures = load_temperatures()
    
    metadata = {
        'version': '1.1',
        'model_type': 'tcn_distribution',
        'created_at': datetime.now().isoformat(),
        'input_shape': [1, SEQ_LEN, N_CHANNELS],
        'seq_len': SEQ_LEN,
        'n_channels': N_CHANNELS,
        'feature_names': ['relative_glucose', 'velocity'],
        'prediction_horizons': PREDICTION_HORIZONS,
        'bin_config': {},
        'temperatures': temperatures  # Per-horizon temperature scaling factors
    }
    
    for horizon in PREDICTION_HORIZONS:
        cfg = BIN_CONFIG[horizon]
        bin_centers = get_bin_centers(horizon)
        
        metadata['bin_config'][str(horizon)] = {
            'min': cfg['min'],
            'max': cfg['max'],
            'n_bins': cfg['n_bins'],
            'bin_centers': bin_centers
        }
    
    # Add trend thresholds (Dexcom-style, rate per minute)
    metadata['trend_thresholds'] = {
        'double_down': -3.0,  # mg/dL per minute
        'down': -2.0,
        'slightly_down': -1.0,
        'flat_low': -1.0,
        'flat_high': 1.0,
        'slightly_up': 1.0,
        'up': 2.0,
        'double_up': 3.0
    }
    
    for output_dir in output_dirs:
        metadata_path = os.path.join(output_dir, 'tcn_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")


def main():
    print("=" * 70)
    print("TCN MODEL EXPORT TO TENSORFLOW LITE")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANDROID_ASSETS_DIR, exist_ok=True)
    
    output_dirs = [OUTPUT_DIR, ANDROID_ASSETS_DIR]
    
    print(f"\nInput directory: {DATA_DIR}")
    print(f"Output directories:")
    for d in output_dirs:
        print(f"  - {d}")
    
    print(f"\nConverting {len(PREDICTION_HORIZONS)} models...")
    
    success_count = 0
    
    for horizon in PREDICTION_HORIZONS:
        print(f"\n--- {horizon}-min horizon ---")
        
        keras_path = os.path.join(DATA_DIR, f'tcn_{horizon}min.keras')
        
        if not os.path.exists(keras_path):
            print(f"  WARNING: Model not found: {keras_path}")
            continue
        
        # Convert and save to all output directories
        for output_dir in output_dirs:
            tflite_path = os.path.join(output_dir, f'tcn_{horizon}min.tflite')
            if convert_model_to_tflite(keras_path, tflite_path, horizon):
                success_count += 1
    
    print("\n" + "=" * 70)
    print("CREATING METADATA")
    print("=" * 70)
    
    create_metadata(output_dirs)
    
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nModels converted: {success_count // len(output_dirs)} / {len(PREDICTION_HORIZONS)}")
    print(f"\nOutput directories:")
    for d in output_dirs:
        print(f"  - {d}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()


