"""
Export E2 Multi-Horizon TCN Model to TensorFlow Lite Format

Converts the trained multi-head TCN model (E2) to a single TFLite model
with 6 named outputs (h5, h10, h15, h20, h25, h30) for on-device inference
in the GlucoDataHandler Android app.

The E2 model uses horizon-specific bins and head adapters for better calibration.

Usage:
    python export_tflite_e2.py
"""
import os
import json
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Enable eager mode for horizon-specific bins
tf.config.run_functions_eagerly(True)

# Import from tcn_distribution
import tcn_distribution
tcn_distribution.USE_SHARED_BINS = False  # E2 uses horizon-specific bins

from tcn_distribution import (
    build_tcn_multihead, MultiHorizonTCN,
    get_bin_config, PREDICTION_HORIZONS, HEAD_NAMES
)

# ============================================================================
# Configuration
# ============================================================================
PREDICTION_HORIZONS = [5, 10, 15, 20, 25, 30]

# Horizon-specific binning configuration (must match tcn_distribution.py)
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

# Paths
DATA_DIR = '../data'
MODELS_DIR = '../models'
E2_MODEL_PATH = os.path.join(DATA_DIR, 'tcn_multihead.keras')
E2_TEMPS_PATH = os.path.join(MODELS_DIR, 'temps_e2_calibrated.json')
E2_TEMPS_FALLBACK_PATH = os.path.join(DATA_DIR, 'tcn_multihead_temps_coverage.json')

OUTPUT_DIR = '../data/tflite_models'
ANDROID_ASSETS_DIR = '../../StudioProjects/GlucoDataHandler/common/src/main/assets/tcn_models'


def get_bin_centers(horizon):
    """Get bin centers for a specific horizon."""
    cfg = BIN_CONFIG[horizon]
    return list(range(cfg['min'], cfg['max'] + 1))


def load_e2_temperatures():
    """Load per-horizon temperature scaling factors for E2 model."""
    # Try primary path first
    if os.path.exists(E2_TEMPS_PATH):
        print(f"  Loading temperatures from {E2_TEMPS_PATH}")
        with open(E2_TEMPS_PATH, 'r') as f:
            temps_data = json.load(f)
    elif os.path.exists(E2_TEMPS_FALLBACK_PATH):
        print(f"  Loading temperatures from {E2_TEMPS_FALLBACK_PATH}")
        with open(E2_TEMPS_FALLBACK_PATH, 'r') as f:
            temps_data = json.load(f)
    else:
        print(f"  WARNING: No temperature file found, using T=1.0 for all horizons")
        return {str(h): 1.0 for h in PREDICTION_HORIZONS}
    
    # Extract temperatures (ignore metadata keys starting with _)
    temperatures = {}
    for key, value in temps_data.items():
        if not key.startswith('_'):
            temperatures[key] = float(value)
    
    print(f"  Loaded temperatures: {temperatures}")
    return temperatures


def build_e2_model():
    """Build and load the E2 multi-horizon model."""
    print("Building E2 model architecture...")
    
    # Build backbone with horizon-specific bins
    backbone = build_tcn_multihead(
        seq_len=SEQ_LEN,
        n_channels=N_CHANNELS,
        use_head_adapter=True,
        head_adapter_dim=64
    )
    
    # Wrap in MultiHorizonTCN (for weight loading)
    model = MultiHorizonTCN(
        backbone,
        lambda_curve=0,
        lambda_varmono=0.001,
        point_loss_enabled=True,
        point_loss_weight=0.1
    )
    
    # Build model by calling it once
    dummy_input = np.zeros((1, SEQ_LEN, N_CHANNELS), dtype=np.float32)
    _ = model(dummy_input)
    
    # Load weights
    print(f"Loading weights from {E2_MODEL_PATH}...")
    model.load_weights(E2_MODEL_PATH)
    print("  Weights loaded successfully")
    
    # Return the backbone for TFLite conversion (it's a functional model)
    return model.backbone


def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite format."""
    print(f"Converting to TFLite...")
    
    try:
        # Convert using TFLiteConverter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Saved: {output_path} ({size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"  ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_tflite_model(tflite_path):
    """Verify TFLite model loads and runs correctly."""
    print(f"\nVerifying TFLite model...")
    
    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Number of outputs: {len(output_details)}")
        
        # Map output names
        output_names = []
        for detail in output_details:
            name = detail['name']
            shape = detail['shape']
            output_names.append(name)
            print(f"    {name}: shape {shape}")
        
        # Run inference with dummy input
        dummy_input = np.zeros((1, SEQ_LEN, N_CHANNELS), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        # Check outputs
        for i, detail in enumerate(output_details):
            output = interpreter.get_tensor(detail['index'])
            print(f"    Output {i} ({detail['name']}): shape {output.shape}, sum={output.sum():.4f}")
        
        print("  Verification passed!")
        return True
        
    except Exception as e:
        print(f"  ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_metadata(output_dirs, temperatures):
    """Create metadata JSON file for E2 model."""
    print("\nCreating E2 metadata...")
    
    metadata = {
        'version': '2.0-E2',
        'model_type': 'tcn_multihead',
        'description': 'E2 Multi-horizon TCN with horizon-specific bins and head adapters',
        'created_at': datetime.now().isoformat(),
        'input_shape': [1, SEQ_LEN, N_CHANNELS],
        'seq_len': SEQ_LEN,
        'n_channels': N_CHANNELS,
        'feature_names': ['relative_glucose', 'velocity'],
        'prediction_horizons': PREDICTION_HORIZONS,
        'output_names': [f'h{h}' for h in PREDICTION_HORIZONS],
        'bin_config': {},
        'temperatures': temperatures
    }
    
    # Add per-horizon bin configuration
    for horizon in PREDICTION_HORIZONS:
        cfg = BIN_CONFIG[horizon]
        bin_centers = get_bin_centers(horizon)
        
        metadata['bin_config'][str(horizon)] = {
            'min': cfg['min'],
            'max': cfg['max'],
            'n_bins': cfg['n_bins'],
            'bin_centers': bin_centers
        }
    
    # Add trend thresholds (same as baseline)
    metadata['trend_thresholds'] = {
        'double_down': -3.0,
        'down': -2.0,
        'slightly_down': -1.0,
        'flat_low': -1.0,
        'flat_high': 1.0,
        'slightly_up': 1.0,
        'up': 2.0,
        'double_up': 3.0
    }
    
    # Save to all output directories
    for output_dir in output_dirs:
        metadata_path = os.path.join(output_dir, 'tcn_multihead_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")


def main():
    print("=" * 70)
    print("E2 MULTI-HORIZON TCN EXPORT TO TENSORFLOW LITE")
    print("=" * 70)
    
    # Verify source model exists
    if not os.path.exists(E2_MODEL_PATH):
        print(f"\nERROR: E2 model not found: {E2_MODEL_PATH}")
        print("Please train the model first with: python tcn_distribution.py --multihead")
        return
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANDROID_ASSETS_DIR, exist_ok=True)
    
    output_dirs = [OUTPUT_DIR, ANDROID_ASSETS_DIR]
    
    print(f"\nSource model: {E2_MODEL_PATH}")
    print(f"Output directories:")
    for d in output_dirs:
        print(f"  - {d}")
    
    # Load temperatures
    print("\n" + "-" * 70)
    print("Loading temperature scaling factors...")
    temperatures = load_e2_temperatures()
    
    # Build and load model
    print("\n" + "-" * 70)
    backbone = build_e2_model()
    
    # Print model summary
    print(f"\nBackbone model: {backbone.name}")
    print(f"  Input shape: {backbone.input_shape}")
    print(f"  Outputs: {list(backbone.output.keys()) if isinstance(backbone.output, dict) else 'single'}")
    
    # Convert to TFLite for each output directory
    print("\n" + "-" * 70)
    print("Converting to TFLite...")
    
    success = True
    for output_dir in output_dirs:
        tflite_path = os.path.join(output_dir, 'tcn_multihead.tflite')
        if not convert_to_tflite(backbone, tflite_path):
            success = False
        else:
            # Verify the model
            if not verify_tflite_model(tflite_path):
                success = False
    
    if not success:
        print("\nERROR: Some conversions failed!")
        return
    
    # Create metadata
    print("\n" + "-" * 70)
    create_metadata(output_dirs, temperatures)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nExported files:")
    for output_dir in output_dirs:
        print(f"  {output_dir}/")
        print(f"    - tcn_multihead.tflite")
        print(f"    - tcn_multihead_metadata.json")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

