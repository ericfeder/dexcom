# Troubleshooting: Model Training Stuck on Epoch 1

## Problem Description

When running multi-horizon TCN training (e.g., `python tcn_distribution.py --multihead`), the training process appears to hang or get stuck at "Epoch 1/100" with no progress shown.

## Root Causes

There are several potential causes for this issue:

### 1. TensorFlow Eager Execution Conflict

The custom `train_step()` in `MultiHorizonTCN` can conflict with TensorFlow's execution mode.

**Fix:** Ensure eager execution is disabled at the start of training:

```python
import tensorflow as tf
tf.config.run_functions_eagerly(False)  # Set BEFORE model creation
```

### 2. Output Buffering (False Positive)

When using `verbose=2` with `model.fit()`, Keras only outputs at the END of each epoch. Combined with Python's output buffering, it can appear the training is stuck when it's actually running.

**Fix:** Use `verbose=1` to see batch-by-batch progress:

```python
model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1  # Shows progress bar for each batch
)
```

### 3. Python Output Buffering

When running with pipes (`| tee logfile.log`) or redirecting output, Python buffers stdout by default.

**Fix:** Force unbuffered output:

```python
import sys
sys.stdout.reconfigure(line_buffering=True)
```

Or run Python with `-u` flag:

```bash
python -u tcn_distribution.py --multihead
```

### 4. Graph Compilation Overhead

The first epoch may take significantly longer due to TensorFlow graph compilation (tracing the `train_step` function).

**Signs:** High CPU usage initially, then progress starts after 1-5 minutes.

**Fix:** This is normal - just wait. To verify it's working:

```bash
# Check if Python is using CPU
ps aux | grep python | grep -v grep
# Should show CPU% > 0 if training is active
```

## Recommended Training Setup

When running training, use this pattern to avoid issues:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging noise

import sys
sys.stdout.reconfigure(line_buffering=True)  # Force unbuffered output

import tensorflow as tf
tf.config.run_functions_eagerly(False)  # Disable eager execution for speed

# ... rest of training code ...

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1  # Use verbose=1 to see batch progress
)
```

## Quick Diagnostic Steps

If training appears stuck:

1. **Check CPU usage:**
   ```bash
   ps aux | grep python | grep -v grep
   ```
   - If CPU% is high (>50%), training is likely running - just slow or buffered
   - If CPU% is 0%, something is genuinely stuck

2. **Wait at least 2-3 minutes** - the first epoch includes graph compilation overhead

3. **Check for actual output:**
   ```bash
   tail -f your_logfile.log
   ```

4. **Test with minimal epochs:**
   ```python
   # Quick test with 2-3 epochs to verify training works
   import tcn_distribution
   tcn_distribution.MAX_EPOCHS = 3
   tcn_distribution.train_multihead()
   ```

5. **Run inline test:**
   ```bash
   cd /Users/eric.feder/dexcom/analysis
   python -c "
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   import tensorflow as tf
   tf.config.run_functions_eagerly(False)
   
   from tcn_distribution import train_multihead
   train_multihead()
   " 2>&1
   ```

## Configuration Changes Made

In `tcn_distribution.py`, the following changes address this issue:

```python
# Line ~49 - Disable eager execution
tf.config.run_functions_eagerly(False)

# In train_multihead() function - Use verbose=1
history = model.fit(
    ...
    verbose=1  # Changed from verbose=2
)
```

## Summary

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Stuck at "Epoch 1/100" forever | Eager execution conflict | `tf.config.run_functions_eagerly(False)` |
| No output, but CPU is active | Output buffering | Use `verbose=1` and `python -u` |
| First epoch very slow | Graph compilation | Wait 2-3 minutes, this is normal |
| CPU at 0%, truly stuck | Unknown hang | Kill and restart with inline test |

---

*Last updated: January 2026*

