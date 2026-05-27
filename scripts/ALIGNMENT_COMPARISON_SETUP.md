# Alignment Comparison Test - Setup Guide

## Overview

Two alignment comparison scripts have been created:

1. **alignment_optimisation.py** - Updated to implement fixed sternum point-to-point alignment
2. **test_alignment_comparison.py** - Test script to compare two alignment methods

## Alignment Methods Being Compared

### Method 1: Fixed Sternum Point-to-Point (alignment_optimisation.py)
- Uses scipy least_squares optimization
- Rotation only around sternum origin (zero sternum drift by design)
- Simple point-to-point correspondence via KD-Tree

### Method 2: Optimal Sternum-Fixed SVD (alignment.py)
- Uses SVD-based closed-form rotation (Kabsch algorithm)
- Iterative with outlier rejection (trimmed ICP)
- Globally optimal rotation solution

## Dependency Issue

The test script requires the `morphic` package which is not currently installed in your environment.

### Options to Fix:

**Option 1: Install morphic**
```bash
# If morphic is available via pip or from a local source
pip install morphic
# OR
pip install git+https://github.com/organization/morphic.git
```

**Option 2: Mock the morphic imports** (for testing only)
Create a mock morphic module in your path that provides minimal functionality needed for loading mesh files.

**Option 3: Use a different approach**
Instead of loading the morphic mesh files directly, you could:
- Pre-extract point clouds from mesh files and save as numpy arrays
- Load the numpy arrays in the test script

## Current Status

### ✓ Completed:
- Updated `alignment_optimisation.py` with fixed sternum implementation
- Created comprehensive test script `test_alignment_comparison.py`
- Added proper error handling for missing data
- Set up comparison metrics (RMSE, sternum drift, iterations, etc.)

### ⚠️ Blocked by:
- Missing `morphic` dependency prevents script execution

## How to Run (once morphic is installed)

```bash
cd motion-landmarks
python scripts/test_alignment_comparison.py
```

The script will:
1. Load subjects that had successful alignment (VL00014, VL00018, VL00019, VL00020, VL00022, VL00025, VL00030, VL00031)
2. Run both alignment methods on each subject
3. Compare RMSE, sternum drift, and computational efficiency
4. Save results to `output/alignment_method_comparison.csv`
5. Print summary statistics

## Expected Output

```
================================================================================
ALIGNMENT METHOD COMPARISON
================================================================================
Processing 8 subjects...

Method 1: Fixed Sternum Point-to-Point (scipy least_squares)
Method 2: Optimal Sternum-Fixed SVD (iterative)

--- Processing VL00014 ---
  Running Method 1: Fixed Sternum Point-to-Point...
    RMSE: X.XXX mm
    Sternum Drift: 0.000000 mm
  Running Method 2: Optimal Sternum-Fixed SVD...
    RMSE: X.XXX mm
    Sternum Drift: 0.000000 mm
    Iterations: XX

...

================================================================================
SUMMARY STATISTICS
================================================================================

--- Method 1: Fixed Sternum Point-to-Point ---
  Success Rate: 8/8 (100.0%)
  RMSE: X.XXX ± X.XXX mm
  Sternum Drift: 0.000000 ± 0.000000 mm
  Function Evaluations: XXX ± XX

--- Method 2: Optimal Sternum-Fixed SVD ---
  Success Rate: 8/8 (100.0%)
  RMSE: X.XXX ± X.XXX mm
  Sternum Drift: 0.000000 ± 0.000000 mm
  Iterations: XX ± X

--- Direct Comparison (Both Methods Succeeded) ---
  N = 8
  RMSE Difference (Method1 - Method2): X.XXX ± X.XXX mm
  Method 1 Better: X/8
  Method 2 Better: X/8
```

## Key Implementation Details

### alignment_optimisation.py changes:
- Centers prone and supine ribcages on their respective sternum superior landmarks
- Optimizes only rotation angles (3 DOF) using scipy.optimize.least_squares
- Translation is fixed as the offset between prone and supine sternum positions
- Sternum drift is mathematically zero (R @ [0,0,0] = [0,0,0])

### Test script features:
- Loads ribcage point clouds from mesh and segmentation files
- Extracts sternum superior landmarks from anatomical data
- Runs both methods with same input data
- Computes metrics: RMSE, mean distance, sternum drift, computational cost
- Statistical comparison with success rates and paired analysis

## Next Steps

1. **Install morphic dependency**
2. **Run test script** on the 8 subjects with successful alignment
3. **Analyze results** to determine which method is more accurate
4. **Update main.py** to use the better method
5. **Document findings** for the paper

