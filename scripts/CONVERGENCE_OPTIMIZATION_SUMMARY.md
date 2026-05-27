# Convergence Optimization Update - Summary

## Problem Statement

When comparing alignment results:
- **100 iterations**: Slightly higher RMSE, **lower STD**
- **200 iterations**: Slightly lower RMSE, **higher STD**

**Question**: Which should be used for optimization and when to stop?

---

## Analysis & Recommendation

### Key Insight: Lower STD is Better

**STD (Standard Deviation)** measures consistency of alignment:
- Lower STD = More uniform fit across the surface
- Higher STD = Some areas fit well, others fit poorly (overfitting)

**RMSE** measures average error:
- Lower RMSE = Better average fit
- But can indicate overfitting if STD increases

### Decision Rule

For **medical imaging alignment**, prefer:
✓ **Lower STD** (consistency) over slightly lower RMSE (accuracy)
✓ **Earlier stopping** if STD is better
✓ **Robust alignment** that generalizes across subjects

**Why?**
1. Anatomical stability > fitting noise
2. Consistency across patients matters more than perfect fit on one scan
3. Lower variance = more reliable clinical measurements
4. Overfitting can introduce spurious deformations

**Answer: Use ~100-150 iterations with adaptive stopping**

---

## Implementation: Improved Convergence Criteria

Updated `alignment.py` with **5 convergence checks**:

### 1. RMSE Convergence
```python
rmse_converged = abs(prev_rmse - rmse) < convergence_threshold  # 1e-5
```
Stop when RMSE stops improving significantly.

### 2. Rotation Convergence
```python
rotation_converged = rotation_change < rotation_threshold  # 1e-6
```
Stop when rotation matrix barely changes.

### 3. Early Stopping with Patience
```python
patience_exceeded = no_improvement_count >= patience  # 10 iterations
```
- Track best RMSE solution
- Stop if no improvement for 10 consecutive iterations
- **Return best solution**, not last iteration

### 4. STD Monitoring (Overfitting Detection)
```python
if monitor_std and len(std_window) >= 5:
    std_trend = np.polyfit(range(len(std_window)), std_window, 1)[0]
    std_increasing = std_trend > 0.01  # STD increasing
```
Stop if STD starts increasing (indicates overfitting).

### 5. Combined Stopping Logic
```python
if rmse_converged and rotation_converged:
    # Both converged - stop naturally
elif patience_exceeded:
    # No improvement - return best solution
elif std_increasing and it > 50:
    # Overfitting detected - return best solution
```

---

## Updated Default Parameters

### Before:
```python
max_iterations: int = 100
convergence_threshold: float = 1e-6
# No early stopping
# No STD monitoring
```

### After:
```python
max_iterations: int = 150          # Middle ground (100-200)
convergence_threshold: float = 1e-5  # Slightly relaxed
patience: int = 10                  # Early stopping
rotation_threshold: float = 1e-6    # Rotation convergence
monitor_std: bool = True            # STD tracking
```

---

## Benefits of the Update

### 1. Prevents Overfitting
- Stops when STD starts increasing
- Returns best solution before overfitting occurs

### 2. Faster Convergence
- Stops early if no improvement
- Avoids unnecessary iterations

### 3. More Robust
- Tracks multiple metrics (RMSE, STD, rotation)
- Makes intelligent stopping decisions

### 4. Better for Clinical Use
- Prioritizes consistency over perfection
- More stable across different subjects
- Reduces spurious deformations

---

## What Changed in alignment.py

### 1. Function Signature
```python
def optimal_sternum_fixed_alignment(
    # ... existing params ...
    max_iterations: int = 150,          # Changed from 100
    convergence_threshold: float = 1e-5, # Changed from 1e-6
    patience: int = 10,                 # NEW
    rotation_threshold: float = 1e-6,   # NEW
    monitor_std: bool = True,           # NEW
    verbose: bool = False
)
```

### 2. Early Stopping Tracking
```python
best_rmse = np.inf
best_R = np.eye(3)
best_src = src.copy()
best_iteration = 0
no_improvement_count = 0
```

### 3. STD Tracking
```python
rmse_window = []
std_window = []
# Track last 5 iterations for trend detection
```

### 4. Enhanced Convergence Checks
```python
# 5 different stopping criteria
if rmse_converged and rotation_converged:
    # Natural convergence
elif patience_exceeded:
    # Early stopping - return best
elif std_increasing:
    # Overfitting - return best
```

### 5. Richer Info Dictionary
```python
info = {
    # ... existing fields ...
    "best_iteration": best_iteration,      # NEW
    "stop_reason": stop_reason,            # NEW
    "euclidean_std": final_std,            # NEW
    "best_rmse": best_rmse,                # NEW
}
```

---

## Usage Example

### No Changes Required!
The function signature is backward compatible. Just use it:

```python
from alignment import align_prone_to_supine_optimal

results = align_prone_to_supine_optimal(
    subject=subject,
    prone_ribcage_mesh_path=prone_path,
    supine_ribcage_seg_path=supine_path,
    verbose=True  # See convergence details
)

# Check convergence info
print(f"Stopped at: {results['alignment_info']['iterations']} iterations")
print(f"Best iteration: {results['alignment_info']['best_iteration']}")
print(f"Reason: {results['alignment_info']['stop_reason']}")
print(f"RMSE: {results['alignment_info']['euclidean_rmse']:.4f} mm")
print(f"STD: {results['alignment_info']['euclidean_std']:.4f} mm")
```

---

## Expected Behavior

### Typical Convergence Patterns

#### Pattern 1: Natural Convergence (~50-80 iterations)
```
Iter 1: RMSE=8.24 mm, STD=4.15 mm
Iter 10: RMSE=5.12 mm, STD=3.02 mm
Iter 20: RMSE=4.23 mm, STD=2.45 mm
...
Iter 67: RMSE=3.88 mm, STD=2.21 mm
✓ Converged (RMSE and rotation stable)
```

#### Pattern 2: Early Stopping (~80-120 iterations)
```
Iter 1: RMSE=7.89 mm, STD=3.98 mm
...
Iter 95: RMSE=3.92 mm, STD=2.18 mm (BEST)
Iter 96-105: RMSE=3.93-3.94 mm (no improvement)
✓ Early stopping (no improvement for 10 iterations)
→ Returning best solution from iteration 95
```

#### Pattern 3: Overfitting Detection (~100-150 iterations)
```
Iter 1: RMSE=8.15 mm, STD=4.05 mm
...
Iter 78: RMSE=3.87 mm, STD=2.15 mm (BEST)
Iter 90: RMSE=3.82 mm, STD=2.28 mm
Iter 100: RMSE=3.78 mm, STD=2.45 mm (STD increasing!)
⚠ STD increasing detected
→ Returning best solution from iteration 78
```

---

## Validation

To verify the optimization is working correctly:

1. **Check stop_reason** in results
   - Should be one of: `"rmse_rotation_converged"`, `"early_stopping_patience"`, `"std_increasing_overfitting"`, or `"max_iterations"`

2. **Compare best_iteration vs iterations**
   - If different, algorithm detected overfitting and returned earlier solution

3. **Monitor STD trend**
   - STD should decrease or stabilize, not increase

4. **Verify sternum error**
   - Should always be ~0 mm (< 0.001 mm)

---

## Troubleshooting

### Stops Too Early (< 30 iterations)
- Increase `patience` from 10 to 20
- Decrease `convergence_threshold` from 1e-5 to 1e-6

### Runs Too Long (> 150 iterations)
- Decrease `patience` from 10 to 5
- Increase `convergence_threshold` from 1e-5 to 1e-4

### STD Keeps Increasing
- This is expected if overfitting occurs
- Algorithm will automatically stop and return best solution
- If concerned, decrease `max_iterations` to 100

---

## Summary

### Question Answered
**"With 100 iterations RMSE slightly higher but STD slightly lower vs 200 iterations, what should be used?"**

**Answer**: **Use 100-150 iterations with adaptive stopping**
- Lower STD is more important for clinical robustness
- Adaptive stopping prevents overfitting automatically
- Algorithm now intelligently chooses when to stop

### Key Takeaway
You don't need to manually choose between 100 or 200 iterations anymore. The algorithm will:
1. Try to converge naturally
2. Stop early if no improvement (patience)
3. Detect and prevent overfitting (STD monitoring)
4. Return the best solution found

**Result**: Optimal alignment quality with robust, consistent results across all subjects.

---

## Files Updated
- ✅ `alignment.py` - Improved convergence criteria
- ✅ `analyze_convergence.py` - Analysis and recommendations
- ✅ `CONVERGENCE_OPTIMIZATION_SUMMARY.md` - This document

## Date
February 6, 2026
