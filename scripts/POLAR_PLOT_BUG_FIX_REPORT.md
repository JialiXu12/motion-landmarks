# Polar Plot Mean Position Bug Fix

## Date: January 20, 2026

## Problem Identified

**Issue:** Mean supine position in polar plots was incorrectly calculated.

**Root Cause:** Using **arithmetic mean** for angles instead of **circular mean**.

---

## The Bug Explained

### What Happens With Arithmetic Mean

When landmarks cluster near **12 o'clock** (0°/360° boundary):

**Example Data:**
- Supine angles: [350°, 355°, 358°, 2°, 5°, 8°, 10°]
- **Arithmetic mean:** 155.4° → plots at **~5 o'clock** ❌
- **Circular mean:** 1.1° → plots at **~12 o'clock** ✓

**Error:** **154° difference** (completely wrong quadrant!)

### Why This Happens

Angles are **circular** (wraps around at 360°/0°), but arithmetic mean treats them as **linear**:

```
Wrong: (350 + 355 + 358 + 2 + 5 + 8 + 10) / 7 = 155.4°
Right: atan2(mean(sin(θ)), mean(cos(θ))) = 1.1°
```

The arithmetic mean falls between the smallest (2°) and largest (358°) values, completely missing the actual cluster.

---

## The Fix

### Added Function: `circular_mean_angle()`

**Location:** `scripts/analysis.py` ~line 1465

```python
def circular_mean_angle(angles_rad):
    """
    Calculate the circular mean of angles (in radians).
    This correctly handles the circular nature of angles.
    
    Args:
        angles_rad: Array of angles in radians
    
    Returns:
        mean_angle_rad: Circular mean angle in radians
    """
    # Convert angles to unit vectors on the unit circle
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    
    # Calculate mean angle from the mean vector
    mean_angle = np.arctan2(sin_mean, cos_mean)
    
    return mean_angle
```

### Mathematical Principle

1. Convert each angle θ to a unit vector: `(cos(θ), sin(θ))`
2. Average the vectors: `(mean(cos(θ)), mean(sin(θ)))`
3. Convert back to angle: `atan2(mean(sin(θ)), mean(cos(θ)))`

This is the **standard circular statistics** method.

---

## Code Changes

### Change 1: Clock Rotation Analysis

**File:** `analysis.py`  
**Line:** ~1868-1871

**Before:**
```python
mean_theta_prone = np.radians(df_subset['angle_prone'].mean())
mean_theta_supine = np.radians(df_subset['angle_supine'].mean())
```

**After:**
```python
mean_theta_prone = circular_mean_angle(theta_prone)
mean_theta_supine = circular_mean_angle(theta_supine)
```

### Change 2: Comparison Plot

**File:** `analysis.py`  
**Line:** ~1947-1950

**Before:**
```python
mean_theta_prone = np.radians(df_subset['angle_prone'].mean())
mean_theta_supine = np.radians(df_subset['angle_supine'].mean())
```

**After:**
```python
mean_theta_prone = circular_mean_angle(theta_prone)
mean_theta_supine = circular_mean_angle(theta_supine)
```

---

## Impact

### Affected Plots

1. **Clock rotation analysis polar plots** (`analyse_clock_position_rotation`)
   - Individual breast plots (left/right)
   - Comparison plot
   
2. **Displacement mechanism schematic** (`plot_displacement_mechanism_schematic`)
   - Panel C: Surgical View polar plot

### What Was Wrong

**Symptoms:**
- Mean supine position appeared in wrong quadrant
- Mean trajectory arrow pointed incorrectly  
- Statistical summaries based on mean angles were inaccurate

**When It Mattered Most:**
- Landmarks clustering near **12 o'clock** (superior, medial positions)
- This is **exactly** where supine landmarks cluster after medial shift!
- Bug was **maximally impactful** for our data

---

## Validation

### Test Results

Created `test_polar_mean.py` to verify the fix:

**Test 1:** Angles near 12 o'clock
- Input: [350°, 355°, 0°, 5°, 10°]
- Arithmetic: 144° ❌
- Circular: 0° ✓

**Test 2:** Angles not near boundary
- Both methods agree (difference < 5°)
- Bug doesn't affect data away from boundaries

**Test 3:** Critical case (our data!)
- Input: [350°, 355°, 358°, 2°, 5°, 8°, 10°]
- Arithmetic: 155.4° ❌ (completely wrong!)
- Circular: 1.1° ✓ (correct!)
- **Error: 154.3°** if using arithmetic mean

---

## Why This Bug Was Serious

### Clinical Impact

1. **Misleading Visualization:**
   - Mean supine position shown in wrong quadrant
   - Could mislead surgical planning decisions

2. **Incorrect Statistics:**
   - Mean rotation angles wrong
   - Hypothesis testing potentially affected

3. **Publication Risk:**
   - Would have been caught by reviewers
   - Fundamental error in circular statistics

### Why It Wasn't Caught Earlier

1. **Subtle for most data:** Only critical near 0°/360° boundary
2. **Our data triggers it:** Landmarks cluster at 12 o'clock in supine
3. **Common mistake:** Many researchers don't know about circular mean

---

## Best Practices Applied

### Circular Statistics

✓ **Always** use circular mean for angles  
✓ Use circular standard deviation for angle spread  
✓ Use circular correlation for angle-angle relationships  
✓ Never apply linear statistics to circular data  

### When Circular Mean Matters

**Critical cases:**
- Angles near 0°, 90°, 180°, 270° (clock positions 12, 3, 6, 9)
- Wind directions
- Time of day
- Compass bearings
- Any periodic data

**Less critical:**
- Angles far from boundaries with low variance
- (But still better to use circular mean!)

---

## Files Modified

1. **`scripts/analysis.py`**
   - Added `circular_mean_angle()` function
   - Updated `analyse_clock_position_rotation()` - 2 locations
   - Updated mean calculations in polar plots

2. **`scripts/test_polar_mean.py`** (new)
   - Test suite demonstrating the bug and fix
   - Can be run anytime to verify correctness

---

## How to Verify Fix

### Run Test Suite
```bash
cd scripts
python test_polar_mean.py
```

Should output:
```
✓ Circular mean is ESSENTIAL for angles near boundaries
✓ Without it, mean supine position would be wrong by ~180°!
```

### Visual Check

1. Run analysis: `python analysis.py`
2. Open polar plots in `../output/figs/clock_analysis/`
3. Check mean supine position (red star):
   - Should be near **12 o'clock** (superior)
   - NOT at 5-6 o'clock (would be wrong!)

---

## Technical Details

### Circular Mean Formula

For angles θ₁, θ₂, ..., θₙ:

```
Circular Mean = atan2(mean(sin(θᵢ)), mean(cos(θᵢ)))
```

### Why It Works

- Each angle maps to point on unit circle
- Mean of points is **centroid** of cluster
- Angle of centroid is circular mean
- Handles wraparound automatically

### Edge Cases

1. **Angles uniformly distributed:** Mean is undefined (centroid at origin)
   - Our implementation returns atan2(0, 0) = 0
   - Acceptable for clustered data
   
2. **Angles in radians vs degrees:** Function expects radians
   - All calls properly convert using `np.radians()`
   
3. **Result range:** Returns [-π, π], convert to [0, 2π] if needed
   - Polar plots handle this automatically

---

## Summary

✅ **Bug Fixed:** Mean supine position now correctly calculated  
✅ **Method:** Circular mean instead of arithmetic mean  
✅ **Impact:** Critical for landmarks near 12 o'clock  
✅ **Validated:** Test suite confirms correction  
✅ **Publication-Ready:** Follows statistical best practices  

The polar plots now accurately represent the mean landmark positions in both prone and supine orientations.
