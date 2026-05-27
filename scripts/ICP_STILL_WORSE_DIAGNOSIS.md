# ICP Still Worse - Diagnosis and Fixes Applied

**Date:** February 5, 2026  
**Status:** 🔧 **DEBUGGING IN PROGRESS**

---

## Problem

Even after fixing the double rotation bug, ICP is still performing worse than initial alignment.

---

## Potential Root Causes Identified

### 1. ❌ **Wrong Normal Orientation**

**Problem:** The normal estimation was forcing all normals to point towards -Z direction:

```python
# OLD (WRONG):
if normal[2] > 0:
    normal = -normal  # Force towards -Z
```

This arbitrary orientation can make point-to-plane ICP go in the **wrong direction** because:
- Point-to-plane distance is **signed**: distance = dot(point_diff, normal)
- If normals point the wrong way, ICP tries to **increase** distance instead of decrease it

**Fix Applied:**
```python
# NEW (BETTER):
# Orient normal towards point cloud center (for convex surfaces)
center = points.mean(axis=0)
to_center = center - points[i]
if np.dot(normal, to_center) < 0:
    normal = -normal
```

This orients normals **inward** (towards the center), which is correct for convex surfaces like ribcages.

---

### 2. ⚠️ **Too Aggressive Parameters**

**Problem:** The ICP parameters were too aggressive, causing overshooting:

- `max_correspondence_distance=15.0` → Too large, includes bad matches
- `huber_delta=3.0` → Too tolerant of outliers
- `trim_percentage=0.15` → Rejecting too many points (15%)
- `optimizer_maxiter=150` → Over-optimizing per iteration
- `optimizer_ftol=1e-10` → Too tight, forcing tiny steps

**Fix Applied: CONSERVATIVE Parameters**

```python
# NEW CONSERVATIVE SETTINGS:
max_correspondence_distance=10.0,   # ← Was 15.0 (smaller initial)
max_iterations=100,                  # ← Was 200 (fewer iterations)
huber_delta=2.0,                     # ← Was 3.0 (stricter outlier rejection)
convergence_threshold=1e-6,          # ← Was 1e-7 (looser, stops earlier)
k_neighbors_normals=30,              # ← Was 50 (less smoothing)
optimizer_ftol=1e-8,                 # ← Was 1e-10 (looser tolerance)
optimizer_maxiter=50,                # ← Was 150 (fewer steps)
trim_percentage=0.05,                # ← Was 0.15 (trim only 5% worst)
```

**Why this helps:**
- Smaller correspondence distance prevents bad matches early
- Looser convergence stops before overshooting
- Less trimming keeps more data for stability
- Fewer optimization steps prevents over-fitting

---

## Other Possible Issues (To Check)

### 3. **Point-to-Plane vs Point-to-Point Distance**

Point-to-plane ICP minimizes **signed distance along normal direction**, which can be problematic if:
- Normals are noisy
- Surfaces are nearly parallel (small normal component)
- Initial alignment is far off

**Diagnostic:** Check if point-to-point distance also gets worse.

---

### 4. **Local Minima**

ICP can get stuck in local minima if:
- Initial alignment is too far from optimal
- Correspondence threshold is too tight
- Optimization is too aggressive

**Solution:** The adaptive correspondence (10mm → 2mm) should help with this.

---

### 5. **Sternum Constraint Conflict**

Since sternum superior is locked at origin, ICP **cannot translate**. This means:
- If the initial rotation places the ribcage off-center relative to sternum, ICP cannot fix it
- ICP can only **rotate around the locked sternum**, which might make ribcage fit worse

**Check:** Is the initial sternum alignment good? If sternum inferior error is large, rotation-only ICP cannot help.

---

## Testing the Fixes

Run your alignment again and check:

### Expected Output (If Fixed):

```
Initial alignment quality:
  Mean ribcage error: 5.23 mm
  Sternum superior error: 0.0000 mm
  Sternum inferior error: 4.52 mm

PHASE 4: Point-to-Plane ICP Refinement
  Estimating surface normals (k=30)...
  Iter 1: fitness=0.88, RMSE=5.10mm, dist_thresh=10.00mm, angle_Δ=0.52°
  Iter 10: fitness=0.90, RMSE=4.85mm, dist_thresh=7.89mm, angle_Δ=0.08°
  Iter 30: fitness=0.91, RMSE=4.72mm, dist_thresh=4.23mm, angle_Δ=0.01°
  Converged at iteration 42
  Final: fitness=0.92, inlier_rmse=4.65mm

Final alignment quality:
  Mean ribcage error: 4.68 mm  ← Should be BETTER than 5.23mm
```

### If Still Worse:

Check these values:
1. **Sternum inferior error:** If large (>10mm), rotation-only ICP can't help
2. **Fitness:** Should be >0.85, if lower, not enough correspondences
3. **Angle changes:** Should be small (<1°), if large, ICP is making big rotations

---

## Fallback Options

### Option 1: Disable ICP (Use Initial Alignment Only)

If ICP consistently makes things worse, just use `R_optimal`:

```python
# Skip ICP, use initial alignment only
R_total = R_optimal
print("  Skipping ICP refinement (using initial alignment only)")
```

### Option 2: Use Open3D ICP (from utils.py)

Open3D's ICP is battle-tested, though it allows sternum to slide:

```python
from utils import run_point_to_plane_icp

T_icp, supine_aligned, icp_info = run_point_to_plane_icp(
    source_pts=prone_rib_rotated,
    target_pts=supine_rib_centered,
    max_correspondence_distance=10.0,
    max_iterations=50
)
```

### Option 3: Two-Stage ICP

First do point-to-point (more robust), then point-to-plane:

```python
# Stage 1: Point-to-point ICP (robust but less accurate)
R_stage1 = run_point_to_point_icp(...)

# Stage 2: Point-to-plane ICP (accurate but sensitive)
R_stage2 = run_point_to_plane_icp(R_init=R_stage1, ...)
```

---

## Changes Made Summary

1. ✅ **Fixed normal orientation** - now points towards center, not arbitrary -Z
2. ✅ **Reduced all parameters** - more conservative to prevent overshooting
3. ✅ **Fewer iterations** - stops before over-fitting
4. ✅ **Less trimming** - keeps more data for stability

---

## Next Steps

1. **Run alignment** with new conservative parameters
2. **Check console output:**
   - Is ICP RMSE decreasing?
   - Is fitness >0.85?
   - Are angle changes small?
3. **If still worse:**
   - Report the exact numbers (initial vs final RMSE)
   - Check sternum inferior error
   - Consider using initial alignment only (skip ICP)

The fixes are applied - please test and report the results!
