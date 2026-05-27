# Point-to-Plane ICP Improvements Guide

## Overview
The improved ICP implementation addresses common failure modes in point cloud alignment and provides extensive tuning parameters for better performance.

---

## Key Improvements

### 1. **Adaptive Correspondence Distance (Coarse-to-Fine Strategy)**
**Problem:** Fixed correspondence distance can cause:
- Early iterations: Missing valid correspondences when surfaces are far apart
- Late iterations: Including outliers that prevent fine-tuning

**Solution:** Start with large distance threshold, gradually reduce it
- `adaptive_correspondence=True`: Enable adaptive distance
- `correspondence_schedule='exponential'`: Fast decay initially, slow refinement later
- Alternative schedules: 'linear' (steady decay) or 'fixed' (constant)

**Parameters:**
```python
max_correspondence_distance=15.0,  # Initial threshold (mm)
# Minimum threshold hardcoded as 2.0mm for final refinement
correspondence_schedule='exponential'  # Fast→slow decay
```

---

### 2. **Trimmed ICP (Outlier Rejection)**
**Problem:** Bad correspondences (e.g., edge artifacts, noise) can dominate the objective

**Solution:** Each iteration, reject worst X% of correspondences
- Compute distances for all correspondences
- Keep only the best (1 - trim_percentage) points
- Prevents outliers from pulling alignment off-target

**Parameters:**
```python
use_trimmed_icp=True,      # Enable outlier trimming
trim_percentage=0.15,      # Reject worst 15% per iteration
```

**Tuning Advice:**
- Increase if you see alignment getting pulled toward outliers (try 0.2-0.3)
- Decrease for cleaner data (try 0.05-0.1)
- Disable if point clouds are very sparse

---

### 3. **Improved Normal Estimation**
**Problem:** Poor surface normals lead to incorrect point-to-plane distances

**Solution:** Use more neighbors for smoother, more stable normal estimation
- More neighbors = smoother normals = less sensitive to noise
- Fewer neighbors = capture fine details but more noise-sensitive

**Parameters:**
```python
k_neighbors_normals=50,  # Number of neighbors for normal estimation
```

**Tuning Advice:**
- Increase for noisy data (try 80-100)
- Decrease for high-resolution clean data (try 20-30)
- Default 50 works well for medical imaging

---

### 4. **More Aggressive Optimization**
**Problem:** Optimizer stops too early or takes imprecise steps

**Solution:** Tighter tolerance, more iterations per ICP step

**Parameters:**
```python
optimizer_ftol=1e-10,       # Function tolerance (smaller = more precise)
optimizer_maxiter=150,      # Max optimizer iterations per ICP step
convergence_threshold=1e-7, # ICP convergence (radians)
```

**Tuning Advice:**
- For faster but less accurate: increase ftol to 1e-8, reduce maxiter to 50
- For maximum precision: decrease ftol to 1e-12, increase maxiter to 200
- Monitor convergence messages to see if hitting iteration limits

---

### 5. **Better Convergence Criteria**
**Problem:** Simple angle change threshold can miss plateaus or oscillations

**Solution:** Check both angle change AND error improvement
```python
angle_change < convergence_threshold AND error_improvement < 1e-4
```

This prevents:
- Oscillating without progress
- Tiny changes that don't improve fit

---

## Recommended Parameter Sets

### **Conservative (High Accuracy, Slower)**
Best for publication-quality results
```python
run_fixed_sternum_icp(
    max_correspondence_distance=20.0,    # Very tolerant initially
    max_iterations=300,                   # Allow full convergence
    huber_delta=2.0,                     # Strict outlier rejection
    convergence_threshold=1e-8,          # Very tight
    adaptive_correspondence=True,
    correspondence_schedule='exponential',
    k_neighbors_normals=80,              # Very smooth normals
    optimizer_ftol=1e-11,
    optimizer_maxiter=200,
    use_trimmed_icp=True,
    trim_percentage=0.2,                 # Aggressive outlier rejection
    verbose=True
)
```

### **Balanced (Current Default)**
Good speed/accuracy tradeoff
```python
run_fixed_sternum_icp(
    max_correspondence_distance=15.0,
    max_iterations=200,
    huber_delta=3.0,
    convergence_threshold=1e-7,
    adaptive_correspondence=True,
    correspondence_schedule='exponential',
    k_neighbors_normals=50,
    optimizer_ftol=1e-10,
    optimizer_maxiter=150,
    use_trimmed_icp=True,
    trim_percentage=0.15,
    verbose=True
)
```

### **Fast (Quick Testing)**
For development/debugging
```python
run_fixed_sternum_icp(
    max_correspondence_distance=12.0,
    max_iterations=100,
    huber_delta=4.0,                     # More tolerant
    convergence_threshold=1e-6,          # Looser
    adaptive_correspondence=True,
    correspondence_schedule='linear',    # Simpler schedule
    k_neighbors_normals=30,              # Faster normals
    optimizer_ftol=1e-8,
    optimizer_maxiter=80,
    use_trimmed_icp=True,
    trim_percentage=0.1,
    verbose=False                        # Less output
)
```

---

## Troubleshooting Guide

### **ICP Converges Too Early (High RMSE)**
**Symptoms:** Stops after 10-20 iterations, RMSE still high (>5mm)
**Solutions:**
- Increase `max_correspondence_distance` (try 20mm)
- Decrease `convergence_threshold` (try 1e-8)
- Increase `max_iterations` (try 300)
- Check if initial rotation was poor

### **ICP Takes Too Long**
**Symptoms:** Runs for 200+ iterations, doesn't converge
**Solutions:**
- Check initial alignment (should be <10mm mean error)
- Reduce `optimizer_maxiter` (try 80)
- Use correspondence_schedule='linear' instead of 'exponential'
- Increase `convergence_threshold` (try 1e-6)

### **Alignment Gets Pulled by Outliers**
**Symptoms:** Visual inspection shows one region fits well, another is badly offset
**Solutions:**
- Increase `trim_percentage` (try 0.2-0.3)
- Decrease `huber_delta` (try 2.0 or 1.5)
- Check for systematic edge artifacts in segmentation

### **Sternum Superior Moves (Should be 0.000mm)**
**Symptoms:** Sternum superior error > 0.01mm
**Solutions:**
- This indicates a bug - the rotation-only constraint is failing
- Check that data is properly centered before ICP
- Verify rotation matrix is applied correctly
- Should NEVER happen with correct implementation

### **Noisy Convergence (Oscillates)**
**Symptoms:** RMSE bounces up and down, doesn't smooth decrease
**Solutions:**
- Increase `k_neighbors_normals` (try 80-100)
- Enable trimmed ICP if not already on
- Reduce `optimizer_ftol` for more stable steps

---

## Diagnostic Outputs

With `verbose=True`, monitor these messages:

```
Iter 1: fitness=0.8234, RMSE=6.42 mm, dist_thresh=15.00 mm, angle_Δ=2.3456°
Iter 10: fitness=0.8891, RMSE=4.12 mm, dist_thresh=10.23 mm, angle_Δ=0.1234°
Iter 50: fitness=0.9234, RMSE=3.15 mm, dist_thresh=3.45 mm, angle_Δ=0.0123°
Converged at iteration 67 (angle: 0.000234°, error improve: 8.234e-05)
Final: fitness=0.9456, inlier_rmse=2.87 mm, iterations=67
```

**What to look for:**
- **Fitness** should increase (more points matching)
- **RMSE** should decrease smoothly
- **dist_thresh** should decrease if adaptive enabled
- **angle_Δ** should decrease over time
- **Convergence** should happen before max_iterations

---

## Advanced: Parameter Sensitivity Analysis

To find optimal parameters for YOUR data:

```python
# Test different parameter combinations
param_grid = {
    'max_correspondence_distance': [12, 15, 18, 20],
    'trim_percentage': [0.1, 0.15, 0.2, 0.25],
    'k_neighbors_normals': [30, 50, 80],
}

results = []
for max_dist in param_grid['max_correspondence_distance']:
    for trim in param_grid['trim_percentage']:
        for k_neigh in param_grid['k_neighbors_normals']:
            R, aligned, info = run_fixed_sternum_icp(
                source, target,
                max_correspondence_distance=max_dist,
                trim_percentage=trim,
                k_neighbors_normals=k_neigh,
                verbose=False
            )
            results.append({
                'max_dist': max_dist,
                'trim': trim,
                'k_neigh': k_neigh,
                'rmse': info['inlier_rmse'],
                'fitness': info['fitness'],
                'iterations': info['iterations']
            })

# Find best combination
best = min(results, key=lambda x: x['rmse'])
print(f"Best parameters: {best}")
```

---

## Summary of Default Changes

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `max_correspondence_distance` | 10.0mm | 15.0mm | Better initial capture |
| `max_iterations` | 50 | 200 | Allow full convergence |
| `huber_delta` | 2.0 | 3.0 | More tolerant of normal data variation |
| `convergence_threshold` | 1e-6 | 1e-7 | Tighter convergence |
| `optimizer_ftol` | 1e-8 | 1e-10 | More precise optimization |
| `optimizer_maxiter` | 50 | 150 | More thorough per-iteration optimization |
| `adaptive_correspondence` | N/A | True | Coarse-to-fine strategy |
| `k_neighbors_normals` | 50 | 50 | Already good, kept |
| `use_trimmed_icp` | N/A | True | Reject outliers |
| `trim_percentage` | N/A | 0.15 | Reject worst 15% |

---

## Expected Performance

With improved parameters, you should see:
- **Initial RMSE:** 6-10mm (after initial rotation)
- **Final RMSE:** 2-4mm (after ICP refinement)
- **Convergence:** 50-150 iterations typically
- **Sternum Superior Error:** <0.001mm (mathematically locked)
- **Sternum Inferior Error:** 2-8mm (allowed to adjust)
- **Fitness:** >0.90 (>90% of points have valid correspondences)

If results are worse than this, review the troubleshooting guide above.
