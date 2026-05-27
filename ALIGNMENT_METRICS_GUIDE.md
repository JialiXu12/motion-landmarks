# Alignment Metrics, Linearization Approximation, and Convergence Behavior

## 1. What Metrics Should We Report for Alignment?

For scientific publication, alignment accuracy should be reported using **multiple complementary metrics**:

### Primary Metrics (MUST REPORT)

| Metric | Definition | Why Report |
|--------|------------|------------|
| **RMSE (Root Mean Square Error)** | `sqrt(mean(errors²))` | Standard metric, sensitive to large errors, widely understood |
| **Mean ± SD** | Average error with standard deviation | Shows central tendency and variability |
| **Sternum Error** | Distance of sternum from fixed position | Validates constraint enforcement (should be ~0 mm) |

### Secondary Metrics (RECOMMENDED)

| Metric | Definition | Why Report |
|--------|------------|------------|
| **Median [IQR]** | 50th percentile with 25th-75th range | Robust to outliers, better for skewed distributions |
| **Range (Min-Max)** | Extreme values | Shows worst-case performance |
| **Inlier Fraction** | % of points within max_correspondence_distance | Measures how much of the anatomy aligned well |

### Example Reporting Text for Methods Section:

> "Prone-to-supine alignment was performed using a sternum-fixed iterative closest point algorithm. Across N=64 subjects, ribcage surface alignment achieved an **RMSE of 5.2 ± 1.8 mm** (median: 4.8 mm, range: 2.1-12.3 mm), with sternum superior error of **0.0 mm** (mathematically fixed), indicating excellent registration quality suitable for landmark displacement analysis."

### Per-Subject vs. Cohort Reporting:

```
Per-Subject:
- RMSE for each individual
- Inlier fraction (varies by anatomy)

Cohort:
- Mean RMSE ± SD across subjects
- Median RMSE [IQR]
- Range of RMSE values
```

---

## 2. What Does Linearization Approximation Mean?

### The Problem: Rotation is Non-Linear

Finding the optimal rotation R that aligns source points to target points involves minimizing:

```
E = Σᵢ ||R · pᵢ - qᵢ||²
```

However, **rotation matrices are non-linear** - they live on the SO(3) manifold, not in Euclidean space.

### The Linearization Trick

For **small rotations**, we can approximate R using first-order Taylor expansion:

```
R ≈ I + [skew(θ)]

where [skew(θ)] = [ 0   -θz   θy ]
                  [ θz   0   -θx ]
                  [-θy   θx   0  ]
```

For small angles θ = (θx, θy, θz):
- `cos(θ) ≈ 1`
- `sin(θ) ≈ θ`

### In Surface-to-Point Alignment

The **plane-to-point error** at each correspondence is:

```
eᵢ = nᵢ · (qᵢ - R · pᵢ)
```

where:
- `nᵢ` = surface normal at source point
- `pᵢ` = source surface point
- `qᵢ` = target point
- `R` = rotation matrix

**Linearizing** this for small rotation (θx, θy, θz):

```
nᵢ · (qᵢ - pᵢ - θ × pᵢ) ≈ 0

Rearranging:
nᵢ · (qᵢ - pᵢ) = nᵢ · (θ × pᵢ) = θ · (pᵢ × nᵢ)
```

This gives a **linear system** `A · θ = b` that can be solved by least squares!

### Why Linearization Works in ICP

1. **ICP is iterative** - each iteration only needs a small correction
2. After solving the linearized system, we project back to a proper rotation via SVD
3. The next iteration refines further

### Comparison: SVD (Point-to-Point) vs. Linearized (Plane-to-Point)

| Aspect | SVD (Kabsch) | Linearized |
|--------|--------------|------------|
| **Error Type** | Point-to-point: ||R·p - q||² | Plane-to-point: (n·(q-R·p))² |
| **Solution** | Closed-form (exact for given correspondences) | Approximate (small-angle assumption) |
| **Normal Use** | Not used | Source surface normals |
| **Pros** | Globally optimal rotation | Allows sliding along surface (more flexible) |
| **Cons** | Doesn't use surface geometry | Linearization error for large rotations |

---

## 3. What Convergence Behavior Should Surface-to-Point Alignment Use?

### Recommended Convergence Criteria

Surface-to-point alignment should use **multiple stopping criteria**:

#### 3.1 Primary: RMSE Change Threshold
```python
if abs(prev_rmse - rmse) < convergence_threshold:
    converged = True
```
- **Recommended value**: `1e-5` to `1e-6` mm
- **Meaning**: Stop when error stops improving

#### 3.2 Secondary: Maximum Iterations
```python
max_iterations = 100 to 200
```
- **Purpose**: Prevent infinite loops
- **Typical behavior**: Should converge in 20-50 iterations

#### 3.3 Tertiary: Rotation Change Threshold
```python
rotation_change = np.linalg.norm(R_delta - np.eye(3))
if rotation_change < rotation_threshold:
    converged = True
```
- **Recommended value**: `1e-6`
- **Meaning**: Rotation corrections become negligible

### Expected Convergence Behavior

```
Iteration 1:  RMSE = 15.2 mm, rotation_change = 0.082
Iteration 2:  RMSE = 8.4 mm,  rotation_change = 0.043
Iteration 5:  RMSE = 5.1 mm,  rotation_change = 0.012
Iteration 10: RMSE = 4.3 mm,  rotation_change = 0.003
Iteration 20: RMSE = 4.1 mm,  rotation_change = 0.0008
Iteration 30: RMSE = 4.08 mm, rotation_change = 0.0001
→ Converged (RMSE change < 0.01 mm)
```

### Why Surface-to-Point May Give Higher Error Than Point-to-Point

Several factors explain this:

1. **Different error metrics**: 
   - Point-to-point: Euclidean distance `||p - q||`
   - Plane-to-point: Projected distance `n · (q - p)`
   - These are mathematically different!

2. **Linearization error accumulates**: The small-angle approximation introduces error

3. **Normal estimation quality**: Poor normals → poor alignment

4. **Sliding behavior**: Plane-to-point allows sliding along the surface, which may not align well with a different geometry

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| **Meshes with accurate normals** | Surface-to-point |
| **Point clouds only** | Point-to-point (SVD) |
| **Large initial misalignment** | Point-to-point first, then refine with surface-to-point |
| **Publication-quality results** | Point-to-point with trimming |

### Hybrid Approach

The `point_to_point_weight` parameter allows blending:

```python
point_to_point_weight = 0.3  # 30% point-to-point, 70% plane-to-point
```

This constrains the tangential sliding that pure plane-to-point misses.

---

## 4. Summary Recommendations

### For Your Study:

1. **Use point-to-point (SVD) alignment** as primary method
   - Closed-form solution = no linearization error
   - More robust for varying anatomy

2. **Report these metrics**:
   - RMSE (primary)
   - Mean ± SD (secondary)
   - Sternum error = 0.0 mm (constraint validation)
   - Inlier fraction (quality indicator)

3. **Convergence settings**:
   ```python
   max_iterations = 100
   convergence_threshold = 1e-5
   trim_percentage = 0.1  # Fixed across all subjects
   max_correspondence_distance = 15.0  # mm
   ```

4. **For cohort reporting**:
   ```
   "Ribcage alignment achieved RMSE of X.X ± Y.Y mm across N subjects
   (median: Z.Z mm, range: A.A-B.B mm), with sternum superior fixed
   at the origin (0.0 mm error)."
   ```

