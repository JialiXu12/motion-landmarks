# Fixed Sternum Alignment: Technical Explanation

## Overview

This document explains the new `align_fixed_sternum.py` approach compared to the original `align_prone_to_supine()` in `utils.py`.

---

## Problem with Original Approach

### Original Method (`utils.py::align_prone_to_supine`)

**Phase 1: Initial Point-to-Point Alignment**
- Uses `combined_objective_function` to minimize:
  - Distance between prone/supine ribcage point clouds
  - Distance between prone/supine sternum landmarks (superior + inferior)
- Optimizes **6 parameters**: 3 rotation angles + 3 translation values
- No explicit weighting - both contribute equally to objective

**Phase 2: Point-to-Plane ICP Refinement**
- Uses Open3D's `registration_icp` with point-to-plane metric
- Optimizes **full 4x4 transformation matrix** (rotation + translation)
- **Problem**: Sternum can "slide" during ICP because there's no hard constraint

**Result**: 
- Sternum superior alignment error can be 6-25mm after ICP
- The algorithm trades sternum accuracy for better overall ribcage fit
- Violates anatomical knowledge that sternum superior is stable

---

## New Approach: Fixed Sternum Alignment

### Mathematical Foundation

**Key Principle**: By centering all coordinates on the sternum superior (moving it to the origin), we mathematically lock it in place during rotation.

**Why This Works**:
For any rotation matrix $R$ and origin point $\mathbf{o} = [0, 0, 0]^T$:

$$R \times \mathbf{o} = [0, 0, 0]^T$$

The origin is a **fixed point** under rotation. No rotation can move it.

---

### Implementation Steps

#### **PHASE 1: Load Data**
- Same as original approach
- Load sternum landmarks, nipples, ribcage meshes, and soft tissue landmarks

#### **PHASE 2: Center on Sternum Superior (Anchor Point)**

```python
# Translate all prone coordinates
prone_rib_centered = prone_rib - sternum_sup_prone
prone_sternum_centered = sternum_prone - sternum_sup_prone
# Now sternum_sup_prone is at [0, 0, 0]

# Translate all supine coordinates
supine_rib_centered = supine_rib - sternum_sup_supine
supine_sternum_centered = sternum_supine - sternum_sup_supine
# Now sternum_sup_supine is at [0, 0, 0]
```

**Result**: Both prone and supine sternum superior are now at the origin.

---

#### **PHASE 3: Rotation-Only Optimization**

We optimize **only 3 rotation angles** $[\theta_x, \theta_y, \theta_z]$ to minimize:

$$E(\theta) = w_{rib} \cdot E_{rib}(\theta) + w_{sternum} \cdot E_{sternum}(\theta)$$

Where:
- $E_{rib}$: Sum of squared distances between rotated prone ribcage and supine ribcage
- $E_{sternum}$: Squared distance between rotated prone sternum inferior and supine sternum inferior
- $w_{sternum} = 100$ (high weight locks the chest axis)

**Key Constraint**: No translation is allowed. We only rotate around the origin (sternum superior).

```python
R = rotation_matrix_from_euler(angles)
prone_rotated = R @ prone_centered  # Only rotation, no translation
```

**Why No Translation?**
- Translation would move the origin (sternum superior)
- By removing translation, we force the sternum superior to stay locked at [0,0,0]

---

#### **PHASE 4: Point-to-Plane ICP with Rotation-Only**

Standard ICP optimizes translation + rotation. We modify it to **rotation-only**:

```python
def point_to_plane_error(angles):
    R = rotation_matrix_from_euler(angles)
    source_rotated = R @ source_centered  # No translation
    # Calculate point-to-plane distances...
    return sum_of_squared_distances

# Use scipy.optimize to find best rotation
result = minimize(point_to_plane_error, initial_angles, method='L-BFGS-B')
```

**Why This Prevents "Slide"**:
- Standard ICP can trade rotation error for translation error
- By removing translation, the algorithm **must** find the best rotation fit
- The sternum superior cannot move because it's locked at the origin

---

#### **PHASE 5: Transform Back to Absolute Coordinates**

After optimization, we un-center the data:

```python
prone_aligned = prone_rotated + sternum_sup_supine
```

We also construct the full 4x4 transformation matrix for compatibility:

$$T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Where $\mathbf{t} = \text{sternum}_{supine} - R \times \text{sternum}_{prone}$

This ensures that when applied to the original prone data:

$$\text{sternum}_{prone}^{aligned} = R \times \text{sternum}_{prone} + \mathbf{t} = \text{sternum}_{supine}$$

---

#### **PHASE 6: Calculate Displacements**

Same as original approach:
- Landmark displacements relative to sternum
- Landmark displacements relative to nipples
- Separate left/right breast

---

## Comparison of Methods

| Aspect | Original (`utils.py`) | Fixed Sternum (`align_fixed_sternum.py`) |
|--------|----------------------|------------------------------------------|
| **Phase 1** | Optimize 6 DOF (rotation + translation) | Optimize 3 DOF (rotation only) |
| **Phase 2** | Open3D ICP (6 DOF) | Custom ICP (3 DOF rotation only) |
| **Sternum Constraint** | Soft (part of objective) | Hard (mathematical lock at origin) |
| **Sternum Superior Error** | 6-25 mm typical | ~0.000001 mm (numerical precision) |
| **Ribcage Fit** | Optimized globally | Optimized under sternum constraint |
| **Anatomical Validity** | Assumes sternum can move | Locks most stable landmark |
| **Optimization Method** | `scipy.optimize` + Open3D | `scipy.optimize` only (full control) |

---

## Expected Outcomes

### Fixed Sternum Method Should Produce:

1. **Perfect sternum superior alignment** (~0 mm error)
2. **Slightly worse ribcage fit** compared to original (trade-off for anatomical validity)
3. **More anatomically realistic displacements** (sternum is truly fixed)
4. **Better interpretation of results** (reference frame is truly stable)

### When to Use Each Method:

**Original Method (`align_prone_to_supine`)**:
- When minimizing overall point cloud error is priority
- When sternum movement is expected/acceptable
- For general-purpose alignment

**Fixed Sternum Method (`align_fixed_sternum`)**:
- When anatomical validity is critical
- For clinical interpretation (sternum as fixed reference)
- When studying breast tissue deformation relative to chest wall
- For publication where reference frame stability is questioned

---

## Testing the New Method

See `test_fixed_sternum_alignment.py` for a comparison script that runs both methods side-by-side and compares:
- Sternum alignment errors
- Ribcage fit quality
- Computational time
- Landmark displacement differences

---

## Mathematical Proof: Sternum Cannot Move

Given:
- Sternum superior in prone: $\mathbf{s}_p$
- Sternum superior in supine: $\mathbf{s}_s$
- We center: $\mathbf{s}'_p = \mathbf{s}_p - \mathbf{s}_p = [0,0,0]^T$
- We center: $\mathbf{s}'_s = \mathbf{s}_s - \mathbf{s}_s = [0,0,0]^T$

After rotation-only optimization:
$$\mathbf{s}'_p(\text{rotated}) = R \times [0,0,0]^T = [0,0,0]^T$$

After un-centering:
$$\mathbf{s}_p(\text{aligned}) = \mathbf{s}'_p(\text{rotated}) + \mathbf{s}_s = [0,0,0]^T + \mathbf{s}_s = \mathbf{s}_s$$

**Therefore**: $\mathbf{s}_p(\text{aligned}) = \mathbf{s}_s$ (exactly)

**Q.E.D.** The sternum superior cannot move from its target position. ∎

---

## References

- Open3D ICP Documentation: http://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html
- Besl & McKay (1992): "A Method for Registration of 3-D Shapes"
- Horn (1987): "Closed-form solution of absolute orientation using unit quaternions"

---

**Author**: Analysis Team  
**Date**: February 3, 2026  
**Version**: 1.0
