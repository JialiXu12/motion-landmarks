# Detailed Alignment Process Analysis

**Date:** February 2, 2026  
**Purpose:** Comprehensive explanation of the prone-to-supine alignment algorithm

---

## Question 1: Initial Transformation - Optimization Strategy

### **Answer: SIMULTANEOUS optimization with EQUAL weighting (but problematic)**

---

## Phase 1: Initial Point-to-Point Alignment

### 1.1 Initial Guess (Lines 1004-1007 in utils.py)

```python
rot_angle_init = [0., 0., 0.]  # No rotation initially
translation_init = list(
    breast_metadata.find_centroid(sternum_supine.T) - 
    breast_metadata.find_centroid(sternum_prone.T)
)
T_init = rot_angle_init + translation_init  # [rx, ry, rz, tx, ty, tz]
```

**What is used for initial translation and rotation:**
- ✅ **Rotation:** `[0°, 0°, 0°]` (identity rotation - no rotation initially)
- ✅ **Translation:** Difference between sternum centroids
  - Translation = Supine Sternum Center - Prone Sternum Center
  - This aligns the sternum centroids as starting point
  - Sternum = average of superior and inferior sternum landmarks (2 points)

**Why sternum?** It's the most stable anatomical landmark on the chest wall.

---

### 1.2 Optimization Process

#### The Objective Function (alignment.py lines 86-107)

```python
def combined_objective_function(T_guess, set1, set2):
    # set1 = [prone_ribcage_mesh_coords, sternum_prone]
    # set2 = [supine_ribcage_pc, sternum_supine]
    
    # Ribcage error
    obj_func1 = objective_function(T_guess, set1[0], set2[0])
    
    # Sternum error
    obj_func2 = objective_function(T_guess, set1[1], set2[1])
    
    # Simple addition - NO WEIGHTING
    return obj_func1 + obj_func2
```

#### What Each objective_function Computes (alignment.py lines 52-84)

```python
def objective_function(T_guess, set1, set2):
    # 1. Apply transformation to prone points
    T_full = rotation_matrix(T_guess[0], T_guess[1], T_guess[2])
    T_full[:-1, -1] = T_guess[3:]  # Add translation
    
    set1_transformed = (T_full @ np.hstack((set1, ones)).T)[:-1, :].T
    
    # 2. Find nearest neighbor distances (KD-tree)
    tree = spatial.KDTree(set1_transformed)
    distances, _ = tree.query(set2, k=1)
    
    # 3. Compute Mean Squared Distance (MSD)
    msd = np.sum(distances**2) / len(distances)
    
    return msd
```

---

### 1.3 **CRITICAL ISSUE: Unbalanced Weighting**

#### The Math:

**Combined Objective:**
```
Total_Error = MSD_ribcage + MSD_sternum

Where:
  MSD_ribcage = Σ(distances_rib²) / N_rib
  MSD_sternum = Σ(distances_sternum²) / N_sternum
```

**The Problem:**
- Ribcage: ~5,000-20,000 points
- Sternum: 2 points (superior + inferior)

**Effective Weight Ratio:**
```
Weight_ribcage : Weight_sternum ≈ N_rib : N_sternum ≈ 10,000 : 1
```

**Even though the code adds them equally, the ribcage DOMINATES because:**
1. MSD is divided by point count (normalization)
2. BUT the optimization sees two separate terms
3. The ribcage MSD has ~10,000× more "pull" during gradient descent

---

### 1.4 The Optimization Algorithm (alignment.py lines 117-137)

```python
def run_optimisation(objective, T_init, prone_coords, supine_coords):
    # Uses scipy.optimize.least_squares
    res = least_squares(
        fun=objective,
        x0=T_init,  # [0, 0, 0, tx, ty, tz]
        args=(prone_coords, supine_coords)
    )
    
    # Extract optimized parameters
    # res.x = [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    
    T = rotation_matrix(res.x[0], res.x[1], res.x[2])
    T[:-1, -1] = res.x[3:]  # Add translation
    
    return T, res
```

**What is optimized:**
- **6 parameters:** 3 rotation angles (°) + 3 translation components (mm)
- **Rotation order:** Z → Y → X (Euler angles)
- **Algorithm:** Levenberg-Marquardt (least_squares default)

**What is used for translation and rotation during optimization:**
- Both are **simultaneously optimized** from the initial guess
- The optimizer adjusts all 6 parameters to minimize the combined MSD
- Uses Jacobian-based gradient descent with trust region

---

## Question 2: Point-to-Plane ICP

### **Answer: NO, sternum is NOT fixed - This is a MAJOR ISSUE**

---

## Phase 2: ICP Refinement

### 2.1 Input to ICP (utils.py lines 1067-1076)

```python
target_pts = np.asarray(prone_ribcage_mesh_transformed, dtype=np.float64)
source_pts = np.asarray(supine_ribcage_pc, dtype=np.float64)

T_icp, supine_ribcage_refined, icp_result = run_point_to_plane_icp(
    source_pts=source_pts,      # Supine ribcage (~10,000-20,000 points)
    target_pts=target_pts,       # Prone ribcage (~5,000 points)
    max_correspondence_distance=10.0,
    max_iterations=200,
    delta=1.0  # Huber loss parameter
)
```

**⚠️ CRITICAL: Sternum is NOT included in ICP refinement!**

Only ribcage points are used. The sternum that was carefully aligned in Phase 1 is now **ignored**.

---

### 2.2 The ICP Algorithm (utils.py lines 1457-1575)

```python
def run_point_to_plane_icp(...):
    # 1. Estimate surface normals on target
    tgt_pcd.estimate_normals(...)
    
    # 2. Set initial transformation
    init_T = np.eye(4)  # Identity - starts from current alignment
    
    # 3. Define convergence criteria
    icp_criteria = ICPConvergenceCriteria(max_iteration=max_iterations)
    
    # 4. Define robust loss (Huber)
    loss = o3d.pipelines.registration.HuberLoss(k=delta)
    
    # 5. Run ICP
    result = o3d.pipelines.registration.registration_icp(
        src_pcd,  # Supine ribcage
        tgt_pcd,  # Prone ribcage (transformed)
        max_correspondence_distance,
        init_T,
        TransformationEstimationPointToPlane(loss),
        icp_criteria
    )
    
    T_icp = result.transformation  # The refinement transformation
```

---

### 2.3 What ICP Optimizes

**Open3D Point-to-Plane ICP minimizes:**

```
E(T) = Σ [n_i · (T*p_i - q_i)]²  with robust loss

Where:
  p_i = source point (supine ribcage)
  q_i = closest target point (prone ribcage)
  n_i = normal at q_i
  T   = transformation matrix [R|t]
```

**What is used for translation and rotation in ICP:**
- Both are **simultaneously optimized** starting from identity
- The algorithm iteratively:
  1. Finds closest point pairs
  2. Computes optimal T to minimize point-to-plane distances
  3. Applies T to source points
  4. Repeats until convergence

**Rotation:** Estimated via singular value decomposition (SVD) of point correspondences
**Translation:** Estimated via weighted least squares of point correspondences

---

### 2.4 **THE FUNDAMENTAL PROBLEM**

#### Sternum is NOT Constrained!

After ICP refinement (utils.py lines 1127-1133):

```python
T_total = T_icp_inv @ T_optimal  # Combine transformations

prone_ribcage_aligned_final = apply_transform(prone_ribcage_mesh_coords, T_total)
prone_sternum_aligned_final = apply_transform(sternum_prone, T_total)

# RE-EVALUATE sternum error AFTER ICP
error, _ = breast_metadata.closest_distances(sternum_supine, prone_sternum_aligned_final)
sternum_error = np.linalg.norm(error, axis=1)
print(f"Sternum alignment error: {sternum_error} mm")
```

**Why is this re-evaluated?** Because ICP **CAN and DOES degrade sternum alignment!**

---

### 2.5 How ICP Can Cause Problems

The ribcage can translate/rotate in ways that minimize ribcage error but **worsen sternum alignment:**

#### Example Scenario:
```
Initial (after Phase 1):
  - Sternum error: 2mm ✅
  - Ribcage error: 8mm

After ICP (Phase 2):
  - Sternum error: 15mm ❌ (degraded!)
  - Ribcage error: 3mm ✅ (improved)
```

**Why does this happen?**

1. **Ribcage deformation:** Ribcage shape changes between prone/supine due to:
   - Breathing state differences
   - Soft tissue compression
   - Rib expansion/contraction
   - Muscle tension

2. **ICP assumes rigid body:** It finds the best **rigid transformation** for the ribcage, which may not preserve sternum alignment

3. **No constraints:** There's nothing preventing ICP from:
   - Translating superior/inferior (up/down)
   - Rotating around sternum
   - Shifting anterior/posterior

---

### 2.6 Current "Safeguards" (Insufficient)

#### Safeguard 1: Correspondence Distance Limit
```python
max_correspondence_distance=10.0  # mm
```
- Only matches points within 10mm
- Prevents huge jumps, but doesn't preserve sternum

#### Safeguard 2: Huber Loss
```python
loss = o3d.pipelines.registration.HuberLoss(k=1.0)
```
- Downweights outliers
- Prevents outliers from dominating, but doesn't preserve sternum

#### Safeguard 3: Good Initial Alignment
- Phase 1 provides a good starting point
- ICP makes only **small refinements**
- But "small" can still be 10-15mm sternum drift (as seen in your data)

---

## Evidence from Your Data

From your terminal output:

| Subject | Sternum Error (mm) | Ribcage Error (mm) | Status | ICP RMSE |
|---------|-------------------|-------------------|---------|----------|
| VL00009 | [24.2, 26.2] | 9.50 | ❌ FAILED | 4.42 |
| VL00011 | [18.2, 20.0] | 3.53 | ❌ FAILED | 3.56 |
| VL00014 | [15.1, 14.3] | 7.99 | ✅ PASS | 3.73 |
| VL00018 | [12.3, 10.1] | 3.25 | ✅ PASS | 3.44 |
| VL00020 | [4.9, 4.7] | 4.29 | ✅ PASS | 3.41 |

**Analysis:**
- Subjects with sternum error > 15mm are marked FAILED
- Even "passing" subjects have 5-15mm sternum error
- For biomechanical analysis, this is **significant** (breast motion is 50-150mm)

---

## Summary: Answers to Your Questions

### Q1: Does initial transformation minimize ribcage and sternum simultaneously or sequentially?

**Answer:** **SIMULTANEOUS** but with problematic weighting

- **Method:** Combined objective function (sum of two MSDs)
- **Weighting:** 
  - **Explicit:** 1:1 (equal addition)
  - **Effective:** ~10,000:1 (ribcage dominates due to point count)
- **Parameters optimized:** 6 DOF (3 rotation + 3 translation)
- **Initial translation:** Sternum centroid difference
- **Initial rotation:** [0°, 0°, 0°] (identity)
- **Algorithm:** Levenberg-Marquardt least squares

### Q2: Is sternum fixed in point-to-plane ICP?

**Answer:** **NO - and this is a major limitation**

- **What's included:** Only ribcage points (~5,000-20,000)
- **What's excluded:** Sternum landmarks (the reference point!)
- **Consequence:** ICP can degrade sternum alignment
- **Parameters optimized:** 6 DOF (3 rotation + 3 translation) from identity
- **Initial state:** Uses result from Phase 1 (T_optimal)
- **Algorithm:** Point-to-plane ICP with Huber loss

### Q3: Does the algorithm account for ribcage translation issues?

**Answer:** **NO - only soft constraints**

- Max correspondence distance (10mm) ❌ Not sufficient
- Huber loss (robust estimation) ❌ Not sufficient  
- Good initial alignment ❌ Not sufficient
- **Result:** 5-26mm sternum drift in your data

---

## Recommendations

### Option 1: Weighted ICP with Sternum Constraints ⭐ **RECOMMENDED**
```python
# Include sternum in ICP with high weight
source_pts = np.vstack([supine_ribcage_pc, sternum_supine])
target_pts = np.vstack([prone_ribcage_mesh, sternum_prone])
weights = np.concatenate([
    np.ones(len(supine_ribcage_pc)),
    np.ones(len(sternum_supine)) * 1000  # 1000× weight
])
```

### Option 2: Constrained Optimization
Add explicit sternum distance constraint to ICP optimization:
```python
minimize: point_to_plane_error(ribcage)
subject to: ||T*sternum_prone - sternum_supine|| < threshold
```

### Option 3: Two-Stage ICP
1. First ICP: Align sternum + nearby ribcage
2. Second ICP: Refine full ribcage with sternum locked

### Option 4: Post-ICP Correction
After ICP, apply corrective transformation to restore sternum alignment:
```python
T_corrected = compute_sternum_correction(T_icp, sternum_prone, sternum_supine)
T_total = T_corrected @ T_icp @ T_optimal
```

---

## Conclusion

Your alignment pipeline has **two fundamental issues:**

1. ✅ **Phase 1:** Ribcage dominates optimization (10,000:1 effective weight)
2. ❌ **Phase 2:** Sternum is completely ignored in ICP refinement

For accurate biomechanical analysis, **sternum alignment is critical** because it's your reference frame for measuring breast displacement. The current 5-26mm sternum drift introduces systematic error into all your displacement measurements.

**Recommendation:** Implement **Option 1 (Weighted ICP)** - it's the simplest and most effective solution that preserves sternum alignment while still refining ribcage fit.
