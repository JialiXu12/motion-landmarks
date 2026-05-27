# Fix for plot_evaluate_alignment Not Showing

## Problem Identified

The `plot_evaluate_alignment` visualization was not showing due to **two critical issues**:

### Issue 1: Incorrect Nearest Neighbor Indices (MAIN ISSUE)

**Location:** `align_fixed_sternum.py`, line ~539 and ~567

**Problem:**
```python
# WRONG: Built tree from supine, queried with prone
tree = cKDTree(supine_rib_centered)
distances_final, _ = tree.query(prone_rib_final, k=1)

# Then passed to visualization
plot_evaluate_alignment(
    supine_pts=supine_rib_vis,
    transformed_prone_mesh=prone_rib_final_vis,
    distances=distances_final,
    idxs=np.arange(len(distances_final)),  # WRONG! Just a sequence, not real indices
    ...
)
```

**Why this is wrong:**
- The `idxs` parameter should contain the index of the nearest **prone** point for each **supine** point
- Passing `np.arange(len(distances_final))` means "supine point i maps to prone point i", which is incorrect
- This caused the arrow visualization to fail or point to wrong locations
- The line `nearest_prone_pts = prone_pts_for_tree[idxs[worst_idx]]` (utils.py:694) would index incorrectly

**Solution:**
```python
# CORRECT: Build tree from prone, query with supine
tree_prone = cKDTree(prone_rib_final)
distances_vis, nearest_indices_vis = tree_prone.query(supine_rib_centered, k=1)

# Now pass the real indices
plot_evaluate_alignment(
    supine_pts=supine_rib_vis,
    transformed_prone_mesh=prone_rib_final_vis,
    distances=distances_vis,
    idxs=nearest_indices_vis,  # CORRECT! Real nearest neighbor indices
    ...
)
```

### Issue 2: Silent Exception Handling (MASKING THE ERROR)

**Location:** `align_fixed_sternum.py`, line ~562-576

**Problem:**
```python
try:
    plot_evaluate_alignment(...)
except Exception as e:
    print(f"  Could not visualize alignment errors: {e}")
```

**Why this is wrong:**
- Any exception thrown by `plot_evaluate_alignment` was caught and only printed
- This prevented us from seeing the actual error (likely an IndexError or shape mismatch due to Issue 1)
- The plot window never appeared, but no obvious error was shown

**Solution:**
- Removed the try-except block
- Added debug prints to show data shapes and ranges
- Now any error will be immediately visible

## Technical Details

### How plot_evaluate_alignment Works

From `utils.py`, lines 603-722:

1. Takes `supine_pts` (N x 3) as main point cloud to visualize
2. Takes `distances` (N,) showing distance from each supine point to nearest prone point
3. Takes `idxs` (N,) showing which prone point index is nearest to each supine point
4. Colors the supine points by distance
5. For the worst N points, draws arrows from supine point to its nearest prone point:
   ```python
   worst_idx = np.argsort(distances)[-worst_n_use:]
   worst_targets = supine_pts[worst_idx]
   nearest_prone_pts = prone_pts_for_tree[idxs[worst_idx]]  # Line 694
   vectors = nearest_prone_pts - worst_targets
   ```

### Why Direction Matters: Prone-to-Supine Alignment

When aligning **prone TO supine**, we transform prone points to match the supine reference frame.

The alignment code builds two different KDTree queries for different purposes:

1. **For alignment quality metric** (How well does prone fit supine?):
   ```python
   tree = cKDTree(supine_rib_centered)  # Target (supine) is reference
   distances_final, _ = tree.query(prone_rib_final, k=1)  # Query with source (prone)
   ```
   - **Meaning**: "For each transformed prone point, what's the nearest supine target point?"
   - **Purpose**: Measures how well the source (prone) has been moved to match target (supine)
   - **Used for**: Alignment error statistics, optimization objective

2. **For visualization** (What does the error field look like?):
   ```python
   tree_prone = cKDTree(prone_rib_final)  # Build tree from transformed prone
   distances_vis, nearest_indices_vis = tree_prone.query(supine_rib_centered, k=1)
   ```
   - **Meaning**: "For each supine point, what's the nearest transformed prone point?"
   - **Purpose**: Visualize error distribution from the target (supine) perspective
   - **Used for**: Color-coded visualization and error arrows in plot_evaluate_alignment

Both are correct but serve different purposes. The visualization query matches the original implementation in `utils.py:1010`:
```python
error, mapped_idx = breast_metadata.closest_distances(supine_ribcage_pc, prone_ribcage_mesh_transformed)
```

## Changes Made

### File: `align_fixed_sternum.py`

1. **Line ~539-543**: Added second KDTree query for visualization
   ```python
   # For alignment quality: find nearest supine point for each prone point
   tree = cKDTree(supine_rib_centered)
   distances_final, _ = tree.query(prone_rib_final, k=1)
   
   # For visualization: find nearest prone point for each supine point
   tree_prone = cKDTree(prone_rib_final)
   distances_vis, nearest_indices_vis = tree_prone.query(supine_rib_centered, k=1)
   ```

2. **Line ~565-583**: Removed try-except, added debug info, fixed parameters
   ```python
   # Visualize alignment quality with error colors
   print("\n  Calling plot_evaluate_alignment...")
   print(f"    supine_pts shape: {supine_rib_vis.shape}")
   print(f"    prone_mesh shape: {prone_rib_final_vis.shape}")
   print(f"    distances_vis shape: {distances_vis.shape}")
   print(f"    distances_vis range: [{distances_vis.min():.2f}, {distances_vis.max():.2f}] mm")
   print(f"    nearest_indices_vis shape: {nearest_indices_vis.shape}")
   
   plot_evaluate_alignment(
       supine_pts=supine_rib_vis,
       transformed_prone_mesh=prone_rib_final_vis,
       distances=distances_vis,
       idxs=nearest_indices_vis,
       worst_n=60,
       cmap="viridis",
       point_size=3,
       arrow_scale=20,
       show_scalar_bar=True,
       return_data=False
   )
   ```

## Testing

To verify the fix works:

1. Run alignment from `main.py`:
   ```python
   from align_fixed_sternum import align_prone_to_supine_fixed_sternum
   results = align_prone_to_supine_fixed_sternum(subject, plot_for_debug=True, ...)
   ```

2. The visualization should now show:
   - Supine point cloud colored by alignment error (distance to nearest prone point)
   - Red arrows from the 60 worst supine points to their nearest prone points
   - The arrows should point in anatomically correct directions
   - A scalar bar showing the distance scale in mm

## Expected Behavior

When `plot_for_debug=True`:
1. First plot: Initial rotation alignment (plot_all)
2. Second plot: ICP refinement result (plot_all)  
3. **Third plot: Alignment quality visualization (plot_evaluate_alignment)** ← NOW WORKS!
   - Interactive PyVista 3D window
   - Color-coded error visualization
   - Arrow indicators for worst alignment regions

Date Fixed: February 4, 2026
