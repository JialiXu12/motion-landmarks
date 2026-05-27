# Anterior Ribcage Mutual Region Filtering

## Purpose

Before ICP alignment, the prone ribcage mesh and supine ribcage point cloud are filtered to their mutual overlapping region. This removes posterior/lateral/inferior points that don't contribute to alignment and can cause ICP to fail or converge poorly.

The anterior direction (Y-min in RAI coordinates) is special: the mesh and point cloud often have different anterior extents, and we want to preserve the most anterior points from **both** datasets.

## Coordinate System

All filtering happens in **sternum-centered** coordinates:
- Origin = sternum superior
- Y axis: Anterior (negative) to Posterior (positive) in RAI convention
- The mesh is also rotated by `R_init` (initial Rodrigues rotation) before filtering

## Pipeline

### 1. Preprocessing (`alignment_preprocessing.py`)

```
src_centered_rotated = R_init @ (mesh_pts - sternum_prone)
tgt_centered = pc_pts - sternum_supine
```

Compute shared anterior boundary:
```
shared_y_min = min(mesh_y_min, pc_y_min)
```

This is the most anterior Y coordinate across both datasets, in sternum-centered space.

### 2. Mutual Region Filtering (`filter_mutual_region` in `alignment_utils.py`)

#### Step 1: Filter point cloud to mesh bounding box

Keeps point cloud points inside the mesh bbox (with padding on each side).

- X: `[mesh_x_min - padding, mesh_x_max + padding]`
- Y: `[shared_y_min - padding_reciprocal, mesh_y_max + padding]`
- Z: `[mesh_z_min - padding_inferior, mesh_z_max + padding]`

The Y-min boundary is set to `shared_y_min - padding_reciprocal` instead of `mesh_y_min - padding`. This ensures point cloud points more anterior than the mesh are kept.

#### Step 2: Filter mesh to (filtered) point cloud bounding box

Keeps mesh points inside the filtered point cloud bbox.

- X: `[pc_x_min, pc_x_max]` (tight, no padding)
- Y: `[shared_y_min - padding_reciprocal, pc_y_max]`
- Z: `[pc_z_min, pc_z_max]` (tight, no padding)

The Y-min boundary uses the same `shared_y_min - padding_reciprocal`, ensuring mesh points more anterior than the point cloud are also kept.

## Parameters

| Parameter | Default | Used In | Effect |
|-----------|---------|---------|--------|
| `padding` | 15 mm | Step 1 | General margin around mesh bbox (X, Y-max, Z-max) |
| `padding_inferior` | 0 mm | Step 1 | Overrides `padding` for Z-min (inferior) |
| `padding_reciprocal` | 0 mm | Both | Extra margin beyond `shared_y_min` on the anterior (Y-min) side |
| `anterior_y_min` | auto | Both | `min(mesh_y_min, pc_y_min)` — computed in preprocessing |

## Key Design Decisions

1. **Shared Y-min**: Both bboxes use the same anterior boundary. Neither the mesh nor the point cloud loses its most anterior points, regardless of which extends further.

2. **`padding_reciprocal` as extra margin**: Applied as `shared_y_min - padding_reciprocal`. Set to 0 if you want the boundary exactly at the most anterior point. Increase for a small safety margin.

3. **Tight X/Z in Step 2**: The reciprocal filter (Step 2) has no padding on X and Z, so mesh points outside the point cloud's lateral/vertical extent are removed. Only the anterior direction gets special treatment.

## Files

- `alignment_preprocessing.py` — computes `shared_y_min`, calls `filter_mutual_region`
- `alignment_utils.py` — `filter_mutual_region()` implements the two-step filtering
- `main_alignment.py` — sets `mutual_region_padding_reciprocal` parameter
