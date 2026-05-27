# Prone-to-Supine Alignment: Query Direction Explained

## The Scenario

We are transforming **prone** points to align with **supine** (the target/reference).

```
Source (Moving):  Prone ribcage  → Transform → Aligned Prone
Target (Fixed):   Supine ribcage (stays in place)
```

## Why Two Different Queries?

### Query 1: Alignment Quality Metric
**Direction**: Prone → Supine (Source to Target)

```python
tree = cKDTree(supine_rib_centered)     # Build tree from TARGET
distances, _ = tree.query(prone_rib_final, k=1)  # Query with SOURCE
```

**Interpretation**: 
- For each **transformed prone point**, find the nearest **supine point**
- "How far is each source point from its best-match target point?"

**Used for**:
- Computing mean alignment error
- Optimization objective (minimize this distance)
- Reporting fit quality: "Mean ribcage error: X mm"

**Analogy**: Like measuring "how far did the moving object travel from its intended destination?"

---

### Query 2: Visualization 
**Direction**: Supine → Prone (Target to Source)

```python
tree_prone = cKDTree(prone_rib_final)   # Build tree from transformed SOURCE
distances_vis, idxs = tree_prone.query(supine_rib_centered, k=1)  # Query with TARGET
```

**Interpretation**:
- For each **supine point**, find the nearest **transformed prone point**
- "How well is each target point covered by the aligned source?"

**Used for**:
- Coloring supine points by alignment error
- Drawing arrows from supine to nearest prone (shows gaps/overlaps)
- Interactive visualization with PyVista

**Analogy**: Like a "coverage map" showing which target areas are well-matched vs poorly-matched

---

## Visual Example

```
Alignment Quality (Query 1: prone → supine):
    Prone Point A  ----→  [finds] Supine Point X  (distance = 3mm)
    Prone Point B  ----→  [finds] Supine Point Y  (distance = 5mm)
    Mean Error: 4mm ✓

Visualization (Query 2: supine → prone):
    Supine Point X  ←----  [finds] Prone Point A  (distance = 3mm)
    Supine Point Y  ←----  [finds] Prone Point B  (distance = 5mm)
    Color supine points: X=green (3mm), Y=yellow (5mm) ✓
    Draw arrows: X→A, Y→B ✓
```

## Why Not Just Use Query 1 for Everything?

**Problem**: Query 1 gives `distances[i]` for **prone point i**, but `plot_evaluate_alignment` expects `distances[j]` for **supine point j**.

The visualization function **must** receive:
- `supine_pts[j]` = position of supine point j
- `distances[j]` = distance from supine point j to nearest prone point
- `idxs[j]` = index of nearest prone point for supine point j

Using Query 1 would give distances indexed by prone points, which doesn't match the supine point cloud being visualized!

---

## Original Code Reference

This approach matches the original `align_prone_to_supine` in `utils.py`:

```python
# Line 1010 in utils.py
error, mapped_idx = breast_metadata.closest_distances(
    supine_ribcage_pc,              # Query points (target)
    prone_ribcage_mesh_transformed  # Search tree (transformed source)
)
# Returns: for each supine point, nearest prone point
```

Then passed to visualization:
```python
plot_evaluate_alignment(
    supine_pts=supine_ribcage_pc,
    transformed_prone_mesh=prone_ribcage_mesh_transformed,
    distances=rib_error_mag,  # Distance for each supine point
    idxs=mapped_idx,          # Nearest prone index for each supine point
    ...
)
```

---

## Conclusion

✅ **Both queries are correct** - they serve different purposes:
- Query 1 (prone→supine): Measures alignment quality
- Query 2 (supine→prone): Enables visualization

This is **not a bug**, it's the proper implementation for prone-to-supine alignment with visualization!

Date: February 4, 2026
