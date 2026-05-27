# Coordinate Systems in alignment.py - Complete Explanation

## Your Question

> "In alignment.py the sternum positions are fixed, then rotation matrix is calculated, after alignment, transform prone data to supine frame, are these all have sternum as origin? or in original supine frame?"

## Short Answer

**The final transformed data is in the ORIGINAL SUPINE FRAME, NOT sternum-centered.**

The sternum is **NOT at origin (0,0,0)** in the final output. It's at the supine sternum's actual physical position.

---

## Detailed Explanation

### Phase 1: During Alignment (Lines 109-199)

```python
# Center both point clouds on their respective sternums
src_centered = source_pts - source_ss  # Prone sternum → origin (0,0,0)
tgt_centered = target_pts - target_ss  # Supine sternum → origin (0,0,0)

# Optimize rotation around origin
# ... ICP iterations ...
# Returns: R_total (rotation matrix) and aligned_prone_centered
```

**At this stage**: Both point clouds are in **sternum-centered coordinates** (sternum at origin).

### Phase 2: Transform to Supine Frame (Lines 626-680)

```python
def apply_transform_to_coords(coords, R, source_anchor, target_anchor):
    coords_centered = coords - source_anchor      # [1] Center on prone sternum
    coords_rotated = (R @ coords_centered.T).T    # [2] Rotate around origin
    coords_final = coords_rotated + target_anchor # [3] ← KEY: Add supine sternum
    return coords_final
```

**Step [3] is crucial**: It **ADDS** `target_anchor` (supine sternum position), which moves everything from origin to the supine sternum's actual position.

**Result**: All transformed prone data is now in the **ORIGINAL SUPINE COORDINATE SYSTEM**.

### Test Verification

```
Original positions:
  Prone sternum:  [100, 200, 300]
  Supine sternum: [50, 150, 250]

After transformation:
  Prone sternum transformed: [50, 150, 250]  ← Matches supine sternum!
  
Distance to origin:        295.80 mm  ← NOT at origin
Distance to supine sternum: 0.00 mm  ← AT supine sternum position
```

---

## What About "Relative to Sternum"? (Lines 690-705)

Later in the code, you calculate:

```python
ref_sternum_prone = prone_sternum_transformed[0]  # At [50, 150, 250]
ref_sternum_supine = sternum_supine[0]            # At [50, 150, 250]

# Subtract to get sternum-centered coordinates
lm_pos_prone_rel_sternum = landmark_prone_transformed - ref_sternum_prone
lm_pos_supine_rel_sternum = landmark_supine_ave_raw - ref_sternum_supine
```

**Now these ARE in sternum-centered coordinates** because you explicitly subtracted the sternum position!

---

## Summary Table

| Stage | Coordinate System | Sternum Position |
|-------|------------------|------------------|
| **During alignment** | Sternum-centered | (0, 0, 0) |
| **After apply_transform_to_coords** | Original supine frame | Supine sternum's actual position (e.g., [50, 150, 250]) |
| **After subtracting sternum** | Sternum-centered | (0, 0, 0) again |

---

## Why This Design?

This is actually a **good design** because:

1. **Alignment works in optimal coordinates** (centered, no numerical issues)
2. **Output is in familiar coordinates** (original supine frame)
3. **Easy to visualize** with MRI images (all in same coordinate system)
4. **Can convert to sternum-centered** anytime by subtracting sternum position

---

## The Potential Confusion

The variable names suggest sternum-centered coordinates:

```python
# These names suggest "relative to sternum" but they're NOT (yet)
'ld_ave_prone_transformed'  # In original supine frame, not sternum-centered
'nipple_prone_transformed'  # In original supine frame, not sternum-centered

# Only THESE are actually sternum-centered (after subtraction):
'ld_ave_prone_rel_sternum'  # Truly sternum-centered
'nipple_prone_rel_sternum'  # Truly sternum-centered
```

---

## Recommendation: Improve Code Clarity

To avoid confusion, consider:

1. **Rename variables** to be more explicit:
   ```python
   # Instead of:
   landmark_prone_transformed  
   
   # Use:
   landmark_prone_in_supine_frame
   ```

2. **Add comments** clarifying coordinate systems:
   ```python
   # Transform to original supine frame (NOT sternum-centered)
   prone_sternum_transformed = apply_transform_to_coords(...)
   
   # Now in sternum-centered coordinates
   lm_pos_prone_rel_sternum = landmark_prone_transformed - ref_sternum_prone
   ```

3. **Document the coordinate system** in function docstrings:
   ```python
   def apply_transform_to_coords(...):
       """
       ...
       Returns:
           Coordinates in the original target frame (NOT centered on target_anchor)
       """
   ```

---

## Visual Diagram

```
ALIGNMENT PHASE:
┌─────────────────────────────────┐
│  Prone (centered)               │  Supine (centered)
│  Sternum at (0,0,0)            │  Sternum at (0,0,0)
│         *                       │         *
│      landmark                   │      landmark
└─────────────────────────────────┘
         ↓ Optimize rotation R
         
TRANSFORMATION PHASE:
┌─────────────────────────────────┐
│  Original Supine Frame          │
│  Sternum at (50, 150, 250)     │
│         *                       │
│      landmark                   │
│                                 │
│  ← All prone data moved here   │
└─────────────────────────────────┘

AFTER "RELATIVE TO STERNUM":
┌─────────────────────────────────┐
│  Sternum-Centered               │
│  Sternum at (0,0,0)            │
│         *                       │
│      landmark                   │
└─────────────────────────────────┘
```

---

## Answer to Your Question

**After alignment and transformation**:
- ✗ NOT in sternum-centered coordinates (sternum is not at origin)
- ✓ IN ORIGINAL SUPINE FRAME (sternum at its physical position)
- ✓ You get sternum-centered coords only when you subtract sternum position

The sternum is **fixed during rotation** (rotation happens around sternum at origin), but the **final output is translated** to put everything in the original supine coordinate system.

