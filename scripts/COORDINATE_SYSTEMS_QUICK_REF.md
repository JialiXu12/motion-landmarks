# Coordinate Systems in alignment.py - Quick Reference

## Direct Answer to Your Question

**Q:** "After alignment, transform prone data to supine frame - are these all have sternum as origin? or in original supine frame?"

**A:** **ORIGINAL SUPINE FRAME** (sternum is NOT at origin)

---

## Three Coordinate Systems Used

### 1️⃣ During Alignment (Internal)
- **Coordinate system**: Sternum-centered
- **Sternum position**: (0, 0, 0)
- **Used for**: Optimization (SVD/ICP iterations)
- **Code location**: Lines 109-199 (`optimal_sternum_fixed_alignment`)

### 2️⃣ After Transformation (Main Output)
- **Coordinate system**: Original supine frame
- **Sternum position**: Supine sternum's actual position (e.g., [50, 150, 250])
- **Used for**: All transformed prone data
- **Code location**: Lines 626-680 (`apply_transform_to_coords`)
- **Variables**: 
  - `landmark_prone_transformed`
  - `nipple_prone_transformed`
  - `prone_sternum_transformed`

### 3️⃣ Relative to Sternum (Derived)
- **Coordinate system**: Sternum-centered
- **Sternum position**: (0, 0, 0)
- **Used for**: Displacement calculations
- **Code location**: Lines 690-705
- **Variables**:
  - `lm_pos_prone_rel_sternum`
  - `lm_pos_supine_rel_sternum`
  - `nipple_pos_prone_rel_sternum`

---

## Key Code Snippet

```python
# Step 1: Transform to ORIGINAL SUPINE FRAME
prone_sternum_transformed = apply_transform_to_coords(
    sternum_prone, R, source_anchor, target_anchor
)
# Result: prone_sternum_transformed = supine_sternum position (NOT origin)

# Step 2: Convert to STERNUM-CENTERED coords by subtracting sternum
lm_pos_prone_rel_sternum = landmark_prone_transformed - prone_sternum_transformed
# Result: Now sternum is at (0, 0, 0)
```

---

## Visual Example

```
ALIGNMENT (Internal):
  Prone sternum:  (0, 0, 0) ← centered
  Supine sternum: (0, 0, 0) ← centered
  → Rotation optimized here

AFTER TRANSFORMATION (Main Output):
  Prone sternum transformed:  (50, 150, 250) ← NOT origin!
  Supine sternum:             (50, 150, 250) ← Match
  → In original supine coordinate system

RELATIVE TO STERNUM (Derived):
  Prone relative:  landmark - (50, 150, 250) = (x, y, z)
  Supine relative: landmark - (50, 150, 250) = (x, y, z)
  → Sternum effectively at (0, 0, 0)
```

---

## Why This Design?

✅ **Advantages**:
1. Alignment optimizes in ideal coordinates (no large offsets)
2. Output matches original image coordinates (easy visualization)
3. Can derive sternum-centered coords anytime (just subtract)

❌ **Disadvantage**:
- Variable names can be misleading (`_transformed` suggests centered, but it's not)

---

## Code Changes Made

Added clarifying comments to `alignment.py`:

1. **`apply_transform_to_coords` docstring** (line ~470):
   - Clarified output is in "TARGET'S ORIGINAL FRAME"
   - NOT anchor-centered

2. **Section 3 comment** (line ~626):
   - "After transformation, all prone data is in ORIGINAL SUPINE COORDINATE SYSTEM"

3. **Section 5 comment** (line ~690):
   - "By subtracting sternum, we convert to STERNUM-CENTERED coordinates"

---

## Test Results

Run `test_coordinate_simple.py` to verify:

```bash
cd scripts
python test_coordinate_simple.py
```

Output shows:
```
Prone sternum distance to origin: 295.80 mm  ← NOT at origin
Prone sternum distance to supine sternum: 0.00 mm  ← At supine position

→ OUTPUT IS IN ORIGINAL SUPINE FRAME
```

---

## Summary

| Question | Answer |
|----------|---------|
| Is sternum at origin after transformation? | ❌ NO |
| Is data in original supine frame? | ✅ YES |
| When is sternum at origin? | Only after subtracting sternum position |
| Does alignment work correctly? | ✅ YES - sternum drift is 0.0 mm |

The alignment is **mathematically correct** - sternum is fixed during optimization, and the final output is properly transformed to the supine coordinate system.

