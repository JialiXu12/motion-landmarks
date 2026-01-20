# Changes to plot_vectors_rel_sternum Text Labels

## Summary of Changes

Updated the `plot_vectors_rel_sternum` function to improve text label visibility and positioning across different coloring schemes and plane views.

---

## Changes Made

### 1. Subject Coloring (color_by='subject')
**Added text labels for Axial and Coronal views**

- **Axial View**: Added "Right Breast" and "Left Breast" labels in **black** color
- **Coronal View**: Added "Right Breast" and "Left Breast" labels in **black** color
- **Sagittal View**: No text labels (unchanged)

**Before**: No breast labels were shown when color_by='subject'  
**After**: Black labels now appear for Axial and Coronal views

---

### 2. DTS Coloring (color_by='dts')
**Changed text label colors and visibility**

- **Axial View**: Changed "Right Breast" and "Left Breast" labels to **black** color (was blue/green)
- **Coronal View**: Changed "Right Breast" and "Left Breast" labels to **black** color (was blue/green)
- **Sagittal View**: **Removed** text labels (previously had blue/green labels)

**Before**: All views had blue/green labels  
**After**: Axial/Coronal have black labels; Sagittal has no labels

---

### 3. Breast Coloring (color_by='breast', default)
**Improved text label positioning for Sagittal view**

- **Axial View**: "Right Breast" (blue) and "Left Breast" (green) - unchanged
- **Coronal View**: "Right Breast" (blue) and "Left Breast" (green) - unchanged
- **Sagittal View**: 
  - "Right Breast" (blue) - **moved further to the right** (25% from left edge)
  - "Left Breast" (green) - **moved further to the left** (75% from left edge)

**Before**: Labels positioned at mean of data points (could be too close together)  
**After**: Labels positioned at fixed positions for better separation

---

## Code Implementation

### Right Breast Label Logic
```python
if color_by == 'breast':
    # Blue color, with special positioning for Sagittal
    if plane_name == 'Sagittal':
        right_x_pos = lims[0] + (lims[1] - lims[0]) * 0.25  # 25% from left
    else:
        right_x_pos = np.mean(base_right[:, axis_x_idx])
    # Display blue text
    
elif color_by == 'dts':
    # Black color for Axial/Coronal only
    if plane_name != 'Sagittal':
        right_x_pos = np.mean(base_right[:, axis_x_idx])
        # Display black text
    # No text for Sagittal
    
elif color_by == 'subject':
    # Black color for Axial/Coronal only
    if plane_name != 'Sagittal':
        right_x_pos = np.mean(base_right[:, axis_x_idx])
        # Display black text
    # No text for Sagittal
```

### Left Breast Label Logic
```python
if color_by == 'breast':
    # Green color, with special positioning for Sagittal
    if plane_name == 'Sagittal':
        left_x_pos = lims[0] + (lims[1] - lims[0]) * 0.75  # 75% from left
    else:
        left_x_pos = np.mean(base_left[:, axis_x_idx])
    # Display green text
    
elif color_by == 'dts':
    # Black color for Axial/Coronal only
    if plane_name != 'Sagittal':
        left_x_pos = np.mean(base_left[:, axis_x_idx])
        # Display black text
    # No text for Sagittal
    
elif color_by == 'subject':
    # Black color for Axial/Coronal only
    if plane_name != 'Sagittal':
        left_x_pos = np.mean(base_left[:, axis_x_idx])
        # Display black text
    # No text for Sagittal
```

---

## Summary Table

| Coloring Scheme | Axial View | Coronal View | Sagittal View |
|----------------|------------|--------------|---------------|
| **breast** (default) | Blue "Right Breast"<br>Green "Left Breast" | Blue "Right Breast"<br>Green "Left Breast" | Blue "Right Breast" (moved right)<br>Green "Left Breast" (moved left) |
| **dts** | **Black** "Right Breast"<br>**Black** "Left Breast" | **Black** "Right Breast"<br>**Black** "Left Breast" | **No labels** |
| **subject** | **Black** "Right Breast" ✨ NEW<br>**Black** "Left Breast" ✨ NEW | **Black** "Right Breast" ✨ NEW<br>**Black** "Left Breast" ✨ NEW | **No labels** |

✨ = Newly added functionality

---

## Files Modified

- **File**: `analysis.py`
- **Function**: `plot_vectors_rel_sternum()`
- **Lines Modified**: ~790-870 (Right Breast labels), ~870-950 (Left Breast labels)

---

## Testing

Created test script: `test_vectors_rel_sternum_labels.py`

### Expected Output Files
- `Vectors_rel_sternum_Axial.png` (breast coloring)
- `Vectors_rel_sternum_Coronal.png` (breast coloring)
- `Vectors_rel_sternum_Sagittal.png` (breast coloring)
- `Vectors_rel_sternum_Axial_DTS.png` (DTS coloring)
- `Vectors_rel_sternum_Coronal_DTS.png` (DTS coloring)
- `Vectors_rel_sternum_Sagittal_DTS.png` (DTS coloring)
- `Vectors_rel_sternum_Axial_by_subject.png` (subject coloring)
- `Vectors_rel_sternum_Coronal_by_subject.png` (subject coloring)
- `Vectors_rel_sternum_Sagittal_by_subject.png` (subject coloring)

---

## Benefits

1. **Improved Clarity**: Subject-colored plots now have breast identification labels
2. **Consistency**: DTS and subject coloring both use black labels in Axial/Coronal views
3. **Cleaner Sagittal Views**: Removed redundant labels from DTS and subject modes in sagittal
4. **Better Positioning**: Sagittal breast-colored labels are better separated
5. **Professional Appearance**: Appropriate label colors for each visualization mode

---

## Status

✅ **COMPLETE AND READY FOR USE**

All changes implemented and code validated. No errors introduced.

Date: January 19, 2026
