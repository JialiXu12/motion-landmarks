# plot_vectors_rel_sternum Function - Usage Guide

## Overview
The `plot_vectors_rel_sternum` function has been enhanced with multiple coloring options to visualize landmark displacement vectors from prone to supine positions relative to the sternum.

## Function Signature
```python
def plot_vectors_rel_sternum(df_ave, color_by='breast', vl_id=None):
    """
    Plots displacement vectors (Prone -> Supine) for both breasts.
    
    Args:
        df_ave: DataFrame with landmark data
        color_by: Coloring scheme - 'breast' (default, blue/green), 
                  'subject' (color by VL_ID), or 'dts' (color by distance to skin)
        vl_id: Optional subject ID to filter data. If None, uses all subjects.
    """
```

## Features

### 1. Default Coloring (by breast)
Colors landmarks by breast side:
- **Blue**: Right breast
- **Green**: Left breast

**Usage:**
```python
plot_vectors_rel_sternum(df_ave, color_by='breast')
# or simply:
plot_vectors_rel_sternum(df_ave)
```

**Output files:**
- `Vectors_rel_sternum_Coronal.png`
- `Vectors_rel_sternum_Sagittal.png`
- `Vectors_rel_sternum_Axial.png`

---

### 2. Color by Subject
Colors each subject's landmarks with a unique color using the viridis colormap.
Useful for visualizing inter-subject variability and identifying patterns.

**Usage:**
```python
plot_vectors_rel_sternum(df_ave, color_by='subject')
```

**Output files:**
- `Vectors_rel_sternum_Coronal_by_subject.png`
- `Vectors_rel_sternum_Sagittal_by_subject.png`
- `Vectors_rel_sternum_Axial_by_subject.png`

**Features:**
- Automatically detects unique subjects from VL_ID column
- Assigns distinct colors from viridis colormap
- Shows number of unique subjects in console output

---

### 3. Color by DTS (Distance to Skin)
Colors landmarks based on their distance to skin (DTS) in prone position.
Useful for analyzing depth-related displacement patterns.

**Usage:**
```python
plot_vectors_rel_sternum(df_ave, color_by='dts')
```

**Output files:**
- `Vectors_rel_sternum_Coronal_DTS.png`
- `Vectors_rel_sternum_Sagittal_DTS.png`
- `Vectors_rel_sternum_Axial_DTS.png`

**Features:**
- Color scale: 0-40mm (viridis colormap)
- Includes colorbar showing DTS values
- Superficial landmarks (low DTS) appear darker
- Deep landmarks (high DTS) appear lighter

---

### 4. Subject Filtering
Filter data to plot landmarks for a specific subject only.
Can be combined with any coloring scheme.

**Usage:**
```python
# Plot only subject VL_9 with DTS coloring
plot_vectors_rel_sternum(df_ave, color_by='dts', vl_id=9)

# Plot only subject VL_20 with default breast coloring
plot_vectors_rel_sternum(df_ave, color_by='breast', vl_id=20)
```

---

## Anatomical Planes

All three coloring schemes generate plots for three anatomical planes:

1. **Coronal Plane** (X vs Z)
   - X-axis: Right-Left (mm)
   - Y-axis: Inf-Sup (mm)
   - View: Frontal view of both breasts

2. **Sagittal Plane** (Y vs Z)
   - X-axis: Ant-Post (mm)
   - Y-axis: Inf-Sup (mm)
   - View: Side view showing anterior-posterior motion

3. **Axial Plane** (X vs Y)
   - X-axis: Right-Left (mm)
   - Y-axis: Ant-Post (mm)
   - View: Top-down view

---

## Output Details

### Common Features (All Plots)
- Sternum marked at origin (0,0) with black dot
- 50mm tick intervals on all axes
- Axis limits: -250 to 250 mm
- Dashed grid (alpha=0.5)
- Reference lines through sternum
- High resolution (300 dpi)

### Plot Elements
- **Base points**: Prone landmark positions (shown as scatter points)
- **Vectors**: Arrows showing displacement from prone to supine
- **Origin**: Black dot at (0,0) representing sternum

---

## Testing

A comprehensive test file has been created: `test_plot_vectors_rel_sternum.py`

### Run Tests
```bash
cd C:\Users\jxu759\Documents\motion-landmarks\scripts
python test_plot_vectors_rel_sternum.py
```

### Test Coverage
1. ✅ Default breast coloring
2. ✅ Subject-based coloring
3. ✅ DTS-based coloring
4. ✅ Subject filtering
5. ✅ Output file validation

---

## Example Use Cases

### Research Questions

**Q1: How do landmarks move differently for superficial vs deep structures?**
```python
# Use DTS coloring to see depth-related patterns
plot_vectors_rel_sternum(df_ave, color_by='dts')
```

**Q2: Is there high inter-subject variability?**
```python
# Use subject coloring to visualize each subject
plot_vectors_rel_sternum(df_ave, color_by='subject')
```

**Q3: What's the motion pattern for a specific subject?**
```python
# Filter for subject and use DTS coloring
plot_vectors_rel_sternum(df_ave, color_by='dts', vl_id=9)
```

**Q4: What's the general left vs right breast difference?**
```python
# Use default breast coloring
plot_vectors_rel_sternum(df_ave, color_by='breast')
```

---

## Output Location
All plots are saved to:
```
C:\Users\jxu759\Documents\motion-landmarks\output\figs\landmark vectors\
```

---

## Technical Details

### Required DataFrame Columns
- `landmark ave prone transformed x/y/z`: Prone positions relative to sternum
- `landmark ave supine x/y/z`: Supine positions relative to sternum
- `landmark side (prone)`: 'LB' or 'RB'
- `Distance to skin (prone) [mm]`: For DTS coloring
- `VL_ID`: For subject coloring and filtering

### Coordinate System
- X (axis 0): Right/Left (positive = right, negative = left)
- Y (axis 1): Anterior/Posterior (positive = anterior, negative = posterior)
- Z (axis 2): Inferior/Superior (positive = superior, negative = inferior)

---

## Notes

1. **Colormap Choice**: The viridis colormap is used for both subject and DTS coloring as it's:
   - Perceptually uniform
   - Colorblind-friendly
   - Sequential (for DTS) or categorical (for subjects)

2. **Arrow Styling**:
   - Width: 0.003 (for quiver plots)
   - Head width: 3 (for arrow plots)
   - Alpha: 0.7 (semi-transparent)

3. **Performance**: Subject coloring may be slower for large datasets as it plots vectors individually rather than using quiver.

4. **Consistency**: All plots use consistent styling (font sizes, grid style, tick intervals) for easy comparison.

---

## Troubleshooting

### Issue: "No data found for subject VL_X"
- **Solution**: Check that the VL_ID exists in the dataset using `df_ave['VL_ID'].unique()`

### Issue: DTS coloring shows uniform color
- **Solution**: Verify that 'Distance to skin (prone) [mm]' column exists and has valid values

### Issue: Subject coloring uses only a few colors
- **Solution**: This is expected if you have many subjects (>20). The colormap cycles through colors.

---

## Changes from Previous Version

1. ✅ Added `color_by` parameter with three options
2. ✅ Added `vl_id` parameter for subject filtering
3. ✅ Implemented subject-specific colormap using viridis
4. ✅ Implemented DTS-based coloring with colorbar
5. ✅ Enhanced console output with status messages
6. ✅ Created comprehensive test suite
7. ✅ Consistent styling across all coloring schemes

---

## Contact & Support

For issues or questions about this function, please refer to the test file or check the main analysis.py documentation.

**Test File**: `test_plot_vectors_rel_sternum.py`
**Source File**: `analysis.py` (lines 598-918)

---

Last Updated: January 19, 2026
