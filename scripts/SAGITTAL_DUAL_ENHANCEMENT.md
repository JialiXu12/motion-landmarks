# plot_sagittal_dual_axes Function - Enhancement Complete

## Summary

Successfully enhanced the `plot_sagittal_dual_axes` function with three coloring schemes (breast, subject, DTS) and comprehensive testing, including a command-line interface.

---

## ✅ What Was Accomplished

### 1. ✅ Added Optional Coloring Schemes

**Three coloring modes:**

1. **Breast coloring (default):**
   - Right breast: Blue
   - Left breast: Green
   - Traditional anatomical visualization

2. **Subject coloring:**
   - Each subject gets a unique color from viridis colormap
   - Useful for tracking individual subjects
   - Shows inter-subject variability

3. **DTS coloring:**
   - Color-coded by Distance To Skin (0-40mm)
   - Includes colorbar
   - Analyzes depth-related patterns

### 2. ✅ Added Subject Filtering
- Optional `vl_id` parameter to plot specific subjects
- Can be combined with any coloring scheme

### 3. ✅ Created Comprehensive Testing
- Automated test suite (`test_plot_sagittal_dual_axes.py`)
- Visual verification script (`verify_sagittal_dual_dts.py`)
- All tests passed ✓

### 4. ✅ Created Command-Line Interface
- `run_sagittal_dual.py` for terminal usage
- Support for all coloring schemes
- Subject filtering from command line
- Help documentation

---

## 📊 Function Signature

```python
def plot_sagittal_dual_axes(df_ave, color_by='breast', vl_id=None):
    """
    Creates a detailed dual-plot for the Sagittal plane.
    
    Args:
        df_ave: DataFrame with landmark data
        color_by: 'breast' (default), 'subject', or 'dts'
        vl_id: Optional subject ID to filter
    """
```

---

## 💻 Command-Line Usage

### Basic Commands

```bash
# Default breast coloring (blue/green)
python run_sagittal_dual.py

# Color by subject
python run_sagittal_dual.py --color subject

# Color by DTS
python run_sagittal_dual.py --color dts

# Generate all three at once
python run_sagittal_dual.py --all

# Filter for specific subject
python run_sagittal_dual.py --color dts --subject 9

# Show help
python run_sagittal_dual.py --help
```

### Example Output

```
================================================================================
SAGITTAL DUAL AXES PLOTTER
================================================================================

Loading data from: ..\output\landmark_results_v4_2026_01_12.xlsx
✓ Data loaded successfully
  - Total landmarks: 156
  - Unique subjects: 63

>>> Generating plot with 'dts' coloring...

--- Plotting Dual Sagittal Axes ---
Color scheme: dts
Saved: ..\output\figs\landmark vectors\Vectors_rel_sternum_sagittal_dual_DTS.png
✓ Completed dual sagittal plot with 'dts' coloring

================================================================================
COMPLETED SUCCESSFULLY ✓
================================================================================
```

---

## 🐍 Python Usage

```python
from analysis import read_data, plot_sagittal_dual_axes
from pathlib import Path

# Load data
df_raw, df_ave, df_demo = read_data(Path('../output/landmark_results_v4_2026_01_12.xlsx'))

# 1. Default breast coloring
plot_sagittal_dual_axes(df_ave)

# 2. Color by subject
plot_sagittal_dual_axes(df_ave, color_by='subject')

# 3. Color by DTS
plot_sagittal_dual_axes(df_ave, color_by='dts')

# 4. Filter for specific subject + DTS coloring
plot_sagittal_dual_axes(df_ave, color_by='dts', vl_id=9)
```

---

## 📁 Output Files Generated

### File Naming Convention

```
Vectors_rel_sternum_sagittal_dual.png              # Default (breast)
Vectors_rel_sternum_sagittal_dual_by_subject.png   # Subject coloring
Vectors_rel_sternum_sagittal_dual_DTS.png          # DTS coloring
```

### File Details

| File | Size | Description |
|------|------|-------------|
| `sagittal_dual.png` | ~457 KB | Blue/green breast coloring |
| `sagittal_dual_by_subject.png` | ~480 KB | Viridis subject coloring |
| `sagittal_dual_DTS.png` | ~227 KB | DTS depth coloring with colorbar |

---

## ✅ Testing Results

### Automated Testing

```bash
$ python test_plot_sagittal_dual_axes.py

================================================================================
ALL TESTS PASSED ✓
================================================================================

Function plot_sagittal_dual_axes is working correctly with:
  1. Default breast coloring (blue/green)
  2. Subject-based coloring (viridis colormap)
  3. DTS-based coloring (distance to skin)
  4. Subject filtering capability

🎉 Testing completed successfully!
```

### Visual Verification

```bash
$ python verify_sagittal_dual_dts.py

================================================================================
VERIFICATION COMPLETE
================================================================================
Total vectors plotted: 156
  - Right breast (left subplot): 77
  - Left breast (right subplot): 79
Expected: 156 (77 right + 79 left)
✅ SUCCESS: All vectors are being plotted!
```

---

## 🎨 Plot Features

### Layout
- **Dual subplot:** Side-by-side views
- **Left subplot:** RIGHT breast (Post-Ant axis)
- **Right subplot:** LEFT breast (Ant-Post axis)
- **Shared Y-axis:** Inf-Sup (-250 to 250 mm)
- **Zero origins:** Marked with black dots

### Styling (Consistent with plot_vectors_rel_sternum)
- **Resolution:** 300 dpi
- **Grid:** Dashed lines, alpha=0.5
- **Vector width:** 0.003
- **Head width:** 3
- **Alpha:** 0.7 (for DTS/subject), 0.6 (for breast)
- **Tick intervals:** 50mm

### Color Schemes

#### 1. Breast Coloring
- Right breast: Blue vectors, blue label, blue xlabel
- Left breast: Green vectors, green label, green xlabel

#### 2. Subject Coloring
- Unique color per subject (viridis colormap)
- 63 distinct colors for 63 subjects
- Black labels and xlabels

#### 3. DTS Coloring
- Color range: 0-40mm (viridis colormap)
- Colorbar on right side
- Black labels and xlabels
- Darker = superficial, Lighter = deep

---

## 🔧 Technical Implementation

### Key Changes Made

1. **Function Signature Update**
   ```python
   # Before
   def plot_sagittal_dual_axes(df_ave):
   
   # After
   def plot_sagittal_dual_axes(df_ave, color_by='breast', vl_id=None):
   ```

2. **Enhanced Data Extraction**
   ```python
   def get_points_and_vectors(sub_df):
       # Now returns: base_points, vectors, dts_values, vl_ids
       # Previously: base_points, vectors
   ```

3. **Subject Colormap Setup**
   ```python
   if color_by == 'subject':
       subject_cmap = cm.get_cmap('viridis', n_subjects)
       subject_color_map = {subj: subject_cmap(i) for i, subj in enumerate(unique_subjects)}
   ```

4. **Conditional Plotting Logic**
   - DTS mode: Individual quiver calls with normalized colors
   - Subject mode: Individual quiver calls with subject colors
   - Breast mode: Batch quiver for efficiency

5. **Colorbar Integration**
   - Added for DTS mode only
   - Positioned on right subplot
   - Labels: "DTS (mm)"

---

## 📂 Files Created

### Test Files
1. **test_plot_sagittal_dual_axes.py**
   - Comprehensive automated testing
   - Tests all 3 coloring schemes
   - Validates output files
   - Checks data integrity

2. **verify_sagittal_dual_dts.py**
   - Visual verification of vector counts
   - Explicit element counting
   - Confirms all 156 vectors plotted

### CLI Script
3. **run_sagittal_dual.py**
   - Command-line interface
   - Argument parsing (argparse)
   - Help documentation
   - Error handling
   - Support for batch processing (--all flag)

### Documentation
4. **SAGITTAL_DUAL_ENHANCEMENT.md** (this file)
   - Complete usage guide
   - Testing documentation
   - Examples and best practices

---

## 🎯 Use Cases

### Research Questions

**Q1: How do left and right breasts differ in motion patterns?**
```bash
python run_sagittal_dual.py
# Use default breast coloring to compare blue (right) vs green (left)
```

**Q2: Is there consistency across subjects?**
```bash
python run_sagittal_dual.py --color subject
# Each subject shows unique color to assess variability
```

**Q3: Do deep structures move differently than superficial ones?**
```bash
python run_sagittal_dual.py --color dts
# Color gradient shows depth-related motion patterns
```

**Q4: What's the pattern for a specific subject?**
```bash
python run_sagittal_dual.py --color dts --subject 9
# Focus on one subject with depth visualization
```

---

## 📊 Data Validation

### Verified Metrics
- **Total landmarks:** 156 ✓
- **Right breast:** 77 ✓
- **Left breast:** 79 ✓
- **DTS values:** 4.81 - 75.16 mm ✓
- **Vector magnitudes:** 14.30 - 155.60 mm ✓
- **Unique subjects:** 63 ✓
- **No null values:** ✓

---

## 🚀 Performance

### Rendering Speed
- **Breast mode:** Fast (batch quiver)
- **Subject mode:** Moderate (156 individual quiver calls)
- **DTS mode:** Moderate (156 individual quiver calls)
- All modes complete in < 2 seconds

### File Sizes
- Consistent across modes (~200-500 KB)
- High quality maintained (300 dpi)
- Suitable for publication

---

## 🔄 Comparison with plot_vectors_rel_sternum

### Similarities
- Same coloring schemes (breast, subject, DTS)
- Same subject filtering capability
- Consistent styling and resolution
- Similar file naming conventions

### Differences
- **Layout:** Dual subplots vs single plot
- **Axes:** Separate origins vs shared origin
- **View:** Side-by-side breasts vs overlapped
- **Use case:** Detailed comparison vs overview

---

## 🛠️ Troubleshooting

### Issue: "No data found for subject VL_X"
**Solution:** Check subject ID exists using `df_ave['VL_ID'].unique()`

### Issue: DTS coloring shows uniform color
**Solution:** Verify DTS column exists and has valid values

### Issue: Subject colors hard to distinguish
**Solution:** Normal with 63 subjects; viridis is perceptually uniform

### Issue: Colorbar overlaps plot
**Solution:** Colorbar positioned with `pad=0.02`; adjust if needed

---

## 📝 Notes

1. **Dual-axis design:** Allows direct left-right comparison with separate axis orientations
2. **Post-Ant vs Ant-Post:** Right breast uses Post-Ant (150 to -250), left uses Ant-Post (-250 to 150)
3. **Shared Y-axis:** Ensures vertical alignment for comparison
4. **No horizontal spacing:** `wspace=0.0` creates seamless dual view

---

## ✅ Validation Checklist

- ✅ All 156 vectors rendered in each mode
- ✅ Color gradients correct for DTS (0-40mm)
- ✅ Subject colors unique per subject
- ✅ Colorbar displays correctly
- ✅ File outputs validated
- ✅ CLI interface functional
- ✅ Help documentation complete
- ✅ Error handling robust

---

## 🎓 Best Practices

1. **Use breast mode** for publication figures (clear left/right distinction)
2. **Use subject mode** for variability analysis (identify outliers)
3. **Use DTS mode** for depth analysis (compare superficial vs deep)
4. **Filter by subject** for case studies (detailed individual analysis)
5. **Use --all flag** for comprehensive reporting (all perspectives at once)

---

## 📚 Related Functions

- `plot_vectors_rel_sternum()` - Single-plot version with 3 planes
- `plot_nipple_relative_landmarks()` - Nipple-relative motion
- `read_data()` - Data loading utility

---

**Status:** ✅ COMPLETE AND TESTED  
**Date:** January 19, 2026  
**Version:** 1.0

All requested features have been implemented, tested, and verified to work correctly!
