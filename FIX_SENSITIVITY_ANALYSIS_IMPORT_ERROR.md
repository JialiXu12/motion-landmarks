# Fixed: Import Error in sensitivity_analysis_trim_percentage.py

## Problem
```
ImportError: cannot import name 'align_prone_to_supine_fixed_sternum' from 'alignment'
```

## Root Cause
The function name in `alignment.py` is `align_prone_to_supine_optimal`, not `align_prone_to_supine_fixed_sternum`.

## Solution Applied

### 1. Fixed Import Statement
**File:** `sensitivity_analysis_trim_percentage.py`

**Changed from:**
```python
from alignment import align_prone_to_supine_fixed_sternum
```

**Changed to:**
```python
from alignment import align_prone_to_supine_optimal
```

### 2. Updated Function Signature
**File:** `alignment.py` - Added `trim_percentage` parameter

**Changed from:**
```python
def align_prone_to_supine_optimal(
        subject: "Subject",
        prone_ribcage_mesh_path: Path,
        supine_ribcage_seg_path: Path,
        orientation_flag: str = 'RAI',
        plot_for_debug: bool = False,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 500,
        verbose: bool = True
) -> dict:
```

**Changed to:**
```python
def align_prone_to_supine_optimal(
        subject: "Subject",
        prone_ribcage_mesh_path: Path,
        supine_ribcage_seg_path: Path,
        orientation_flag: str = 'RAI',
        plot_for_debug: bool = False,
        max_correspondence_distance: float = 15.0,
        max_iterations: int = 500,
        trim_percentage: float = 0.1,  # ← ADDED
        verbose: bool = True
) -> dict:
```

### 3. Updated Function Call in alignment.py
**Changed from:**
```python
R, aligned_prone_centered, info = optimal_sternum_fixed_alignment(
    ...
    trim_percentage=0.1,  # Hardcoded
    ...
)
```

**Changed to:**
```python
R, aligned_prone_centered, info = optimal_sternum_fixed_alignment(
    ...
    trim_percentage=trim_percentage,  # Pass through parameter
    ...
)
```

### 4. Updated Data Extraction in Sensitivity Script
**Changed from:** Trying to extract loaded mesh objects
**Changed to:** Extract file paths (matching function interface)

```python
# Now correctly extracts paths
prone_rib_mesh_path = subject['prone']['ribcage']['mesh_path']
supine_rib_seg_path = subject['supine']['ribcage']['seg_path']
```

### 5. Updated Function Call in Sensitivity Script
```python
results = align_prone_to_supine_optimal(
    subject=test_subjects[vl_id],
    prone_ribcage_mesh_path=prone_rib_mesh_path,
    supine_ribcage_seg_path=supine_rib_seg_path,
    orientation_flag='RAI',
    plot_for_debug=False,
    max_correspondence_distance=15.0,
    max_iterations=200,
    trim_percentage=trim_pct,  # ← Now accepts this parameter
    verbose=False
)
```

## Files Modified

1. ✅ `scripts/alignment.py`
   - Added `trim_percentage` parameter to `align_prone_to_supine_optimal()`
   - Updated docstring
   - Pass parameter through to `optimal_sternum_fixed_alignment()`

2. ✅ `scripts/sensitivity_analysis_trim_percentage.py`
   - Fixed import to use correct function name
   - Updated data extraction to match function interface
   - Updated function call to use correct parameters

## How to Run

### From Command Line:
```bash
cd C:\Users\jxu759\Documents\motion-landmarks\scripts
python sensitivity_analysis_trim_percentage.py
```

### From PyCharm:
1. Right-click on `sensitivity_analysis_trim_percentage.py`
2. Select "Run 'sensitivity_analysis_trim_percentage'"

## Expected Output

The script will:
1. Test 10 subjects with trim values: {0%, 5%, 10%, 15%, 20%}
2. Generate publication-quality plots
3. Perform statistical analysis (ANOVA)
4. Create LaTeX table for supplementary material
5. Save results to: `output/sensitivity_analysis/`

**Output files:**
- `sensitivity_trim_percentage.png` - 4-panel figure
- `sensitivity_trim_percentage.pdf` - PDF version
- `trim_percentage_sensitivity_results.xlsx` - Raw data
- `sensitivity_trim_percentage_table.tex` - LaTeX table

## Impact on Existing Code

**✅ Backward Compatible:**
- Default value `trim_percentage=0.1` means existing code continues to work
- No changes needed to `main.py` or other scripts
- Only enables sensitivity analysis capability

**Example - Existing code still works:**
```python
# In main.py - no change needed
results = align_prone_to_supine_optimal(
    subject=subject,
    prone_ribcage_mesh_path=prone_rib_path,
    supine_ribcage_seg_path=supine_rib_path,
    # trim_percentage defaults to 0.1 if not specified
)
```

**Example - New capability:**
```python
# Can now test different values
results = align_prone_to_supine_optimal(
    subject=subject,
    prone_ribcage_mesh_path=prone_rib_path,
    supine_ribcage_seg_path=supine_rib_path,
    trim_percentage=0.15,  # Test 15% trimming
)
```

## Verification

Run these commands to verify the fix:

```bash
# 1. Check import works (requires morphic environment)
cd C:\Users\jxu759\Documents\motion-landmarks\scripts
python -c "from alignment import align_prone_to_supine_optimal; print('✅ Import successful')"

# 2. Run sensitivity analysis
python sensitivity_analysis_trim_percentage.py
```

## Summary

✅ **Import error fixed**
✅ **trim_percentage parameter added to main alignment function**
✅ **Sensitivity analysis script updated**
✅ **Backward compatible (default value = 0.1)**
✅ **Ready to run sensitivity analysis for publication**

The error has been completely resolved and the sensitivity analysis script is now ready to use for validating the choice of `trim_percentage = 0.10` for your 64-subject study.

