# Analysis File Comparison Summary
**Date:** January 26, 2026  
**Backup File:** `analysis_backup_20260123_232525.py` (3945 lines, 174,629 bytes)  
**Current File:** `analysis.py` (3148 lines, 140,261 bytes)  
**Difference:** 797 lines removed (20% smaller)

---

## 🔍 KEY FINDINGS

### **MAJOR LOSS: Missing Function**
The backup file contains **1 critical function** that is NOT in the current file:

#### ❌ **MISSING: `plot_anatomical_correlation_matrix(df)`**
- **Purpose:** Advanced biomechanical analysis of landmark displacement drivers
- **Features:**
  - Calculates Delta values (Supine - Prone) for:
    - Delta_Skin: Change in distance to skin
    - Delta_Rib: Change in distance to rib cage
    - Delta_Nipple: Change in distance to nipple
  - Generates correlation matrix heatmap analyzing relationships between:
    - Displacement vectors (vx, vy, vz, magnitude)
    - Biomechanical changes (compression, expansion, tethering)
  - Prints statistical analysis including:
    - Mean and SD of all deltas
    - Key correlation coefficients
    - Strength and direction of relationships
  - Creates publication-quality visualization with:
    - Custom color map (coolwarm: red/blue/white)
    - Triangular mask for cleaner display
    - Readable labels for anatomical terms
    - Output saved as `anatomical_correlation_matrix.png`

**⚠️ This is a SIGNIFICANT ANALYSIS FUNCTION that should likely be restored.**

---

## 📊 FILE STRUCTURE COMPARISON

### Backup File Has:
- **Section 7:** Advanced Biomechanical Analysis (completely missing in current)
- `plot_anatomical_correlation_matrix()` function (~150 lines)
- Additional error handling code fragments at the beginning
- Incomplete code snippets (fragments of print statements)

### Current File:
- Cleaner structure without code fragments
- All other 15 functions are present and appear intact
- Missing the correlation matrix analysis entirely

---

## 🔬 DETAILED BREAKDOWN

### Functions Present in BOTH Files (15 total):
1. ✅ `plot_vectors_rel_sternum`
2. ✅ `plot_sagittal_dual_axes`
3. ✅ `plot_vectors_for_vl81_combined`
4. ✅ `plot_nipple_relative_landmarks`
5. ✅ `get_nipple_relative_points_and_vectors`
6. ✅ `plot_3panel_anatomical_views`
7. ✅ `compute_clock_positions`
8. ✅ `analyse_clock_position_rotation`
9. ✅ `perform_repeated_measures_analysis`
10. ✅ All statistical analysis functions
11. ✅ All plotting helper functions

### ❌ Function ONLY in Backup:
- `plot_anatomical_correlation_matrix` - **LOST IN CURRENT VERSION**

---

## 🧬 WHAT THE MISSING FUNCTION DOES

The `plot_anatomical_correlation_matrix` function answers the research question:
**"WHY do landmarks move the way they do?"**

### Key Analysis Components:

1. **Delta Calculations** (Change from Prone → Supine):
   ```python
   Delta_Skin   = Distance_Supine - Distance_Prone  # Compression effect
   Delta_Rib    = Distance_Supine - Distance_Prone  # Chest wall interaction
   Delta_Nipple = Distance_Supine - Distance_Prone  # Tethering effect
   ```

2. **Correlation Analysis** (Pearson coefficients):
   - Effect variables: Total displacement, X/Y/Z vectors
   - Cause variables: Delta_Rib, Delta_Skin, Delta_Nipple, Initial depth
   - Identifies which anatomical changes drive landmark motion

3. **Visual Output**:
   - Heatmap showing correlation strength (-1 to +1)
   - Color coding: Red (positive), Blue (negative), White (no correlation)
   - Statistical annotations with r-values
   - Professional publication-ready format

4. **Clinical Insights**:
   - Compression vs expansion effects
   - Tissue tethering mechanisms
   - Pendulum effects for deeper structures
   - Biomechanical drivers of deformation

---

## 🎯 RECOMMENDATION

**ACTION REQUIRED:** The backup file contains important analysis code that was lost.

### Option 1: Restore the Missing Function
Copy the `plot_anatomical_correlation_matrix()` function from the backup file to the current file under a new **Section 7: Advanced Biomechanical Analysis**.

### Option 2: Verify This Was Intentional
If the correlation matrix analysis was deliberately removed (perhaps moved to another file), document the reason.

---

## 📝 ADDITIONAL OBSERVATIONS

### Backup File Issues:
- Starts with incomplete code fragments (print statements without context)
- Appears to have been saved mid-edit or during debugging
- Has some dangling statements at the top

### Current File:
- Cleaner, more organized structure
- All main analysis functions intact
- Missing advanced correlation analysis
- Appears to be a working version but incomplete

---

## 🚨 CONCLUSION

**The current `analysis.py` is missing ~800 lines of code, primarily:**
1. The complete `plot_anatomical_correlation_matrix()` function
2. Associated Section 7 documentation
3. Some error handling fragments (which may have been cleanup)

**RECOMMENDED ACTION:**  
Restore the `plot_anatomical_correlation_matrix()` function from the backup file to preserve this valuable biomechanical analysis capability.

---

## 📂 FILES ANALYZED
- **Backup:** `analysis_backup_20260123_232525.py` (January 23, 2026)
- **Current:** `analysis.py` (current working version)
- **This Report:** `DIFF_SUMMARY.md`
