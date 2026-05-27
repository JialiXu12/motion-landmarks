# DETAILED FUNCTION-BY-FUNCTION COMPARISON WITH LINE NUMBERS
Generated: January 26, 2026

## SUMMARY OF ALL FUNCTIONS

### ✅ IDENTICAL FUNCTIONS (4 functions - 100% match)
1. **analyze_3d_stability**
   - Backup: Lines 2356-2381 (26 lines)
   - Current: Lines 1749-1774 (26 lines)
   - Status: 100% IDENTICAL

2. **calculate_clock_position**
   - Backup: Lines 2438-2473 (36 lines)
   - Current: Lines 1831-1866 (36 lines)
   - Status: 100% IDENTICAL

3. **circular_mean_angle**
   - Backup: Lines 2414-2437 (24 lines)
   - Current: Lines 1807-1830 (24 lines)
   - Status: 100% IDENTICAL

4. **plot_vectors_for_vl81**
   - Backup: Lines 926-1054 (129 lines)
   - Current: Lines 470-598 (129 lines)
   - Status: 100% IDENTICAL

---

### 🟡 MINOR DIFFERENCES (5 functions - 90-99% similar)

#### 5. **read_data** - 90.5% similar
- Backup: Lines 467-489 (23 lines)
- Current: Lines 23-41 (19 lines)
- **Change:** -4 lines

**Differences:**
- Lines 486-489 in backup: Section header removed (cleanup)
```python
# REMOVED from current:
# ==============================================================================
# SECTION 2: STATISTICAL ANALYSIS FUNCTIONS
# ==============================================================================
```

---

#### 6. **investigate_proximity_effect** - 91.8% similar
- Backup: Lines 894-925 (32 lines)
- Current: Lines 441-469 (29 lines)
- **Change:** -3 lines

**Differences:**
- Line 895 vs 442: Comment text changed
- Lines 922-924 in backup: Section header removed
```python
# REMOVED from current:
# ==============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS - STERNUM REFERENCE FRAME
# ==============================================================================
```

---

#### 7. **plot_bmi_correlations** - 94.5% similar
- Backup: Lines 839-893 (55 lines)
- Current: Lines 386-440 (55 lines)
- **Change:** 0 lines (same length)

**Differences:**
- Line 866 vs 413: Added `fontweight='bold'` to title
  ```python
  # Backup:
  axes[0].set_title('Impact of BMI on Difference in DTN\n(β = -4.01, p < 0.001)', fontsize=14)
  # Current:
  axes[0].set_title('Impact of BMI on Difference in DTN\n(β = -4.01, p < 0.001)', fontsize=14, fontweight='bold')
  ```

- Line 879 vs 426: Added `fontweight='bold'` to title
- Line 887 vs 434: Path changed from `"figs" / "v5" / "BMI"` to `"figs" / "BMI"`

---

#### 8. **perform_repeated_measures_analysis** - 96.6% similar
- Backup: Lines 676-838 (163 lines)
- Current: Lines 228-385 (158 lines)
- **Change:** -5 lines

**Differences:**
- Line 739 vs 291: Unicode character difference
  ```python
  # Backup: print("eta² = partial eta squared (effect size)")
  # Current: print("η² = partial eta squared (effect size)")
  ```
- Lines 797, 800 vs 349, 352: Warning symbol changed
- Lines 834-838: Section header removed

---

#### 9. **plot_nipple_relative_landmarks** - 98.6% similar
- Backup: Lines 2213-2355 (143 lines)
- Current: Lines 1610-1748 (139 lines)
- **Change:** -4 lines

**Differences:**
- Lines 2352-2355 in backup: Section header removed
```python
# REMOVED from current:
# ==============================================================================
# SECTION 6: CLOCK POSITION ANALYSIS FUNCTIONS
# ==============================================================================
```

---

#### 10. **plot_sagittal_dual_axes** - 97.6% similar
- Backup: Lines 1914-2212 (299 lines)
- Current: Lines 1315-1609 (295 lines)
- **Change:** -4 lines

**Differences:**
- Line 2008 vs 1409: Added `fontweight='bold'` to suptitle
- Line 2200 vs 1601: Path changed from `"figs" / "v5" / "landmark vectors"` to `"figs" / "landmark vectors"`
- Lines 2206-2212 vs 1607-1609: Section header removed and print message changed

---

#### 11. **perform_two_group_analysis** - 98.8% similar
- Backup: Lines 591-675 (85 lines)
- Current: Lines 143-227 (85 lines)
- **Change:** 0 lines

**Differences:**
- Line 609 vs 161: Warning symbol changed (⚠️ → different encoding)

---

#### 12. **perform_group_analysis** - 99.0% similar
- Backup: Lines 490-590 (101 lines)
- Current: Lines 42-142 (101 lines)
- **Change:** 0 lines

**Differences:**
- Line 501 vs 53: Warning symbol encoding difference

---

### 🟠 SIGNIFICANT DIFFERENCES (2 functions)

#### 13. **plot_vectors_rel_sternum** - 84.4% similar ⚠️
- Backup: Lines 1055-1493 (439 lines)
- Current: Lines 599-972 (374 lines)
- **Change:** **-65 lines** (15% reduction!)

**Major Differences:**

1. **Function signature changed:**
   ```python
   # Backup:
   def plot_vectors_rel_sternum(df_ave, color_by='breast', vl_id=None, data_type='landmarks', include_dual_sagittal=False):
   
   # Current:
   def plot_vectors_rel_sternum(df_ave, color_by='breast', vl_id=None):
   ```
   - Removed `data_type` parameter
   - Removed `include_dual_sagittal` parameter

2. **Removed nipple plotting capability:**
   - Lines 1095-1140 in backup: Full nipple position extraction logic REMOVED
   - Current only handles landmarks

3. **Simplified helper function:**
   ```python
   # Backup:
   def get_points_and_vectors(sub_df, is_left_breast=True):
   
   # Current:
   def get_points_and_vectors(sub_df):
   ```
   - Removed `is_left_breast` parameter

4. **Removed dual sagittal view:**
   - Lines 1472-1493 in backup: Call to `_plot_dual_sagittal_view_sternum()` REMOVED

5. **Styling changes:**
   - Added `fontweight='bold'` to multiple text labels
   - Changed save path from `"v5" / "landmark vectors"` to `"landmark vectors"`

**Impact:** MODERATE - Functionality removed but may have been refactored elsewhere

---

#### 14. **analyse_clock_position_rotation** - 52.0% similar ⚠️⚠️
- Backup: Lines 2474-3099 (626 lines)
- Current: Lines 1867-3148 (1282 lines)
- **Change:** **+656 lines** (105% growth!)

**Major Differences:**

This function was MASSIVELY EXPANDED. Key changes:

1. **Path changes:** `"v5" / "clock_analysis"` → `"clock_analysis"`

2. **Visualization improvements:**
   - Line 2711 → 2104: Added `fontweight='bold'` to titles
   - Lines 2835-2842 → 2228-2233: Changed from `annotate()` to `arrow()` for polar arrows
   - Line 2881 → 2272: Figure size increased (14,6) → (14,7)

3. **Enhanced polar plots:**
   - Lines 2896-2900 → 2287-2293: Individual points styling improved (alpha, edgecolors, linewidths)
   - Lines 2910-2924 → 2303-2307: Mean shift arrow rendering changed
   - Multiple label and legend improvements

4. **Text improvements:**
   - "Clockwise" → "CW", "Counterclockwise" → "CCW" (abbreviated)
   - Enhanced titles with `fontweight='bold'` and color coding

5. **NEW CONTENT (lines 3096-3099 → 2505-3148):**
   - **ENTIRE MAIN EXECUTION BLOCK ADDED** (640+ lines!)
   - This explains the massive size increase
   - The backup ends with a section header, current continues with full execution code

**Impact:** MAJOR ENHANCEMENT - Significant improvements + main execution added

---

## ❌ MISSING FUNCTIONS (4 functions - 1,266 lines total)

### 1. **_plot_dual_sagittal_view_sternum** 
- Backup: Lines 1494-1676 (183 lines)
- Status: COMPLETELY MISSING
- Type: Helper function (private)
- Purpose: Creates dual sagittal view plots for sternum reference
- **Action:** Likely integrated into other functions (acceptable)

---

### 2. **plot_3panel_anatomical_views**
- Backup: Lines 1677-1913 (237 lines)
- Status: COMPLETELY MISSING
- Purpose: Creates 3-panel anatomical visualization
- **Action:** REPLACED by `plot_3panel_displacement_mechanism` (342 lines) in current

---

### 3. **plot_anatomical_correlation_matrix** ⚠️⚠️⚠️
- Backup: Lines 3100-3249 (150 lines)
- Status: **COMPLETELY MISSING - CRITICAL ANALYSIS**
- Purpose: 
  - Calculates biomechanical deltas (Skin, Rib, Nipple distance changes)
  - Performs Pearson correlation analysis
  - Generates correlation heatmap
  - Identifies compression, tethering, pendulum effects
- Output: `correlation_matrix_anatomical.png`
- **Action:** ❌ **SHOULD BE RESTORED** - Essential biomechanical analysis

---

### 4. **compare_reference_frames** ⚠️⚠️⚠️
- Backup: Lines 3250-3945 (696 lines!)
- Status: **COMPLETELY MISSING - CRITICAL VALIDATION**
- Purpose:
  - Compares sternum vs nipple reference frames
  - Performs statistical validation (t-test, Cohen's d, Bland-Altman)
  - Creates 3-panel comparison visualization
  - Provides clinical recommendations
- Output: `reference_frame_comparison.png`
- **Action:** ❌ **SHOULD BE RESTORED** - Essential methodological validation

---

## ✅ NEW FUNCTIONS (1 function)

### **plot_3panel_displacement_mechanism**
- Current: Lines 973-1314 (342 lines)
- Status: NEW (replaces `plot_3panel_anatomical_views`)
- Purpose: Enhanced 3-panel displacement visualization
- **Action:** ✅ Keep - Appears to be improved version

---

## 📊 STATISTICS SUMMARY

| Category | Count | Lines | Status |
|----------|-------|-------|--------|
| Identical | 4 | 215 | ✅ Keep |
| Minor differences | 8 | 1,060 → 993 | ✅ Keep |
| Significant changes | 2 | 1,065 → 1,656 | ⚠️ Review |
| Missing (should restore) | 2 | **846** | ❌ **RESTORE** |
| Missing (replaced/integrated) | 2 | 420 | ✅ OK |
| New functions | 1 | 342 | ✅ Keep |

---

## 🎯 CRITICAL ACTIONS REQUIRED

### ✅ RESTORE IMMEDIATELY:
1. **plot_anatomical_correlation_matrix()** (150 lines)
   - Location in backup: Lines 3100-3249
   - Add to current: After line 3148 (new Section 7)

2. **compare_reference_frames()** (696 lines)
   - Location in backup: Lines 3250-3945
   - Add to current: After correlation matrix function

### ⚠️ INVESTIGATE:
3. **plot_vectors_rel_sternum()** (lost 65 lines)
   - Verify nipple plotting capability is available elsewhere
   - Verify dual sagittal view is available via `plot_sagittal_dual_axes()`

### ✅ KEEP AS IS:
- All identical functions
- All minor differences (mostly cleanup and style improvements)
- New `plot_3panel_displacement_mechanism` function
- Enhanced `analyse_clock_position_rotation` function

---

## 📁 FILES GENERATED

1. `function_comparison_detailed.txt` - Full detailed comparison (this file)
2. `compare_functions.py` - Comparison script (reusable)

---

**END OF DETAILED COMPARISON**
