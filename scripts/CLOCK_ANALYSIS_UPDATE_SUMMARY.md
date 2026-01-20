# Clock Position Rotation Analysis Update

## Summary of Changes

The `analyze_clock_position_rotation` function in `analysis.py` has been updated to use the existing `Time (prone)` and `Time (supine)` columns from the dataset instead of recalculating clock positions from coordinates.

---

## Technical Details

### 1. New Helper Function: `parse_clock_time`
A new utility function `parse_clock_time` was added to handle clock time parsing:
- Inputs: Time strings like `"2:30"`, `"9:00"`, `"central"`
- Outputs: 
  - `clock_hours`: Decimal hours (e.g., `2.5` for 2:30)
  - `angle_degrees`: Rotation degrees where 12 o'clock = 0°, moving clockwise (e.g., 3 o'clock = 90°)
- Handles edge cases like "central" or missing data (returns NaN)

### 2. Updated Analysis Logic
The `analyze_clock_position_rotation` function now:
1. Iterates through the data for Left vs Right breasts.
2. Uses `parse_clock_time` to extract angles from `Time (prone)` and `Time (supine)`.
3. Uses existing columns `Distance to nipple (prone) [mm]` and `Distance to nipple (supine) [mm]` directly.
4. Calculates rotation angles, handling the 360°/0° boundary correctly.
5. Computes statistical significance of rotation.
6. Generates polar plots (Rose plots) visualizing the shift.

### 3. Warning Fixes
- Addressed `SettingWithCopyWarning` by properly creating independent copies (`df_clean`) before modification.
- Ensured operations are performed on the clean copies.

---

## Results

**Left Breast (n=73):**
- **Mean Rotation:** -0.88 hours (-26.5°) (Counter-clockwise)
- **Significance:** p = 0.0097 (Significant)
- **Hypothesis:** Results show significant rotation, supporting the hypothesis of gravity-induced displacement.

**Right Breast (n=69):**
- **Mean Rotation:** 0.62 hours (18.7°) (Clockwise)
- **Significance:** p = 0.088 (Not significant at p<0.05 level)

**Visualization:**
- Polar plots generated in `../output/figs/clock_analysis/` showing the vector of movement for each landmark and the mean trend.

---

## Status
✅ **Complete**
- Time parsing integrated.
- Statistical analysis updated.
- Visualizations generated.
- Code verified with test script.
