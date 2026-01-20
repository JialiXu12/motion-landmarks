# Clock Position Frequency & Rotation Analysis Update

## Summary of Changes

The `analyze_clock_position_rotation` function has been enhanced to include a **Frequency Analysis** of clock positions and to ensure geometric consistency in distance/radius calculations.

---

## 1. Frequency Analysis (New)
- **What it does:** Calculates and visualizes the distribution of tumors across the "Clock Face" (1-12 hours).
- **Visualization:** Generates **Rose Plots (Circular Histograms)** showing the count of landmarks at each hour for Prone (Blue) vs Supine (Red) positions.
- **Output:**
  - Console table showing counts for each hour (1-12).
  - Plots saved as `clock_frequency_left_breast.png` and `clock_frequency_right_breast.png`.

## 2. Distance Calculation Update (Geometric Consistency)
- **Question:** Should we use 3D distance or 2D coronal plane distance?
- **Decision:** **2D Coronal Plane Distance** (Magnitude on Y-Z plane).
- **Reasoning:** A "Clock Face" polar plot is inherently a 2D projection. Using 3D Euclidean distance for the radius ($R$) would indiscriminately mix "depth" (anterior-posterior) with "radial distance" (nipple-to-periphery). A tumor deep directly behind the nipple has 3D distance > 0 but is at the *center* ($R=0$) of the 2D clock face. Using 2D projection accurately represents the "Distance from Nipple" on the clock face diagram used in surgery.
- **Implementation:**
  - The code now calculates `distance_prone` and `distance_supine` using $\sqrt{(\Delta y)^2 + (\Delta z)^2}$ relative to the nipple.
  - This ensures the Polar Plot ($R, \theta$) is a true representation of the Coronal View.

## 3. Visualizations
- **Frequency Rose Plots:** Show where tumors are most commonly located.
- **Rotation Vectors:** Show how they move (rotate and shift radius) from prone to supine.

---

## Status
✅ **Code Updated:** `analysis.py` updated.
✅ **Distance Metric:** Switched to 2D Coronal Projection for Polar Plots.
✅ **Analysis Added:** Frequency counts and Rose Plots added.
