# 3-Panel Displacement Mechanism Schematic - Documentation

## Overview

**Status:** ✅ **IMPLEMENTED** in `analysis.py`

The analysis.py file now includes the comprehensive 3-panel schematic figure as requested. This visualization explains the "medial movement paradox" – why landmarks appear to move medially relative to the nipple even though both nipple and landmarks move laterally in absolute terms.

---

## Function Details

### Function Name
```python
plot_displacement_mechanism_schematic(df_ave, save_path=None)
```

### Location
- File: `scripts/analysis.py`
- Lines: ~1265-1575

### Purpose
Creates a publication-quality 3-panel figure that explains:
1. **Absolute displacement** (sternum reference frame)
2. **Vector subtraction mechanics** (mathematical explanation)
3. **Surgical view** (nipple reference frame)

---

## Panel Specifications

### **Panel A: Absolute Displacement (Sternum Reference)**

#### Goal
Show that the nipple moves MORE laterally than deep landmarks in absolute terms (sternum-fixed frame).

#### Visual Elements
1. **Sternum**: Black vertical line at x=0 (fixed reference)
2. **Breast Outlines**:
   - Prone breast: Light blue wedge
   - Supine breast: Light coral wedge (dashed)
3. **Nipple Markers**:
   - Prone: Red circle (●)
   - Supine: Red triangle (▲)
4. **Nipple Vector**: Long RED arrow (~179 mm lateral displacement)
5. **Landmark Markers**:
   - Prone: Blue circle (●)
   - Supine: Blue triangle (▲)
6. **Landmark Vector**: Short BLUE arrow (~26 mm lateral displacement)

#### Key Annotation
```
"Key Insight:
Nipple moves MORE laterally
than deep landmarks"
```

#### Measured Values (From Data)
- Nipple displacement: **179 mm** lateral (X-direction)
- Landmark displacement: **26 mm** lateral (X-direction)
- **Difference: 153 mm** (nipple "outruns" landmark)

---

### **Panel B: Vector Subtraction (Differential Mechanism)**

#### Goal
Explain why the relative direction flips using vector mathematics.

#### Visual Elements
1. **Nipple Vector** (Red): 
   - Equation: $\vec{V}_{Nipple} = 179$ mm
   - Drawn horizontally to the right
   
2. **Landmark Vector** (Blue):
   - Equation: $\vec{V}_{Landmark} = 26$ mm  
   - Drawn horizontally (shorter than nipple vector)

3. **Subtraction Operation**: Large "−" symbol

4. **Relative Vector** (Green):
   - Equation: $\vec{V}_{Relative} = \vec{V}_{Landmark} - \vec{V}_{Nipple} = -153$ mm
   - Points BACKWARDS (medially/leftward)
   - Shows the result of vector subtraction

#### Mathematical Equation (Displayed)
```
V_Relative = V_Landmark − V_Nipple
```

#### Explanation Box
```
"Vector Subtraction:

Because nipple moves MORE laterally,
the DIFFERENCE vector points MEDIALLY
(backwards relative to nipple motion).

Superficial tissue 'outruns' deep tissue."
```

---

### **Panel C: Surgical View (Nipple Reference)**

#### Goal
Show what the surgeon sees – landmarks appearing to move medially on the clock face.

#### Visual Format
- **Polar (rose) plot** in clock-face convention
- Center: Nipple at origin (0,0)
- Radial axis: Distance from nipple (mm)
- Angular axis: Clock positions (12, 1, 2, ..., 11)

#### Visual Elements
1. **Prone Positions**: Blue scatter points (●)
2. **Supine Positions**: Red scatter points (●)
3. **Trajectories**: Gray lines connecting prone→supine for sample landmarks
4. **Mean Trajectory**: Thick green line with black arrow
5. **Mean Markers**:
   - Prone mean: Large blue star (★)
   - Supine mean: Large red star (★)

#### Special Annotation
- **"Ghost" sternum indicator**: Purple double-arrow showing sternum appears to shift away from nipple
- Text: "Sternum shifts away from nipple"

#### Clock Convention
- 12 o'clock = Superior
- 3 o'clock (left breast) = Lateral
- 6 o'clock = Inferior  
- 9 o'clock (left breast) = Medial

#### Key Observation
Landmarks cluster more medially (toward 12 o'clock) in supine position compared to prone (more lateral, ~3 o'clock region).

---

## Comprehensive Title

```
"Displacement Mechanism: Why Landmarks Appear to Move Medially
(Despite Both Nipple and Landmarks Moving Laterally in Absolute Terms)"
```

---

## Output

### File Location
```
../output/figs/displacement_mechanism_schematic.png
```

### Figure Specifications
- **Size**: 18" × 6" (wide format, 3 panels side-by-side)
- **DPI**: 300 (publication quality)
- **Format**: PNG with tight bounding box

---

## Return Values

The function returns a dictionary with computed displacement metrics:

```python
{
    'nipple_displacement_abs': 197.0 mm,      # 3D magnitude
    'landmark_displacement_abs': 68.9 mm,     # 3D magnitude  
    'relative_displacement': 154.4 mm,        # 3D magnitude
    'nipple_dx': 178.6 mm,                    # X-component (lateral)
    'landmark_dx': 25.5 mm,                   # X-component (lateral)
    'relative_dx': -153.0 mm                  # X-component (MEDIAL!)
}
```

---

## Key Insights Demonstrated

### 1. The Paradox Explained
- **Absolute frame (sternum)**: Both nipple and landmarks move laterally
- **Relative frame (nipple)**: Landmarks move medially  
- **Why?** Nipple moves 7× more than landmarks (179 mm vs. 26 mm)

### 2. Vector Subtraction
The relative movement is simply:
```
V_relative = V_landmark - V_nipple  
          = 26 mm − 179 mm  
          = −153 mm (MEDIAL direction)
```

### 3. Clinical Relevance
- **Surgeons** use nipple as reference → see medial movement
- **MRI scans** (prone) show different position than **surgery** (supine)
- **~153 mm medial shift** is clinically significant for tumor localization

---

## Usage Example

```python
from analysis import *

# Load data
df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)

# Generate the 3-panel schematic
result = plot_displacement_mechanism_schematic(df_ave)

# Print displacement summary
print(f"Nipple moves: {result['nipple_dx']:.1f} mm laterally")
print(f"Landmark moves: {result['landmark_dx']:.1f} mm laterally")
print(f"Relative movement: {result['relative_dx']:.1f} mm (medial!)")
```

### Console Output
```
================================================================================
GENERATING DISPLACEMENT MECHANISM SCHEMATIC
================================================================================

Calculated Displacements (Left Breast, n=79):
  Nipple displacement (absolute): X=178.6, Y=81.0, Z=-19.2 mm
  Landmark displacement (absolute): X=25.5, Y=63.4, Z=-8.9 mm
  Relative displacement: X=-153.0, Y=-17.7, Z=10.3 mm

✓ Saved schematic figure: ..\output\figs\displacement_mechanism_schematic.png

================================================================================
SCHEMATIC GENERATION COMPLETE
================================================================================
```

---

## Implementation Notes

### Data Sources
- **Left breast data only** (n=79 landmarks)
- Uses mean displacement across all landmarks
- Coordinates relative to sternum in prone/supine positions

### Coordinate System
- **X-axis**: Right (−) ← → Left (+)
- **Y-axis**: Posterior (−) ← → Anterior (+)
- **Z-axis**: Inferior (−) ← → Superior (+)

### Panel A Scaling
- Vectors scaled by 50% for visualization (actual values in labels)
- Breast outlines are schematic representations
- Maintains proportions between nipple and landmark vectors

---

## Scientific Accuracy

✅ **All measurements are real data** from the analysis  
✅ **Vector mathematics is correct** (standard subtraction)  
✅ **Polar plot uses actual coordinates** (no approximation)  
✅ **Labels show actual mm values** (not scaled)  

---

## Publication Readiness

This figure is designed for:
- ✅ Scientific journals (high DPI, clear labels)
- ✅ Conference presentations (large format, readable)
- ✅ Supplementary materials (comprehensive explanation)
- ✅ Educational purposes (step-by-step visualization)

---

## Comparison to Request

| Requirement | Status | Implementation |
|------------|--------|----------------|
| **Panel A: Absolute Reality** | ✅ Complete | Sternum reference, long nipple vector, short landmark vector |
| **Panel B: Differential Mechanism** | ✅ Complete | Vector subtraction diagram with equations |
| **Panel C: Surgical View** | ✅ Complete | Polar plot, nipple-centered, medial trajectories |
| **Sternum reference line** | ✅ Complete | Black vertical line at x=0 |
| **Nipple vector (red)** | ✅ Complete | 179 mm, clearly labeled |
| **Landmark vector (blue)** | ✅ Complete | 26 mm, clearly labeled |
| **Vector subtraction** | ✅ Complete | Mathematical equation + visual demonstration |
| **Ghost sternum** | ✅ Complete | Purple annotation in Panel C |
| **Clock face convention** | ✅ Complete | 12=Superior, clockwise numbering |
| **Publication quality** | ✅ Complete | 300 DPI, 18×6 inches, tight layout |

---

## Summary

**✅ YES**, the analysis.py now includes the requested 3-panel schematic plot that comprehensively explains:

1. **What actually happens** (Panel A): Nipple and landmarks both move laterally, nipple moves more
2. **Why it looks different** (Panel B): Vector subtraction creates reversed direction  
3. **What surgeons see** (Panel C): Medial movement in nipple-relative frame

The implementation is complete, tested, and ready for publication use.
