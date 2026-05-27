# Trim Percentage vs. Inlier Fraction: Scientific Explanation

## Quick Answer
**Trim percentage is ALWAYS 10% for all subjects (fixed parameter).**  
The "inlier percentage" you see varying between subjects (e.g., 75%-95%) is a **quality metric**, not the trim percentage.

---

## Two Different Concepts

### 1. **Trim Percentage** (Fixed at 10%)
- **When**: Applied DURING each iteration of ICP optimization
- **Purpose**: Robust outlier rejection to prevent poor correspondences from corrupting the alignment
- **How it works**: In each iteration, find point correspondences, then reject the worst 10%
- **Consistency**: This is FIXED at 10% for ALL subjects in the cohort
- **For reporting**: "All alignments used 10% trimming for outlier rejection during ICP optimization"

### 2. **Final Inlier Fraction** (Varies by Subject: typically 75-95%)
- **When**: Calculated AFTER alignment is complete
- **Purpose**: Quality metric showing what percentage of ribcage points are within 15mm distance threshold
- **How it works**: Count how many source points have a nearest neighbor within 15mm in the target
- **Variability**: Naturally varies between subjects based on:
  - Anatomical differences (breast size, chest shape)
  - Posture changes (breathing, shoulder position)
  - Deformation magnitude (larger breasts = more soft tissue motion)
  - Segmentation quality (cleaner masks = better correspondences)
- **For reporting**: "Final alignment quality varied by subject, with 75-95% of ribcage points within 15mm correspondence distance"

---

## Why They're Different

### Trimming During ICP (10% fixed)
```
Iteration 1:
├─ Find 10,000 correspondences
├─ Sort by distance
├─ Reject worst 1,000 (10%)  <-- TRIM PERCENTAGE
└─ Optimize using best 9,000

Iteration 2:
├─ Find 10,000 correspondences
├─ Sort by distance
├─ Reject worst 1,000 (10%)  <-- TRIM PERCENTAGE (always 10%)
└─ Optimize using best 9,000
...
```

### Final Quality Assessment (varies)
```
After convergence:
├─ Count points with distance < 15mm
├─ Subject A: 9,200/10,000 = 92% inliers  <-- HIGH QUALITY
├─ Subject B: 7,800/10,000 = 78% inliers  <-- ACCEPTABLE QUALITY
└─ Subject C: 9,500/10,000 = 95% inliers  <-- EXCELLENT QUALITY
```

---

## For Scientific Journal Reporting

### Methods Section
"Prone-to-supine registration was performed using Iterative Closest Point (ICP) with the sternum superior landmark fixed at the origin. To ensure robustness against outliers, we used trimmed ICP with a fixed 10% trim percentage (Chetverikov et al., 2002), rejecting the worst 10% of correspondences in each iteration. The maximum correspondence distance threshold was set to 15 mm, and convergence was defined as RMSE change < 0.01 mm between iterations."

### Results Section
"Alignment quality varied by subject anatomy and posture differences. The mean ribcage RMSE was X.X ± Y.Y mm (range: A.A-B.B mm). On average, Z.Z% ± W.W% of ribcage surface points were within the 15 mm correspondence threshold after alignment, indicating [excellent/good/acceptable] registration quality."

### Supplementary Material
Include a table showing per-subject metrics:
| Subject ID | Ribcage RMSE (mm) | Inliers (%) | Sternum Error (mm) |
|------------|-------------------|-------------|---------------------|
| VL00009    | 5.2              | 87%         | 0.00               |
| VL00011    | 4.1              | 92%         | 0.00               |
| ...        | ...              | ...         | ...                |

Note: "All subjects used 10% trimming during optimization. Inlier percentage represents final alignment quality."

---

## Why Inlier Percentage Varies

**It's EXPECTED and SCIENTIFICALLY VALID** for the inlier percentage to vary:

1. **Larger breasts** → More soft tissue deformation → Harder to align → Lower inlier %
2. **Different breathing states** → Ribcage expansion varies → Affects correspondence quality
3. **Posture differences** → Shoulder rotation, arm position → Changes chest wall geometry
4. **BMI variation** → More adipose tissue → More non-rigid deformation → May lower inlier %

**This is a FEATURE, not a bug!** It tells you which subjects had:
- Excellent alignment (>90% inliers)
- Good alignment (80-90% inliers)
- Acceptable alignment (70-80% inliers)
- Potential issues (<70% inliers - may need manual inspection)

---

## Key Takeaway

✅ **Trim percentage = 10%** (FIXED for all subjects)  
✅ **Inlier fraction = 75-95%** (VARIES by subject quality)  
✅ **Both are correct and expected!**

The trim percentage ensures robust optimization. The inlier fraction measures how well the anatomy actually matched after optimal alignment. Both metrics are valuable for scientific reporting.

---

## Reference
Chetverikov, D., Svirko, D., Stepanov, D., & Krsek, P. (2002). The trimmed iterative closest point algorithm. In Object recognition supported by user interaction for service robots (Vol. 3, pp. 545-548). IEEE.

