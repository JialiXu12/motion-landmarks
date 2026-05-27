# Motion Landmarks - Project Summary

## Purpose

Medical imaging research pipeline that analyzes breast tissue landmark displacement between prone (face-down) and supine (face-up) body positions using 3D MRI data. The pipeline loads DICOM images and manually-annotated landmarks, performs rigid-body alignment of ribcage structures, computes displacement vectors, and runs statistical analyses to characterize tissue deformation patterns.

## Data Pipeline

```
1. LOAD DATA (main.py)
   DICOM images + JSON landmarks + mesh files
        |
2. FIND CORRESPONDENCES (utils.py)
   Match landmarks across registrars (<3mm threshold)
        |
3. AVERAGE LANDMARKS (utils.py)
   Average positions across registrars -> "ld_ave"
        |
4. CALCULATE DISTANCES (utils.py)
   Distance from each landmark to skin and ribcage surfaces
        |
5. CLOCK COORDINATES (utils.py)
   Convert 3D positions to clock-face positions relative to nipple
        |
6. PRONE-TO-SUPINE ALIGNMENT (alignment.py)
   SVD/ICP registration with sternum-fixed constraint
        |
7. SAVE RESULTS (utils.py)
   Excel workbook + transformation matrices (.npy)
        |
8. STATISTICAL ANALYSIS (analysis.py)
   ANOVA, t-tests, BMI correlations, clock rotation analysis
        |
9. VISUALIZATION (utils_plot.py, plot_*.py)
   Vector plots, polar plots, 3-panel views
```

## Project Structure

### Core Scripts (`scripts/`)

| File | Lines | Role |
|------|-------|------|
| `main.py` | 277 | Entry point - orchestrates the full pipeline |
| `structures.py` | 49 | Data classes: `Subject`, `ScanData`, `AnatomicalLandmarks`, `RegistrarData` |
| `readers.py` | ~200 | DICOM/JSON loading, subject assembly |
| `utils.py` | 1,875 | Core utilities: distances, ICP, correspondence, Excel I/O |
| `alignment.py` | 1,528 | Sternum-fixed optimal alignment (SVD + ICP) |
| `align_fixed_sternum.py` | 765 | Simplified sternum-fixed alignment (point-to-point) |
| `analysis.py` | 3,530 | Statistical analysis: ANOVA, correlations, clock rotation |
| `utils_plot.py` | 821 | Plotting utilities for landmarks and vectors |
| `plot_nipple_relative_vectors.py` | 675 | Nipple-relative displacement visualization |
| `plot_polar_plots.py` | 824 | Circular/polar clock position analysis |
| `partial_correlation.py` | 163 | Partial correlation tests |

### Supporting Scripts

| File | Role |
|------|------|
| `alignment_optimisation.py` | Optimization method exploration |
| `constrained_icp_fixed_sternum.py` | ICP variant with sternum constraints |
| `compare_alignment_versions.py` | Compare alignment algorithms |
| `sensitivity_analysis_trim_percentage.py` | ICP trim parameter sensitivity |
| `analyze_convergence.py` | ICP convergence behavior |
| `generate_point_cloud.py` | Create point clouds from segmentations |
| `fix_nipple_labels_in_json.py` | Data cleanup utility |

### External Dependencies (`external/`)

| Submodule | Purpose |
|-----------|---------|
| `breast_metadata_mdv` | Breast-specific DICOM metadata and image handling |
| `bmw` | Breast biomechanics library |
| `automesh` | Automated mesh generation |

### Output (`output/`)

- `landmark_results_v6_*.xlsx` - Main results (multi-sheet workbook)
- `transformation_matrix_v6/` - Alignment matrices (.npy per subject)
- `figs/` - Visualization outputs organized by analysis type
- `analysis_output_v6_*.txt` - Timestamped analysis logs

## Key Algorithms

### Alignment (alignment.py)
- **SVD-based rotation** (Kabsch algorithm) for initial alignment
- **Iterative Closest Point (ICP)** for refinement
- **Sternum-fixed constraint** - sternum superior pinned to origin
- **Trimmed Least Squares** - rejects 10% worst correspondences
- Key params: `max_correspondence_distance=15mm`, `max_iterations=200`, `convergence_threshold=1e-6`

### Statistical Tests (analysis.py)
- Normality: Shapiro-Wilk
- Homogeneity: Levene's (Brown-Forsythe)
- Parametric: ANOVA, Student's/Welch's t-test
- Non-parametric: Kruskal-Wallis, Mann-Whitney U
- Post-hoc: Tukey's HSD, Games-Howell, Bonferroni correction
- Repeated measures: RM-ANOVA, Friedman
- Correlation: Pearson, Spearman, partial correlation

## Input Data Formats

| Format | Content |
|--------|---------|
| DICOM | 3D breast MRI scans (prone and supine) |
| JSON | Anatomical landmarks (nipples, sternum points) |
| JSON (picker) | Soft tissue landmarks from manual annotation |
| `.mesh` (morphic) | Ribcage surface models |
| `.nii.gz` (NIfTI) | Ribcage segmentations |

## Dependencies

Declared in `requirements.txt`:
- Scientific: `numpy 1.26.4`, `scipy 1.16.2`, `pandas 1.5.3`, `scikit-learn 1.7.2`
- Medical imaging: `pydicom 3.0.1`, `SimpleITK 2.5.2`, `itk 5.4.4`, `nibabel 5.3.2`
- Mesh/3D: `pyvista 0.46.3`, `trimesh`
- Visualization: `matplotlib 3.9.4`, `seaborn`
- Statistics: `pingouin`
- Data: `h5py 3.14.0`, `tables 3.10.2`

Python >= 3.10 required.

## Known Issues

See the issues list below for items that need attention.
