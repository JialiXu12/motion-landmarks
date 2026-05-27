"""
Test alignment parameters on actual cohort data.

This script tests different combinations of max_correspondence_distance and trim_percentage
on the real cohort to determine:
1. Optimal parameters across the cohort
2. Whether individual subjects need different parameters
3. Parameter sensitivity analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Tuple
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from readers import load_subject
import external.breast_metadata_mdv.breast_metadata as breast_metadata


# ============================================================================
# UTILITY FUNCTIONS (Self-contained)
# ============================================================================

def extract_contour_points(mask, nb_points):
    """
    Extract surface/boundary points from a segmentation mask.

    Args:
        mask: Image mask object with .values, .spacing, and .origin attributes
        nb_points: Target number of points to return

    Returns:
        (N, 3) array of boundary point coordinates in world space
    """
    labels = mask.values.copy()
    boundaries = find_boundaries(labels, mode='inner').astype(np.uint8)

    boundary_indices = np.argwhere(boundaries > 0)

    if len(boundary_indices) == 0:
        print("Warning: No boundary points found in mask")
        return np.array([]).reshape(0, 3)

    spacing = np.array(mask.spacing)
    origin = np.array(mask.origin)

    points = boundary_indices.astype(np.float64) * spacing + origin

    if nb_points < len(points):
        step = max(1, len(points) // nb_points)
        indices = np.arange(0, len(points), step)[:nb_points]
        return points[indices, :]
    else:
        return points


# ============================================================================
# ALIGNMENT FUNCTIONS (Self-contained for testing)
# ============================================================================

def svd_rotation_point_to_point(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute optimal rotation R such that R @ P ≈ Q using SVD."""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def get_surface_mesh_coords(morphic_mesh, res, elems=None):
    """Extract 3D coordinates from morphic mesh."""
    if elems is None:
        elems = []
    Xi = morphic_mesh.grid(res, method='center')
    NPPE = Xi.shape[0]

    if len(elems) == 0:
        NE = morphic_mesh.elements.size()
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(morphic_mesh.elements):
            eid = element.id
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[eid].evaluate(Xi)
    else:
        NE = len(elems)
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(elems):
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[element].evaluate(Xi)

    return mesh_coords


def alignment_with_params(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
        source_sternum: np.ndarray,
        target_sternum: np.ndarray,
        max_correspondence_distance: float,
        trim_percentage: float,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6
) -> Dict:
    """
    Run alignment with specified parameters and return metrics.
    """
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    source_ss = np.asarray(source_sternum).flatten()
    target_ss = np.asarray(target_sternum).flatten()

    # Center on sternum
    src_centered = source_pts - source_ss
    tgt_centered = target_pts - target_ss

    tree = cKDTree(tgt_centered)
    src = src_centered.copy()
    R_total = np.eye(3)
    prev_rmse = np.inf

    for it in range(max_iterations):
        dists, idxs = tree.query(src)

        # Filter by max distance
        if max_correspondence_distance > 0:
            valid = dists <= max_correspondence_distance
        else:
            valid = np.ones(len(dists), dtype=bool)

        # Trimmed ICP
        if trim_percentage > 0 and np.sum(valid) > 100:
            valid_dists = dists[valid]
            threshold = np.percentile(valid_dists, (1.0 - trim_percentage) * 100)
            valid = valid & (dists <= threshold)

        n_valid = np.sum(valid)
        if n_valid < 10:
            break

        P = src[valid]
        Q = tgt_centered[idxs[valid]]

        R_delta = svd_rotation_point_to_point(P, Q)
        src = (R_delta @ src.T).T
        R_total = R_delta @ R_total

        # Compute RMSE on inliers
        dists_new, _ = tree.query(src)
        if max_correspondence_distance > 0:
            valid_new = dists_new <= max_correspondence_distance
        else:
            valid_new = np.ones(len(dists_new), dtype=bool)

        rmse = np.sqrt(np.mean(dists_new[valid_new] ** 2)) if np.any(valid_new) else np.inf

        if abs(prev_rmse - rmse) < convergence_threshold:
            break
        prev_rmse = rmse

    # Final metrics on ALL points (for fair comparison)
    dists_final, _ = tree.query(src)

    # Compute various metrics
    return {
        "rmse_all": np.sqrt(np.mean(dists_final ** 2)),
        "mean_all": np.mean(dists_final),
        "median_all": np.median(dists_final),
        "std_all": np.std(dists_final),
        "max_all": np.max(dists_final),
        "p90_all": np.percentile(dists_final, 90),
        "p95_all": np.percentile(dists_final, 95),
        "iterations": it + 1,
        "n_inliers": int(np.sum(dists_final <= 15)),  # Standard threshold for reporting
        "inlier_fraction": float(np.sum(dists_final <= 15)) / len(dists_final),
        "R_total": R_total,
        "aligned_source": src
    }


# ============================================================================
# MAIN TEST FUNCTIONS
# ============================================================================

def test_subject_with_params(
        subject,
        prone_mesh_file: Path,
        supine_seg_file: Path,
        param_grid: list,
        prone_sternum: np.ndarray,
        supine_sternum: np.ndarray
) -> pd.DataFrame:
    """
    Test a single subject with multiple parameter combinations.

    Args:
        subject: Subject object with anatomical data
        prone_mesh_file: Path to prone ribcage mesh
        supine_seg_file: Path to supine ribcage segmentation
        param_grid: List of (max_dist, trim_pct) tuples to test
        prone_sternum: Sternum position in prone
        supine_sternum: Sternum position in supine

    Returns:
        DataFrame with results for each parameter combination
    """
    import morphic

    # Load prone mesh
    prone_mesh = morphic.Mesh(str(prone_mesh_file))
    prone_pts = get_surface_mesh_coords(prone_mesh, res=10)

    # Load supine point cloud using breast_metadata.readNIFTIImage
    supine_mask = breast_metadata.readNIFTIImage(
        str(supine_seg_file), 'RAI', swap_axes=True
    )
    supine_pts = extract_contour_points(supine_mask, 20000)

    results = []

    for max_dist, trim_pct in param_grid:
        start_time = time.time()

        try:
            metrics = alignment_with_params(
                prone_pts, supine_pts,
                prone_sternum, supine_sternum,
                max_correspondence_distance=max_dist,
                trim_percentage=trim_pct
            )

            elapsed = time.time() - start_time

            results.append({
                "max_dist": max_dist,
                "trim_pct": trim_pct,
                "rmse": metrics["rmse_all"],
                "mean": metrics["mean_all"],
                "median": metrics["median_all"],
                "std": metrics["std_all"],
                "max": metrics["max_all"],
                "p90": metrics["p90_all"],
                "p95": metrics["p95_all"],
                "iterations": metrics["iterations"],
                "inlier_fraction": metrics["inlier_fraction"],
                "time_sec": elapsed,
                "success": True
            })
        except Exception as e:
            results.append({
                "max_dist": max_dist,
                "trim_pct": trim_pct,
                "rmse": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "max": np.nan,
                "p90": np.nan,
                "p95": np.nan,
                "iterations": 0,
                "inlier_fraction": 0,
                "time_sec": 0,
                "success": False,
                "error": str(e)
            })

    return pd.DataFrame(results)


def run_cohort_parameter_test(
        vl_ids: list,
        segmentation_root: Path,
        prone_mesh_root: Path,
        supine_ribcage_root: Path,
        dicom_root: Path,
        anatomical_json_base_root: Path,
        soft_tissue_root: Path,
        param_grid: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test all parameter combinations across the entire cohort.

    Returns:
        - all_results: DataFrame with per-subject, per-parameter results
        - summary: DataFrame with aggregated results per parameter combination
    """

    if param_grid is None:
        # Default parameter grid to test
        max_distances = [0, 10, 15, 20, 30, 50]  # 0 = no filtering
        trim_percentages = [0, 0.05, 0.10, 0.15, 0.20]
        param_grid = [(d, t) for d in max_distances for t in trim_percentages]

    all_results = []

    print(f"\n{'='*80}")
    print(f"COHORT PARAMETER OPTIMIZATION TEST")
    print(f"{'='*80}")
    print(f"Subjects: {len(vl_ids)}")
    print(f"Parameter combinations: {len(param_grid)}")
    print(f"Total tests: {len(vl_ids) * len(param_grid)}")
    print(f"{'='*80}\n")

    for i, vl_id in enumerate(vl_ids):
        vl_id_str = f"VL{vl_id:05d}"
        print(f"\n[{i+1}/{len(vl_ids)}] Testing {vl_id_str}...")

        # Build paths
        prone_mesh_file = prone_mesh_root / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = supine_ribcage_root / f"rib_cage_{vl_id_str}.nii.gz"

        # Check files exist
        if not prone_mesh_file.exists():
            print(f"  Skipping: Prone mesh not found")
            continue
        if not supine_seg_file.exists():
            print(f"  Skipping: Supine segmentation not found")
            continue

        try:
            # Load subject with correct parameters
            subject = load_subject(
                vl_id=vl_id,
                positions=["prone", "supine"],
                dicom_root=dicom_root,
                anatomical_json_base_root=anatomical_json_base_root,
                soft_tissue_root=soft_tissue_root
            )

            if "prone" not in subject.scans or "supine" not in subject.scans:
                print(f"  Skipping: Missing prone or supine scan data")
                continue

            # Check anatomical landmarks exist
            prone_scan = subject.scans["prone"]
            supine_scan = subject.scans["supine"]

            if (prone_scan.anatomical_landmarks is None or
                supine_scan.anatomical_landmarks is None):
                print(f"  Skipping: Missing anatomical landmarks")
                continue

            prone_sternum = prone_scan.anatomical_landmarks.sternum_superior
            supine_sternum = supine_scan.anatomical_landmarks.sternum_superior

            if prone_sternum is None or supine_sternum is None:
                print(f"  Skipping: Missing sternum landmarks")
                continue

            # Test all parameter combinations
            df = test_subject_with_params(
                subject, prone_mesh_file, supine_seg_file, param_grid,
                prone_sternum=prone_sternum,
                supine_sternum=supine_sternum
            )
            df["vl_id"] = vl_id
            df["subject"] = vl_id_str

            all_results.append(df)

            # Print best result for this subject
            best_idx = df["rmse"].idxmin()
            best = df.loc[best_idx]
            print(f"  Best: max_dist={best['max_dist']}, trim={best['trim_pct']:.0%} → RMSE={best['rmse']:.2f}mm")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results collected!")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Create summary statistics per parameter combination
    summary = all_results_df.groupby(["max_dist", "trim_pct"]).agg({
        "rmse": ["mean", "std", "min", "max"],
        "median": ["mean"],
        "inlier_fraction": ["mean"],
        "iterations": ["mean"],
        "success": ["sum"]
    }).round(3)

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    return all_results_df, summary


def analyze_subject_variability(all_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether different subjects need different parameters.

    For each subject, find the best parameter combination and compare
    to the cohort-wide best.
    """

    # Find best params per subject
    best_per_subject = []

    for vl_id in all_results_df["vl_id"].unique():
        subject_df = all_results_df[all_results_df["vl_id"] == vl_id]
        best_idx = subject_df["rmse"].idxmin()
        best = subject_df.loc[best_idx].copy()
        best["is_best_for_subject"] = True
        best_per_subject.append(best)

    best_df = pd.DataFrame(best_per_subject)

    # Find cohort-wide best parameters
    cohort_best = all_results_df.groupby(["max_dist", "trim_pct"])["rmse"].mean().idxmin()
    cohort_best_max_dist, cohort_best_trim = cohort_best

    # For each subject, compare their best to cohort best
    comparison = []

    for vl_id in all_results_df["vl_id"].unique():
        subject_df = all_results_df[all_results_df["vl_id"] == vl_id]

        # Subject's best
        best_idx = subject_df["rmse"].idxmin()
        subject_best = subject_df.loc[best_idx]

        # Cohort best applied to this subject
        cohort_row = subject_df[
            (subject_df["max_dist"] == cohort_best_max_dist) &
            (subject_df["trim_pct"] == cohort_best_trim)
        ]

        if len(cohort_row) > 0:
            cohort_rmse = cohort_row["rmse"].values[0]
        else:
            cohort_rmse = np.nan

        comparison.append({
            "vl_id": vl_id,
            "subject_best_max_dist": subject_best["max_dist"],
            "subject_best_trim": subject_best["trim_pct"],
            "subject_best_rmse": subject_best["rmse"],
            "cohort_params_rmse": cohort_rmse,
            "rmse_penalty": cohort_rmse - subject_best["rmse"],  # How much worse is cohort best?
            "same_as_cohort": (
                subject_best["max_dist"] == cohort_best_max_dist and
                subject_best["trim_pct"] == cohort_best_trim
            )
        })

    comparison_df = pd.DataFrame(comparison)

    return comparison_df, (cohort_best_max_dist, cohort_best_trim)


def print_final_report(all_results_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print comprehensive analysis report."""

    print("\n" + "="*80)
    print("FINAL PARAMETER OPTIMIZATION REPORT")
    print("="*80)

    # 1. Best overall parameters
    print("\n1. COHORT-WIDE BEST PARAMETERS")
    print("-"*60)

    best_idx = summary_df["rmse_mean"].idxmin()
    best = summary_df.loc[best_idx]

    print(f"   max_correspondence_distance = {best['max_dist']} mm")
    print(f"   trim_percentage = {best['trim_pct']:.0%}")
    print(f"   Mean RMSE across cohort: {best['rmse_mean']:.2f} ± {best['rmse_std']:.2f} mm")
    print(f"   Mean inlier fraction: {best['inlier_fraction_mean']:.1%}")

    # 2. Parameter sensitivity
    print("\n2. PARAMETER SENSITIVITY ANALYSIS")
    print("-"*60)

    # Effect of max_distance
    print("\n   Effect of max_correspondence_distance (trim=10%):")
    trim_10 = summary_df[summary_df["trim_pct"] == 0.10].sort_values("max_dist")
    for _, row in trim_10.iterrows():
        dist_str = f"{row['max_dist']:>3}" if row['max_dist'] > 0 else "None"
        print(f"     max_dist={dist_str}mm: RMSE = {row['rmse_mean']:.2f} ± {row['rmse_std']:.2f} mm")

    # Effect of trim_percentage
    print("\n   Effect of trim_percentage (max_dist=15mm):")
    dist_15 = summary_df[summary_df["max_dist"] == 15].sort_values("trim_pct")
    for _, row in dist_15.iterrows():
        print(f"     trim={row['trim_pct']:>4.0%}: RMSE = {row['rmse_mean']:.2f} ± {row['rmse_std']:.2f} mm")

    # 3. Subject variability analysis
    print("\n3. SUBJECT VARIABILITY ANALYSIS")
    print("-"*60)

    comparison_df, cohort_best = analyze_subject_variability(all_results_df)

    n_same = comparison_df["same_as_cohort"].sum()
    n_total = len(comparison_df)

    print(f"   Cohort best parameters: max_dist={cohort_best[0]}mm, trim={cohort_best[1]:.0%}")
    print(f"   Subjects where cohort best is optimal: {n_same}/{n_total} ({100*n_same/n_total:.0f}%)")
    print(f"   Mean RMSE penalty using cohort params: {comparison_df['rmse_penalty'].mean():.3f} mm")
    print(f"   Max RMSE penalty: {comparison_df['rmse_penalty'].max():.3f} mm")

    # Show subjects that differ
    different = comparison_df[~comparison_df["same_as_cohort"]]
    if len(different) > 0:
        print(f"\n   Subjects requiring different parameters:")
        for _, row in different.iterrows():
            print(f"     VL{row['vl_id']:05d}: best={row['subject_best_max_dist']}mm/{row['subject_best_trim']:.0%}, "
                  f"penalty={row['rmse_penalty']:.2f}mm")

    # 4. Recommendation
    print("\n4. RECOMMENDATION")
    print("-"*60)
    print(f"""
   Based on the analysis:
   
   FIXED PARAMETERS (use for all subjects):
     max_correspondence_distance = {cohort_best[0]} mm
     trim_percentage = {cohort_best[1]:.0%}
   
   RATIONALE:
   - The RMSE penalty for using fixed vs optimal per-subject params is minimal
     (mean penalty: {comparison_df['rmse_penalty'].mean():.3f} mm)
   - Using consistent parameters across the cohort is preferred for:
     a) Reproducibility
     b) Fair comparison between subjects
     c) Simpler reporting in publications
   
   FOR PUBLICATION:
   "All alignments used a maximum correspondence distance of {cohort_best[0]} mm
   and {cohort_best[1]:.0%} trimmed ICP for robust outlier rejection. These
   parameters were optimized across the cohort and applied consistently to
   all subjects."
""")

    return comparison_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define paths (matching main.py)
    ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
    SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
    ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
    SEGMENTATION_ROOT = Path(r'U:\sandbox\jxu759\volunteer_seg\results')
    PRONE_MESH_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
    OUTPUT_DIR = Path(r"C:\Users\jxu759\Documents\motion-landmarks\output")

    # Define cohort
    VL_IDS_BATCH1 = [9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25, 29, 30, 31]
    VL_IDS_BATCH2 = [32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]
    VL_IDS_BATCH3 = [51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69]
    VL_IDS_BATCH4 = [70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 89]

    # Full cohort analysis:
    VL_IDS = VL_IDS_BATCH1 + VL_IDS_BATCH2 + VL_IDS_BATCH3 + VL_IDS_BATCH4

    # Define parameter grid to test
    PARAM_GRID = [
        # (max_correspondence_distance, trim_percentage)
        (0, 0),       # No filtering at all
        (0, 0.05),    # Trim only
        (0, 0.10),
        (10, 0),      # Max dist only
        (10, 0.05),
        (10, 0.10),
        (15, 0),      # Current default distance
        (15, 0.05),
        (15, 0.10),   # Current default
        (15, 0.15),
        (20, 0),
        (20, 0.05),
        (20, 0.10),
        (30, 0.10),
        (50, 0.10),   # Very permissive
    ]

    print("Starting cohort parameter optimization test...")
    print(f"Testing {len(VL_IDS)} subjects with {len(PARAM_GRID)} parameter combinations")

    # Run the test
    all_results_df, summary_df = run_cohort_parameter_test(
        vl_ids=VL_IDS,
        segmentation_root=SEGMENTATION_ROOT,
        prone_mesh_root=PRONE_MESH_ROOT,
        supine_ribcage_root=SUPINE_RIBCAGE_ROOT,
        dicom_root=ROOT_PATH_MRI,
        anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
        soft_tissue_root=SOFT_TISSUE_ROOT,
        param_grid=PARAM_GRID
    )

    # Save raw results
    results_file = OUTPUT_DIR / "alignment_parameter_optimization_results.xlsx"
    with pd.ExcelWriter(results_file) as writer:
        all_results_df.to_excel(writer, sheet_name="All Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    print(f"\nResults saved to: {results_file}")

    # Print final report
    if len(all_results_df) > 0:
        comparison_df = print_final_report(all_results_df, summary_df)

        # Save comparison
        comparison_df.to_excel(
            OUTPUT_DIR / "alignment_parameter_subject_comparison.xlsx",
            index=False
        )








