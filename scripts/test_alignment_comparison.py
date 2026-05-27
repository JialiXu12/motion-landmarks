"""
Test Script to Compare Alignment Methods

Compares two alignment approaches across the cohort:
1. Fixed Sternum Point-to-Point (alignment_optimisation.py)
2. Optimal Sternum-Fixed SVD (alignment.py)

Generates comparison statistics and saves results to CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

# Setup paths similar to how other scripts do it
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Ensure imports work
sys.path.insert(0, str(project_root / "external" / "automesh"))
sys.path.insert(0, str(script_dir))

from readers import load_subject
from structures import Subject
from alignment_optimisation import align_ribcage_fixed_sternum, calculate_alignment_error
from alignment import optimal_sternum_fixed_alignment
from utils import extract_contour_points, filter_point_cloud_asymmetric
import external.breast_metadata_mdv.breast_metadata as breast_metadata


# Define paths
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
SEGMENTATION_ROOT = Path(r'U:\sandbox\jxu759\volunteer_seg\results')
PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
OUTPUT_DIR = Path("../output")


def load_ribcage_data(vl_id: int, vl_id_str: str, subject: Subject) -> Dict:
    """
    Load ribcage point clouds and sternum positions for alignment.

    Returns:
        Dictionary with prone_ribcage, supine_ribcage, prone_sternum, supine_sternum
        or None if data cannot be loaded
    """
    # Import morphic here to avoid module-level import issues
    import morphic
    from breast_metadata_mdv.examples.images.visualise_image_and_mesh import align_prone_supine as aps

    # Build file paths
    prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
    supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

    # Check files exist
    if not prone_mesh_file.exists() or not supine_seg_file.exists():
        print(f"  Missing files for {vl_id_str}")
        return None

    # Get anatomical landmarks
    anat_prone = subject.scans["prone"].anatomical_landmarks
    anat_supine = subject.scans["supine"].anatomical_landmarks

    if anat_prone.sternum_superior is None or anat_supine.sternum_superior is None:
        print(f"  Missing sternum landmarks for {vl_id_str}")
        return None

    # Load prone ribcage mesh
    prone_ribcage = morphic.Mesh(str(prone_mesh_file))
    prone_ribcage_coords = aps.get_surface_mesh_coords(prone_ribcage, res=26)

    # Load supine ribcage segmentation
    supine_ribcage_mask = breast_metadata.readNIFTIImage(
        str(supine_seg_file), 'RAI', swap_axes=True
    )
    supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

    # Filter supine point cloud
    supine_ribcage_pc = filter_point_cloud_asymmetric(
        points=supine_ribcage_pc,
        reference=supine_ribcage_pc,
        tol_min=0,
        tol_max=5,
        axis=2
    )

    return {
        'prone_ribcage': prone_ribcage_coords,
        'supine_ribcage': supine_ribcage_pc,
        'prone_sternum': anat_prone.sternum_superior,
        'supine_sternum': anat_supine.sternum_superior
    }


def compare_alignment_for_subject(vl_id: int, vl_id_str: str, subject: Subject) -> Dict:
    """
    Run both alignment methods on one subject and compare results.

    Returns:
        Dictionary with comparison metrics or None if alignment fails
    """
    print(f"\n--- Processing {vl_id_str} ---")

    # Load data
    data = load_ribcage_data(vl_id, vl_id_str, subject)
    if data is None:
        return None

    prone_ribcage = data['prone_ribcage']
    supine_ribcage = data['supine_ribcage']
    prone_sternum = data['prone_sternum']
    supine_sternum = data['supine_sternum']

    results = {'vl_id': vl_id, 'vl_id_str': vl_id_str}

    # ========================================
    # Method 1: Fixed Sternum Point-to-Point
    # ========================================
    try:
        print("  Running Method 1: Fixed Sternum Point-to-Point...")
        T_fixed, info_fixed = align_ribcage_fixed_sternum(
            prone_ribcage=prone_ribcage,
            supine_ribcage=supine_ribcage,
            prone_sternum=prone_sternum,
            supine_sternum=supine_sternum,
            verbose=False
        )

        results['method1_success'] = info_fixed['success']
        results['method1_rmse'] = info_fixed['rmse']
        results['method1_mean_dist'] = info_fixed['mean_distance']
        results['method1_sternum_drift'] = info_fixed['sternum_drift']
        results['method1_nfev'] = info_fixed['nfev']

        print(f"    RMSE: {info_fixed['rmse']:.3f} mm")
        print(f"    Sternum Drift: {info_fixed['sternum_drift']:.6f} mm")

    except Exception as e:
        print(f"  Method 1 FAILED: {e}")
        results['method1_success'] = False
        results['method1_rmse'] = np.nan
        results['method1_mean_dist'] = np.nan
        results['method1_sternum_drift'] = np.nan
        results['method1_nfev'] = np.nan

    # ========================================
    # Method 2: Optimal Sternum-Fixed SVD
    # ========================================
    try:
        print("  Running Method 2: Optimal Sternum-Fixed SVD...")
        R_optimal, src_aligned, info_optimal = optimal_sternum_fixed_alignment(
            source_pts=prone_ribcage,
            target_pts=supine_ribcage,
            source_sternum_sup=prone_sternum,
            target_sternum_sup=supine_sternum,
            max_correspondence_distance=15.0,
            max_iterations=200,
            verbose=False
        )

        results['method2_success'] = info_optimal['converged']
        results['method2_rmse'] = info_optimal['euclidean_rmse']
        results['method2_sternum_drift'] = info_optimal['sternum_error_mm']
        results['method2_iterations'] = info_optimal['iterations']
        results['method2_inlier_fraction'] = info_optimal['inlier_fraction']

        print(f"    RMSE: {info_optimal['euclidean_rmse']:.3f} mm")
        print(f"    Sternum Drift: {info_optimal['sternum_error_mm']:.6f} mm")
        print(f"    Iterations: {info_optimal['iterations']}")

    except Exception as e:
        print(f"  Method 2 FAILED: {e}")
        results['method2_success'] = False
        results['method2_rmse'] = np.nan
        results['method2_sternum_drift'] = np.nan
        results['method2_iterations'] = np.nan
        results['method2_inlier_fraction'] = np.nan

    return results


def compare_alignment_cohort(vl_ids: List[int]) -> pd.DataFrame:
    """
    Compare alignment methods across entire cohort.

    Args:
        vl_ids: List of volunteer IDs to process

    Returns:
        DataFrame with comparison results
    """
    print("="*80)
    print("ALIGNMENT METHOD COMPARISON")
    print("="*80)
    print(f"Processing {len(vl_ids)} subjects...")
    print("\nMethod 1: Fixed Sternum Point-to-Point (scipy least_squares)")
    print("Method 2: Optimal Sternum-Fixed SVD (iterative)")

    # Load all subjects
    all_subjects = {}
    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=ROOT_PATH_MRI,
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT
        )
        if subject.scans:
            all_subjects[vl_id] = subject

    # Run comparison for each subject
    results = []
    for vl_id, subject in all_subjects.items():
        vl_id_str = f"VL{vl_id:05d}"
        result = compare_alignment_for_subject(vl_id, vl_id_str, subject)
        if result is not None:
            results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\n--- Method 1: Fixed Sternum Point-to-Point ---")
    successful_1 = df[df['method1_success'] == True]
    if len(successful_1) > 0:
        print(f"  Success Rate: {len(successful_1)}/{len(df)} ({100*len(successful_1)/len(df):.1f}%)")
        print(f"  RMSE: {successful_1['method1_rmse'].mean():.3f} ± {successful_1['method1_rmse'].std():.3f} mm")
        print(f"  Mean Distance: {successful_1['method1_mean_dist'].mean():.3f} ± {successful_1['method1_mean_dist'].std():.3f} mm")
        print(f"  Sternum Drift: {successful_1['method1_sternum_drift'].mean():.6f} ± {successful_1['method1_sternum_drift'].std():.6f} mm")
        print(f"  Function Evaluations: {successful_1['method1_nfev'].mean():.0f} ± {successful_1['method1_nfev'].std():.0f}")

    print("\n--- Method 2: Optimal Sternum-Fixed SVD ---")
    successful_2 = df[df['method2_success'] == True]
    if len(successful_2) > 0:
        print(f"  Success Rate: {len(successful_2)}/{len(df)} ({100*len(successful_2)/len(df):.1f}%)")
        print(f"  RMSE: {successful_2['method2_rmse'].mean():.3f} ± {successful_2['method2_rmse'].std():.3f} mm")
        print(f"  Sternum Drift: {successful_2['method2_sternum_drift'].mean():.6f} ± {successful_2['method2_sternum_drift'].std():.6f} mm")
        print(f"  Iterations: {successful_2['method2_iterations'].mean():.0f} ± {successful_2['method2_iterations'].std():.0f}")
        print(f"  Inlier Fraction: {successful_2['method2_inlier_fraction'].mean():.3f} ± {successful_2['method2_inlier_fraction'].std():.3f}")

    # Compare methods (where both succeeded)
    both_success = df[(df['method1_success'] == True) & (df['method2_success'] == True)]
    if len(both_success) > 0:
        print("\n--- Direct Comparison (Both Methods Succeeded) ---")
        print(f"  N = {len(both_success)}")
        print(f"  RMSE Difference (Method1 - Method2): {(both_success['method1_rmse'] - both_success['method2_rmse']).mean():.3f} ± {(both_success['method1_rmse'] - both_success['method2_rmse']).std():.3f} mm")
        print(f"  Method 1 Better: {sum(both_success['method1_rmse'] < both_success['method2_rmse'])}/{len(both_success)}")
        print(f"  Method 2 Better: {sum(both_success['method2_rmse'] < both_success['method1_rmse'])}/{len(both_success)}")

    # Save to CSV
    output_csv = OUTPUT_DIR / "alignment_method_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")

    return df


if __name__ == "__main__":
    # Test on subset first
    VL_IDS = [9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25, 29, 30, 31]

    # Run comparison
    df = compare_alignment_cohort(VL_IDS)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)




