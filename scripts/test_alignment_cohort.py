"""
Comprehensive Alignment Parameter Testing for Full Cohort

This script runs alignment tests across all subjects with different parameters
to find optimal settings for:
1. Point-to-point alignment (optimal_sternum_fixed_alignment)
2. Surface-to-point (plane-to-point) alignment

Records RMSE, mean, std for both full mesh and selected elements.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
import traceback
import sys

# Add project paths - order matters!
scripts_dir = Path(__file__).parent
project_root = scripts_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(project_root))  # For 'external' module
sys.path.insert(0, str(project_root / "external"))
sys.path.insert(0, str(project_root / "external" / "breast_metadata_mdv"))  # For internal imports
sys.path.insert(0, str(project_root / "src" / "morphic"))
sys.path.insert(0, str(project_root / "src" / "mesh-tools"))

import morphic
from breast_metadata import *  # Import directly since it's in path

# Import alignment functions - use try/except to handle different import scenarios
try:
    from alignment import (
        get_surface_mesh_coords,
        get_mesh_with_selected_elements,
        get_mesh_elements_2,
        optimal_sternum_fixed_alignment,
        apply_transform_to_coords,
        filter_point_cloud_to_match_mesh_region,
    )
except ImportError as e:
    print(f"Warning: Could not import from alignment: {e}")

try:
    from surface_to_point_alignment import surface_to_point_align
except ImportError as e:
    print(f"Warning: Could not import surface_to_point_align: {e}")
    surface_to_point_align = None

from structures import Subject
from readers import load_subject
from skimage.segmentation import find_boundaries


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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# All VL IDs from the cohort
ALL_VL_IDS = [
    # Batch 1
    9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25, 29, 30, 31,
    # Batch 2
    32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50,
    # Batch 3
    51, 52, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69,
    # Batch 4
    70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 89
]

# Paths
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
OUTPUT_ROOT = Path(r"..\output")

# Selected elements for alignment
SELECTED_ELEMENTS = [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23]

# Parameter grid to test
PARAM_GRID = {
    'max_correspondence_distance': [10.0, 15.0, 20.0, 25.0],
    'trim_percentage': [0.0, 0.05, 0.1, 0.15, 0.2],
    'res': [6, 10, 14, 18, 26],  # For surface-to-point alignment
}

# Default parameters
DEFAULT_PARAMS = {
    'max_correspondence_distance': 15.0,
    'trim_percentage': 0.1,
    'res': 10,
    'max_iterations': 200,
}


# =============================================================================
# Helper Functions
# =============================================================================

def load_alignment_data(vl_id: int) -> Tuple[dict, bool]:
    """
    Load all data needed for alignment for a single subject.

    Returns:
        data_dict: Dictionary with mesh, point cloud, sternum positions, etc.
        success: Boolean indicating if loading was successful
    """
    try:
        # Load subject
        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=ROOT_PATH_MRI,
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT
        )

        if subject is None:
            return None, False

        # Get anatomical landmarks
        anat_prone = subject.scans["prone"].anatomical_landmarks
        anat_supine = subject.scans["supine"].anatomical_landmarks

        if anat_prone.sternum_superior is None or anat_supine.sternum_superior is None:
            print(f"  WARNING: VL{vl_id:05d} missing sternum landmarks")
            return None, False

        sternum_prone = np.vstack([anat_prone.sternum_superior, anat_prone.sternum_inferior])
        sternum_supine = np.vstack([anat_supine.sternum_superior, anat_supine.sternum_inferior])

        # Load prone ribcage mesh
        vl_id_str = f"VL{vl_id:05d}"
        prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        if not prone_mesh_file.exists():
            print(f"  WARNING: VL{vl_id:05d} prone mesh not found: {prone_mesh_file}")
            return None, False

        if not supine_seg_file.exists():
            print(f"  WARNING: VL{vl_id:05d} supine segmentation not found: {supine_seg_file}")
            return None, False

        prone_ribcage = morphic.Mesh(str(prone_mesh_file))
        prone_ribcage_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)

        # Load supine ribcage segmentation
        supine_ribcage_mask = breast_metadata.readNIFTIImage(
            str(supine_seg_file), 'RAI', swap_axes=True
        )
        supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, 20000)

        # Get element centers
        centers_array, num_elements = get_mesh_elements_2(prone_ribcage)

        # Get selected elements coordinates
        prone_ribcage_selected_coords = get_mesh_with_selected_elements(
            prone_ribcage, SELECTED_ELEMENTS, res=26
        )

        # Filter supine point cloud to match selected mesh region
        supine_ribcage_pc_filtered = filter_point_cloud_to_match_mesh_region(
            point_cloud=supine_ribcage_pc,
            mesh_region_coords=prone_ribcage_selected_coords,
            pc_sternum_sup=sternum_supine[0],
            mesh_sternum_sup=sternum_prone[0],
            padding=30.0,
            verbose=False,
        )

        return {
            'subject': subject,
            'prone_ribcage': prone_ribcage,
            'prone_ribcage_mesh_coords': prone_ribcage_mesh_coords,
            'prone_ribcage_selected_coords': prone_ribcage_selected_coords,
            'supine_ribcage_pc': supine_ribcage_pc,
            'supine_ribcage_pc_filtered': supine_ribcage_pc_filtered,
            'sternum_prone': sternum_prone,
            'sternum_supine': sternum_supine,
            'centers_array': centers_array,
            'num_elements': num_elements,
        }, True

    except Exception as e:
        print(f"  ERROR loading VL{vl_id:05d}: {str(e)}")
        traceback.print_exc()
        return None, False


def run_point_to_point_alignment(
    data: dict,
    max_correspondence_distance: float,
    trim_percentage: float,
    max_iterations: int = 200,
) -> dict:
    """
    Run point-to-point alignment with given parameters.

    Returns dict with RMSE, mean, std for full mesh and selected elements.
    """
    try:
        R, aligned_prone_centered, info = optimal_sternum_fixed_alignment(
            source_pts=data['prone_ribcage_selected_coords'],
            target_pts=data['supine_ribcage_pc_filtered'],
            source_sternum_sup=data['sternum_prone'][0],
            target_sternum_sup=data['sternum_supine'][0],
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            convergence_threshold=1e-6,
            trim_percentage=trim_percentage,
            verbose=False,
            visualize_iterations=False,
        )

        # Transform full mesh and selected mesh
        source_anchor = data['sternum_prone'][0]
        target_anchor = data['sternum_supine'][0]

        prone_ribcage_transformed = apply_transform_to_coords(
            data['prone_ribcage_mesh_coords'], R, source_anchor, target_anchor
        )

        prone_selected_transformed = apply_transform_to_coords(
            data['prone_ribcage_selected_coords'], R, source_anchor, target_anchor
        )

        # Calculate errors for full mesh
        error_full, _ = breast_metadata.closest_distances(
            prone_ribcage_transformed, data['supine_ribcage_pc']
        )
        error_mag_full = np.linalg.norm(error_full, axis=1)

        # Calculate errors for selected mesh
        error_selected, _ = breast_metadata.closest_distances(
            prone_selected_transformed, data['supine_ribcage_pc_filtered']
        )
        error_mag_selected = np.linalg.norm(error_selected, axis=1)

        # Sternum error
        sternum_prone_transformed = apply_transform_to_coords(
            data['sternum_prone'], R, source_anchor, target_anchor
        )
        sternum_error = np.linalg.norm(sternum_prone_transformed[0] - data['sternum_supine'][0])

        return {
            'success': True,
            'method': 'point_to_point',
            'iterations': info.get('iterations', -1),
            'converged': info.get('converged', False),
            'sternum_error': sternum_error,
            # Full mesh metrics
            'full_rmse': float(np.sqrt(np.mean(error_mag_full ** 2))),
            'full_mean': float(np.mean(error_mag_full)),
            'full_std': float(np.std(error_mag_full)),
            'full_median': float(np.median(error_mag_full)),
            'full_n_points': len(error_mag_full),
            # Selected mesh metrics
            'selected_rmse': float(np.sqrt(np.mean(error_mag_selected ** 2))),
            'selected_mean': float(np.mean(error_mag_selected)),
            'selected_std': float(np.std(error_mag_selected)),
            'selected_median': float(np.median(error_mag_selected)),
            'selected_n_points': len(error_mag_selected),
            # Inlier metrics from alignment
            'inlier_rmse': info.get('euclidean_rmse', np.nan),
            'inlier_fraction': info.get('inlier_fraction', np.nan),
        }

    except Exception as e:
        return {
            'success': False,
            'method': 'point_to_point',
            'error_message': str(e),
        }


def run_surface_to_point_alignment(
    data: dict,
    max_distance: float,
    trim_percentage: float,
    res: int,
    max_iterations: int = 200,
    point_to_point_weight: float = 0.0,
) -> dict:
    """
    Run surface-to-point (plane-to-point) alignment with given parameters.

    Returns dict with RMSE, mean, std for full mesh and selected elements.
    """
    try:
        R, T_total, info = surface_to_point_align(
            mesh=data['prone_ribcage'],
            target_pts=data['supine_ribcage_pc_filtered'],
            source_sternum_sup=data['sternum_prone'][0],
            target_sternum_sup=data['sternum_supine'][0],
            max_distance=max_distance,
            max_iterations=max_iterations,
            convergence_threshold=1e-6,
            trim_percentage=trim_percentage,
            res=res,
            verbose=False,
            elems=SELECTED_ELEMENTS,
            point_to_point_weight=point_to_point_weight,
        )

        # Transform full mesh and selected mesh
        source_anchor = data['sternum_prone'][0]
        target_anchor = data['sternum_supine'][0]

        prone_ribcage_transformed = apply_transform_to_coords(
            data['prone_ribcage_mesh_coords'], R, source_anchor, target_anchor
        )

        prone_selected_transformed = apply_transform_to_coords(
            data['prone_ribcage_selected_coords'], R, source_anchor, target_anchor
        )

        # Calculate errors for full mesh
        error_full, _ = breast_metadata.closest_distances(
            prone_ribcage_transformed, data['supine_ribcage_pc']
        )
        error_mag_full = np.linalg.norm(error_full, axis=1)

        # Calculate errors for selected mesh
        error_selected, _ = breast_metadata.closest_distances(
            prone_selected_transformed, data['supine_ribcage_pc_filtered']
        )
        error_mag_selected = np.linalg.norm(error_selected, axis=1)

        # Sternum error
        sternum_prone_transformed = apply_transform_to_coords(
            data['sternum_prone'], R, source_anchor, target_anchor
        )
        sternum_error = np.linalg.norm(sternum_prone_transformed[0] - data['sternum_supine'][0])

        return {
            'success': True,
            'method': 'surface_to_point',
            'iterations': info.get('iterations', -1),
            'converged': info.get('converged', False),
            'sternum_error': sternum_error,
            # Full mesh metrics
            'full_rmse': float(np.sqrt(np.mean(error_mag_full ** 2))),
            'full_mean': float(np.mean(error_mag_full)),
            'full_std': float(np.std(error_mag_full)),
            'full_median': float(np.median(error_mag_full)),
            'full_n_points': len(error_mag_full),
            # Selected mesh metrics
            'selected_rmse': float(np.sqrt(np.mean(error_mag_selected ** 2))),
            'selected_mean': float(np.mean(error_mag_selected)),
            'selected_std': float(np.std(error_mag_selected)),
            'selected_median': float(np.median(error_mag_selected)),
            'selected_n_points': len(error_mag_selected),
            # Inlier metrics from alignment
            'inlier_rmse': info.get('inlier_rmse', np.nan),
            'inlier_fraction': info.get('inlier_fraction', np.nan),
        }

    except Exception as e:
        return {
            'success': False,
            'method': 'surface_to_point',
            'error_message': str(e),
        }


# =============================================================================
# Main Test Functions
# =============================================================================

def test_single_subject_all_params(
    vl_id: int,
    data: dict,
) -> List[dict]:
    """
    Test all parameter combinations for a single subject.
    """
    results = []

    # Test default parameters first
    print(f"  Testing default parameters...")

    # Point-to-point with defaults
    result = run_point_to_point_alignment(
        data,
        max_correspondence_distance=DEFAULT_PARAMS['max_correspondence_distance'],
        trim_percentage=DEFAULT_PARAMS['trim_percentage'],
        max_iterations=DEFAULT_PARAMS['max_iterations'],
    )
    result['vl_id'] = vl_id
    result['max_correspondence_distance'] = DEFAULT_PARAMS['max_correspondence_distance']
    result['trim_percentage'] = DEFAULT_PARAMS['trim_percentage']
    result['res'] = 'N/A'
    result['point_to_point_weight'] = 'N/A'
    results.append(result)

    # Surface-to-point with defaults
    result = run_surface_to_point_alignment(
        data,
        max_distance=DEFAULT_PARAMS['max_correspondence_distance'],
        trim_percentage=DEFAULT_PARAMS['trim_percentage'],
        res=DEFAULT_PARAMS['res'],
        max_iterations=DEFAULT_PARAMS['max_iterations'],
    )
    result['vl_id'] = vl_id
    result['max_correspondence_distance'] = DEFAULT_PARAMS['max_correspondence_distance']
    result['trim_percentage'] = DEFAULT_PARAMS['trim_percentage']
    result['res'] = DEFAULT_PARAMS['res']
    result['point_to_point_weight'] = 0.0
    results.append(result)

    # Test parameter variations for point-to-point
    print(f"  Testing point-to-point parameter grid...")
    for max_dist in PARAM_GRID['max_correspondence_distance']:
        for trim in PARAM_GRID['trim_percentage']:
            if max_dist == DEFAULT_PARAMS['max_correspondence_distance'] and \
               trim == DEFAULT_PARAMS['trim_percentage']:
                continue  # Skip default (already tested)

            result = run_point_to_point_alignment(
                data,
                max_correspondence_distance=max_dist,
                trim_percentage=trim,
            )
            result['vl_id'] = vl_id
            result['max_correspondence_distance'] = max_dist
            result['trim_percentage'] = trim
            result['res'] = 'N/A'
            result['point_to_point_weight'] = 'N/A'
            results.append(result)

    # Test parameter variations for surface-to-point
    print(f"  Testing surface-to-point parameter grid...")
    for max_dist in PARAM_GRID['max_correspondence_distance']:
        for trim in PARAM_GRID['trim_percentage']:
            for res in PARAM_GRID['res']:
                if max_dist == DEFAULT_PARAMS['max_correspondence_distance'] and \
                   trim == DEFAULT_PARAMS['trim_percentage'] and \
                   res == DEFAULT_PARAMS['res']:
                    continue  # Skip default

                result = run_surface_to_point_alignment(
                    data,
                    max_distance=max_dist,
                    trim_percentage=trim,
                    res=res,
                )
                result['vl_id'] = vl_id
                result['max_correspondence_distance'] = max_dist
                result['trim_percentage'] = trim
                result['res'] = res
                result['point_to_point_weight'] = 0.0
                results.append(result)

    return results


def test_cohort_default_params(vl_ids: List[int]) -> pd.DataFrame:
    """
    Test all subjects with default parameters only.
    Quick test to validate alignment across cohort.
    """
    all_results = []

    print(f"\n{'='*70}")
    print(f"COHORT ALIGNMENT TEST - DEFAULT PARAMETERS")
    print(f"Selected elements: {SELECTED_ELEMENTS}")
    print(f"Default params: {DEFAULT_PARAMS}")
    print(f"{'='*70}\n")

    for i, vl_id in enumerate(vl_ids):
        print(f"[{i+1}/{len(vl_ids)}] Processing VL{vl_id:05d}...")

        # Load data
        data, success = load_alignment_data(vl_id)
        if not success:
            # Add failed entry
            all_results.append({
                'vl_id': vl_id,
                'success': False,
                'method': 'point_to_point',
                'error_message': 'Failed to load data',
            })
            all_results.append({
                'vl_id': vl_id,
                'success': False,
                'method': 'surface_to_point',
                'error_message': 'Failed to load data',
            })
            continue

        # Point-to-point alignment
        result_ptp = run_point_to_point_alignment(
            data,
            max_correspondence_distance=DEFAULT_PARAMS['max_correspondence_distance'],
            trim_percentage=DEFAULT_PARAMS['trim_percentage'],
        )
        result_ptp['vl_id'] = vl_id
        result_ptp['max_correspondence_distance'] = DEFAULT_PARAMS['max_correspondence_distance']
        result_ptp['trim_percentage'] = DEFAULT_PARAMS['trim_percentage']
        result_ptp['res'] = 'N/A'
        all_results.append(result_ptp)

        if result_ptp['success']:
            print(f"  Point-to-Point: RMSE={result_ptp['full_rmse']:.2f}mm (full), "
                  f"{result_ptp['selected_rmse']:.2f}mm (selected)")

        # Surface-to-point alignment
        result_stp = run_surface_to_point_alignment(
            data,
            max_distance=DEFAULT_PARAMS['max_correspondence_distance'],
            trim_percentage=DEFAULT_PARAMS['trim_percentage'],
            res=DEFAULT_PARAMS['res'],
        )
        result_stp['vl_id'] = vl_id
        result_stp['max_correspondence_distance'] = DEFAULT_PARAMS['max_correspondence_distance']
        result_stp['trim_percentage'] = DEFAULT_PARAMS['trim_percentage']
        result_stp['res'] = DEFAULT_PARAMS['res']
        all_results.append(result_stp)

        if result_stp['success']:
            print(f"  Surface-to-Point: RMSE={result_stp['full_rmse']:.2f}mm (full), "
                  f"{result_stp['selected_rmse']:.2f}mm (selected)")

    df = pd.DataFrame(all_results)
    return df


def test_cohort_full_param_search(vl_ids: List[int]) -> pd.DataFrame:
    """
    Full parameter search across all subjects.
    WARNING: This will take a long time!
    """
    all_results = []

    print(f"\n{'='*70}")
    print(f"FULL PARAMETER SEARCH - ALL SUBJECTS")
    print(f"Selected elements: {SELECTED_ELEMENTS}")
    print(f"Parameter grid: {PARAM_GRID}")
    print(f"{'='*70}\n")

    total_subjects = len(vl_ids)

    for i, vl_id in enumerate(vl_ids):
        print(f"\n[{i+1}/{total_subjects}] Processing VL{vl_id:05d}...")

        # Load data
        data, success = load_alignment_data(vl_id)
        if not success:
            continue

        # Test all parameter combinations
        subject_results = test_single_subject_all_params(vl_id, data)
        all_results.extend(subject_results)

        # Print summary for this subject
        successful = [r for r in subject_results if r.get('success', False)]
        if successful:
            ptp_results = [r for r in successful if r['method'] == 'point_to_point']
            stp_results = [r for r in successful if r['method'] == 'surface_to_point']

            if ptp_results:
                best_ptp = min(ptp_results, key=lambda x: x.get('full_rmse', np.inf))
                print(f"  Best Point-to-Point: RMSE={best_ptp['full_rmse']:.2f}mm "
                      f"(dist={best_ptp['max_correspondence_distance']}, "
                      f"trim={best_ptp['trim_percentage']})")

            if stp_results:
                best_stp = min(stp_results, key=lambda x: x.get('full_rmse', np.inf))
                print(f"  Best Surface-to-Point: RMSE={best_stp['full_rmse']:.2f}mm "
                      f"(dist={best_stp['max_correspondence_distance']}, "
                      f"trim={best_stp['trim_percentage']}, res={best_stp['res']})")

    df = pd.DataFrame(all_results)
    return df


def analyze_results(df: pd.DataFrame) -> dict:
    """
    Analyze results to find optimal parameters.
    """
    analysis = {}

    # Filter successful results
    df_success = df[df['success'] == True].copy()

    if len(df_success) == 0:
        print("No successful alignments to analyze!")
        return analysis

    # Separate by method
    df_ptp = df_success[df_success['method'] == 'point_to_point']
    df_stp = df_success[df_success['method'] == 'surface_to_point']

    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")

    # Point-to-Point Analysis
    if len(df_ptp) > 0:
        print(f"\n--- Point-to-Point Alignment ---")
        print(f"Total successful: {len(df_ptp)}")

        # Cohort-level statistics
        print(f"\nCohort Statistics (Full Mesh):")
        print(f"  RMSE: {df_ptp['full_rmse'].mean():.2f} Â+/- {df_ptp['full_rmse'].std():.2f} mm")
        print(f"  Mean: {df_ptp['full_mean'].mean():.2f} Â+/- {df_ptp['full_mean'].std():.2f} mm")
        print(f"  Range: {df_ptp['full_rmse'].min():.2f} - {df_ptp['full_rmse'].max():.2f} mm")

        print(f"\nCohort Statistics (Selected Mesh):")
        print(f"  RMSE: {df_ptp['selected_rmse'].mean():.2f} Â+/- {df_ptp['selected_rmse'].std():.2f} mm")
        print(f"  Mean: {df_ptp['selected_mean'].mean():.2f} Â+/- {df_ptp['selected_mean'].std():.2f} mm")

        # Find optimal parameters (by mean RMSE across subjects)
        if 'max_correspondence_distance' in df_ptp.columns:
            param_summary = df_ptp.groupby(['max_correspondence_distance', 'trim_percentage']).agg({
                'full_rmse': ['mean', 'std', 'count'],
                'selected_rmse': ['mean', 'std'],
            }).reset_index()

            # Find best parameter combination
            param_summary.columns = ['_'.join(col).strip('_') for col in param_summary.columns.values]
            best_idx = param_summary['full_rmse_mean'].idxmin()
            best_params = param_summary.iloc[best_idx]

            print(f"\nOptimal Parameters (Point-to-Point):")
            print(f"  max_correspondence_distance: {best_params['max_correspondence_distance']}")
            print(f"  trim_percentage: {best_params['trim_percentage']}")
            print(f"  â†’ Full RMSE: {best_params['full_rmse_mean']:.2f} Â+/- {best_params['full_rmse_std']:.2f} mm")

            analysis['ptp_optimal_params'] = {
                'max_correspondence_distance': best_params['max_correspondence_distance'],
                'trim_percentage': best_params['trim_percentage'],
                'full_rmse_mean': best_params['full_rmse_mean'],
                'full_rmse_std': best_params['full_rmse_std'],
            }

    # Surface-to-Point Analysis
    if len(df_stp) > 0:
        print(f"\n--- Surface-to-Point Alignment ---")
        print(f"Total successful: {len(df_stp)}")

        # Cohort-level statistics
        print(f"\nCohort Statistics (Full Mesh):")
        print(f"  RMSE: {df_stp['full_rmse'].mean():.2f} Â+/- {df_stp['full_rmse'].std():.2f} mm")
        print(f"  Mean: {df_stp['full_mean'].mean():.2f} Â+/- {df_stp['full_mean'].std():.2f} mm")
        print(f"  Range: {df_stp['full_rmse'].min():.2f} - {df_stp['full_rmse'].max():.2f} mm")

        print(f"\nCohort Statistics (Selected Mesh):")
        print(f"  RMSE: {df_stp['selected_rmse'].mean():.2f} Â+/- {df_stp['selected_rmse'].std():.2f} mm")
        print(f"  Mean: {df_stp['selected_mean'].mean():.2f} Â+/- {df_stp['selected_mean'].std():.2f} mm")

        # Find optimal parameters
        if 'res' in df_stp.columns and df_stp['res'].dtype != object:
            param_summary = df_stp.groupby(['max_correspondence_distance', 'trim_percentage', 'res']).agg({
                'full_rmse': ['mean', 'std', 'count'],
                'selected_rmse': ['mean', 'std'],
            }).reset_index()

            param_summary.columns = ['_'.join(col).strip('_') for col in param_summary.columns.values]
            best_idx = param_summary['full_rmse_mean'].idxmin()
            best_params = param_summary.iloc[best_idx]

            print(f"\nOptimal Parameters (Surface-to-Point):")
            print(f"  max_correspondence_distance: {best_params['max_correspondence_distance']}")
            print(f"  trim_percentage: {best_params['trim_percentage']}")
            print(f"  res: {best_params['res']}")
            print(f"  â†’ Full RMSE: {best_params['full_rmse_mean']:.2f} Â+/- {best_params['full_rmse_std']:.2f} mm")

            analysis['stp_optimal_params'] = {
                'max_correspondence_distance': best_params['max_correspondence_distance'],
                'trim_percentage': best_params['trim_percentage'],
                'res': best_params['res'],
                'full_rmse_mean': best_params['full_rmse_mean'],
                'full_rmse_std': best_params['full_rmse_std'],
            }

    # Per-subject optimal
    print(f"\n--- Per-Subject Optimal Parameters ---")
    for vl_id in df_success['vl_id'].unique():
        df_subj = df_success[df_success['vl_id'] == vl_id]
        if len(df_subj) > 0:
            best = df_subj.loc[df_subj['full_rmse'].idxmin()]
            print(f"  VL{vl_id:05d}: RMSE={best['full_rmse']:.2f}mm, method={best['method']}, "
                  f"dist={best['max_correspondence_distance']}, trim={best['trim_percentage']}")

    return analysis


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test alignment parameters across cohort')
    parser.add_argument('--mode', choices=['default', 'full', 'quick'], default='default',
                       help='Test mode: default (default params only), full (all params), quick (subset)')
    parser.add_argument('--vl_ids', type=str, default=None,
                       help='Comma-separated list of VL IDs to test (default: all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output Excel filename')

    args = parser.parse_args()

    # Determine VL IDs to test
    if args.vl_ids:
        vl_ids = [int(x.strip()) for x in args.vl_ids.split(',')]
    else:
        vl_ids = ALL_VL_IDS

    # Quick mode: test first 5 subjects only
    if args.mode == 'quick':
        vl_ids = vl_ids[:5]
        print(f"Quick mode: testing {len(vl_ids)} subjects")

    # Run tests
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == 'full':
        df_results = test_cohort_full_param_search(vl_ids)
    else:
        df_results = test_cohort_default_params(vl_ids)

    # Analyze results
    analysis = analyze_results(df_results)

    # Save results
    if args.output:
        output_file = OUTPUT_ROOT / args.output
    else:
        output_file = OUTPUT_ROOT / f"alignment_test_results_{timestamp}.xlsx"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All Results', index=False)

        # Summary by method
        df_success = df_results[df_results['success'] == True]
        if len(df_success) > 0:
            # Point-to-point summary
            df_ptp = df_success[df_success['method'] == 'point_to_point']
            if len(df_ptp) > 0:
                summary_ptp = df_ptp.groupby('vl_id').agg({
                    'full_rmse': 'min',
                    'full_mean': 'min',
                    'full_std': 'min',
                    'selected_rmse': 'min',
                    'selected_mean': 'min',
                    'selected_std': 'min',
                    'sternum_error': 'min',
                }).reset_index()
                summary_ptp.to_excel(writer, sheet_name='PTP Summary', index=False)

            # Surface-to-point summary
            df_stp = df_success[df_success['method'] == 'surface_to_point']
            if len(df_stp) > 0:
                summary_stp = df_stp.groupby('vl_id').agg({
                    'full_rmse': 'min',
                    'full_mean': 'min',
                    'full_std': 'min',
                    'selected_rmse': 'min',
                    'selected_mean': 'min',
                    'selected_std': 'min',
                    'sternum_error': 'min',
                }).reset_index()
                summary_stp.to_excel(writer, sheet_name='STP Summary', index=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")







