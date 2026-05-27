"""
Standalone script for cleaning ribcage point clouds.

Loads a supine ribcage segmentation mask, extracts surface points,
and applies a sequence of geometric and statistical filters to produce
a clean point cloud suitable for alignment.

Filtering steps:
    1. Remove top and bottom axial slices
    2. Remove spine region (posterior midline)
    3. Remove posterior-inferior corner points
    4. Statistical outlier removal (SOR)

Each step is visualised when run_plot_all=True. Parameters can be
loaded from / saved to a JSON config file for reproducibility.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from scipy.spatial import KDTree

import pyvista as pv
import external.breast_metadata_mdv.breast_metadata as breast_metadata
from utils import extract_contour_points
from utils_plot import plot_all


def filter_point_cloud_asymmetric(points, reference, tol_min, tol_max, axis):
    """
    Filters points along an axis using different tolerances for min and max bounds.
    """
    min_ref = np.min(reference[:, axis])
    max_ref = np.max(reference[:, axis])

    keep_idx = [idx for idx, pt in enumerate(points)
                if min_ref + tol_min < pt[axis] < max_ref - tol_max]

    return points[keep_idx]


def clean_ribcage_point_cloud(
        pc_data: np.ndarray,
        image_grid,
        config_filepath: Optional[str] = None,
        # --- Geometric Cropping Parameters ---
        z_voxels_bottom: float = 20,
        z_voxels_top: float = 5,
        x_spine_offset: float = 25.0,
        y_spine_offset: float = 60.0,
        x_lateral_margin: float = 90.0,
        y_posterior_margin: float = 60.0,
        z_inferior_margin: float = 25.0,
        # --- Statistical Outlier Removal (SOR) Parameters ---
        sor_k_neighbors: int = 50,
        sor_std_multiplier: float = 1.0,
        run_plot_all: bool = True
) -> np.ndarray:
    """
    Clean a ribcage point cloud by removing noisy regions and outliers.

    Args:
        pc_data: (N, 3) raw point cloud from segmentation contour
        image_grid: object with .spacing attribute (voxel sizes in mm)
        config_filepath: optional JSON file to load/save parameters
        z_voxels_bottom: voxels to remove from inferior (bottom) edge
        z_voxels_top: voxels to remove from superior (top) edge
        x_spine_offset: lateral offset from median X defining spine region (mm)
        y_spine_offset: anterior offset from posterior edge for spine (mm)
        x_lateral_margin: lateral margin for posterior-inferior removal (mm)
        y_posterior_margin: posterior margin for posterior-inferior removal (mm)
        z_inferior_margin: inferior margin for posterior-inferior removal (mm)
        sor_k_neighbors: number of neighbours for SOR
        sor_std_multiplier: std multiplier for SOR threshold
        run_plot_all: show diagnostic plots after each step

    Returns:
        (M, 3) cleaned point cloud
    """
    # ----------------------------------------------------
    # I. Configuration Loading and Setup
    # ----------------------------------------------------
    params = {
        'z_voxels_bottom': z_voxels_bottom,
        'z_voxels_top': z_voxels_top,
        'x_spine_offset': x_spine_offset,
        'y_spine_offset': y_spine_offset,
        'x_lateral_margin': x_lateral_margin,
        'y_posterior_margin': y_posterior_margin,
        'z_inferior_margin': z_inferior_margin,
        'sor_k_neighbors': sor_k_neighbors,
        'sor_std_multiplier': sor_std_multiplier,
    }

    if config_filepath and os.path.exists(config_filepath):
        try:
            with open(config_filepath, 'r') as f:
                loaded_params = json.load(f)
            params.update(loaded_params)
            print(f"INFO: Loaded parameters from {config_filepath}")
        except Exception as e:
            print(f"WARNING: Could not load parameters from file: {e}. Using defaults.")

    # Unpack parameters
    z_voxels_bottom = params['z_voxels_bottom']
    z_voxels_top = params['z_voxels_top']
    x_spine_offset = params['x_spine_offset']
    y_spine_offset = params['y_spine_offset']
    x_lateral_margin = params['x_lateral_margin']
    y_posterior_margin = params['y_posterior_margin']
    z_inferior_margin = params['z_inferior_margin']
    sor_k_neighbors = params['sor_k_neighbors']
    sor_std_multiplier = params['sor_std_multiplier']

    # ----------------------------------------------------
    # II. Filtering Logic
    # ----------------------------------------------------
    print(f"Original point cloud size: {pc_data.shape}")
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    # Step 1: Remove top and bottom axial slices
    pc_data = filter_point_cloud_asymmetric(
        points=pc_data,
        reference=pc_data,
        tol_min=image_grid.spacing[2] * z_voxels_bottom,
        # tol_max=image_grid.spacing[2] * z_voxels_top,
        tol_max=0,
        axis=2,
    )
    print(f"After removing top/bottom axial slices: {pc_data.shape}")
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    # Step 2: Remove spine region (posterior midline)
    x_median = np.median(pc_data[:, 0])
    pc_data = pc_data[
        (pc_data[:, 0] <= (x_median - x_spine_offset)) |
        (pc_data[:, 0] >= (x_median + x_spine_offset)) |
        (pc_data[:, 1] < np.max(pc_data[:, 1]) - y_spine_offset)
    ]
    print(f"After removing spine region: {pc_data.shape}")
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    # Step 3: Remove posterior-inferior corner points
    x_median = np.median(pc_data[:, 0])
    pc_data = pc_data[
        (pc_data[:, 0] <= (x_median - x_lateral_margin)) |
        (pc_data[:, 0] >= (x_median + x_lateral_margin)) |
        (pc_data[:, 1] < np.max(pc_data[:, 1]) - y_posterior_margin) |
        (pc_data[:, 2] >= (np.min(pc_data[:, 2]) + z_inferior_margin))
    ]
    print(f"After removing posterior-inferior points: {pc_data.shape}")
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    # Step 4: Statistical outlier removal (SOR)
    try:
        tree = KDTree(pc_data)
    except Exception as e:
        print(f"ERROR: Could not build KDTree: {e}")
        return pc_data

    distances, _ = tree.query(pc_data, k=sor_k_neighbors + 1)
    kth_dist = distances[:, sor_k_neighbors]

    mean_dist = np.mean(kth_dist)
    std_dev = np.std(kth_dist)
    threshold = mean_dist + sor_std_multiplier * std_dev

    inliers_mask = kth_dist < threshold
    pc_data = pc_data[inliers_mask]
    print(f"After statistical outlier removal: {pc_data.shape}")
    if run_plot_all:
        plot_all(point_cloud=pc_data)

    # ----------------------------------------------------
    # III. Configuration Saving
    # ----------------------------------------------------
    if config_filepath and not os.path.exists(config_filepath):
        try:
            config_dir = os.path.dirname(config_filepath)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            with open(config_filepath, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"INFO: Saved current parameters to {config_filepath}")
        except Exception as e:
            print(f"WARNING: Could not save parameters to file: {e}")

    return pc_data


if __name__ == "__main__":
    # ======================================================================
    # Configuration
    # ======================================================================
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
    OUTPUT_ROOT = Path(__file__).parent.parent / "output"
    CONFIG_ROOT = OUTPUT_ROOT / "clean_up_pc_config"

    orientation_flag = 'RAI'
    nb_points = 20000
    run_plot_all = True

    # ======================================================================
    # Mode selection
    # ======================================================================
    # Mode 1: "batch"  - Load existing configs and apply to all subjects
    # Mode 2: "new"    - Run with defaults for specific subjects, save configs
    MODE = "batch"

    # For "new" mode: specify subjects to process (configs will be saved)
    NEW_VL_IDS = [9]

    # ======================================================================
    # Determine subject list based on mode
    # ======================================================================
    if MODE == "batch":
        # Auto-detect subjects from existing config files
        config_files = sorted(CONFIG_ROOT.glob("VL*_config.json"))
        if not config_files:
            print(f"ERROR: No config files found in {CONFIG_ROOT}")
            exit(1)

        vl_ids = []
        for cf in config_files:
            vl_str = cf.stem.replace("_config", "")  # "VL00009"
            vl_num = int(vl_str.replace("VL", ""))
            vl_ids.append(vl_num)

        print(f"MODE: batch — applying existing configs")
        print(f"Found {len(vl_ids)} config files in {CONFIG_ROOT}")
        print(f"Subjects: {['VL' + str(v).zfill(5) for v in vl_ids]}")

    elif MODE == "new":
        vl_ids = NEW_VL_IDS
        print(f"MODE: new — running with defaults, will save configs")
        print(f"Subjects: {['VL' + str(v).zfill(5) for v in vl_ids]}")

    else:
        print(f"ERROR: Unknown MODE '{MODE}'. Use 'batch' or 'new'.")
        exit(1)

    # ======================================================================
    # Process subjects
    # ======================================================================
    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        print(f"\n{'='*60}")
        print(f"CLEAN RIBCAGE POINT CLOUD")
        print(f"Subject: {vl_id_str}")
        print(f"{'='*60}")

        config_filepath = str(CONFIG_ROOT / f"{vl_id_str}_config.json")

        if MODE == "batch":
            # Config must exist
            if not os.path.exists(config_filepath):
                print(f"  ERROR: Config not found: {config_filepath}")
                continue
            with open(config_filepath, 'r') as f:
                config = json.load(f)
            print(f"  Loaded config: {config}")

        elif MODE == "new":
            # Use defaults; config_filepath passed to clean function
            # will save if file doesn't exist yet
            if os.path.exists(config_filepath):
                print(f"  WARNING: Config already exists: {config_filepath}")
                print(f"  Will load existing config. Delete it first to use fresh defaults.")
            else:
                print(f"  No existing config — will use defaults and save to:")
                print(f"    {config_filepath}")

        # Load supine ribcage segmentation
        supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"
        if not supine_seg_file.exists():
            print(f"  ERROR: Segmentation not found: {supine_seg_file}")
            continue

        supine_ribcage_mask = breast_metadata.readNIFTIImage(
            str(supine_seg_file), orientation_flag, swap_axes=True
        )

        # Extract surface contour points
        supine_ribcage_pc = extract_contour_points(supine_ribcage_mask, nb_points)
        print(f"  Extracted {supine_ribcage_pc.shape[0]} contour points")

        # Build image grid for voxel spacing
        image_grid = breast_metadata.SCANToPyvistaImageGrid(
            supine_ribcage_mask, orientation_flag
        )

        # Clean the point cloud
        cleaned_pc = clean_ribcage_point_cloud(
            pc_data=supine_ribcage_pc,
            image_grid=image_grid,
            config_filepath=config_filepath,
            run_plot_all=run_plot_all,
        )

        print(f"\n  Final: {cleaned_pc.shape[0]} points "
              f"({cleaned_pc.shape[0]/supine_ribcage_pc.shape[0]*100:.0f}% retained)")
