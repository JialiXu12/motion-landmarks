"""
Test script for alignment.py element selection and visualization
"""
import sys
from pathlib import Path

# Add src folder to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "morphic"))
sys.path.insert(0, str(project_root / "src" / "breast-metadata"))
sys.path.insert(0, str(project_root / "src" / "mesh-tools"))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from typing import Dict
from readers import load_subject
from structures import Subject
from alignment import (
    align_prone_to_supine_optimal,
    select_elements_by_region,
    get_mesh_elements_2,
    plot_mesh_elements,
    get_surface_mesh_coords,
    get_mesh_with_selected_elements
)
import morphic
import numpy as np

# Test paths
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")


def test_plot_mesh_elements_all():
    """Test plot_mesh_elements with all elements"""
    print("=" * 60)
    print("TEST 1: plot_mesh_elements with all elements")
    print("=" * 60)

    vl_id = 22
    vl_id_str = f"VL{vl_id:05d}"

    # Load prone mesh
    prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
    print(f"Loading mesh from {prone_mesh_file}...")
    prone_ribcage = morphic.Mesh(str(prone_mesh_file))

    # Get mesh coordinates
    prone_ribcage_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)
    print(f"Mesh has {prone_ribcage_mesh_coords.shape[0]} points")

    # Get element centers
    centers_array, num_elements = get_mesh_elements_2(prone_ribcage)
    print(f"Mesh has {num_elements} elements")
    print(f"Centers array shape: {centers_array.shape}")

    # Test: element_indices = range(num_elements) should work
    element_indices = range(num_elements)
    labels = [str(i) for i in element_indices]
    print(f"Number of labels: {len(labels)}")
    print(f"Number of centers: {len(centers_array)}")

    assert len(labels) == len(centers_array), \
        f"Mismatch: {len(labels)} labels vs {len(centers_array)} centers"

    print("✓ Labels and centers match!")

    # Plot all elements
    print("\nPlotting all elements...")
    plot_mesh_elements(prone_ribcage_mesh_coords, centers_array, element_indices)
    print("✓ Plot completed successfully!")

    return True


def test_plot_mesh_elements_selected():
    """Test plot_mesh_elements with selected elements"""
    print("\n" + "=" * 60)
    print("TEST 2: plot_mesh_elements with selected elements")
    print("=" * 60)

    vl_id = 22
    vl_id_str = f"VL{vl_id:05d}"

    # Load prone mesh
    prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
    prone_ribcage = morphic.Mesh(str(prone_mesh_file))

    # Get mesh coordinates and element centers
    prone_ribcage_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)
    centers_array, num_elements = get_mesh_elements_2(prone_ribcage)

    # Select elements
    print("\nSelecting elements (anterior + superior)...")
    selected_elements = select_elements_by_region(
        prone_ribcage,
        y_region='anterior',
        y_percentile=70.0,
        z_region='superior',
        z_percentile=40.0,
        verbose=True
    )

    # Get selected mesh coordinates and centers
    selected_mesh_coords = get_mesh_with_selected_elements(prone_ribcage, selected_elements, res=26)
    selected_centers = centers_array[selected_elements]

    # Test: labels should match selected centers
    labels = [str(i) for i in selected_elements]
    print(f"\nNumber of selected elements: {len(selected_elements)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Number of selected centers: {len(selected_centers)}")

    assert len(labels) == len(selected_centers), \
        f"Mismatch: {len(labels)} labels vs {len(selected_centers)} centers"

    print("✓ Labels and centers match!")

    # Plot selected elements
    print("\nPlotting selected elements...")
    plot_mesh_elements(selected_mesh_coords, selected_centers, selected_elements)
    print("✓ Plot completed successfully!")

    return True


def test_full_alignment():
    """Test full alignment workflow"""
    print("\n" + "=" * 60)
    print("TEST 3: Full alignment workflow")
    print("=" * 60)

    vl_id = 22
    vl_id_str = f"VL{vl_id:05d}"

    # Load subject
    print(f"Loading subject {vl_id_str}...")
    subject = load_subject(
        vl_id=vl_id,
        positions=["prone", "supine"],
        dicom_root=ROOT_PATH_MRI,
        anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
        soft_tissue_root=SOFT_TISSUE_ROOT
    )

    prone_mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
    supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

    print("\nRunning alignment...")
    alignment_results = align_prone_to_supine_optimal(
        subject=subject,
        prone_ribcage_mesh_path=prone_mesh_file,
        supine_ribcage_seg_path=supine_seg_file,
        y_region='anterior',
        y_percentile=70.0,
        z_region='superior',
        z_percentile=40.0,
        orientation_flag='RAI',
        plot_for_debug=True,
        visualize_iterations=False,
        verbose=True
    )

    print("\n✓ Alignment completed successfully!")
    print(f"  Sternum error: {alignment_results['sternum_error']:.4f} mm")
    print(f"  Ribcage RMSE: {alignment_results['ribcage_error_rmse']:.4f} mm")

    return True


if __name__ == "__main__":
    print("Running alignment tests...\n")

    try:
        # Test 1: All elements
        test_plot_mesh_elements_all()

        # Test 2: Selected elements
        test_plot_mesh_elements_selected()

        # Test 3: Full alignment
        test_full_alignment()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


