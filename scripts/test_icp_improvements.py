"""
Test script for improved ICP parameters

This script demonstrates the improved ICP with different parameter sets
and shows how to interpret the results.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.align_fixed_sternum import align_prone_to_supine_fixed_sternum
from scripts.readers import load_all_subjects


def test_icp_improvements():
    """Test the improved ICP implementation"""

    print("="*80)
    print("TESTING IMPROVED ICP PARAMETERS")
    print("="*80)

    # Load one subject for testing
    SEGMENTATION_ROOT = Path("U:/sandbox/jxu759/volunteer_seg/results")
    METADATA_JSON = Path("../external/breast_metadata_mdv/metadata_volunteers.json")

    print("\nLoading test subject...")
    all_subjects = load_all_subjects(METADATA_JSON)

    # Filter to subjects with complete data
    test_subjects = {
        vl_id: subj for vl_id, subj in all_subjects.items()
        if vl_id in [14, 18, 19, 20]  # Known good subjects
    }

    if not test_subjects:
        print("Error: No test subjects found")
        return

    # Test with first available subject
    test_id = min(test_subjects.keys())
    test_subject = test_subjects[test_id]

    print(f"\nTesting with subject VL{test_id:05d}")
    print("-"*80)

    # Setup paths
    prone_rib_mesh = SEGMENTATION_ROOT / "prone" / "fitted_mesh" / f"VL{test_id:05d}_rib_fitted_mesh_prone_t2.mesh"
    supine_rib_seg = SEGMENTATION_ROOT / "supine" / "landmarks" / f"VL{test_id:05d}_seg_ribs_supine_t2.nii.gz"

    if not prone_rib_mesh.exists():
        print(f"Error: Prone mesh not found: {prone_rib_mesh}")
        return
    if not supine_rib_seg.exists():
        print(f"Error: Supine segmentation not found: {supine_rib_seg}")
        return

    # Run alignment with improved parameters
    print("\nRunning alignment with IMPROVED parameters...")
    print("Expected: Better convergence, lower RMSE, more stable")
    print()

    results = align_prone_to_supine_fixed_sternum(
        subject=test_subject,
        prone_ribcage_mesh_path=prone_rib_mesh,
        supine_ribcage_seg_path=supine_rib_seg,
        plot_for_debug=False,
        w_rib=1.0,
        w_sternum=100.0
    )

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nAlignment Quality:")
    print(f"  Sternum Superior Error: {results['sternum_error'][0]:.6f} mm (should be ~0)")
    print(f"  Sternum Inferior Error: {results['sternum_error'][1]:.2f} mm")
    print(f"  Mean Ribcage Error: {results['ribcage_error_mean']:.2f} mm")
    print(f"  Std Ribcage Error: {results['ribcage_error_std']:.2f} mm")
    print(f"  Inlier RMSE: {results['ribcage_inlier_RMSE']:.2f} mm")

    print(f"\nExpected Ranges:")
    print(f"  Sternum Superior: < 0.001 mm (locked)")
    print(f"  Sternum Inferior: 2-8 mm (flexible)")
    print(f"  Mean Ribcage: 4-7 mm")
    print(f"  Inlier RMSE: 2-4 mm")

    # Check if results are in expected range
    checks = {
        "Sternum locked": results['sternum_error'][0] < 0.01,
        "Inlier RMSE acceptable": results['ribcage_inlier_RMSE'] < 5.0,
        "Mean error acceptable": results['ribcage_error_mean'] < 10.0,
    }

    print(f"\nValidation Checks:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    if all(checks.values()):
        print("\n✓ All checks passed! ICP improvements working correctly.")
    else:
        print("\n⚠ Some checks failed. Review parameters or input data quality.")

    return results


def demonstrate_parameter_effects():
    """
    Show how different parameters affect ICP performance

    This is for educational purposes - demonstrates parameter sensitivity
    """

    print("\n" + "="*80)
    print("PARAMETER TUNING DEMONSTRATION")
    print("="*80)

    print("\nThis demonstrates how each parameter affects ICP performance:")
    print()

    print("1. max_correspondence_distance (Initial: 15mm)")
    print("   - Controls which point pairs are considered")
    print("   - Too small: May miss valid correspondences initially")
    print("   - Too large: Includes too many outliers")
    print("   - Recommended: 15-20mm for initial, reduce to 2mm by end")
    print()

    print("2. trim_percentage (Default: 0.15)")
    print("   - Rejects worst X% of correspondences each iteration")
    print("   - Too small: Outliers pull alignment off-target")
    print("   - Too large: May reject valid correspondences")
    print("   - Recommended: 0.10-0.20 for medical imaging")
    print()

    print("3. k_neighbors_normals (Default: 50)")
    print("   - Number of neighbors for surface normal estimation")
    print("   - Too small: Noisy normals, unstable point-to-plane distances")
    print("   - Too large: Over-smoothed, may miss fine details")
    print("   - Recommended: 30-80 depending on resolution")
    print()

    print("4. huber_delta (Default: 3.0mm)")
    print("   - Threshold for outlier rejection in loss function")
    print("   - Too small: Normal variations treated as outliers")
    print("   - Too large: True outliers affect optimization")
    print("   - Recommended: 2.0-4.0mm for ribcage alignment")
    print()

    print("5. convergence_threshold (Default: 1e-7 radians)")
    print("   - Minimum angle change to continue iterating")
    print("   - Too large: Stops before reaching optimal alignment")
    print("   - Too small: Wastes computation on negligible improvements")
    print("   - Recommended: 1e-7 to 1e-8 for precision")
    print()

    print("6. adaptive_correspondence (Default: True)")
    print("   - Enables coarse-to-fine correspondence distance")
    print("   - False: Fixed distance threshold (simpler, may fail)")
    print("   - True: Gradually reduces threshold (more robust)")
    print("   - Recommended: Always True unless testing")
    print()

    print("TIP: Start with conservative (slow but accurate) parameters")
    print("     Then optimize for speed once you have ground truth")
    print()


if __name__ == "__main__":
    try:
        # Run test
        test_results = test_icp_improvements()

        # Show parameter guide
        demonstrate_parameter_effects()

        print("\n" + "="*80)
        print("For detailed parameter tuning guide, see:")
        print("  scripts/ICP_IMPROVEMENTS_GUIDE.md")
        print("="*80)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
