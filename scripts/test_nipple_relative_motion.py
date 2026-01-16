"""
Test file to verify the calculation of landmark motion relative to nipple.

This test creates synthetic data to verify that the differential motion vectors
are calculated correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))


def test_nipple_relative_motion():
    """
    Test the calculation of landmark motion relative to nipple.
    
    Scenario:
    - Sternum doesn't move (reference point)
    - Nipple moves 10mm in +X direction
    - Landmark moves 15mm in +X direction
    - Expected: Landmark moves 5mm relative to nipple in +X direction
    """
    
    print("=" * 80)
    print("TEST: Landmark Motion Relative to Nipple")
    print("=" * 80)
    
    # Setup: Sternum as reference (doesn't move)
    ref_sternum_prone = np.array([0., 0., 0.])
    ref_sternum_supine = np.array([0., 0., 0.])
    
    # Nipple positions
    # Prone: at (50, 0, 0)
    # Supine: at (60, 0, 0) - moved 10mm in +X
    nipple_prone = np.array([50., 0., 0.])
    nipple_supine = np.array([60., 0., 0.])
    
    # Landmark positions
    # Prone: at (70, 0, 0)
    # Supine: at (85, 0, 0) - moved 15mm in +X
    landmark_prone = np.array([70., 0., 0.])
    landmark_supine = np.array([85., 0., 0.])
    
    print("\n1. POSITIONS:")
    print(f"   Sternum (prone):   {ref_sternum_prone}")
    print(f"   Sternum (supine):  {ref_sternum_supine}")
    print(f"   Nipple (prone):    {nipple_prone}")
    print(f"   Nipple (supine):   {nipple_supine}")
    print(f"   Landmark (prone):  {landmark_prone}")
    print(f"   Landmark (supine): {landmark_supine}")
    
    # Calculate positions relative to sternum
    nipple_pos_prone_rel_sternum = nipple_prone - ref_sternum_prone
    nipple_pos_supine_rel_sternum = nipple_supine - ref_sternum_supine
    
    lm_pos_prone_rel_sternum = landmark_prone - ref_sternum_prone
    lm_pos_supine_rel_sternum = landmark_supine - ref_sternum_supine
    
    print("\n2. POSITIONS RELATIVE TO STERNUM:")
    print(f"   Nipple (prone):    {nipple_pos_prone_rel_sternum}")
    print(f"   Nipple (supine):   {nipple_pos_supine_rel_sternum}")
    print(f"   Landmark (prone):  {lm_pos_prone_rel_sternum}")
    print(f"   Landmark (supine): {lm_pos_supine_rel_sternum}")
    
    # Calculate displacements relative to sternum
    nipple_disp_rel_sternum = nipple_pos_supine_rel_sternum - nipple_pos_prone_rel_sternum
    lm_disp_rel_sternum = lm_pos_supine_rel_sternum - lm_pos_prone_rel_sternum
    
    print("\n3. DISPLACEMENTS RELATIVE TO STERNUM:")
    print(f"   Nipple displacement:   {nipple_disp_rel_sternum}")
    print(f"   Landmark displacement: {lm_disp_rel_sternum}")
    
    # Calculate displacement relative to nipple (CORRECT METHOD)
    lm_disp_rel_nipple_correct = lm_disp_rel_sternum - nipple_disp_rel_sternum
    
    print("\n4. DISPLACEMENT RELATIVE TO NIPPLE (CORRECT):")
    print(f"   lm_disp_rel_nipple = lm_disp_rel_sternum - nipple_disp_rel_sternum")
    print(f"   lm_disp_rel_nipple = {lm_disp_rel_sternum} - {nipple_disp_rel_sternum}")
    print(f"   lm_disp_rel_nipple = {lm_disp_rel_nipple_correct}")
    print(f"   Magnitude: {np.linalg.norm(lm_disp_rel_nipple_correct):.2f} mm")
    
    # For plotting: calculate V vectors
    # X: Initial position relative to prone nipple
    X = landmark_prone - nipple_prone
    
    print("\n5. FOR PLOTTING:")
    print(f"   X (initial pos rel to prone nipple): {X}")
    
    # V: Differential movement vector
    # CORRECT: V = landmark_displacement - nipple_displacement
    V_correct = lm_disp_rel_sternum - nipple_disp_rel_sternum
    
    # INCORRECT (current code): V = nipple_displacement - landmark_displacement
    V_incorrect = nipple_disp_rel_sternum - lm_disp_rel_sternum
    
    print(f"\n   V_correct (lm_disp - nipple_disp):   {V_correct}")
    print(f"   V_incorrect (nipple_disp - lm_disp): {V_incorrect}")
    
    # Verify the results
    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)
    
    expected_relative_motion = np.array([5., 0., 0.])  # 15mm - 10mm = 5mm in +X
    
    print(f"\nExpected: Landmark moves 5mm relative to nipple in +X direction")
    print(f"Expected vector: {expected_relative_motion}")
    print(f"\nCorrect calculation result:   {V_correct}")
    print(f"Incorrect calculation result: {V_incorrect}")
    
    # Check if correct method matches expected
    if np.allclose(V_correct, expected_relative_motion):
        print("\n✓ CORRECT method produces expected result!")
        print(f"  V_correct = lm_disp - nipple_disp = {V_correct}")
    else:
        print("\n✗ CORRECT method does NOT match expected result!")
        return False
    
    # Check if incorrect method is wrong
    if not np.allclose(V_incorrect, expected_relative_motion):
        print(f"✗ INCORRECT method produces wrong result!")
        print(f"  V_incorrect = nipple_disp - lm_disp = {V_incorrect}")
        print(f"  This is the OPPOSITE direction!")
    else:
        print("✓ INCORRECT method somehow matches (this shouldn't happen)")
        return False
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("\nThe current code in lines 1208 and 1217 is INCORRECT.")
    print("\nCurrent code:")
    print("  V_left = nipple_disp_left_vec - lm_disp_left")
    print("  V_right = nipple_disp_right_vec - lm_disp_right")
    print("\nShould be:")
    print("  V_left = lm_disp_left - nipple_disp_left_vec")
    print("  V_right = lm_disp_right - nipple_disp_right_vec")
    print("\nThis matches the calculation in line 1193:")
    print("  lm_disp_rel_nipple = lm_disp_rel_sternum - closest_nipple_disp_vec")
    print("=" * 80)
    
    return True


def test_multiple_landmarks():
    """
    Test with multiple landmarks on left and right breasts.
    """
    print("\n\n" + "=" * 80)
    print("TEST 2: Multiple Landmarks (Left and Right Breasts)")
    print("=" * 80)
    
    # Reference points
    ref_sternum = np.array([0., 0., 0.])
    
    # Nipples (left is index 0, right is index 1)
    # Left nipple: moves 8mm in +X
    nipple_prone = np.array([
        [60., 30., 0.],   # Left nipple (prone)
        [60., -30., 0.]   # Right nipple (prone)
    ])
    nipple_supine = np.array([
        [68., 30., 0.],   # Left nipple (supine) - moved 8mm in +X
        [65., -30., 0.]   # Right nipple (supine) - moved 5mm in +X
    ])
    
    # Landmarks (3 on left, 2 on right)
    landmark_prone = np.array([
        [80., 30., 10.],   # Left breast landmark 1
        [75., 35., 0.],    # Left breast landmark 2
        [70., 25., -5.],   # Left breast landmark 3
        [85., -25., 5.],   # Right breast landmark 1
        [78., -35., 0.]    # Right breast landmark 2
    ])
    
    landmark_supine = np.array([
        [90., 30., 10.],   # Left landmark 1 - moved 10mm in +X
        [86., 35., 0.],    # Left landmark 2 - moved 11mm in +X
        [82., 25., -5.],   # Left landmark 3 - moved 12mm in +X
        [92., -25., 5.],   # Right landmark 1 - moved 7mm in +X
        [84., -35., 0.]    # Right landmark 2 - moved 6mm in +X
    ])
    
    # Determine which breast each landmark belongs to
    dist_to_left = np.linalg.norm(landmark_supine - nipple_supine[0], axis=1)
    dist_to_right = np.linalg.norm(landmark_supine - nipple_supine[1], axis=1)
    is_left_breast = dist_to_left < dist_to_right
    
    print(f"\nLandmark assignment:")
    print(f"  is_left_breast: {is_left_breast}")
    print(f"  Left breast landmarks: {np.where(is_left_breast)[0]}")
    print(f"  Right breast landmarks: {np.where(~is_left_breast)[0]}")
    
    # Calculate displacements
    nipple_disp = nipple_supine - nipple_prone
    lm_disp = landmark_supine - landmark_prone
    
    print(f"\nNipple displacements:")
    print(f"  Left nipple:  {nipple_disp[0]}")
    print(f"  Right nipple: {nipple_disp[1]}")
    
    print(f"\nLandmark displacements:")
    for i, disp in enumerate(lm_disp):
        side = "Left" if is_left_breast[i] else "Right"
        print(f"  Landmark {i} ({side}): {disp}")
    
    # Assign closest nipple displacement to each landmark
    nipple_disp_left_vec = nipple_disp[0]
    nipple_disp_right_vec = nipple_disp[1]
    
    closest_nipple_disp_vec = np.where(
        is_left_breast[:, np.newaxis],
        nipple_disp_left_vec,
        nipple_disp_right_vec
    )
    
    # Calculate relative displacements
    lm_disp_rel_nipple = lm_disp - closest_nipple_disp_vec
    
    print(f"\nLandmark displacements RELATIVE TO NIPPLE:")
    for i, rel_disp in enumerate(lm_disp_rel_nipple):
        side = "Left" if is_left_breast[i] else "Right"
        mag = np.linalg.norm(rel_disp)
        print(f"  Landmark {i} ({side}): {rel_disp}, magnitude: {mag:.2f} mm")
    
    # Expected results:
    # Left landmarks moved 10, 11, 12 mm; left nipple moved 8mm
    # So relative motion should be 2, 3, 4 mm in +X
    # Right landmarks moved 7, 6 mm; right nipple moved 5mm
    # So relative motion should be 2, 1 mm in +X
    
    expected_rel_motion = np.array([
        [2., 0., 0.],   # Left 1: 10 - 8 = 2
        [3., 0., 0.],   # Left 2: 11 - 8 = 3
        [4., 0., 0.],   # Left 3: 12 - 8 = 4
        [2., 0., 0.],   # Right 1: 7 - 5 = 2
        [1., 0., 0.]    # Right 2: 6 - 5 = 1
    ])
    
    print(f"\nExpected relative motion:")
    for i, exp in enumerate(expected_rel_motion):
        side = "Left" if is_left_breast[i] else "Right"
        print(f"  Landmark {i} ({side}): {exp}")
    
    # Verify
    if np.allclose(lm_disp_rel_nipple, expected_rel_motion):
        print("\n✓ All relative motions calculated correctly!")
        return True
    else:
        print("\n✗ Relative motions do NOT match expected values!")
        print(f"\nDifference:")
        print(lm_disp_rel_nipple - expected_rel_motion)
        return False


if __name__ == "__main__":
    test1_passed = test_nipple_relative_motion()
    test2_passed = test_multiple_landmarks()
    
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    print(f"Test 1 (Single Landmark): {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (Multiple Landmarks): {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    
    if test1_passed and test2_passed:
        print("\nAll tests PASSED! ✓")
        print("\nThe fix needed:")
        print("  Line 1208: V_left = lm_disp_left - nipple_disp_left_vec")
        print("  Line 1217: V_right = lm_disp_right - nipple_disp_right_vec")
    else:
        print("\nSome tests FAILED! ✗")
    print("=" * 80)
