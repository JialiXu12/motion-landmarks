"""
Fixed Sternum Point-to-Point Alignment

This module provides point-to-point alignment with sternum strictly fixed.
Uses scipy least_squares optimization for rotation only around sternum origin.
"""

import numpy as np
from scipy import spatial
from scipy.optimize import least_squares
from typing import Tuple, Dict


def rotation_matrix(rot_x, rot_y, rot_z):
    """
    Construct rotation matrix from the specified rotation angles about the
    X, Y and Z axes.

    :param rot_x: rotation angle about the X axis in degrees
    :type rot_x: float
    :param rot_y: rotation angle about the Y axis in degrees
    :type rot_y: float
    :param rot_z: rotation angle about the Z axis in degrees
    :type rot_z: float
    :return: rotation matrix
    :rtype: ndarray
    """

    R_combined = np.zeros((4, 4))
    R_combined[-1, -1] = 1.

    #   Rotation about the z-axis
    R_z = np.array([[np.cos(np.deg2rad(rot_z)), -np.sin(np.deg2rad(rot_z)), 0.],
                    [np.sin(np.deg2rad(rot_z)), np.cos(np.deg2rad(rot_z)), 0.],
                    [0., 0., 1.]])

    #   Rotation about the y-axis
    R_y = np.array([[np.cos(np.deg2rad(rot_y)), 0., np.sin(np.deg2rad(rot_y))],
                    [0., 1., 0.],
                    [-np.sin(np.deg2rad(rot_y)), 0., np.cos(np.deg2rad(rot_y))]])

    #   Rotation about the x-axis
    R_x = np.array([[1., 0., 0.],
                    [0., np.cos(np.deg2rad(rot_x)), -np.sin(np.deg2rad(rot_x))],
                    [0., np.sin(np.deg2rad(rot_x)), np.cos(np.deg2rad(rot_x))]])

    #   Combined rotation
    R_combined[:-1, :-1] = R_z @ R_y @ R_x

    return R_combined


def objective_fixed_sternum(rotation_angles, set1_centered, set2_centered, tree):
    """
    Objective function for optimization with fixed sternum (rotation only).

    :param rotation_angles: 3 rotation angles (x, y, z) in degrees
    :param set1_centered: prone coordinates centered at sternum
    :param set2_centered: supine coordinates centered at sternum
    :param tree: KDTree for nearest neighbor search
    :return: residuals (distances) for least_squares
    """
    # Create rotation matrix ONLY (no translation)
    R = rotation_matrix(rotation_angles[0], rotation_angles[1], rotation_angles[2])

    # Apply rotation around the origin (sternum at 0,0,0)
    set1_transformed = (R[:3, :3] @ set1_centered.T).T

    # Calculate error
    distances, _ = tree.query(set1_transformed)
    return distances  # Return residuals for least_squares


def align_ribcage_fixed_sternum(
        prone_ribcage: np.ndarray,
        supine_ribcage: np.ndarray,
        prone_sternum: np.ndarray,
        supine_sternum: np.ndarray,
        verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Align prone ribcage to supine ribcage with fixed sternum position.
    Point-to-point correspondence using scipy least_squares optimization.

    :param prone_ribcage: prone rib cage coordinates (N, 3)
    :param supine_ribcage: supine rib cage coordinates (M, 3)
    :param prone_sternum: prone sternum superior coordinate (3,)
    :param supine_sternum: supine sternum superior coordinate (3,)
    :param verbose: print optimization details
    :return: (T, info) - transformation matrix and info dictionary
    """
    # Center both point sets at their respective sternums
    prone_centered = prone_ribcage - prone_sternum
    supine_centered = supine_ribcage - supine_sternum

    # Build KDTree for supine centered points
    tree = spatial.KDTree(supine_centered)

    # Initial rotation angles (start with no rotation)
    rotation_init = [0.0, 0.0, 0.0]

    if verbose:
        print(f"Optimizing rotation with {len(prone_centered)} prone and {len(supine_centered)} supine points")

    # Perform optimization (rotation only)
    res = least_squares(
        fun=objective_fixed_sternum,
        x0=rotation_init,
        args=(prone_centered, supine_centered, tree),
        verbose=2 if verbose else 0
    )

    if verbose:
        print(f"Optimization success: {res.success}")
        print(f"Optimization message: {res.message}")
        print(f"Final rotation angles (deg): {res.x}")

    # Construct transformation matrix
    # The correct transformation is:
    #   T * p = R * (p - prone_sternum) + supine_sternum
    # In matrix form: T = [R | supine_sternum - R*prone_sternum]
    #                     [0 |            1                    ]
    # This ensures: T * prone_sternum = supine_sternum (zero drift)
    T = rotation_matrix(res.x[0], res.x[1], res.x[2])
    R = T[:3, :3]  # Extract rotation matrix

    # Translation that ensures prone_sternum maps exactly to supine_sternum
    T[:3, 3] = supine_sternum - R @ prone_sternum

    # Calculate final alignment metrics
    rmse, mean_dist, sternum_drift = calculate_alignment_error(
        prone_ribcage, supine_ribcage, prone_sternum, supine_sternum, T
    )

    info = {
        'method': 'fixed_sternum_point_to_point',
        'success': res.success,
        'message': res.message,
        'rotation_angles_deg': res.x,
        'rmse': rmse,
        'mean_distance': mean_dist,
        'sternum_drift': sternum_drift,
        'cost': res.cost,
        'optimality': res.optimality,
        'nfev': res.nfev
    }

    return T, info


def calculate_alignment_error(
        prone_coords: np.ndarray,
        supine_coords: np.ndarray,
        prone_sternum: np.ndarray,
        supine_sternum: np.ndarray,
        T: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate alignment error after transformation.

    :param prone_coords: prone coordinates (N, 3)
    :param supine_coords: supine coordinates (M, 3)
    :param prone_sternum: prone sternum position (3,)
    :param supine_sternum: supine sternum position (3,)
    :param T: transformation matrix (4, 4)
    :return: (rmse, mean_dist, sternum_drift)
    """
    # Transform prone coordinates
    prone_homogeneous = np.hstack([prone_coords, np.ones((prone_coords.shape[0], 1))])
    prone_transformed = (T @ prone_homogeneous.T).T[:, :3]

    # Calculate distances to nearest supine points
    tree = spatial.KDTree(supine_coords)
    distances, _ = tree.query(prone_transformed)

    rmse = np.sqrt(np.mean(distances ** 2))
    mean_dist = np.mean(distances)

    # Calculate sternum drift
    prone_sternum_hom = np.append(prone_sternum, 1)
    prone_sternum_transformed = (T @ prone_sternum_hom)[:3]
    sternum_drift = np.linalg.norm(prone_sternum_transformed - supine_sternum)

    return rmse, mean_dist, sternum_drift
