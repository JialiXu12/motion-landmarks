"""
Author: Max Dang Vu (mdan066@aucklanduni.ac.nz)
Affiliation: Auckland Bioengineering Institute, The University of Auckland, New Zealand
"""

import argparse
import os
import sys
# # Add the parent folder 3 levels up to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import breast_metadata
import tools
import json

import mesh_tools
import morphic
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import SimpleITK as sitk
from tools import landmarks as ld
import copy

from tools.landmarks import displacement


def get_surface_mesh_coords(morphic_mesh, res, elems=[]):
    """
    Extracts the 3D coordinates of a surface mesh

    :param morphic_mesh: surface mesh
    :type morphic_mesh: morphic.Mesh
    :param res: number of material points per element axis
    :type res: int
    :param elems: specified elements to extract coordinates (leave empty if want all elements)
    :type elems: list
    :return: mesh_coords
    :rtype: ndarray
    """

    #   local coordinates for each element
    Xi = morphic_mesh.grid(res, method='center')
    NPPE = Xi.shape[0]

    #   if looking at all elements of mesh
    if elems == []:

        #   evaluate spatial coordinates
        NE = morphic_mesh.elements.size()
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(morphic_mesh.elements):
            eid = element.id
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[eid].evaluate(Xi)

    #   if looking at specific elements
    else:
        NE = len(elems)

        #   evaluate spatial coordinates
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(elems):
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[element].evaluate(Xi)

    return mesh_coords

def filter_point_cloud(points, reference, tol, axis):

    keep_idx = [idx for idx, pt in enumerate(points)
                  if np.min(reference[:, axis]) + tol < pt[axis] < np.max(reference[:, axis]) - tol]
    points = points[keep_idx]

    return points

def summary_stats(data):
    """
    Print summary statistics of the data, in the case the projection error between two point clouds
    :param data: projection error of data points
    :type data: ndarray
    """

    #   compute the interquartile range and median of the data
    quantiles = np.percentile(data, [0, 25, 50, 75, 100])
    error_iqr = quantiles[3] - quantiles[1]

    #   print summary statistics
    print(f"Min error: {quantiles[0]} mm")
    print(f"LQ error: {quantiles[1]} mm")
    print(f"Median error: {quantiles[2]} mm")
    print(f"UQ error: {quantiles[3]} mm")
    print(f"Max error: {quantiles[4]} mm")

def plot_histogram(data, interval):
    """
    Plots a histogram of the distribution of data points in groups, defined by the interval size.
    :param data: projection error of data points
    :type data: ndarray
    :param interval: size of the histogram bins
    :type interval: float
    """

    #   determine the size of histogram bins
    plt.hist(data, bins=np.arange(0, np.max(data)+interval, interval), edgecolor='black')
    plt.xlabel("Error (mm)")
    plt.ylabel("Number of points")
    plt.title(f"Projection error (n={data.shape[0]})")
    plt.show()


def read_landmarks_metadata(vl_ids, soft_landmarks_r1_path, soft_landmarks_r2_path, mri_t2_images_root_path, nipple_path, sternum_path):
    # Process both prone and supine positions
    positions = ['prone', 'supine']
    results = {}

    for position in positions:
        # Load breast tissue landmarks identified by registrars
        # User008 = Ben, User007 = Clark
        # registrar1_landmarks = ld.Landmarks('user008', vl_ids, position, soft_landmarks_path)
        # registrar2_landmarks = ld.Landmarks('user007', vl_ids, position, soft_landmarks_path)

        registrar1_landmarks = ld.Landmarks('user008', vl_ids, position, soft_landmarks_r1_path)
        vl_ids = registrar1_landmarks.get_valid_volunteers()
        registrar2_landmarks = ld.Landmarks('user007', vl_ids, position, soft_landmarks_r2_path)

        # Load metadata from mr images and nipple, sternum, and spinal cord landmarks for each volunteer
        metadata = ld.read_metadata(vl_ids, position, mri_t2_images_root_path, nipple_path, sternum_path)

        # Store results in a dictionary by position
        results[position] = (registrar1_landmarks, registrar2_landmarks, metadata)

    # Access results for prone and supine
    (registrar1_landmarks_prone, registrar2_landmarks_prone, prone_metadata) = results['prone']
    (registrar1_landmarks_supine, registrar2_landmarks_supine, supine_metadata) = results['supine']

    # Corresponding landmarks between registrars
    # This function returns a dictionary where each volunteer's key maps to a list of corresponding landmark pairs.
    # Each pair contains two values: the first represents the local landmark number identified by the first registrar,
    # and the second represents the corresponding landmark number identified by the second registrar.
    cor_1_2 = ld.corresponding_landmarks_between_registrars(registrar1_landmarks_prone, registrar2_landmarks_prone,
        registrar1_landmarks_supine, registrar2_landmarks_supine)
    vl_ids = list(cor_1_2.keys())

    # Make a deep copy to retain all attributes
    registrar1_prone_landmarks = copy.deepcopy(registrar1_landmarks_prone)
    registrar2_prone_landmarks = copy.deepcopy(registrar2_landmarks_prone)
    registrar1_supine_landmarks = copy.deepcopy(registrar1_landmarks_supine)
    registrar2_supine_landmarks = copy.deepcopy(registrar2_landmarks_supine)

    # For registrar 1, the landmarks are stored in the first index of the list
    for vl_id in vl_ids:
        # Reset the landmarks to only include the paired landmarks
        registrar1_prone_landmarks.landmarks[vl_id] = []
        registrar2_prone_landmarks.landmarks[vl_id] = []
        registrar1_supine_landmarks.landmarks[vl_id] = []
        registrar2_supine_landmarks.landmarks[vl_id] = []

        registrar1_prone_landmarks.landmark_types[vl_id] = []
        registrar2_prone_landmarks.landmark_types[vl_id] = []
        registrar1_supine_landmarks.landmark_types[vl_id] = []
        registrar2_supine_landmarks.landmark_types[vl_id] = []

        # Add only the landmarks specified in cor_1_2[vl_id]
        for indx_pair in cor_1_2[vl_id]:
            registrar1_prone_landmark = registrar1_landmarks_prone.landmarks[vl_id][indx_pair[0]]
            registrar2_prone_landmark = registrar2_landmarks_prone.landmarks[vl_id][indx_pair[1]]
            registrar1_supine_landmark = registrar1_landmarks_supine.landmarks[vl_id][indx_pair[0]]
            registrar2_supine_landmark = registrar2_landmarks_supine.landmarks[vl_id][indx_pair[1]]

            registrar1_prone_landmark_type = registrar1_landmarks_prone.landmark_types[vl_id][indx_pair[0]]
            registrar2_prone_landmark_type = registrar2_landmarks_prone.landmark_types[vl_id][indx_pair[1]]
            registrar1_supine_landmark_type = registrar1_landmarks_supine.landmark_types[vl_id][indx_pair[0]]
            registrar2_supine_landmark_type = registrar2_landmarks_supine.landmark_types[vl_id][indx_pair[1]]

            registrar1_prone_landmarks.landmarks[vl_id].append(registrar1_prone_landmark)
            registrar2_prone_landmarks.landmarks[vl_id].append(registrar2_prone_landmark)
            registrar1_supine_landmarks.landmarks[vl_id].append(registrar1_supine_landmark)
            registrar2_supine_landmarks.landmarks[vl_id].append(registrar2_supine_landmark)

            registrar1_prone_landmarks.landmark_types[vl_id].append(registrar1_prone_landmark_type)
            registrar2_prone_landmarks.landmark_types[vl_id].append(registrar2_prone_landmark_type)
            registrar1_supine_landmarks.landmark_types[vl_id].append(registrar1_supine_landmark_type)
            registrar2_supine_landmarks.landmark_types[vl_id].append(registrar2_supine_landmark_type)

    return (registrar1_prone_landmarks, registrar2_prone_landmarks, registrar1_supine_landmarks,
            registrar2_supine_landmarks, prone_metadata, supine_metadata)



def align_prone_supine(config_file_path, subject, sequence, body_pos, arms_pos, orientation_flag):

    #%% 1. GETTING IMAGE DATA
    #   =================
    #   load study configuration
    with open(config_file_path) as config_file:
        study_cfg = json.load(config_file)
    config_file.close()

    study = breast_metadata.Study(study_cfg['study_metadata_path'])
    metadata = study.get_participant_metadata(id=subject)

    #   load scan
    if 'clinical_duke' in study_cfg['label']:
        scan = breast_metadata.Scan(metadata['mri_t1_prone_image_path'])
    elif 'volunteers_camri' in study_cfg['label']:
        prone_scan = breast_metadata.Scan(metadata[f'mri_{sequence}_{body_pos[0]}_{arms_pos}_image_path'])
        supine_scan = breast_metadata.Scan(metadata[f'mri_{sequence}_{body_pos[1]}_{arms_pos}_image_path'])

    #   convert Scan to Pyvista Image Grid in desired orientation
    prone_image_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan, orientation_flag)
    supine_image_grid = breast_metadata.SCANToPyvistaImageGrid(supine_scan, orientation_flag)

    #%% 2. SELECT RIBCAGE AS RIGID LANDMARKS FOR ALIGNMENT
    #   ===================
    #   centre point cloud at sternum landmarks (all in RAI)
    sternum_path = os.path.join(
        "U:" + os.sep, "sandbox", "mdan066", "volunteer_camri",
        "anna_data", "sternum landmarks", subject + os.sep)
    sternum_prone, sternum_supine, _ = breast_metadata.load_landmarks(sternum_path)

    nipple_path = os.path.join(
        "U:" + os.sep, "sandbox", "mdan066", "volunteer_camri",
        "anna_data", "nipple landmarks", subject + os.sep)
    nipple_prone, nipple_supine, _ = breast_metadata.load_landmarks(nipple_path)

    vl_ids = [49]
    mri_t2_images_root_path = r'U:\projects\volunteer_camri\old_data\mri_t2'
    soft_landmarks_r1_path = r'U:\projects\dashboard\picker_points\ben_reviewed'
    soft_landmarks_r2_path = r"U:\projects\dashboard\picker_points\holly"
    rigid_landmarks_root_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\picker\points_wc_raf'
    nipple_path = os.path.join(rigid_landmarks_root_path, 'user002')
    sternum_path = os.path.join(rigid_landmarks_root_path, 'user001')

    (registrar1_prone_landmarks, registrar2_prone_landmarks, registrar1_supine_landmarks, registrar2_supine_landmarks,
     prone_metadata, supine_metadata)= \
        (read_landmarks_metadata(vl_ids, soft_landmarks_r1_path, soft_landmarks_r2_path, mri_t2_images_root_path,
                                 nipple_path, sternum_path))

    landmark_prone = np.array(registrar1_prone_landmarks.landmarks[49])
    landmark_supine = np.array(registrar1_supine_landmarks.landmarks[49])

    #   prone ribcage mesh
    prone_ribcage_path = os.path.join(
        "U:" + os.sep, "sandbox", "fpan017", "meshes",
        "new_workflow", "ribcage", "iter2", f"{subject}_ribcage_prone.mesh")
    prone_ribcage = morphic.Mesh(prone_ribcage_path)
    prone_ribcage_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=26)
    # elem_pos, elem_ids = mesh_tools.get_elem_pos_ids(prone_ribcage, [0.5, 0.5])

    #   Visualise prone ribcage mesh
    cam_pos = [
        [423.68069174271386, -504.87649973445264, 1099.0746665133568],
        [-283.7873767166018, -103.829530710408, 120.87664926612001],
        [-0.16051295643760097, -0.9485816080426248, -0.27281591540621086]
    ]

    cam_pos = [
        [72.39115428359347, 777.7556050896267, 3.9735210540261776],
        [-42.11093756321733, -266.340069906558, -40.97397730572022],
        [-0.014962440507869636, -0.04136666705034152, 0.9990319935973996]
    ]

    # plotter = pv.Plotter()
    # plotter.camera_position = cam_pos
    # plotter.add_points(prone_ribcage_mesh_coords, color='black', render_points_as_spheres=True, point_size=5)
    # plotter.add_mesh(mesh_tools.morphic_to_trimesh(prone_ribcage), opacity=0.7)
    # plotter.add_axes()
    # plotter.add_text(f"{subject}: Prone ribcage mesh (n={prone_ribcage_mesh_coords.shape[0]})")
    # plotter.show()
    # #
    # cam_pos = plotter.camera_position
    # print("Camera position:", cam_pos[0])
    # print("Camera focal point:", cam_pos[1])
    # print("Camera view up:", cam_pos[2])


    #supine ribcage segmentation
    supine_ribcage_path = os.path.join(
        "U:" + os.sep, "sandbox", "mdan066", "volunteer_camri", "workflow_vl", "segmentations",
        "ribcage", "supine_sean", "point_clouds", f"{subject}_ribcage_pts_20000.txt")
    supine_ribcage_pc = np.loadtxt(supine_ribcage_path, delimiter=" ")

    print(min(supine_ribcage_pc[:, 0]), max(supine_ribcage_pc[:, 0]))
    print(min(supine_ribcage_pc[:, 1]), max(supine_ribcage_pc[:, 1]))
    print(min(supine_ribcage_pc[:, 2]), max(supine_ribcage_pc[:, 2]))

    supine_ribcage_pc = supine_ribcage_pc[supine_ribcage_pc[:, 0] > -120.]      #   remove random segmentation points
    #   remove points on the top and bottom axial slices
    supine_ribcage_pc = filter_point_cloud(
        supine_ribcage_pc, supine_ribcage_pc, supine_image_grid.spacing[2], axis=2)
    supine_ribcage_pc =  supine_ribcage_pc[
        (supine_ribcage_pc[:, 0] <= -10.) | (supine_ribcage_pc[:, 0] >= 40.) |
        (supine_ribcage_pc[:, 1] < np.max(supine_ribcage_pc[:, 1]) - 60)]


    supine_ribcage_path = os.path.join(
        r"U:\sandbox\jxu759\motion_of_landmarks\anna_data\automatic_segmentation_CNN\T2\supine\rib_cage", f"rib_cage_{subject}.nii")
    supine_ribcage_mask = breast_metadata.readNIFTIImage(supine_ribcage_path, orientation_flag, swap_axes=True)
    supine_ribcage_pc2 = tools.landmarks.extract_contour_points(supine_ribcage_mask, 20000)

    # print(min(supine_ribcage_pc2[:, 0]), max(supine_ribcage_pc2[:, 0]))
    # print(min(supine_ribcage_pc2[:, 1]), max(supine_ribcage_pc2[:, 1]))
    # print(min(supine_ribcage_pc2[:, 2]), max(supine_ribcage_pc2[:, 2]))

    supine_ribcage_pc2 = supine_ribcage_pc2[
        (supine_ribcage_pc2[:, 2] > (np.min(supine_ribcage_pc2[:, 2]) + 10.0)) &
        (supine_ribcage_pc2[:, 2] < (np.max(supine_ribcage_pc2[:, 2]) - 10.0))
        ]
    supine_ribcage_pc2 = filter_point_cloud(
        supine_ribcage_pc2, supine_ribcage_pc2, supine_image_grid.spacing[2], axis=2)

    # plotter = pv.Plotter()
    # plotter.add_points(
    #     supine_ribcage_pt, color='gray', render_points_as_spheres=True,
    #     point_size=5, label='Ribcage')
    # plotter.add_points(
    #     sternum_supine, render_points_as_spheres=True,
    #     point_size=10, color='red', label='Sternum')
    # plotter.add_points(
    #     nipple_supine, render_points_as_spheres=True,
    #     point_size=10, color='black', label='Nipple')
    # plotter.add_volume(
    #     supine_image_grid, opacity='sigmoid_6',
    #     cmap='gray', show_scalar_bar=False)
    # plotter.add_axes()
    # plotter.add_legend(bcolor=None)
    # plotter.add_text(f"{subject}: Supine")
    # plotter.show()

    supine_ribcage_path = r"U:\sandbox\jxu759\volunteer_images_for_align_n_density_nifti_for_nnunet\rib_seg\rib_093.nii.gz"
    supine_ribcage_mask = breast_metadata.readNIFTIImage(supine_ribcage_path, orientation_flag, swap_axes=True)
    supine_ribcage_pc3 = tools.landmarks.extract_contour_points(supine_ribcage_mask, 20000)

    print(min(supine_ribcage_pc3[:, 0]), max(supine_ribcage_pc3[:, 0]))
    print(min(supine_ribcage_pc3[:, 1]), max(supine_ribcage_pc3[:, 1]))
    print(min(supine_ribcage_pc3[:, 2]), max(supine_ribcage_pc3[:, 2]))

    supine_ribcage_pc3 = supine_ribcage_pc3[supine_ribcage_pc3[:, 0] > -120.]      #   remove random segmentation points
    #   remove points on the top and bottom axial slices
    supine_ribcage_pc4 = filter_point_cloud(
        supine_ribcage_pc3, supine_ribcage_pc3, supine_image_grid.spacing[2]*10, axis=2)

    print(min(supine_ribcage_pc4[:, 0]), max(supine_ribcage_pc4[:, 0]))
    print(min(supine_ribcage_pc4[:, 1]), max(supine_ribcage_pc4[:, 1]))
    print(min(supine_ribcage_pc4[:, 2]), max(supine_ribcage_pc4[:, 2]))

    supine_ribcage_pc4 =  supine_ribcage_pc4[
        (supine_ribcage_pc4[:, 0] <= -10.) | (supine_ribcage_pc4[:, 0] >= 40.) |
        (supine_ribcage_pc4[:, 1] < np.max(supine_ribcage_pc4[:, 1]) - 60)]


    # plotter = pv.Plotter()
    # plotter.add_points(
    #     supine_ribcage_pc, color='gray', render_points_as_spheres=True,
    #     point_size=5, label='Ribcage')
    # plotter.add_points(
    #     supine_ribcage_pc2, color='blue', render_points_as_spheres=True,
    #     point_size=5, label='Ribcage')
    # plotter.add_points(
    #     supine_ribcage_pc3, color='green', render_points_as_spheres=True,
    #     point_size=5, label='Ribcage')
    # plotter.add_points(
    #     supine_ribcage_pc4, color='red', render_points_as_spheres=True,
    #     point_size=5, label='Ribcage')
    #
    # plotter.add_legend(bcolor=None)
    # plotter.add_text(f"{subject}: Supine")
    # plotter.show()

    supine_ribcage_pc = supine_ribcage_pc4
    #   Visualise supine ribcage point cloud
    # plotter = pv.Plotter()
    # plotter.camera_position = cam_pos
    # plotter.add_points(supine_ribcage_pc, color='gray', render_points_as_spheres=True, point_size=5)
    # plotter.add_volume(supine_image_grid, opacity='sigmoid_6', cmap='coolwarm', show_scalar_bar=False)
    # plotter.add_axes()
    # plotter.add_text(f"{subject}: Supine ribcage (n={supine_ribcage_pc.shape[0]})")
    # plotter.show()

    #%% 3. ESTIMATE TRANSFORMATION MATRIX BY ALIGNING RIGID LANDMARKS
    #   initial guess of the rotation angles
    rot_angle_init = [0., 0., 0.]
    translation_init = list(breast_metadata.find_centroid(sternum_supine.T) - breast_metadata.find_centroid(sternum_prone.T))
    T_init = rot_angle_init + translation_init

    #   a. FIRST ITERATION (blind optimisation)
    print("\nPERFORMING OPTIMISATION\n============")

    #   optimise transformation matrix by performing kd-tree of point clouds between
    #   the landmarks in prone and supine
    prone_points = [prone_ribcage_mesh_coords, sternum_prone]
    supine_points = [supine_ribcage_pc, sternum_supine]
    T_optimal, res_optimal = breast_metadata.run_optimisation(
        breast_metadata.combined_objective_function, T_init,
        prone_points, supine_points)
    print(f"\nProne-to-supine ribcage transformation:\n {T_optimal}")
    print("6 DoFs:", res_optimal.x)

    #   apply transformation matrix to prone
    ones = np.ones((len(prone_ribcage_mesh_coords), 1))
    ribcage_mesh_rotated = (T_optimal @ np.hstack((prone_ribcage_mesh_coords, ones)).T)[:-1, :].T
    ones = np.ones((len(sternum_prone), 1))
    sternum_rotated = (T_optimal @ np.hstack((sternum_prone, ones)).T)[:-1, :].T
    sternum_supine_rotated = (np.linalg.inv(T_optimal) @ np.hstack((sternum_supine, ones)).T)[:-1, :].T

    nipple_rotated = (T_optimal @ np.hstack((nipple_prone, ones)).T)[:-1, :].T

    ones = np.ones((len(landmark_prone), 1))
    landmark_rotated = (T_optimal @ np.hstack((landmark_prone, ones)).T)[:-1, :].T
    landmark_supine_rotated = (np.linalg.inv(T_optimal) @ np.hstack((landmark_supine, ones)).T)[:-1, :].T

    #   visualise the error between prone rotated and supine
    labels = dict(xlabel="X", ylabel="Y", zlabel="Z")

    print(f"Prone ribcage mesh: {ribcage_mesh_rotated.shape[0]} points")
    print(f"Supine ribcage point cloud: {supine_ribcage_pc.shape[0]} points")

    import pyvista as pv
    plotter = pv.Plotter()
    plotter.add_points(
        supine_ribcage_pc, color='gray', render_points_as_spheres=True,
        point_size=5, label='Supine ribcage')
    plotter.add_points(
        ribcage_mesh_rotated, render_points_as_spheres=True,
        color='black', point_size=5, label='Prone ribcage mesh transformed')
    plotter.add_points(
        landmark_rotated, render_points_as_spheres=True,
        color='lightblue', point_size=7, label='Prone landmarks transformed')
    plotter.add_points(
        landmark_supine, render_points_as_spheres=True,
        color='navy', point_size=7, label='Supine landmarks')
    plotter.add_points(
        nipple_rotated, render_points_as_spheres=True,
        color='lightcoral', point_size=7, label='Prone nipple transformed')
    plotter.add_points(
        nipple_supine, render_points_as_spheres=True,
        color='crimson', point_size=7, label='Supine nipple')
    # plotter.add_volume(supine_image_grid, opacity='sigmoid_6', cmap='coolwarm', show_scalar_bar=False)
    plotter.add_volume(
        supine_image_grid, opacity='sigmoid_6',
        cmap='gray', show_scalar_bar=False)
    plotter.add_axes(**labels)
    plotter.add_text(f"{subject}: aligned prone and supine ribcage")
    plotter.show()

    #   evaluate displacement of landmarks
    landmark_displacement_vectors = landmark_supine - landmark_rotated
    landmark_displacement_magnitudes = np.linalg.norm(landmark_displacement_vectors, axis=1)
    # --- Print the results ---
    print("--- Landmark Displacement ---")
    for i in range(len(landmark_supine)):
        print(f"Landmark {i + 1}:")
        print(f"  Displacement Vector: {landmark_displacement_vectors[i]}")
        print(f"  Displacement Magnitude: {landmark_displacement_magnitudes[i]:.2f} mm")

    import pyvista as pv

    # --- Create the 3D plot ---
    plotter = pv.Plotter()
    plotter.add_text("Landmark Displacement After Alignment", font_size=24)

    # Plot the target supine landmarks (e.g., in green)
    plotter.add_points(
        landmark_supine,
        render_points_as_spheres=True,
        color='green',
        point_size=10,
        label='Target Supine Landmarks'
    )

    # Plot the aligned prone landmarks (e.g., in red)
    plotter.add_points(
        landmark_rotated,
        render_points_as_spheres=True,
        color='red',
        point_size=10,
        label='Aligned Prone Landmarks'
    )

    # Add arrows to show the displacement vectors
    for start_point, vector in zip(landmark_rotated, landmark_displacement_vectors):
        plotter.add_arrows(start_point, vector, mag=1.0, color='yellow')
    plotter.add_volume(
        supine_image_grid, opacity='sigmoid_8',
        cmap='coolwarm', show_scalar_bar=False)
    plotter.add_legend()
    plotter.show()

    #   evaluate sternum fit
    error, mapped_idx = breast_metadata.closest_distances(sternum_supine, sternum_rotated)
    print(f"Sternum fit error: {np.linalg.norm(error, axis=1)} mm")
    plotter = pv.Plotter()
    plotter.add_points(
        sternum_supine, render_points_as_spheres=True,
        point_size=8, label='Supine sternum')
    plotter.add_points(
        sternum_rotated, render_points_as_spheres=True,
        color='black', point_size=8, label='Prone sternum transformed')
    plotter.add_arrows(
        cent=sternum_rotated, direction=error,
        cmap='turbo', scalar_bar_args={'title': 'Error (mm)'})
    plotter.add_volume(
        supine_image_grid, opacity='sigmoid_8',
        cmap='coolwarm', show_scalar_bar=False)
    plotter.add_axes(**labels)
    plotter.add_legend(bcolor=None)
    plotter.add_text(f"{subject}: Image and Sternum Point Cloud in Supine ({sternum_supine.shape[0]} points)")
    plotter.show()

    #   evaluate ribcage fit
    error, mapped_idx = breast_metadata.closest_distances(supine_ribcage_pc, ribcage_mesh_rotated)
    error_mag = np.linalg.norm(error, axis=1)
    print(f"Ribcage fit error: {error_mag} mm")

    plotter = pv.Plotter()
    plotter.camera_position = cam_pos
    plotter.add_points(
        supine_ribcage_pc, color='gray', render_points_as_spheres=True,
        point_size=8, label='Supine ribcage')
    plotter.add_points(ribcage_mesh_rotated[mapped_idx], render_points_as_spheres=True,
        color='black', point_size=8, label='Prone ribcage')
    plotter.add_arrows(
        cent=supine_ribcage_pc, direction=error,
        cmap='turbo', scalar_bar_args={'title': 'Error (mm)'})
    plotter.add_volume(
        supine_image_grid, opacity='sigmoid_8',
        cmap='coolwarm', show_scalar_bar=False)
    plotter.add_axes(**labels)
    plotter.add_legend(bcolor=None)
    plotter.add_text(f"{subject}: Image and Ribcage Point Cloud in Supine ({supine_ribcage_pc.shape[0]} points)")
    plotter.show()

    # 	show statistics and distribution of projection errors
    summary_stats(error_mag)
    plot_histogram(error_mag, 5)

    #%% 4. SHOW GRAVITY VECTOR FOR UNLOADING AND LOADING
    g_unloaded = np.array([-6.00689255e-16, -9.81000000e+00, - 0.00000000e+00])
    #g_t = np.linalg.inv(rotation[:3, :3]) @ g
    g_loaded = (np.linalg.inv(T_optimal) @ np.hstack((-g_unloaded, 0)))[:-1]

    #   T_prone_supine, g_loaded = [ 0.25895435 -9.74647427  1.0841042 ]
    #   T_supine_prone, g_loaded = [-0.25129934 -9.74647427 -1.08590419]
    # print("Transformed gravity vector:", g_loaded)

    # plotter = pv.Plotter(shape=(1, 2))
    #
    # plotter.subplot(0, 0)
    # plotter.add_arrows(cent=np.array([0, 0, 0]),
    #                    direction=-g_unloaded,
    #                    mag=10,
    #                    color='blue',
    #                    label='g_load (in supine orientation)')
    # plotter.add_volume(supine_image_grid,
    #                    opacity='sigmoid_6',
    #                    cmap='coolwarm',
    #                    show_scalar_bar=False)
    # plotter.add_axes()
    # plotter.add_legend(bcolor=None, size=(0.4, 0.4))
    # plotter.add_text(f"{subject}: Supine")
    #
    # plotter.subplot(0, 1)
    # # plotter.add_arrows(cent=np.array([0, 0, 0]),
    # #                    direction=g_unloaded,
    # #                    mag=10,
    # #                    color='orange',
    # #                    label='g_unload (in supine orientation)')
    # plotter.add_arrows(cent=np.array([0, 0, 0]),
    #                    direction=g_loaded,
    #                    mag=10,
    #                    color='blue',
    #                    label='g_load (transformed from supine to prone)')
    # plotter.add_volume(prone_image_grid,
    #                    opacity='sigmoid_6',
    #                    cmap='bone',
    #                    show_scalar_bar=False)
    # plotter.add_legend(bcolor=None, size=(0.4, 0.4))
    # plotter.add_axes()
    # plotter.add_text(f"{subject}: Prone")
    #
    # plotter.link_views()
    # plotter.show()

    #%% 5. RESAMPLE PRONE IMAGE TO SUPINE POSITION
    #   ======================
    #   convert Pyvista image grid to SITK image
    prone_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(prone_image_grid)
    prone_image_sitk = sitk.Cast(prone_image_sitk, sitk.sitkUInt8)
    supine_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(supine_image_grid)
    supine_image_sitk = sitk.Cast(supine_image_sitk, sitk.sitkUInt8)

    #   initialise affine transformation matrix
    dimensions = 3
    affine = sitk.AffineTransform(dimensions)

    #   set transformation matrix from prone to supine
    # T_supine_prone = np.linalg.inv(T_optimal)
    T_supine_prone = T_optimal
    affine.SetTranslation(T_supine_prone[:3, 3])
    affine.SetMatrix(T_supine_prone[:3, :3].ravel())

    #   transform prone image to supine coordinate system
    #   sitk.Resample(input_image, reference_image, transform) takes a transformation matrix that maps points
    #   from the reference_image (output space) to it's corresponding location on the input_image (input space)
    rotated_image = sitk.Resample(supine_image_sitk, prone_image_sitk, affine, sitk.sitkLinear, 1.0)
    rotated_image = sitk.Cast(rotated_image, sitk.sitkUInt8)
    print("Rotated image size:", rotated_image.GetSize())

    #   get pixel coordinates of landmarks
    rotated_scan = breast_metadata.SITKToScan(rotated_image, orientation_flag, load_dicom=False, swap_axes=True)
    # sternum_rotated_px = rotated_scan.getPixelCoordinates(sternum_rotated)
    # sternum_supine_px = supine_scan.getPixelCoordinates(sternum_supine)
    sternum_prone_px = prone_scan.getPixelCoordinates(sternum_prone)
    sternum_supine_rotated_px = rotated_scan.getPixelCoordinates(sternum_supine_rotated)

    #   scalar colour map (red-blue) to show alignment of prone rotated and supine MRIs
    # breast_metadata.visualise_alignment_with_landmarks(
    #     supine_image_sitk, rotated_image, sternum_supine_px[0], sternum_rotated_px[0], orientation='axial')
    # breast_metadata.visualise_alignment_with_landmarks(
    #     supine_image_sitk, rotated_image, sternum_supine_px[1], sternum_rotated_px[1], orientation='axial')
    breast_metadata.visualise_alignment_with_landmarks(
        rotated_image, prone_image_sitk, sternum_supine_rotated_px[0], sternum_prone_px[0], orientation='axial')
    breast_metadata.visualise_alignment_with_landmarks(
        rotated_image, prone_image_sitk, sternum_supine_rotated_px[1], sternum_prone_px[1], orientation='axial')

    supine_image_rotated_grid = breast_metadata.SCANToPyvistaImageGrid(rotated_scan, orientation_flag)


    plotter = pv.Plotter()
    # plotter.add_volume(prone_image_grid,
    #                    opacity='sigmoid_6',
    #                    cmap='bone',
    #                    show_scalar_bar=False)
    plotter.add_volume(
        supine_image_rotated_grid, opacity='sigmoid_8',
        cmap='grey', show_scalar_bar=False)
    plotter.add_volume(
        prone_image_grid, opacity='sigmoid_8',
        cmap='coolwarm', show_scalar_bar=False)
    plotter.add_points(landmark_supine_rotated, render_points_as_spheres=True,  color='blue', point_size=8, label='supine transformed' )
    plotter.add_text(f"{subject}: Supine MRI and Transformed Prone MRI")
    plotter.show()


    #   save rotated image
    # image_path = os.path.join(
    #     "U:" + os.sep, "sandbox", "mdan066", "volunteer_camri",
    #     "workflow_vl", "synthetic_images", subject)
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)
    #
    # #   Save rotated image as NIFTI
    # sitk.WriteImage(rotated_image, os.path.join(image_path, 'supine_rotated.nii.gz'), imageIO='NiftiImageIO')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Alignment of prone and supine breast MRIs, segmentations and meshes',
        description='An example script that can bee used to bring breast MRIs, segmentations and meshes from'
                    'different body positions into the same coordinate system for visualisation and analysis.')

    parser.add_argument(
        '-c', '--config_file_path',
        default='../../../external/breast_metadata_mdv/study_configurations/volunteers_camri.config')
    parser.add_argument('-s', '--subject', default='VL00049')
    parser.add_argument('-seq', '--sequence', default='t2')
    parser.add_argument('-b', '--body_pos', default=['prone', 'supine'])
    parser.add_argument('-a', '--arms_pos', default='arms_down')
    parser.add_argument('-0', '--orientation_flag', default='RAI')
    args = parser.parse_args()

    align_prone_supine(
        config_file_path=args.config_file_path,  subject=args.subject, sequence=args.sequence,
        body_pos=args.body_pos, arms_pos=args.arms_pos, orientation_flag=args.orientation_flag)
