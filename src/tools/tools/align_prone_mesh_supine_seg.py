from breast_metadata_mdv.examples.images.visualise_image_and_mesh import align_prone_supine as aps

import os
import sys
import breast_metadata
import tools
import morphic
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from breast_metadata_mdv.examples.images.visualise_image_and_mesh import align_prone_supine as aps

def align_prone_mesh_supine_mask(vl_id, mri_t2_images_prone_path, mri_t2_images_supine_path,
                           prone_ribcage_mesh_path, supine_ribcage_seg_path, registrar1_prone_landmarks,
                                 registrar1_supine_landmarks, registrar2_prone_landmarks, registrar2_supine_landmarks,
                                 prone_metadata, supine_metadata, orientation_flag):
    #%%   load data
    #   load prone and supine scans
    prone_scan = breast_metadata.Scan(mri_t2_images_prone_path)
    supine_scan = breast_metadata.Scan(mri_t2_images_supine_path)

    #   convert Scan to Pyvista Image Grid in desired orientation
    prone_image_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan, orientation_flag)
    supine_image_grid = breast_metadata.SCANToPyvistaImageGrid(supine_scan, orientation_flag)

    #   load sternum, nipple, and landmarks position from metadata
    sternum_prone = np.vstack([prone_metadata[vl_id].jugular_notch, prone_metadata[vl_id].sternum_position])
    sternum_supine = np.vstack([supine_metadata[vl_id].jugular_notch, supine_metadata[vl_id].sternum_position])
    nipple_prone = np.vstack([prone_metadata[vl_id].left_nipple, prone_metadata[vl_id].right_nipple])
    nipple_supine = np.vstack([supine_metadata[vl_id].left_nipple, supine_metadata[vl_id].right_nipple])

    landmark_prone_r1 = np.array(registrar1_prone_landmarks.landmarks[vl_id])
    landmark_supine_r1 = np.array(registrar1_supine_landmarks.landmarks[vl_id])
    landmark_prone_r2 = np.array(registrar2_prone_landmarks.landmarks[vl_id])
    landmark_supine_r2 = np.array(registrar2_supine_landmarks.landmarks[vl_id])

    #   load prone ribcage mesh in morphic format
    prone_ribcage = morphic.Mesh(prone_ribcage_mesh_path)
    prone_ribcage_mesh_coords = aps.get_surface_mesh_coords(prone_ribcage, res=26)

    #   load supine ribcage segmentation mask
    supine_ribcage_mask = breast_metadata.readNIFTIImage(supine_ribcage_seg_path, orientation_flag, swap_axes=True)
    supine_ribcage_pc = tools.landmarks.extract_contour_points(supine_ribcage_mask, 20000)

    #   clean up supine ribcage mask point cloud
    supine_ribcage_pc = supine_ribcage_pc[supine_ribcage_pc[:, 0] > -120.]
    #   remove points on the top and bottom axial slices
    supine_ribcage_pc = aps.filter_point_cloud(
        supine_ribcage_pc, supine_ribcage_pc, supine_image_grid.spacing[2] * 10, axis=2)
    #   remove points on the back side of the ribcage around the spine
    supine_ribcage_pc =  supine_ribcage_pc[
        (supine_ribcage_pc[:, 0] <= -10.) | (supine_ribcage_pc[:, 0] >= 40.) |
        (supine_ribcage_pc[:, 1] < np.max(supine_ribcage_pc[:, 1]) - 60)]

    #%%   initial alignment using sternum points
    #   initial guess of the rotation angles and translation vector
    rot_angle_init = [0., 0., 0.]
    translation_init = list(
        breast_metadata.find_centroid(sternum_supine.T) - breast_metadata.find_centroid(sternum_prone.T))
    T_init = rot_angle_init + translation_init

    #   First iteration: optimise transformation matrix by performing kd-tree of point clouds between
    #   the landmarks in prone and supine
    print("\nPERFORMING OPTIMISATION\n============")
    prone_points = [prone_ribcage_mesh_coords, sternum_prone]
    supine_points = [supine_ribcage_pc, sternum_supine]
    T_optimal, res_optimal = breast_metadata.run_optimisation(breast_metadata.combined_objective_function, T_init,
        prone_points, supine_points)
    print(f"\nProne-to-supine ribcage transformation:\n {T_optimal}")
    print("6 DoFs:", res_optimal.x)

    #%%   apply transformation matrix to prone or supine points
    ones = np.ones((len(prone_ribcage_mesh_coords), 1))
    ribcage_prone_mesh_transformed = (T_optimal @ np.hstack((prone_ribcage_mesh_coords, ones)).T)[:-1, :].T

    ones = np.ones((len(sternum_prone), 1))
    sternum_prone_transformed = (T_optimal @ np.hstack((sternum_prone, ones)).T)[:-1, :].T
    sternum_supine_transformed = (np.linalg.inv(T_optimal) @ np.hstack((sternum_supine, ones)).T)[:-1, :].T

    nipple_prone_transformed = (T_optimal @ np.hstack((nipple_prone, ones)).T)[:-1, :].T
    nipple_supine_transformed = (np.linalg.inv(T_optimal) @ np.hstack((nipple_supine, ones)).T)[:-1, :].T

    ones = np.ones((len(landmark_prone_r1), 1))
    landmark_prone_r1_transformed = (T_optimal @ np.hstack((landmark_prone_r1, ones)).T)[:-1, :].T
    landmark_prone_r2_transformed = (T_optimal @ np.hstack((landmark_prone_r2, ones)).T)[:-1, :].T
    landmark_supine_r1__transformed = (np.linalg.inv(T_optimal) @ np.hstack((landmark_supine_r1, ones)).T)[:-1, :].T
    landmark_supine_r2_transformed = (np.linalg.inv(T_optimal) @ np.hstack((landmark_supine_r2, ones)).T)[:-1, :].T

    #%%   evaluate displacement of landmarks
    landmark_r1_displacement_vectors = landmark_supine_r1 - landmark_prone_r1_transformed
    landmark_r1_displacement_magnitudes = np.linalg.norm(landmark_r1_displacement_vectors, axis=1)
    landmark_r2_displacement_vectors = landmark_prone_r2_transformed - landmark_supine_r2
    landmark_r2_displacement_magnitudes = np.linalg.norm(landmark_r2_displacement_vectors, axis=1)

    # --- Print the results ---
    print("--- Landmark Displacement ---")
    print("--- Registrar 1 ---")
    for i in range(len(landmark_supine_r1)):
        print(f"Landmark {i + 1}:")
        print(f"  Displacement Vector: {landmark_r1_displacement_vectors[i]}")
        print(f"  Displacement Magnitude: {landmark_r1_displacement_magnitudes[i]:.2f} mm")

    print("--- Registrar 2 ---")
    for i in range(len(landmark_supine_r2)):
        print(f"Landmark {i + 1}:")
        print(f"  Displacement Vector: {landmark_r2_displacement_vectors[i]}")
        print(f"  Displacement Magnitude: {landmark_r2_displacement_magnitudes[i]:.2f} mm")

    #%%   evaluate sternum fit
    error, mapped_idx = breast_metadata.closest_distances(sternum_supine, nipple_prone_transformed)
    print(f"Sternum fit error: {np.linalg.norm(error, axis=1)} mm")

    #%%   evaluate ribcage fit
    error, mapped_idx = breast_metadata.closest_distances(supine_ribcage_pc, ribcage_prone_mesh_transformed)
    error_mag = np.linalg.norm(error, axis=1)
    print(f"Ribcage fit error: {error_mag} mm")

    #%%   show statistics and distribution of projection errors
    aps.summary_stats(error_mag)
    aps.plot_histogram(error_mag, 5)


    #%%   resample
    #   convert Pyvista image grid to SITK image
    prone_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(prone_image_grid)
    prone_image_sitk = sitk.Cast(prone_image_sitk, sitk.sitkUInt8)
    supine_image_sitk = breast_metadata.PyvistaImageGridToSITKImage(supine_image_grid)
    supine_image_sitk = sitk.Cast(supine_image_sitk, sitk.sitkUInt8)

    #   initialise affine transformation matrix
    dimensions = 3
    affine = sitk.AffineTransform(dimensions)

    #   set transformation matrix from prone to supine
    T_prone_to_supine = T_optimal
    affine.SetTranslation(T_prone_to_supine[:3, 3])
    affine.SetMatrix(T_prone_to_supine[:3, :3].ravel())

    #   transform prone image to supine coordinate system
    #   sitk.Resample(input_image, reference_image, transform) takes a transformation matrix that maps points
    #   from the reference_image (output space) to it's corresponding location on the input_image (input space)
    prone_image_transformed = sitk.Resample(prone_image_sitk,supine_image_sitk, affine, sitk.sitkLinear, 1.0)
    prone_image_transformed = sitk.Cast(prone_image_transformed, sitk.sitkUInt8)
    print("Transformed image size:", prone_image_transformed.GetSize())

    #   get pixel coordinates of landmarks
    prone_scan_transformed = breast_metadata.SITKToScan(prone_image_transformed, orientation_flag, load_dicom=False, swap_axes=True)
    prone_image_transformed_grid = breast_metadata.SCANToPyvistaImageGrid(prone_scan_transformed, orientation_flag)

    sternum_prone_transformed_px = prone_scan_transformed.getPixelCoordinates(sternum_prone_transformed)
    sternum_supine_px = supine_scan.getPixelCoordinates(sternum_supine)
    nipple_prone_transformed_px = prone_scan_transformed.getPixelCoordinates(nipple_prone_transformed)
    nipple_supine_px = supine_scan.getPixelCoordinates(nipple_supine_transformed)
    landmark_prone_r1_transformed_px = prone_scan_transformed.getPixelCoordinates(landmark_prone_r1_transformed)
    landmark_prone_r2_transformed_px = prone_scan_transformed.getPixelCoordinates(landmark_prone_r2_transformed)
    landmark_supine_r1_px = supine_scan.getPixelCoordinates(landmark_supine_r1)
    landmark_supine_r2_px = supine_scan.getPixelCoordinates(landmark_supine_r2)


    #%%   plot
    # #the prone and supine ribcage point clouds before and after alignment
    plotter = pv.Plotter()
    plotter.add_text("Landmark Displacement After Alignment", font_size=24)

    # Plot the target supine landmarks (e.g., in green)
    plotter.add_points(landmark_supine_r1, render_points_as_spheres=True, color='green', point_size=10,
        label='Target Supine Landmarks'
    )

    # Plot the aligned prone landmarks (e.g., in red)
    plotter.add_points(landmark_prone_r1_transformed, render_points_as_spheres=True, color='red', point_size=10,
        label='Aligned Prone Landmarks'
    )

    # Add arrows to show the displacement vectors
    for start_point, vector in zip(landmark_prone_r1_transformed, landmark_r1_displacement_vectors):
        plotter.add_arrows(start_point, vector, mag=1.0, color='yellow')

    plotter.add_volume(prone_image_transformed_grid, opacity='sigmoid_8', cmap='grey', show_scalar_bar=False)
    plotter.add_volume(supine_image_grid, opacity='sigmoid_8', cmap='coolwarm', show_scalar_bar=False)

    plotter.add_points(sternum_prone_transformed, render_points_as_spheres=True, color='red', point_size=10,
        label='Aligned Prone Landmarks'
    )

    plotter.add_legend()
    plotter.show()


    #%%   scalar colour map (red-blue) to show alignment of prone transformed and supine MRIs
    breast_metadata.visualise_alignment_with_landmarks(
        supine_image_sitk, prone_image_transformed, sternum_supine_px[0], sternum_prone_transformed_px[0], orientation='axial')
    breast_metadata.visualise_alignment_with_landmarks(
        supine_image_sitk, prone_image_transformed, sternum_supine_px[1], sternum_prone_transformed_px[1], orientation='axial')

    # Loop through each landmark and create a visualization
    for i in range(len(sternum_supine_px)):
        print(f"Visualizing alignment for landmark #{i + 1}")
        breast_metadata.visualise_alignment_with_landmarks(
            supine_image_sitk,
            prone_image_transformed,
            landmark_supine_r1_px[i],
            landmark_prone_r1_transformed_px[i],
            orientation='axial'
        )

    return landmark_r1_displacement_vectors, landmark_r1_displacement_magnitudes, \
           landmark_r2_displacement_vectors, landmark_r2_displacement_magnitudes, \
           error, error_mag, T_optimal, res_optimal, prone_image_transformed