import os
import numpy as np
import automesh
from tools import landmarks_old as ld
from tools import realignment_tools
from tools import subjectModel
import h5py
import breast_metadata
import pyvista as pv
import plot

'''
    'soft_landmarks_path': contain all volunteer breast tissue landmarks (e.g., cyst, lymph node) 
                           identified by the registrars in the image coordinate system
    'rigid_landmarks_path': contain all volunteer anatomical landmarks - nipple (user002) and sternum (user001) 
                           identified by the registrars in the image coordinate system
    'masks_path': contain all volunteer masks (i.e., skin and rib cage) in the image coordinate system
    'prone_model_path': contain all volunteer prone skin and rib cage mesh and segmented data points (lung, skin, nipple)
                        in the model coordinate system


    'registrar1_prone': PRONE breast tissue landmarks (e.g., cyst, lymph node) identified by the registrar 1 
    'registrar2_prone': PRONE anatomical landmarks - nipple (user002) and sternum (user001) identified by the registrar 1

    Load:
    1) breast tissue landmarks identified by registrars (soft landmarks)
    2) 'prone_metadata': metadata for each volunteer (anatomical landmarks: nipple and sternum landmarks (rigid landmarks))
    3)

    The metadata includes the age, height, weight, nipple landmark position, and sternum landmark position 
    in the MRI coordinate. The metadata is used to transform the landmarks from the image coordinate system to the model
    coordinate system. The model coordinate system is defined by the skin mesh and the tissue landmarks.

    'registrar1_prone_landmarks': transformed breast tissue landmarks in the model coordinate system (from RAF 
                                  image coordinate system).
'''

if __name__ == '__main__':
    # volunteer IDs
    vl_ids = [12]

    # define all the paths
    mri_t2_images_root_path = r'U:\projects\volunteer_camri\old_data\mri_t2'
    soft_landmarks_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\T2-landmark-analysis-study\picker\points'
    rigid_landmarks_root_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\picker\points_wc_raf'
    nipple_path = os.path.join(rigid_landmarks_root_path, 'user002')
    sternum_path = os.path.join(rigid_landmarks_root_path, 'user001')
    # masks_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\automatic_segmentation_CNN\T2_from_T1'
    masks_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\automatic_segmentation_CNN\T2'
    # rib_mask_path = os.path.join(masks_path, r'prone\rib_cage\rib_cage_VL{0:05d}.nii'.format(vl_ids))
    skin_mask_path = os.path.join(masks_path, r'prone\body\body_VL{0:05d}.nii'.format(vl_ids[0]))
    prone_mesh_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\prone_to_supine_t2\2017_09_06\volunteer_meshes'
    supine_mesh_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\supine_to_prone_t2\2017-07-31'
    output_dir = r'C:\Users\jxu759\Documents\Motion_of_landmarks\output'

    position = 'prone'

    # Load T2 MR images
    mri_t2_images_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl_ids[0]), 'prone')

    # Load breast tissue landmarks identified by registrars
    # User008 = Ben, User007 = Clark
    registrar1_prone = ld.Landmarks('user008', vl_ids, position, soft_landmarks_path)
    registrar2_prone = ld.Landmarks('user007', vl_ids, position, soft_landmarks_path)

    # Load metadata from mr images and nipple, sternum, and spinal cord landmarks for each volunteer
    prone_metadata = ld.read_metadata(vl_ids, position, mri_t2_images_root_path, nipple_path, sternum_path)

    # Transform landmarks from RAI imaging coordinates system to ALS model coordinates system
    registrar1_prone_landmarks = registrar1_prone.getModellandmarks(prone_metadata)
    registrar2_prone_landmarks = registrar2_prone.getModellandmarks(prone_metadata)

    # Load anatomical landmarks identified by registrars

    # Plot MR images
    orientation_flag = "RAI"
    mri_t2_images = breast_metadata.Scan(mri_t2_images_path)
    mri_t2_images_grid = breast_metadata.SCANToPyvistaImageGrid(mri_t2_images, orientation_flag)

    registrar1_prone_landmark_1 = registrar1_prone.landmarks[vl_ids[0]]
    # slice_index = int(round((registrar1_prone_landmark_1[2]- mri_t2_images_grid.origin[2]) / mri_t2_images_grid.spacing[2]))
    #
    # # Extract an axial slice
    # axial_slice = mri_t2_images.values[:, :, slice_index]  # Slice through the Z-axis
    #
    # # Create a 2D PyVista ImageData object for the slice
    # slice_grid = pv.ImageData(dimensions=(axial_slice.shape[0], axial_slice.shape[1], 1))  # Set depth to 1
    # slice_grid.point_data['values'] = axial_slice.flatten(order='F')

    skin_mask = breast_metadata.readNIFTIImage(skin_mask_path, swap_axes=True)
    skin_mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(skin_mask, orientation_flag)

    # # Plot the MR images and landmarks
    # plotter = pv.Plotter()
    #
    # # plotter.add_volume(segmentation_mask_volume,
    # #                    cmap="tab10",
    # #                    opacity="sigmoid")
    #
    # opacity = np.linspace(0, 0.4, 100)
    # plotter.add_volume(mri_t2_images_grid, scalars='values', cmap='gray', opacity=opacity)
    #
    # left_nipple = np.array([[81.9419555664063, -66.75008704742683, -33.680382224490515]])
    # plotter.add_points(registrar1_prone_landmark_1,
    #                    color='red',
    #                    # color='#66B2FF',
    #                    render_points_as_spheres=True,
    #                    label='Point cloud',
    #                    point_size=12)
    #
    # legend_entries = [['Images', 'grey'], ['Point cloud', 'red']]
    # plotter.add_legend(legend_entries, bcolor='w')
    # plotter.add_text('VL00012', position='upper_left', font_size=10, color='black')
    #
    # plotter.show()

    # p = pv.Plotter()
    #
    # # Add the 2D slice to the plot
    # p.add_mesh(slice_grid, scalars='values', cmap='gray', opacity=1)
    #
    # # Add the landmark as a point or small sphere
    # # Extract (x, y) from the landmark since we’re in 2D on this slice
    # landmark_x, landmark_y = int(registrar1_prone_landmark_1[0]), int(registrar1_prone_landmark_1[1])
    # landmark_position = (landmark_x, landmark_y, 0)  # Set Z to 0 for 2D
    #
    # # Add a small sphere at the landmark position
    # landmark_sphere = pv.Sphere(radius=2, center=landmark_position)
    # p.add_mesh(landmark_sphere, color='red', label='Landmark')
    #
    # # Display the plot with the landmark
    # p.add_legend()
    # p.show()

    plotter = pv.Plotter()
    opacity = np.linspace(0, 0.4, 100)
    plotter.add_volume(mri_t2_images_grid, scalars='values', cmap='gray', opacity=opacity)
    thresholded_mask = skin_mask_image_grid.threshold(value=0.5)
    plotter.add_mesh(thresholded_mask, color='lightskyblue', opacity=0.2, show_scalar_bar=False)
    left_nipple = np.array([[81.9419555664063, -66.75008704742683, -33.680382224490515]])
    plotter.add_points(left_nipple, color='pink', render_points_as_spheres=True, label='Point cloud', point_size=14)
    plotter.add_points(registrar1_prone_landmark_1, color='red', render_points_as_spheres=True,
                                          label='Point cloud', point_size=12)
    legend_entries = [['Images', 'grey'], ['Segmentation mask', 'lightskyblue'],['point','pink']]
    plotter.add_legend(legend_entries, bcolor='w')
    plotter.add_text('VL00012', position='upper_left', font_size=10, color='black')

    plotter.show()