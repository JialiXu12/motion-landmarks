import os
import numpy as np
import automesh
from tools import landmarks as ld
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

    The metadata includes the age, height, weight, nipple landmark position, and sternum landmark position 
    in the MRI coordinate. The metadata is used to transform the landmarks from the image coordinate system to the model
    coordinate system. The model coordinate system is defined by the skin mesh and the tissue landmarks.

    'registrar1_prone_landmarks': transformed breast tissue landmarks in the model coordinate system (from RAI 
                                  image coordinate system).
'''

if __name__ == '__main__':
    # volunteer IDs
    vl_ids = [9]

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

    def process_landmark_position(position, vl_ids, mri_t2_images_root_path, soft_landmarks_path, nipple_path, sternum_path):
        # # Load T2 MR images
        # mri_t2_images_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl_ids[0]), position)

        # Load breast tissue landmarks identified by registrars
        # User008 = Ben, User007 = Clark
        registrar1_prone = ld.Landmarks('user008', vl_ids, position, soft_landmarks_path)
        registrar2_prone = ld.Landmarks('user007', vl_ids, position, soft_landmarks_path)

        # Load metadata from mr images and transformed nipple, sternum, and spinal cord landmarks for each volunteer
        metadata = ld.read_metadata(vl_ids, position, mri_t2_images_root_path, nipple_path, sternum_path)

        # # Transform landmarks from RAI imaging coordinates system to ALS model coordinates system
        # registrar1_landmarks = registrar1_prone.getModellandmarks(metadata)
        # registrar2_landmarks = registrar2_prone.getModellandmarks(metadata)

        return registrar1_prone, registrar2_prone, metadata

    # Process both prone and supine positions
    positions = ['prone', 'supine']
    results = {}

    for pos in positions:
        results[pos] = process_landmark_position(pos, vl_ids, mri_t2_images_root_path, soft_landmarks_path, nipple_path,
                                        sternum_path)

    # Access results for prone and supine
    registrar1_prone_landmarks, registrar2_prone_landmarks, prone_metadata = results['prone']
    registrar1_supine_landmarks, registrar2_supine_landmarks, supine_metadata = results['supine']

    # Corresponding landmarks between registrars
    # This function returns a dictionary where each volunteer's key maps to a list of corresponding landmark pairs.
    # Each pair contains two values: the first represents the local landmark number identified by the first registrar,
    # and the second represents the corresponding landmark number identified by the second registrar.
    cor_1_2 = ld.corresponding_landmarks_between_registrars(registrar1_prone_landmarks, registrar2_prone_landmarks,
        registrar1_supine_landmarks, registrar2_supine_landmarks)

    #To be deleted
    ld_indices = cor_1_2[vl_ids[0]][0]
    ld_indices_r1 = ld_indices[0]
    ld_indices_r2 = ld_indices[1]

    # # find the corresponding quadrant for each soft tissue landmark
    # registrar1_prone_landmarks.find_quadrants(prone_metadata)
    # registrar2_prone_landmarks.find_quadrants(prone_metadata)
    # registrar1_supine_landmarks.find_quadrants(supine_metadata)
    # registrar2_supine_landmarks.find_quadrants(supine_metadata)


    registrar1_dist_output = os.path.join(output_dir, 'registrar_1_t2')
    if not os.path.exists(registrar1_dist_output):
        os.mkdir(registrar1_dist_output)
    registrar2_dist_output = os.path.join(output_dir, 'registrar_2_t2')
    if not os.path.exists(registrar2_dist_output):
        os.mkdir(registrar2_dist_output)


    # Load MRI images
    mri_t2_images_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl_ids[0]), 'prone')
    mri_t2_images = breast_metadata.Scan(mri_t2_images_path)
    orientation_flag = "RAI"
    mri_t2_images_grid = breast_metadata.SCANToPyvistaImageGrid(mri_t2_images, orientation_flag)

    skin_mask = breast_metadata.readNIFTIImage(skin_mask_path, swap_axes=True)
    skin_mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(skin_mask, orientation_flag)

    from scipy.spatial import cKDTree
    from tools import sitkTools

    # Closest distances from landmark to skin and chest wall
    # def find_shortest_distances(metadata, masks_path, landmarks, output):
    distances_skin = {}
    closest_points_skin = {}
    distances_rib = {}
    closest_points_rib = {}
    #     dist_landmark_skin, points_skin, dist_landmark_cw, points_cw = ld.shortest_distances(metadata, masks_path,
    for vl_id in registrar1_prone_landmarks.vl_ids:
        skin_mask_path = os.path.join(masks_path, prone_metadata[vl_id].position,
                                      'body\\body_VL{0:05d}.nii'.format(vl_id))
        rib_mask_path = os.path.join(masks_path, prone_metadata[vl_id].position,
                                    'rib_cage\\rib_cage_VL{0:05d}.nii'.format(vl_id))

        if os.path.exists(skin_mask_path):
            skin_mask = breast_metadata.readNIFTIImage(skin_mask_path, swap_axes=True)
            skin_points = sitkTools.extract_contour_points(skin_mask, 100000)
            skin_points_kd_tree = cKDTree(skin_points)

            rib_mask = breast_metadata.readNIFTIImage(rib_mask_path, swap_axes=True)
            rib_points = sitkTools.extract_contour_points(rib_mask, 100000)
            rib_points_kd_tree = cKDTree(rib_points)

            for indx, landmarks in enumerate(registrar1_prone_landmarks.landmarks[vl_id]):
                # Query the KDTree to find the shortest distance to the landmark
                skin_distance, skin_index = skin_points_kd_tree.query(landmarks)
                # Get the coordinates of the nearest point on the skin
                distances_skin.setdefault(vl_id, []).append(skin_distance)
                skin_closest_point = skin_points[skin_index]
                closest_points_skin.setdefault(vl_id, []).append(skin_closest_point)

                # Query the KDTree to find the shortest distance to the landmark
                rib_distance, rib_index = rib_points_kd_tree.query(landmarks)
                # Get the coordinates of the nearest point on the rib cage
                distances_rib.setdefault(vl_id, []).append(rib_distance)
                rib_closest_point = rib_points[rib_index]
                closest_points_rib.setdefault(vl_id, []).append(rib_closest_point)

            # print('closest_points_skin: ', closest_points_skin)
            # print('distances_skin: ', distances_skin)
            # print('closest_points_rib: ', closest_points_rib)
            # print('distances_rib: ', distances_rib)
            # plotter = pv.Plotter()
            # opacity = np.linspace(0, 0.2, 100)
            # plotter.add_volume(mri_t2_images_grid, scalars='values', cmap='gray', opacity=opacity)
            # skin_mask_threshold = skin_mask_image_grid.threshold(value=0.5)
            # plotter.add_mesh(skin_mask_threshold, color='lightskyblue', opacity=0.2, show_scalar_bar=False)
            # # left_nipple = np.array([[81.9419555664063, -66.75008704742683, -33.680382224490515]])
            # # right_nipple = np.array([[-74.22015156855718, -65.497997141762, -28.90346372127533]])
            # left_nipple = prone_metadata[vl_id].left_nipple
            # right_nipple = prone_metadata[vl_id].right_nipple
            # plotter.add_points(left_nipple, color='pink', render_points_as_spheres=True, label='Point cloud',
            #                    point_size=14)
            # plotter.add_points(right_nipple, color='pink', render_points_as_spheres=True, label='Point cloud',
            #                    point_size=14)
            # plotter.add_points(np.array(registrar1_prone_landmarks.landmarks[vl_ids[0]]), color='red', render_points_as_spheres=True,
            #                    label='Point cloud', point_size=12)
            # plotter.add_points(skin_points, color='#F5CBA7', render_points_as_spheres=True, opacity=0.2,
            #                    label='Point cloud', point_size=5)
            # plotter.add_points(np.array(closest_points_skin[vl_ids[0]]), color='green', render_points_as_spheres=True,
            #                    label='Point cloud', point_size=10)
            # plotter.add_points(np.array(closest_points_rib[vl_ids[0]]), color='blue', render_points_as_spheres=True,
            #                    label='Point cloud', point_size=10)
            # # Draw lines between corresponding points in landmarks and skin
            # for i in range(len(registrar1_prone_landmarks.landmarks[vl_ids[0]])):
            #     line = pv.Line(registrar1_prone_landmarks.landmarks[vl_ids[0]][i], np.array(closest_points_skin[vl_ids[0]])[i])
            #     plotter.add_mesh(line, color="yellow", line_width=3, label='Line')
            # # Draw lines between corresponding points in landmarks and rib cage
            # for i in range(len(registrar1_prone_landmarks.landmarks[vl_ids[0]])):
            #     line = pv.Line(registrar1_prone_landmarks.landmarks[vl_ids[0]][i], np.array(closest_points_rib[vl_ids[0]])[i])
            #     plotter.add_mesh(line, color="magenta", line_width=3, label='Line')
            # #
            # for i in range(len(registrar1_prone_landmarks.landmarks[vl_ids[0]])):
            #     if abs(left_nipple[0]-registrar1_prone_landmarks.landmarks[vl_ids[0]][i][0]) < abs(right_nipple[0]-registrar1_prone_landmarks.landmarks[vl_ids[0]][i][0]):
            #         line = pv.Line(registrar1_prone_landmarks.landmarks[vl_ids[0]][i], left_nipple)
            #     else:
            #         line = pv.Line(registrar1_prone_landmarks.landmarks[vl_ids[0]][i], right_nipple)
            #     plotter.add_mesh(line, color="cyan", line_width=3, label='Line')
            # legend_entries = [['Images', 'grey'], ['Segmentation mask', 'lightskyblue'], ['Left_nipple', 'pink'],
            #                   ['Right_nipple', 'pink'], ['Skin points', '#F5CBA7'], ['Closest skin points', 'green'],
            #                   ['Closest rib cage points', 'blue'],['Distance between landmarks and skin','yellow'],
            #                   ['Distance between landmarks and rib cage','magenta'],['Distance between landmarks and nipple','cyan']]
            # plotter.add_legend(legend_entries, bcolor='w')
            # plotter.add_text('VL00012', position='upper_left', font_size=10, color='black')
            #
            # plotter.show()


            v_ld_nipple = prone_metadata[vl_id].left_nipple - registrar1_prone_landmarks.landmarks[vl_ids[0]][1]
            v_ld_skin = closest_points_skin[vl_ids[0]][1] - registrar1_prone_landmarks.landmarks[vl_ids[0]][1]
            v_ld_rib = closest_points_rib[vl_ids[0]][1] - registrar1_prone_landmarks.landmarks[vl_ids[0]][1]
            n = np.array([0, 0, 1]) # The normal vector of the plane (e.g., z-axis plane)
            projection_nipple = v_ld_nipple - np.dot(v_ld_nipple, n) * n
            projection_skin = v_ld_skin - np.dot(v_ld_skin, n) * n
            projection_rib = v_ld_rib - np.dot(v_ld_rib, n) * n

            origin = [0, 0, int(registrar1_prone_landmarks.landmarks[vl_ids[0]][1][2])]
            cent = np.array([registrar1_prone_landmarks.landmarks[vl_ids[0]][1][0], registrar1_prone_landmarks.landmarks[vl_ids[0]][1][1],
                            int(registrar1_prone_landmarks.landmarks[vl_ids[0]][1][2])])
            p = pv.Plotter()
            p.add_mesh(mri_t2_images_grid.slice(origin=origin,
                            normal=[0, 0, 1]), cmap='coolwarm', show_scalar_bar=False)
            p.add_points(np.array(registrar1_prone_landmarks.landmarks[vl_ids[0]][1]), color='red', render_points_as_spheres=True,
                                label='Point cloud', point_size=6)

            # # Add the projected vectors (arrows) to the plotter
            # p.add_arrows(cent=cent, direction=projection_nipple, color="red",
            #                    label="Projected Nipple Vector")
            #
            # p.add_arrows(cent=cent, direction=projection_skin, color="blue",
            #                    label="Projected Skin Vector")
            #
            # p.add_arrows(cent=cent, direction=projection_rib, color="green",
            #                    label="Projected Rib Vector")

            # Draw a line between the landmark and the nipple, projected onto the z-axis plane
            line_skin = pv.Line(cent, np.array(closest_points_skin[vl_ids[0]])[1])
            p.add_mesh(line_skin, color="yellow", line_width=3, label='Line')
            projected_skin = [closest_points_skin[vl_ids[0]][1][0], closest_points_skin[vl_ids[0]][1][1], cent[2]]
            line_projected_skin = pv.Line(cent, projected_skin)
            p.add_mesh(line_projected_skin, color="blue", line_width=5, label='Line')

            line_rib_cage = pv.Line(cent, np.array(closest_points_rib[vl_ids[0]])[1])
            p.add_mesh(line_rib_cage, color="magenta", line_width=3, label='Line')
            projected_rib_cage = [closest_points_rib[vl_ids[0]][1][0], closest_points_rib[vl_ids[0]][1][1], cent[2]]
            line_projected_rib_cage = pv.Line(cent, projected_rib_cage)
            p.add_mesh(line_projected_rib_cage, color="green", line_width=5, label='Line')

            line_nipple = pv.Line(cent, prone_metadata[vl_id].left_nipple)
            p.add_mesh(line_nipple, color="cyan", line_width=3, label='Line')
            projected_nipple = [prone_metadata[vl_id].left_nipple[0], prone_metadata[vl_id].left_nipple[1],
                                cent[2]]
            line_projected_nipple = pv.Line(cent, projected_nipple)
            p.add_mesh(line_projected_nipple, color="red", line_width=5, label='Line')




            labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
                          ylabel=f"{orientation_flag[1]} (mm)",
                          zlabel=f"{orientation_flag[2]} (mm)")
            legend_entries = [['Projected Skin Vector', 'blue'],
                            ['Projected Rib Vector', 'green'], ['Projected Nipple Vector', 'red']]
            p.add_legend(legend_entries, bcolor='w')
            p.show_grid(**labels)
            p.add_axes(**labels)
            p.view_xy()
            p.show()




    dist_landmark_skin_p1, points_skin_p1, \
        dist_landmark_cw_p1, points_cw_p1 = ld.shortest_distances(prone_metadata, masks_path,
                                                                  registrar1_prone_landmarks,
                                                                  registrar1_dist_output)
    dist_landmark_skin_p2, points_skin_p2, \
        dist_landmark_cw_p2, points_cw_p2 = ld.shortest_distances(prone_metadata, masks_path,
                                                                  registrar2_prone_landmarks,
                                                                  registrar2_dist_output)
    dist_landmark_skin_s1, points_skin_s1, \
        dist_landmark_cw_s1, points_cw_s1 = ld.shortest_distances(supine_metadata, masks_path,
                                                                  registrar1_supine_landmarks,
                                                                  registrar1_dist_output)
    dist_landmark_skin_s2, points_skin_s2, \
        dist_landmark_cw_s2, points_cw_s2 = ld.shortest_distances(supine_metadata, masks_path,
                                                                  registrar2_supine_landmarks,
                                                                  registrar2_dist_output)

    # Landmark positions in time coordinates, distance to the nipple
    time_1p, quadrants_1p, dist_landmark_nipple_1p = ld.clock(registrar1_prone_landmarks, prone_metadata)
    time_2p, dist_landmark_nipple_2p = ld.clock(registrar2_prone_landmarks, prone_metadata)
    time_1s, dist_landmark_nipple_1s = ld.clock(registrar1_supine_landmarks, supine_metadata)
    time_2s, dist_landmark_nipple_2s = ld.clock(registrar2_supine_landmarks, supine_metadata)


    dist_skin = dist_landmark_skin_p1[vl_ids[0]]





















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


