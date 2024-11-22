import os
import numpy as np
import automesh
from tools import landmarks as ld
from tools import realignment_tools
from tools import subjectModel
import pandas as pd
import breast_metadata
import pyvista as pv
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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
    plot_figures = True
    # volunteer IDs
    # vl_ids = [11,12,17,31,64,77]
    # vl_ids = [9,11,12,14,15,17,18,19,20,22,25,29,30,31,32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,
    #           54,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
    vl_ids = [9,11,12,14,15,17,18,19,20,22,25,29,30,31,32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,
              54,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,81,84,85,86,87,88,89]

    # define all the paths
    mri_t2_images_root_path = r'U:\projects\volunteer_camri\old_data\mri_t2'
    # soft_landmarks_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\T2-landmark-analysis-study\picker\points'

    soft_landmarks_r1_path = r'U:\projects\dashboard\picker_points\ben_reviewed'
    soft_landmarks_r2_path = r"U:\projects\dashboard\picker_points\holly"

    rigid_landmarks_root_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\picker\points_wc_raf'
    nipple_path = os.path.join(rigid_landmarks_root_path, 'user002')
    sternum_path = os.path.join(rigid_landmarks_root_path, 'user001')

    # masks_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\automatic_segmentation_CNN\T2_from_T1'
    masks_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\automatic_segmentation_CNN\T2'

    prone_mesh_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\prone_to_supine_t2\2017_09_06\volunteer_meshes'
    supine_mesh_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\supine_to_prone_t2\2017-07-31'
    output_dir = r'..\output'

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


    # Closest distances from landmark to skin and chest wall
    dist_landmark_skin_r1_p, closest_points_skin_r1_p, \
        dist_landmark_rib_r1_p, closest_points_rib_r1_p = ld.shortest_distances(prone_metadata, masks_path,
                                                                                registrar1_prone_landmarks)
    dist_landmark_skin_r2_p, closest_points_r2_p, \
        dist_landmark_rib_r2_p, closest_points_rib_r2_p = ld.shortest_distances(prone_metadata, masks_path,
                                                                                registrar2_prone_landmarks)
    dist_landmark_skin_r1_s, closest_points_skin_r1_s, \
        dist_landmark_rib_r1_s, closest_points_rib_r1_s = ld.shortest_distances(supine_metadata, masks_path,
                                                                                registrar1_supine_landmarks)
    dist_landmark_skin_r2_s, closest_points_skin_r2_s, \
        dist_landmark_rib_r2_s, closest_points_rib_r2_s = ld.shortest_distances(supine_metadata, masks_path,
                                                                                registrar2_supine_landmarks)

    # Landmark positions in time coordinates, distance to the nipple
    side_r1_p, time_r1_p, quadrants_r1_p, dist_landmark_nipple_r1_p = ld.clock(registrar1_prone_landmarks, prone_metadata)
    side_r2_p, time_r2_p, quadrants_r2_p, dist_landmark_nipple_r2_p = ld.clock(registrar2_prone_landmarks, prone_metadata)
    side_r1_s, time_r1_s, quadrants_r1_s, dist_landmark_nipple_r1_s = ld.clock(registrar1_supine_landmarks, supine_metadata)
    side_r2_s, time_r2_s, quadrants_r2_s, dist_landmark_nipple_r2_s = ld.clock(registrar2_supine_landmarks, supine_metadata)

    # Calculate the displacement of the landmarks between prone and supine positions
    # landmarks_displacement = ld.landmark_displacement(registrar1_prone_landmarks, registrar1_supine_landmarks)
    landmarks_displacement_r1 = [np.arange(1, len(registrar1_prone_landmarks.landmarks[vl_id])+1) for vl_id in vl_ids]
    landmarks_displacement_r2 = [np.arange(1, len(registrar1_prone_landmarks.landmarks[vl_id])+1) for vl_id in vl_ids]

    # define output paths
    # registrar1_dist_output = os.path.join(output_dir, 'registrar_1_t2')
    # if not os.path.exists(registrar1_dist_output):
    #     os.mkdir(registrar1_dist_output)
    # registrar2_dist_output = os.path.join(output_dir, 'registrar_2_t2')
    # if not os.path.exists(registrar2_dist_output):
    #     os.mkdir(registrar2_dist_output)
    excel_file = "../output/landmarks_results.xlsx"
    fig_path = "../output/figs"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if plot_figures:
        from tools import sitkTools
        for vl_id in vl_ids:
            prone_skin_mask_path = os.path.join(masks_path, r'prone\body\body_VL{0:05d}.nii'.format(vl_id))
            supine_skin_mask_path = os.path.join(masks_path, r'supine\body\body_VL{0:05d}.nii'.format(vl_id))
            prone_rib_mask_path = os.path.join(masks_path, r'prone\rib_cage\rib_cage_VL{0:05d}.nii'.format(vl_id))
            supine_rib_mask_path = os.path.join(masks_path, r'supine\rib_cage\rib_cage_VL{0:05d}.nii'.format(vl_id))

            for position in positions:
                if position == 'prone':
                    closest_points_skin = closest_points_skin_r1_p
                    closest_points_rib = closest_points_rib_r1_p
                    skin_mask_path = prone_skin_mask_path
                    rib_mask_path = prone_rib_mask_path
                    metadata = prone_metadata
                else:
                    closest_points_skin = closest_points_skin_r1_s
                    closest_points_rib = closest_points_rib_r1_s
                    skin_mask_path = supine_skin_mask_path
                    rib_mask_path = supine_rib_mask_path
                    metadata = supine_metadata

                # Load example prone MRI images
                mri_t2_images_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl_id), position)
                mri_t2_images = breast_metadata.Scan(mri_t2_images_path)
                orientation_flag = "RAI"
                mri_t2_images_grid = breast_metadata.SCANToPyvistaImageGrid(mri_t2_images, orientation_flag)

                skin_mask = breast_metadata.readNIFTIImage(skin_mask_path, swap_axes=True)
                skin_mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(skin_mask, orientation_flag)
                rib_mask = breast_metadata.readNIFTIImage(rib_mask_path, swap_axes=True)
                rib_mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(rib_mask, orientation_flag)

                plotter = pv.Plotter()
                opacity = np.linspace(0, 0.2, 100)
                plotter.add_volume(mri_t2_images_grid, scalars='values', cmap='gray', opacity=opacity)
                skin_mask_threshold = skin_mask_image_grid.threshold(value=0.5)
                rib_mask_threshold = rib_mask_image_grid.threshold(value=0.5)

                plotter.add_mesh(skin_mask_threshold, color='lightskyblue', opacity=0.2, show_scalar_bar=False)
                plotter.add_mesh(rib_mask_threshold, color='lavender', opacity=0.2, show_scalar_bar=False)

                left_nipple = metadata[vl_id].left_nipple
                right_nipple = metadata[vl_id].right_nipple
                plotter.add_points(left_nipple, color='pink', render_points_as_spheres=True, label='Point cloud',
                                   point_size=14)
                plotter.add_points(right_nipple, color='pink', render_points_as_spheres=True, label='Point cloud',
                                   point_size=14)
                plotter.add_points(np.array(registrar1_prone_landmarks.landmarks[vl_id]), color='red', render_points_as_spheres=True,
                                   label='Point cloud', point_size=12)
                plotter.add_points(np.array(closest_points_skin[vl_id]), color='green', render_points_as_spheres=True,
                                   label='Point cloud', point_size=10)
                plotter.add_points(np.array(closest_points_rib[vl_id]), color='blue', render_points_as_spheres=True,
                                   label='Point cloud', point_size=10)

                # Draw lines between corresponding points in landmarks and skin
                for i in range(len(registrar1_prone_landmarks.landmarks[vl_id])):
                    line = pv.Line(registrar1_prone_landmarks.landmarks[vl_id][i], np.array(closest_points_skin[vl_id])[i])
                    plotter.add_mesh(line, color="yellow", line_width=3, label='Line')
                # Draw lines between corresponding points in landmarks and rib cage
                for i in range(len(registrar1_prone_landmarks.landmarks[vl_id])):
                    line = pv.Line(registrar1_prone_landmarks.landmarks[vl_id][i], np.array(closest_points_rib[vl_id])[i])
                    plotter.add_mesh(line, color="magenta", line_width=3, label='Line')

                for i in range(len(registrar1_prone_landmarks.landmarks[vl_id])):
                    if abs(left_nipple[0]-registrar1_prone_landmarks.landmarks[vl_id][i][0]) < abs(right_nipple[0]-registrar1_prone_landmarks.landmarks[vl_id][i][0]):
                        line = pv.Line(registrar1_prone_landmarks.landmarks[vl_id][i], left_nipple)
                    else:
                        line = pv.Line(registrar1_prone_landmarks.landmarks[vl_id][i], right_nipple)
                    plotter.add_mesh(line, color="cyan", line_width=3, label='Line')
                legend_entries = [['Images', 'grey'], ['Segmentation mask', 'lightskyblue'], ['Left_nipple', 'pink'],
                                  ['Right_nipple', 'pink'], ['Closest skin points', 'green'],
                                  ['Closest rib cage points', 'blue'],['Distance between landmarks and skin','yellow'],
                                  ['Distance between landmarks and rib cage','magenta'],['Distance between landmarks and nipple','cyan']]

                plotter.add_legend(legend_entries, bcolor='w')
                plotter.add_text(f"VL{vl_id:05d} {position}", position='upper_left', font_size=10, color='black')
                labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
                              ylabel=f"{orientation_flag[1]} (mm)",
                              zlabel=f"{orientation_flag[2]} (mm)")
                plotter.add_axes(**labels)
                plotter.view_xy()

                plotter.show()

    # Save the results to an Excel file
    df_r1 = pd.DataFrame({
        'Registrar': [1] * len(vl_ids),
        'VL_ID': vl_ids,
        'Age': [prone_metadata[vl_id].age for vl_id in vl_ids],
        'Height [m]': [prone_metadata[vl_id].height for vl_id in vl_ids],
        'Weight [kg]': [prone_metadata[vl_id].weight for vl_id in vl_ids],
        'Landmark number': [np.arange(1, len(registrar1_prone_landmarks.landmarks[vl_id])+1) for vl_id in vl_ids],
        'Landmark type': [registrar1_prone_landmarks.landmark_types[vl_id] for vl_id in vl_ids],
        'Distance to nipple (prone) [mm]': [dist_landmark_nipple_r1_p[vl_id] for vl_id in vl_ids],
        'Distance to nipple (supine) [mm]': [dist_landmark_nipple_r1_s[vl_id] for vl_id in vl_ids],
        'Distance to rib cage (prone) [mm]': [dist_landmark_rib_r1_p[vl_id] for vl_id in vl_ids],
        'Distance to rib cage (supine) [mm]': [dist_landmark_rib_r1_s[vl_id] for vl_id in vl_ids],
        'Distance to skin (prone) [mm]': [dist_landmark_skin_r1_p[vl_id] for vl_id in vl_ids],
        'Distance to skin (supine) [mm]': [dist_landmark_skin_r1_s[vl_id] for vl_id in vl_ids],
        'Time (prone)': [time_r1_p[vl_id] for vl_id in vl_ids],
        'Time (supine)': [time_r1_s[vl_id] for vl_id in vl_ids],
        'Quadrant (prone)': [quadrants_r1_p[vl_id] for vl_id in vl_ids],
        'Quadrant (supine)': [quadrants_r1_s[vl_id] for vl_id in vl_ids],
        'Landmark displacement [mm]': landmarks_displacement_r1
    })
    print(df_r1)

    # Explode the list columns
    list_columns = [
        'Landmark number', 'Landmark type', 'Distance to nipple (prone) [mm]', 'Distance to nipple (supine) [mm]',
        'Distance to rib cage (prone) [mm]', 'Distance to rib cage (supine) [mm]',
        'Distance to skin (prone) [mm]', 'Distance to skin (supine) [mm]',
        'Time (prone)', 'Time (supine)', 'Quadrant (prone)', 'Quadrant (supine)', 'Landmark displacement [mm]'
    ]

    df_r1 = df_r1.explode(list_columns, ignore_index=True)

    df_r2 = pd.DataFrame({
        'Registrar': [2] * len(vl_ids),
        'VL_ID': vl_ids,
        'Age': [prone_metadata[vl_id].age for vl_id in vl_ids],
        'Height [m]': [prone_metadata[vl_id].height for vl_id in vl_ids],
        'Weight [kg]': [prone_metadata[vl_id].weight for vl_id in vl_ids],
        'Landmark number': [np.arange(1, len(registrar2_prone_landmarks.landmarks[vl_id]) + 1) for vl_id in vl_ids],
        'Landmark type': [registrar2_prone_landmarks.landmark_types[vl_id] for vl_id in vl_ids],
        'Distance to nipple (prone) [mm]': [dist_landmark_nipple_r2_p[vl_id] for vl_id in vl_ids],
        'Distance to nipple (supine) [mm]': [dist_landmark_nipple_r2_s[vl_id] for vl_id in vl_ids],
        'Distance to rib cage (prone) [mm]': [dist_landmark_rib_r2_p[vl_id] for vl_id in vl_ids],
        'Distance to rib cage (supine) [mm]': [dist_landmark_rib_r2_s[vl_id] for vl_id in vl_ids],
        'Distance to skin (prone) [mm]': [dist_landmark_skin_r2_p[vl_id] for vl_id in vl_ids],
        'Distance to skin (supine) [mm]': [dist_landmark_skin_r2_s[vl_id] for vl_id in vl_ids],
        'Time (prone)': [time_r2_p[vl_id] for vl_id in vl_ids],
        'Time (supine)': [time_r2_s[vl_id] for vl_id in vl_ids],
        'Quadrant (prone)': [quadrants_r2_p[vl_id] for vl_id in vl_ids],
        'Quadrant (supine)': [quadrants_r2_s[vl_id] for vl_id in vl_ids],
        'Landmark displacement [mm]': landmarks_displacement_r2
    })
    df_r2 = df_r2.explode(list_columns, ignore_index=True)

    df_combined = pd.concat([df_r1, df_r2], ignore_index=True)

    # df_combined.to_excel(excel_file, index=False)
    print(f"Data saved to {excel_file}")

    ''' Plot the landmarks with respect to nipples in three planes '''
    # prone registrar 1
    landmarks_left = []
    landmarks_right = []
    distance_to_skin_left = []
    distance_to_skin_right = []
    for vl_id in vl_ids:
        for i, side in enumerate(side_r1_p[vl_id]):
            if side == 'LB':
                nipple = prone_metadata[vl_id].left_nipple
                landmarks_left.append(registrar1_prone_landmarks.landmarks[vl_id][i] - nipple)
                distance_to_skin_left.append(dist_landmark_skin_r1_p[vl_id][i])
            else:
                nipple = prone_metadata[vl_id].right_nipple
                landmarks_right.append(registrar1_prone_landmarks.landmarks[vl_id][i] - nipple)
                distance_to_skin_right.append(dist_landmark_skin_r1_p[vl_id][i])

    landmarks_left = np.array(landmarks_left)
    landmarks_right = np.array(landmarks_right)
    distance_to_skin_left = np.array(distance_to_skin_left)
    distance_to_skin_right = np.array(distance_to_skin_right)
    distance_to_skin_all = np.concatenate([distance_to_skin_left, distance_to_skin_right])

    min_x = np.min(landmarks_right[:, 0])
    max_x = np.max(landmarks_left[:, 0])
    min_y = np.min(landmarks_right[:, 1])
    max_y = np.max(landmarks_left[:, 1])
    min_z = np.min(landmarks_right[:, 2])
    max_z = np.max(landmarks_left[:, 2])
    print('Min X:', min_x, 'Max X:', max_x)
    print('Min Y:', min_y, 'Max Y:', max_y)
    print('Min Z:', min_z, 'Max Z:', max_z)


    # Define color map based on distances
    norm = Normalize(vmin=distance_to_skin_all.min(), vmax=distance_to_skin_all.max())
    cmap = plt.cm.viridis

    fig_titles = ["Coronal View (Right Breast)", "Coronal View (Left Breast)",
                  "Sagittal View (Right Breast)", "Sagittal View (Left Breast)",
                  "Axial View (Right Breast)", "Axial View (Left Breast)"]
    views = [(0, 2), (1, 2), (0, 1)]  # X-Z, Y-Z, X-Y indices for each view
    x_labels = ["Right - Left", "Anterior - Posterior", " Right - Left"]
    y_labels = ["Inferior - Superior", "Inferior - Superior", "Anterior - Posterior"]
    # Axis limits and ticks for each view
    axis_settings = [
        {"xlim": (-60, 60), "ylim": (-60, 60), "xticks": np.arange(-60, 61, 20), "yticks": np.arange(-60, 61, 20)},
        {"xlim": (0, 140), "ylim": (-60, 60), "xticks": np.arange(0, 141, 20), "yticks": np.arange(-60, 61, 20)},
        {"xlim": (-60, 60), "ylim": (0, 140), "xticks": np.arange(-60, 61, 20), "yticks": np.arange(0, 141, 20)}
    ]

    # Plot in coronal, sagittal, and axial views
    for i, view in enumerate(views):
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        # Plot Left Breast
        axs[0].scatter(landmarks_right[:, view[0]], landmarks_right[:, view[1]], c=distance_to_skin_right, cmap=cmap, s=100, edgecolor="k")
        axs[0].set_title(f"{fig_titles[i]}")
        axs[0].set_xlabel(f"{x_labels[i]}")
        axs[0].set_ylabel(f"{y_labels[i]}")
        axs[0].set_xlim(axis_settings[i]["xlim"])
        axs[0].set_ylim(axis_settings[i]["ylim"])
        axs[0].set_xticks(axis_settings[i]["xticks"])
        axs[0].set_yticks(axis_settings[i]["yticks"])

        # Plot Right Breast
        axs[1].scatter(landmarks_left[:, view[0]], landmarks_left[:, view[1]], c=distance_to_skin_left, s=100, edgecolor="k")
        axs[1].set_title(f"{fig_titles[i+1]}")
        axs[1].set_xlabel(f"{x_labels[i]}")
        axs[1].set_ylabel(f"{y_labels[i]}")
        axs[1].set_xlim(axis_settings[i]["xlim"])
        axs[1].set_ylim(axis_settings[i]["ylim"])
        axs[1].set_xticks(axis_settings[i]["xticks"])
        axs[1].set_yticks(axis_settings[i]["yticks"])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array for the colorbar
        fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='vertical', label='DTS (mm)')

        # Save the figure
        fig_name = ["Coronal_View.png", "Sagittal_View.png", "Axial_View.png"]
        fig_filename = os.path.join(fig_path, fig_name[i])
        # plt.savefig(fig_filename, dpi=300)
        print(f"Saved {fig_filename}")
        plt.show()
        # Close the figure to free up memory
        plt.close(fig)

'''

    #PRONE POSITION
    # Plot the MR images and landmarks
    vl_id = vl_ids[0]
    closest_points_skin = closest_points_skin_r1_p
    closest_points_rib = closest_points_rib_r1_p

    # Load example prone MRI images
    mri_t2_images_prone_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl_ids[0]), 'prone')
    mri_t2_images_prone = breast_metadata.Scan(mri_t2_images_prone_path)
    orientation_flag = "RAI"
    mri_t2_images_grid_prone = breast_metadata.SCANToPyvistaImageGrid(mri_t2_images_prone, orientation_flag)

    skin_mask_prone = breast_metadata.readNIFTIImage(prone_skin_mask_path, swap_axes=True)
    skin_mask_image_grid_prone = breast_metadata.SCANToPyvistaImageGrid(skin_mask_prone, orientation_flag)

    from tools import sitkTools
    skin_points = sitkTools.extract_contour_points(skin_mask_prone, 100000)

    # plotter = pv.Plotter()
    # opacity = np.linspace(0, 0.15, 100)
    # # plotter.add_volume(mri_t2_images_grid_prone, scalars='values', cmap='gray', opacity=opacity)
    # skin_mask_threshold = skin_mask_image_grid_prone.threshold(value=0.5)
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
    # # plotter.add_points(skin_points, color='#F5CBA7', render_points_as_spheres=True, opacity=0.2,
    # #                    label='Point cloud', point_size=5)
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
    # # legend_entries = [['Images', 'grey'], ['Segmentation mask', 'lightskyblue'], ['Left_nipple', 'pink'],
    # #                   ['Right_nipple', 'pink'], ['Skin points', '#F5CBA7'], ['Closest skin points', 'green'],
    # #                   ['Closest rib cage points', 'blue'],['Distance between landmarks and skin','yellow'],
    # #                   ['Distance between landmarks and rib cage','magenta'],['Distance between landmarks and nipple','cyan']]
    # legend_entries = [['Segmentation mask', 'lightskyblue'], ['Left_nipple', 'pink'],
    #                   ['Right_nipple', 'pink'], ['Closest skin points', 'green'],
    #                   ['Closest rib cage points', 'blue'],['Distance between landmarks and skin','yellow'],
    #                   ['Distance between landmarks and rib cage','magenta'],['Distance between landmarks and nipple','cyan']]
    # plotter.add_legend(legend_entries, bcolor='w')
    # plotter.add_text(f"VL{vl_id:05d}", position='upper_left', font_size=10, color='black')
    # labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
    #               ylabel=f"{orientation_flag[1]} (mm)",
    #               zlabel=f"{orientation_flag[2]} (mm)")
    # plotter.add_axes(**labels)
    # plotter.view_xy()
    #
    # plotter.show()



    v_ld_nipple = prone_metadata[vl_id].left_nipple - registrar1_prone_landmarks.landmarks[vl_ids[0]][1]
    v_ld_skin = closest_points_skin[vl_ids[0]][1] - registrar1_prone_landmarks.landmarks[vl_ids[0]][1]
    v_ld_rib = closest_points_rib[vl_ids[0]][1] - registrar1_prone_landmarks.landmarks[vl_ids[0]][1]
    n = np.array([0, 0, 1]) # The normal vector of the plane (e.g., z-axis plane)
    projection_nipple = v_ld_nipple - np.dot(v_ld_nipple, n) * n
    projection_skin = v_ld_skin - np.dot(v_ld_skin, n) * n
    projection_rib = v_ld_rib - np.dot(v_ld_rib, n) * n

    origin_landmark = [0, 0, int(registrar1_prone_landmarks.landmarks[vl_ids[0]][1][2])]
    cent = np.array([registrar1_prone_landmarks.landmarks[vl_ids[0]][1][0], registrar1_prone_landmarks.landmarks[vl_ids[0]][1][1],
                    int(registrar1_prone_landmarks.landmarks[vl_ids[0]][1][2])])
    origin_nipple = [0, 0, int(prone_metadata[vl_id].left_nipple[2])]
    origin_skin = [0, 0, int(closest_points_skin[vl_ids[0]][1][2])]
    origin_rib = [0, 0, int(closest_points_rib[vl_ids[0]][1][2])]

    p = pv.Plotter()
    # opacity = np.linspace(0, 0.9, 50)
    opacity = 0.6
    # plotter.add_volume(mri_t2_images_grid_prone, scalars='values', cmap='gray', opacity=opacity)

    p.add_mesh(mri_t2_images_grid_prone.slice(origin=origin_landmark, normal=[0, 0, 1]),
               cmap='coolwarm', opacity=opacity, show_scalar_bar=False)
    p.add_mesh(mri_t2_images_grid_prone.slice(origin=origin_nipple, normal=[0, 0, 1]),
               cmap='coolwarm', opacity=opacity, show_scalar_bar=False)
    p.add_mesh(mri_t2_images_grid_prone.slice(origin=origin_skin, normal=[0, 0, 1]),
               cmap='coolwarm', opacity=opacity, show_scalar_bar=False)
    p.add_mesh(mri_t2_images_grid_prone.slice(origin=origin_rib, normal=[0, 0, 1]),
               cmap='coolwarm', opacity=opacity, show_scalar_bar=False)
    p.add_points(np.array(registrar1_prone_landmarks.landmarks[vl_ids[0]][1]), color='red',
                 render_points_as_spheres=True, label='Point cloud', point_size=6)

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
    p.add_mesh(line_projected_skin, color="blue", line_width=3, label='Line')

    line_rib_cage = pv.Line(cent, np.array(closest_points_rib[vl_ids[0]])[1])
    p.add_mesh(line_rib_cage, color="magenta", line_width=3, label='Line')
    projected_rib_cage = [closest_points_rib[vl_ids[0]][1][0], closest_points_rib[vl_ids[0]][1][1], cent[2]]
    line_projected_rib_cage = pv.Line(cent, projected_rib_cage)
    p.add_mesh(line_projected_rib_cage, color="green", line_width=3, label='Line')

    line_nipple = pv.Line(cent, prone_metadata[vl_id].left_nipple)
    p.add_mesh(line_nipple, color="cyan", line_width=3, label='Line')
    projected_nipple = [prone_metadata[vl_id].left_nipple[0], prone_metadata[vl_id].left_nipple[1],
                        cent[2]]
    line_projected_nipple = pv.Line(cent, projected_nipple)
    p.add_mesh(line_projected_nipple, color="red", line_width=3, label='Line')


    labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
                  ylabel=f"{orientation_flag[1]} (mm)",
                  zlabel=f"{orientation_flag[2]} (mm)")
    legend_entries = [['Shortest skin distance', 'yellow'], ['Shortest rib distance', 'magenta'],
                      ['Shortest rib distance', 'cyan'], ['Projected skin distance', 'blue'],
                      ['Projected rib distance', 'green'], ['Projected nipple distance', 'red']]
    # legend_entries = [['Projected skin distance', 'blue'],
    #                   ['Projected rib distance', 'green'], ['Projected nipple distance', 'red']]
    p.add_legend(legend_entries, bcolor='w')
    p.show_grid(**labels)
    p.add_axes(**labels)
    p.view_xy()
    p.show()



    #SUPINE POSITION
    # Plot the MR images and landmarks
    vl_id = vl_ids[0]
    closest_points_skin = closest_points_skin_r1_s
    closest_points_rib = closest_points_rib_r1_s

    # Load example prone MRI images
    mri_t2_images_supine_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl_ids[0]), 'supine')
    mri_t2_images_supine = breast_metadata.Scan(mri_t2_images_supine_path)
    orientation_flag = "RAI"
    mri_t2_images_grid_supine = breast_metadata.SCANToPyvistaImageGrid(mri_t2_images_supine, orientation_flag)

    skin_mask_supine = breast_metadata.readNIFTIImage(supine_skin_mask_path, swap_axes=True)
    skin_mask_image_grid_supine = breast_metadata.SCANToPyvistaImageGrid(skin_mask_supine, orientation_flag)

    from tools import sitkTools
    skin_points = sitkTools.extract_contour_points(skin_mask_supine, 100000)

    plotter = pv.Plotter()
    opacity = np.linspace(0, 0.15, 100)
    plotter.add_volume(mri_t2_images_grid_supine, scalars='values', cmap='gray', opacity=opacity)
    skin_mask_threshold = skin_mask_image_grid_supine.threshold(value=0.5)
    # plotter.add_mesh(skin_mask_threshold, color='lightskyblue', opacity=0.2, show_scalar_bar=False)
    # left_nipple = np.array([[81.9419555664063, -66.75008704742683, -33.680382224490515]])
    # right_nipple = np.array([[-74.22015156855718, -65.497997141762, -28.90346372127533]])
    left_nipple = supine_metadata[vl_id].left_nipple
    right_nipple = supine_metadata[vl_id].right_nipple
    plotter.add_points(left_nipple, color='pink', render_points_as_spheres=True, label='Point cloud',
                       point_size=14)
    plotter.add_points(right_nipple, color='pink', render_points_as_spheres=True, label='Point cloud',
                       point_size=14)
    plotter.add_points(np.array(registrar1_supine_landmarks.landmarks[vl_ids[0]]), color='red',
                       render_points_as_spheres=True,
                       label='Point cloud', point_size=12)
    # plotter.add_points(skin_points, color='#F5CBA7', render_points_as_spheres=True, opacity=0.2,
    #                    label='Point cloud', point_size=5)
    plotter.add_points(np.array(closest_points_skin[vl_ids[0]]), color='green', render_points_as_spheres=True,
                       label='Point cloud', point_size=10)
    plotter.add_points(np.array(closest_points_rib[vl_ids[0]]), color='blue', render_points_as_spheres=True,
                       label='Point cloud', point_size=10)
    # Draw lines between corresponding points in landmarks and skin
    for i in range(len(registrar1_supine_landmarks.landmarks[vl_ids[0]])):
        line = pv.Line(registrar1_supine_landmarks.landmarks[vl_ids[0]][i], np.array(closest_points_skin[vl_ids[0]])[i])
        plotter.add_mesh(line, color="yellow", line_width=3, label='Line')
    # Draw lines between corresponding points in landmarks and rib cage
    for i in range(len(registrar1_supine_landmarks.landmarks[vl_ids[0]])):
        line = pv.Line(registrar1_supine_landmarks.landmarks[vl_ids[0]][i], np.array(closest_points_rib[vl_ids[0]])[i])
        plotter.add_mesh(line, color="magenta", line_width=3, label='Line')
    #
    for i in range(len(registrar1_supine_landmarks.landmarks[vl_ids[0]])):
        if abs(left_nipple[0] - registrar1_supine_landmarks.landmarks[vl_ids[0]][i][0]) < abs(
                right_nipple[0] - registrar1_supine_landmarks.landmarks[vl_ids[0]][i][0]):
            line = pv.Line(registrar1_supine_landmarks.landmarks[vl_ids[0]][i], left_nipple)
        else:
            line = pv.Line(registrar1_supine_landmarks.landmarks[vl_ids[0]][i], right_nipple)
        plotter.add_mesh(line, color="cyan", line_width=3, label='Line')
    # legend_entries = [['Images', 'grey'], ['Segmentation mask', 'lightskyblue'], ['Left_nipple', 'pink'],
    #                   ['Right_nipple', 'pink'], ['Skin points', '#F5CBA7'], ['Closest skin points', 'green'],
    #                   ['Closest rib cage points', 'blue'], ['Distance between landmarks and skin', 'yellow'],
    #                   ['Distance between landmarks and rib cage', 'magenta'],
    #                   ['Distance between landmarks and nipple', 'cyan']]
    legend_entries = [['Images', 'grey'], ['Left_nipple', 'pink'],
                      ['Right_nipple', 'pink'], ['Closest skin points', 'green'],
                      ['Closest rib cage points', 'blue'], ['Distance between landmarks and skin', 'yellow'],
                      ['Distance between landmarks and rib cage', 'magenta'],
                      ['Distance between landmarks and nipple', 'cyan']]
    plotter.add_legend(legend_entries, bcolor='w')
    plotter.add_text(f"VL{vl_id:05d}", position='upper_left', font_size=10, color='black')
    labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
                  ylabel=f"{orientation_flag[1]} (mm)",
                  zlabel=f"{orientation_flag[2]} (mm)")
    plotter.add_axes(**labels)
    plotter.view_xy()

    plotter.show()
'''