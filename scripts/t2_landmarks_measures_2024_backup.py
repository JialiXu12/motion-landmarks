import os
import numpy as np
import automesh
from tools import landmarks_old as ld
from tools import realignment_tools
from tools import subjectModel
import h5py
import breast_metadata
import pyvista as pv

if __name__ == '__main__':
    # Define dictionary in which the transformed jugular_landmarks are stored
    # vl_ids = [71]
    vl_ids = [9,11,12]

    # root_path_MRI = '/hpc/psam012/opt/breast-modelling/data/images/vl/mri_t2'
    # root_path_MRI = r'U:\projects\volunteer_camri\old_data\mri_t2'
    # root_path_MRI = r'U:\sandbox\jxu759\motion_of_landmarks'
    root_path_MRI = r'U:\projects\volunteer_camri\old_data\mri_t2'

    # soft_landmarks_path = '/hpc_ntot/psam012/opt/breast-modelling/data/T2-landmark-analysis-study/picker/points'
    # rigid_landmarks_path = '/hpc/amir309/data/picker/points_wc_raf'
    soft_landmarks_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\T2-landmark-analysis-study\picker\points'
    rigid_landmarks_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\picker\points_wc_raf'

    # root_path = 'X:\\prasad_data\\T2-landmark-analysis-study\\'
    # root_path = '/hpc_ntot/psam012/opt/breast-modelling/data/T2-landmark-analysis-study/'
    # root_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\T2-landmark-analysis-study'

    # segmented body and ribcage masks
    mask_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\automatic_segmentation_CNN\T2_from_T1'

    # evaluation_points_path = '/hpc_atog/amir309/'
    # prone_data_path = '/hpc_ntot/psam012/opt/breast-modelling/data/prone_to_supine_t2/2017_09_06/'
    # supine_data_path = '/hpc_ntot/psam012/opt/breast-modelling/data/supine_to_prone_t2/2017_06_25/'
    # prone_data_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\prone_to_supine_t2\2017_09_06'
    # supine_data_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\supine_to_prone_t2\2017_06_25'

    # prone_model_path = '/hpc_ntot/psam012/opt/breast-modelling/data/prone_to_supine_t2/2017_09_06/'
    # supine_model_path = '/hpc_ntot/psam012/opt/breast-modelling/data/supine_to_prone_t2/2017-07-31/'

    # skin mesh and ribcage mesh ( and segmented data points (lung, skin, nipple))
    prone_model_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\prone_to_supine_t2\2017_09_06'
    supine_model_path = r'U:\sandbox\jxu759\motion_of_landmarks\prasad_data\supine_to_prone_t2\2017-07-31'

    '''
    'soft_landmarks_path': contain all volunteer breast tissue landmarks (e.g., cyst, lymph node) 
                           identified by the registrars in the image coordinate system
    'rigid_landmarks_path': contain all volunteer anatomical landmarks - nipple (user002) and sternum (user001) 
                           identified by the registrars in the image coordinate system
                           
    'registrar1_prone': PRONE breast tissue landmarks (e.g., cyst, lymph node) identified by the registrar 1 
    'registrar2_prone': PRONE anatomical landmarks - nipple (user002) and sternum (user001) identified by the registrar 1
    
    Read
    1) breast tissue landmarks identified by registrars (soft landmarks)
    2) 'prone_metadata': metadata for each volunteer (anatomical landmarks: nipple and sternum landmarks (rigid landmarks))
    3)
    
    The metadata includes the age, height, weight, nipple landmark position, and sternum landmark position 
    in the MRI coordinate. The metadata is used to transform the landmarks from the image coordinate system to the model
    coordinate system. The model coordinate system is defined by the skin mesh and the tissue landmarks.
    
    'registrar1_prone_landmarks': transformed breast tissue landmarks in the model coordinate system (from RAF 
                                  image coordinate system).
        '''

    position = 'prone'
    # user002 - nipple landmarks, user001 - sternum landmarks
    nipple_path = os.path.join(rigid_landmarks_path, 'user002')
    sternum_path = os.path.join(rigid_landmarks_path, 'user001')

    # output_dir = 'X:\\anna_opt\\breast_dev\\output'
    output_dir = r'C:\Users\jxu759\Documents\Motion_of_landmarks\output'

    # read landmarks in image coordinate system
    # User008 = Ben, User007 = Clarke
    registrar1_prone = ld.Landmarks('user008', vl_ids, position, soft_landmarks_path)
    registrar2_prone = ld.Landmarks('user007', vl_ids, position, soft_landmarks_path)

    # transform landmarks position in the model coordinates system
    prone_metadata = ld.read_metadata(vl_ids, position, root_path_MRI, nipple_path, sternum_path)
    registrar1_prone_landmarks = registrar1_prone.getModellandmarks(prone_metadata)
    registrar2_prone_landmarks = registrar2_prone.getModellandmarks(prone_metadata)

    # find the corresponding quadrant for each soft tissue landmark
    registrar1_prone_landmarks.find_quadrants(prone_metadata)
    registrar2_prone_landmarks.find_quadrants(prone_metadata)

    # ''' plot MR images, landmarks and jugular notch'''
    # # Load scan
    # path_MRI = os.path.join(root_path_MRI, 'VL{0:05d}'.format(vl_ids[0]), 'prone')
    # scan = breast_metadata.Scan(path_MRI)
    #
    # orientation_flag = "RAI"
    #
    # # Convert Scan to Pyvista Image Grid in desired orientation
    # image_grid = breast_metadata.SCANToPyvistaImageGrid(scan,
    #                                                     orientation_flag)
    #
    # # add landmarks
    # point = registrar1_prone.landmarks[vl_ids[0]][1]
    # #point3 = np.vstack([registrar1_prone_landmarks.landmarks[vl_ids[0]][1], prone_metadata[12].jugular_notch])
    #
    # # Create a point cloud with the single point
    # point_cloud = pv.PolyData(point)
    #
    # plotter = pv.Plotter()
    # plotter.add_volume(image_grid,
    #                    opacity='sigmoid_6',
    #                    cmap='coolwarm',
    #                    show_scalar_bar=False)
    #
    # plotter.add_points(point_cloud, color='blue', point_size=20.0, render_points_as_spheres=True)
    #
    # plotter.add_text(f"Image and landmarks in {orientation_flag} Coordinates")
    # labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
    #               ylabel=f"{orientation_flag[1]} (mm)",
    #               zlabel=f"{orientation_flag[2]} (mm)")
    #
    # plotter.show_grid(**labels)
    # plotter.add_axes(**labels)
    # plotter.show()

    # supine
    position = 'supine'
    registrar1_supine = ld.Landmarks('user008', vl_ids, position, soft_landmarks_path)
    registrar2_supine = ld.Landmarks('user007', vl_ids, position, soft_landmarks_path)
    supine_metadata = ld.read_metadata(vl_ids, position, root_path_MRI, nipple_path, sternum_path)
    registrar1_supine_landmarks = registrar1_supine.getModellandmarks(supine_metadata)
    registrar2_supine_landmarks = registrar2_supine.getModellandmarks(supine_metadata)
    # jugular_landmarks_supine = ld.Landmarks('user001', vl_ids,
    #                                         position, rigid_landmarks_path)
    # sternum_land_sm = jugular_landmarks_supine.getModellandmarks(metadata_file)

    registrar1_supine_landmarks.find_quadrants(supine_metadata)
    registrar2_supine_landmarks.find_quadrants(supine_metadata)


    # Corresponding landmarks between registrars
    # gives a dictionary and per volunteer the corresponding local landmark numbers are giving in a list.
    # The first number on cor_1_2 reffers to the the local landmark number as identified by the first registrar and the second by
    # by the second registrar.
    cor_1_2 = ld.corresponding_landmarks_between_registars(
        registrar1_prone_landmarks, registrar2_prone_landmarks,
        registrar1_supine_landmarks, registrar2_supine_landmarks)



    # Closest distances from landmark to skin and chest wall
    registrar1_dist_output = os.path.join(output_dir, 'registrar_1_t2')
    if not os.path.exists(registrar1_dist_output):
        os.mkdir(registrar1_dist_output)
    registrar2_dist_output = os.path.join(output_dir, 'registrar_2_t2')
    if not os.path.exists(registrar2_dist_output):
        os.mkdir(registrar2_dist_output)

    dist_landmark_skin_p1,points_skin_p1, \
    dist_landmark_cw_p1, points_cw_p1 = ld.shortest_distances(prone_metadata, mask_path,
                                                                       registrar1_prone_landmarks,
                                                                       registrar1_dist_output)

    dist_landmark_skin_p2, points_skin_p2, \
    dist_landmark_cw_p2, points_cw_p2 = ld.shortest_distances(prone_metadata, mask_path,
                                                                       registrar2_prone_landmarks,
                                                                       registrar2_dist_output)
    dist_landmark_skin_s1, points_skin_s1, \
    dist_landmark_cw_s1, points_cw_s1 = ld.shortest_distances(supine_metadata, mask_path,
                                                                       registrar1_supine_landmarks,
                                                                       registrar1_dist_output)
    dist_landmark_skin_s2, points_skin_s2, \
    dist_landmark_cw_s2, points_cw_s2 = ld.shortest_distances(supine_metadata, mask_path,
                                                                        registrar2_supine_landmarks,
                                                                       registrar2_dist_output)
    # Landmark positions in time coordinates, distance to the nipple, and volunteer numbers

    time_1p, dist_landmark_nipple_1p = ld.clock(registrar1_prone_landmarks, prone_metadata)
    time_2p, dist_landmark_nipple_2p = ld.clock(registrar2_prone_landmarks, prone_metadata)
    time_1s, dist_landmark_nipple_1s = ld.clock(registrar1_supine_landmarks, supine_metadata)
    time_2s, dist_landmark_nipple_2s = ld.clock(registrar2_supine_landmarks, supine_metadata)

    ld_indices = cor_1_2[vl_ids[0]][0]
    ld_indices_r1 = ld_indices[0]
    ld_indices_r2 = ld_indices[1]
    dist_skin = dist_landmark_skin_p1[vl_ids[0]]

    # For testing only
    print("Shortest distance between landmark and skin in prone by registrar 1: ", dist_landmark_skin_p1[vl_ids[0]][ld_indices_r1])
    print("Shortest distance between landmark and chest wall in prone by registrar 1: ", dist_landmark_cw_p1[vl_ids[0]][ld_indices_r1])
    print("Shortest distance between landmark and nipple in prone by registrar 1: ", dist_landmark_nipple_1p[vl_ids[0]][ld_indices_r1])

    print("Shortest distance between landmark and skin in supine by registrar 1: ", dist_landmark_skin_s1[vl_ids[0]][ld_indices_r1])
    print("Shortest distance between landmark and chest wall in supine by registrar 1: ", dist_landmark_cw_s1[vl_ids[0]][ld_indices_r1])
    print("Shortest distance between landmark and nipple in supine by registrar 1: ", dist_landmark_nipple_1s[vl_ids[0]][ld_indices_r1])

    print("Shortest distance between landmark and skin in prone by registrar 2: ", dist_landmark_skin_p2[vl_ids[0]][ld_indices_r2])
    print("Shortest distance between landmark and chest wall in prone by registrar 2: ", dist_landmark_cw_p2[vl_ids[0]][ld_indices_r2])
    print("Shortest distance between landmark and nipple in prone by registrar 2: ", dist_landmark_nipple_2p[vl_ids[0]][ld_indices_r2])

    print("Shortest distance between landmark and skin in supine by registrar 2: ", dist_landmark_skin_s2[vl_ids[0]][ld_indices_r2])
    print("Shortest distance between landmark and chest wall in supine by registrar 2: ", dist_landmark_cw_s2[vl_ids[0]][ld_indices_r2])
    print("Shortest distance between landmark and nipple in supine by registrar 2: ", dist_landmark_nipple_2s[vl_ids[0]][ld_indices_r2])

    print("Quadrants of landmark(s) in prone by registrar 1: ", registrar1_prone_landmarks.quadrants[vl_ids[0]][ld_indices_r1])
    print("Quadrants of landmark(s) in supine by registrar 1: ", registrar1_supine_landmarks.quadrants[vl_ids[0]][ld_indices_r1])
    print("Quadrants of landmark(s) in prone by registrar 2: ", registrar2_prone_landmarks.quadrants[vl_ids[0]][ld_indices_r2])
    print("Quadrants of landmark(s) in supine by registrar 2: ", registrar2_supine_landmarks.quadrants[vl_ids[0]][ld_indices_r2])

    print("Clock face of landmark(s) in prone by registrar 1: ", time_1p[vl_ids[0]][ld_indices_r1])
    print("Clock face of landmark(s) in supine by registrar 1: ", time_1s[vl_ids[0]][ld_indices_r1])
    print("Clock face of landmark(s) in prone by registrar 2: ", time_2p[vl_ids[0]][ld_indices_r2])
    print("Clock face of landmark(s) in supine by registrar 2: ", time_2s[vl_ids[0]][ld_indices_r2])

    # ############################prone plot############################################
    # from mayavi import mlab
    # import bmw
    # from morphic import viewer
    #
    # vl_image_path = os.path.join(root_path_MRI, 'VL{0:05d}'.format(vl_ids[0]), 'prone')
    # prone_image = automesh.Scan(vl_image_path)
    # points_indx = [0]#list(range(len(registrar1_prone_landmarks.landmarks[vl_ids[0]])))
    # fig = bmw.add_fig(viewer, label='DISTANCE PLOT')
    # bmw.view_mri(None, fig, prone_image, axes='z_axes')
    # fig.plot_points('skin', [points_skin_p1[vl_ids[0]][x] for x in points_indx],
    #                 color=(1, 1, 0), size=3)
    # fig.plot_points('landmark',[registrar1_prone_landmarks.landmarks[vl_ids[0]][x] for x in points_indx],
    #                 color=(1, 0, 0), size=5)
    # fig.plot_points('cw', [points_cw_p1[vl_ids[0]][x] for x in points_indx],
    #                 color=(1, 0, 1), size=3)
    # fig.plot_points('nipples', [prone_metadata[vl_ids[0]].right_nipple, prone_metadata[vl_ids[0]].left_nipple],
    #                 color=(0, 1, 1), size=3)
    #
    # for indx in points_indx:
    #     line = np.array([points_cw_p1[vl_ids[0]][indx], registrar1_prone_landmarks.landmarks[vl_ids[0]][indx]])
    #     line = np.reshape(line, (2, 3))
    #     fig.plots['line_cw_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
    #                                                       color=(1, 0, 1), tube_radius=1)
    #     line = np.array([points_skin_p1[vl_ids[0]][indx], registrar1_prone_landmarks.landmarks[vl_ids[0]][indx]])
    #     line = np.reshape(line, (2, 3))
    #     fig.plots['line_skin_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
    #                                                           color=(1, 1, 0), tube_radius=1)
    #     if 'R' in registrar1_prone_landmarks.quadrants[vl_ids[0]][indx]:
    #         nipple_point = prone_metadata[vl_ids[0]].right_nipple
    #         label = 'r_nipple'
    #     else:
    #         nipple_point = prone_metadata[vl_ids[0]].left_nipple
    #         label = 'l_nipple'
    #
    #     line = np.array([nipple_point, registrar1_prone_landmarks.landmarks[vl_ids[0]][indx]])
    #     line = np.reshape(line, (2, 3))
    #     fig.plots[label+'_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
    #                                                           color=(0, 1, 1), tube_radius=1)
    #     fig.show()
    #
    # ############################supine plot############################################
    #
    # vl_image_path = os.path.join(root_path_MRI, 'VL{0:05d}'.format(vl_ids[0]), 'supine')
    # supine_image = automesh.Scan(vl_image_path)
    # points_indx = [0]#list(range(len(registrar1_prone_landmarks.landmarks[vl_ids[0]])))
    # fig_supine = bmw.add_fig(viewer, label='DISTANCE PLOT supine')
    # bmw.view_mri(None, fig_supine, supine_image, axes='z_axes')
    # fig_supine.plot_points('skin', [points_skin_s1[vl_ids[0]][x] for x in points_indx],
    #                 color=(1, 1, 0), size=3)
    # fig_supine.plot_points('landmark',[registrar1_supine_landmarks.landmarks[vl_ids[0]][x] for x in points_indx],
    #                 color=(1, 0, 0), size=5)
    # fig_supine.plot_points('cw', [points_cw_s1[vl_ids[0]][x] for x in points_indx],
    #                 color=(1, 0, 1), size=3)
    # fig_supine.plot_points('nipples', [supine_metadata[vl_ids[0]].right_nipple, supine_metadata[vl_ids[0]].left_nipple],
    #                 color=(0, 1, 1), size=3)
    #
    # for indx in points_indx:
    #     line = np.array([points_cw_s1[vl_ids[0]][indx], registrar1_supine_landmarks.landmarks[vl_ids[0]][indx]])
    #     line = np.reshape(line, (2, 3))
    #     fig_supine.plots['line_cw_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
    #                                                       color=(1, 0, 1), tube_radius=1)
    #     line = np.array([points_skin_s1[vl_ids[0]][indx], registrar1_supine_landmarks.landmarks[vl_ids[0]][indx]])
    #     line = np.reshape(line, (2, 3))
    #     fig_supine.plots['line_skin_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
    #                                                           color=(1, 1, 0), tube_radius=1)
    #     if 'R' in registrar1_supine_landmarks.quadrants[vl_ids[0]][indx]:
    #         nipple_point = supine_metadata[vl_ids[0]].right_nipple
    #         label = 'r_nipple'
    #     else:
    #         nipple_point = supine_metadata[vl_ids[0]].left_nipple
    #         label = 'l_nipple'
    #
    #     line = np.array([nipple_point, registrar1_supine_landmarks.landmarks[vl_ids[0]][indx]])
    #     line = np.reshape(line, (2, 3))
    #     fig_supine.plots[label+'_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
    #                                                           color=(0, 1, 1), tube_radius=1)
    #


    perform_registration = True

    if perform_registration :
        vl_id_str = {}
        prone_models = {}
        supine_models = {}
        prone_mesh_path = os.path.join(prone_model_path, 'volunteer_meshes')
        supine_mesh_path = os.path.join(supine_model_path, 'volunteer_meshes')
        for vl_id in vl_ids:
            vl_id_str = 'VL{0:05d}'.format(vl_id)

            # build the model structure for each volunteer taken as input for chest wall realignment
            prone_models[vl_id] = subjectModel.SubjectModel(prone_mesh_path, image_path=root_path_MRI,
                                                            metadata=prone_metadata[vl_id], position='prone',
                                                            vl_id=vl_id)
            supine_models[vl_id] = subjectModel.SubjectModel(supine_mesh_path, image_path=root_path_MRI,
                                                             metadata=supine_metadata[vl_id], position='supine',
                                                             vl_id=vl_id)

        # chest wall realignment from supine to prone configuration
        t_supine_models = realignment_tools.alignModels(prone_models, supine_models, mask_path,
                                                     {}, {}, method='image')
        #pdb.set_trace()

        # apply realignment to soft landmarks
        t_registrar1_supine = realignment_tools.applyTransformToModelsLandmarks(t_supine_models, registrar1_supine_landmarks)
        t_registrar2_supine = realignment_tools.applyTransformToModelsLandmarks(t_supine_models, registrar2_supine_landmarks)

        # Landmark displacements between prone and supine
        landmarks_displacement_1 = ld.displacement(registrar1_prone_landmarks, t_registrar1_supine)
        landmarks_displacement_2 = ld.displacement(registrar2_prone_landmarks, t_registrar2_supine)

        # nipple position in supine and prone configuration
        prone_nipples = ld.Landmarks('user002', vl_ids, 'prone', rigid_landmarks_path)
        prone_nipples = prone_nipples.getModellandmarks(prone_metadata)
        supine_nipples = ld.Landmarks('user002', vl_ids, 'supine', rigid_landmarks_path)
        supine_nipples = supine_nipples.getModellandmarks(supine_metadata)

        # apply after realignment transform to the nipple position
        t_supine_nipples = realignment_tools.applyTransformToModelsLandmarks(t_supine_models, supine_nipples)

        # compute nipple displacement after chest wall realigement
        nipple_displacement = ld.displacement(prone_nipples,t_supine_nipples)

        # rnipple_displacement = ld.displacement(prone_right_nipple,t_supine_rn)
    #
    # from morphic import viewer
    # import bmw
    #
    # vl_id = vl_ids[0]
    # prone_fig = bmw.add_fig(viewer, label='prone')
    # bmw.view_mri(None, prone_fig, prone_models[vl_id].get_scan(), axes='z_axes')
    # # bmw.plot_points(prone_fig, 'prone',
    # #                 t_registrar1_supine.landmarks[vl_id], [1, 2],
    # #                 visualise=True, colours=(1, 0, 0), point_size=3,
    # #                 text_size=5)
    # bmw.plot_points(prone_fig, 'soft_land',
    #                 registrar1_prone_landmarks.landmarks[vl_id], [1],
    #                 visualise=True, colours=(0, 1, 0), point_size=3,
    #                 text_size=5)
    # bmw.plot_points(prone_fig, 'prone_nipple1',
    #                 prone_metadata[vl_id].left_nipple, [1],
    #                 visualise=True, colours=(1, 1, 0), point_size=3,
    #                 text_size=5)
    # bmw.plot_points(prone_fig, 'prone_nipple',
    #                 prone_metadata[vl_id].right_nipple, [1, 2],
    #                 visualise=True, colours=(1, 1, 0), point_size=3,
    #                 text_size=5)
    # bmw.visualise_mesh(prone_models[vl_id].cw_surface_mesh, prone_fig, visualise=True, face_colours=(0, 1, 0), opacity=0.3)
    # bmw.visualise_mesh(prone_models[vl_id].lskin_surface_mesh, prone_fig, visualise=True, face_colours=(0, 0, 1), opacity=0.3)
    # bmw.visualise_mesh(prone_models[vl_id].rskin_surface_mesh, prone_fig, visualise=True, face_colours=(0, 0, 1), opacity=0.3)


#================================================================
# write data to hdf5file
#===============================================================
    hdfname = os.path.join(output_dir,'landmarks_data_25_07_2024.hdf5')
    results_file = h5py.File(hdfname, 'a')
    registrars  = ['Registrar 1','Registrar 2']
    for vl_id in vl_ids:
        registrar_id = 'Registrar 1'
        #=======================
        # registrar 1
        #========================
        vl_id_str = 'VL{0:05d}'.format(vl_id)

        # create the grou corresponding to the first registrar
        if not registrar_id in results_file:
            registrar = results_file.create_group(registrar_id)
        else:
            registrar = results_file[registrar_id]

        # check if there are any landmarks for the specific volume
        skip = not (vl_id in registrar1_prone_landmarks.landmarks and vl_id in registrar1_supine_landmarks.landmarks)
        if not skip:
            skip = not (len(registrar1_supine_landmarks.landmarks[vl_id]) ==
                        len(registrar1_prone_landmarks.landmarks[vl_id]))

        if not skip:

            size = len(registrar1_supine_landmarks.landmarks[vl_id])

            if vl_id_str in registrar:
                dset = registrar[vl_id_str]
                # del registrar[vl_id_str]
            else:

                dset = registrar.create_group(vl_id_str)

            # write age
            if vl_id in prone_metadata:
                try:
                    data = int(prone_metadata[vl_id].age) * np.ones(size)
                except:
                    data = [b'n/a' for x in range(size)]
                if b'Age' in dset:
                    del dset[b'Age']
                dset.create_dataset(b'Age', data=data)

                # write heights
                try:
                    data = float(prone_metadata[vl_id].height) * np.ones(size)
                except:
                    data = [b'n/a' for x in range(size)]
                if b'Height [m]' in dset:
                    del dset[b'Height [m]']
                dset.create_dataset(b'Height [m]', data=data)

                # write weights
                try:
                    data = float(prone_metadata[vl_id].weight) * np.ones(size)
                except:
                    data = [b'n/a' for x in range(size)]
                if b'Weight [kg]' in dset:
                    del dset[b'Weight [kg]']
                dset.create_dataset(b'Weight [kg]', data=data)

            # write quadrants
            if vl_id in registrar1_prone_landmarks.quadrants and len(registrar1_prone_landmarks.quadrants[vl_id]):
                data = registrar1_prone_landmarks.quadrants[vl_id]
            else:
                data = [b'n/a' for x in range(size)]

            if b'Quadrant (prone)' in dset:
                del dset[b'Quadrant (prone)']
            dset.create_dataset(b'Quadrant (prone)', data=np.array(data,dtype = 'S5'))

            if vl_id in registrar1_supine_landmarks.quadrants and len(registrar1_supine_landmarks.quadrants[vl_id]):
                data = registrar1_supine_landmarks.quadrants[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Quadrant (supine)' in dset:
                del dset[b'Quadrant (supine)']
            dset.create_dataset(b'Quadrant (supine)', data=np.array(data,dtype = 'S5'))

            # write landmarks type
            if vl_id in registrar1_supine_landmarks.landmark_types and len(
                    registrar1_supine_landmarks.landmark_types[vl_id]):
                data = [str(x) for x in registrar1_supine_landmarks.landmark_types[vl_id]]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Landmark type' in dset:
                del dset[b'Landmark type']
            dset.create_dataset(b'Landmark type', data=np.array(data,dtype = 'S5'))

            # write distance to nipple
            if vl_id in dist_landmark_nipple_1p and len(dist_landmark_nipple_1p[vl_id]):
                data = dist_landmark_nipple_1p[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to nipple (prone)[mm]' in dset:
                del dset[b'Distance to nipple (prone)[mm]']
            dset.create_dataset(b'Distance to nipple (prone)[mm]', data=data)

            if vl_id in dist_landmark_nipple_1s and len(dist_landmark_nipple_1s[vl_id]):
                data = dist_landmark_nipple_1s[vl_id]
            else:
                data = [b'n/a' for x in range(size)]

            if b'Distance to nipple (supine)[mm]' in dset:
                del dset[b'Distance to nipple (supine)[mm]']
            dset.create_dataset(b'Distance to nipple (supine)[mm]', data=data)

            # write distance to skin
            if vl_id in dist_landmark_skin_p1 and len(dist_landmark_skin_p1[vl_id]):
                data = dist_landmark_skin_p1[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to skin (prone)[mm]' in dset:
                del dset[b'Distance to skin (prone)[mm]']
            dset.create_dataset(b'Distance to skin (prone)[mm]', data=data)

            if vl_id in dist_landmark_skin_s1 and len(dist_landmark_skin_s1[vl_id]):
                data = dist_landmark_skin_s1[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to skin (supine)[mm]' in dset:
                del dset[b'Distance to skin (supine)[mm]']
            dset.create_dataset(b'Distance to skin (supine)[mm]', data=data)

            # write distance to rib cage
            if vl_id in dist_landmark_cw_p1 and len(dist_landmark_cw_p1[vl_id]):
                data = dist_landmark_cw_p1[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to rib cage (prone)[mm]' in dset:
                del dset[b'Distance to rib cage (prone)[mm]']
            dset.create_dataset(b'Distance to rib cage (prone)[mm]',
                                data=data)

            if vl_id in dist_landmark_cw_s1 and len(dist_landmark_cw_s1[vl_id]):
                data = dist_landmark_cw_s1[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to rib cage (supine)[mm]' in dset:
                del dset[b'Distance to rib cage (supine)[mm]']
            dset.create_dataset(b'Distance to rib cage (supine)[mm]', data=data)

            # write clock phase time
            if vl_id in time_1s and len(time_1s[vl_id]):
                data = time_1s[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Time (supine)' in dset:
                del dset[b'Time (supine)']
            dset.create_dataset(b'Time (supine)', data=np.array(data,dtype = 'S5'))

            if vl_id in time_1p and len(time_1p[vl_id]):
                data = time_1p[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Time (prone)' in dset:
                del dset[b'Time (prone)']
            dset.create_dataset(b'Time (prone)', data=np.array(data,dtype = 'S5'))

            if perform_registration:

                # write nipple displacement from prone to supine after realignment
                if vl_id in nipple_displacement and len(nipple_displacement[vl_id]) == 2:
                    data = nipple_displacement[vl_id][0] * np.ones(size)
                else:
                    data = [b'n/a' for x in range(size)]
                if b'Left nipple displacement [mm]' in dset:
                    del dset[b'Left nipple displacement [mm]']
                dset.create_dataset(b'Left nipple displacement [mm]', data=data)

                if vl_id in nipple_displacement and len(nipple_displacement[vl_id]) == 2:
                    data = nipple_displacement[vl_id][1] * np.ones(size)
                else:
                    data = [b'n/a' for x in range(size)]
                if b'Right nipple displacement [mm]' in dset:
                    del dset[b'Right nipple displacement [mm]']
                dset.create_dataset(b'Right nipple displacement [mm]', data=data)

                # write corresponding landmarks and asociated displacements
                corresponding_data = [b'n/a' for x in range(size)]
                displacement_data = [b'n/a' for x in range(size)]

                if vl_id in cor_1_2:
                    for coresponding_indx in cor_1_2[vl_id]:
                        corresponding_data[coresponding_indx[0]] = coresponding_indx[1]+1
                        displacement_data[coresponding_indx[0]] = landmarks_displacement_1[vl_id][coresponding_indx[0]]
                if b'Corresponding landmarks' in dset:
                    del dset[b'Corresponding landmarks']
                dset.create_dataset(b'Corresponding landmarks', data=corresponding_data)

                if b'Displacements [mm]' in dset:
                    del dset[b'Displacements [mm]']
                dset.create_dataset(b'Displacements [mm]', data=displacement_data)

        registrar_id = 'Registrar 2'
        # =======================
        # registrar 2
        # ========================

        # create the grou corresponding to the first registrar
        if not registrar_id in results_file:
            registrar = results_file.create_group(registrar_id)
        else:
            registrar = results_file[registrar_id]

        # check if there are any landmarks for the specific volume
        skip = not (vl_id in registrar2_prone_landmarks.landmarks and vl_id in registrar2_supine_landmarks.landmarks)
        if not skip:
            skip = not (len(registrar2_supine_landmarks.landmarks[vl_id]) ==
                        len(registrar2_prone_landmarks.landmarks[vl_id]))

        if not skip:

            size = len(registrar2_supine_landmarks.landmarks[vl_id])

            if vl_id_str in registrar:
                dset = registrar[vl_id_str]
                # del registrar[vl_id_str]
            else:

                dset = registrar.create_group(vl_id_str)

            # write age
            if vl_id in prone_metadata:
                try:
                    data = int(prone_metadata[vl_id].age) * np.ones(size)
                except:
                    data = [b'n/a' for x in range(size)]
                if b'Age' in dset:
                    del dset[b'Age']
                dset.create_dataset(b'Age', data=data)

                # write heights
                try:
                    data = float(prone_metadata[vl_id].height) * np.ones(size)
                except:
                    data = [b'n/a' for x in range(size)]
                if b'Height [m]' in dset:
                    del dset[b'Height [m]']
                dset.create_dataset(b'Height [m]', data=data)

                # write weights
                try:
                    data = float(prone_metadata[vl_id].weight) * np.ones(size)
                except:
                    data = [b'n/a' for x in range(size)]
                if b'Weight [kg]' in dset:
                    del dset[b'Weight [kg]']
                dset.create_dataset(b'Weight [kg]', data=data)

            # write quadrants
            if vl_id in registrar2_prone_landmarks.quadrants and len(registrar2_prone_landmarks.quadrants[vl_id]):
                data = registrar2_prone_landmarks.quadrants[vl_id]
            else:
                data = [b'n/a' for x in range(size)]

            if b'Quadrant (prone)' in dset:
                del dset[b'Quadrant (prone)']
            dset.create_dataset(b'Quadrant (prone)', data=np.array(data,dtype = 'S5'))

            if vl_id in registrar2_supine_landmarks.quadrants and len(registrar2_supine_landmarks.quadrants[vl_id]):
                data = registrar2_supine_landmarks.quadrants[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Quadrant (supine)' in dset:
                del dset[b'Quadrant (supine)']
            dset.create_dataset(b'Quadrant (supine)', data=np.array(data,dtype = 'S5'))

            # write landmarks type
            if vl_id in registrar2_supine_landmarks.landmark_types and len(
                    registrar2_supine_landmarks.landmark_types[vl_id]):
                data = [str(x) for x in registrar2_supine_landmarks.landmark_types[vl_id]]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Landmark type' in dset:
                del dset[b'Landmark type']
            dset.create_dataset(b'Landmark type', data=np.array(data,dtype = 'S5'))

            # write distance to nipple
            if vl_id in dist_landmark_nipple_2p and len(dist_landmark_nipple_2p[vl_id]):
                data = dist_landmark_nipple_2p[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to nipple (prone)[mm]' in dset:
                del dset[b'Distance to nipple (prone)[mm]']
            dset.create_dataset(b'Distance to nipple (prone)[mm]', data=data)

            if vl_id in dist_landmark_nipple_2s and len(dist_landmark_nipple_2s[vl_id]):
                data = dist_landmark_nipple_2s[vl_id]
            else:
                data = [b'n/a' for x in range(size)]

            if b'Distance to nipple (supine)[mm]' in dset:
                del dset[b'Distance to nipple (supine)[mm]']
            dset.create_dataset(b'Distance to nipple (supine)[mm]', data=data)

            # write distance to skin
            if vl_id in dist_landmark_skin_p2 and len(dist_landmark_skin_p2[vl_id]):
                data = dist_landmark_skin_p2[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to skin (prone)[mm]' in dset:
                del dset[b'Distance to skin (prone)[mm]']
            dset.create_dataset(b'Distance to skin (prone)[mm]', data=data)

            if vl_id in dist_landmark_skin_s2 and len(dist_landmark_skin_s2[vl_id]):
                data = dist_landmark_skin_s2[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to skin (supine)[mm]' in dset:
                del dset[b'Distance to skin (supine)[mm]']
            dset.create_dataset(b'Distance to skin (supine)[mm]', data=data)

            # write distance to rib cage
            if vl_id in dist_landmark_cw_p2 and len(dist_landmark_cw_p2[vl_id]):
                data = dist_landmark_cw_p2[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to rib cage (prone)[mm]' in dset:
                del dset[b'Distance to rib cage (prone)[mm]']
            dset.create_dataset(b'Distance to rib cage (prone)[mm]',
                                data=data)

            if vl_id in dist_landmark_cw_s2 and len(dist_landmark_cw_s2[vl_id]):
                data = dist_landmark_cw_s2[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Distance to rib cage (supine)[mm]' in dset:
                del dset[b'Distance to rib cage (supine)[mm]']
            dset.create_dataset(b'Distance to rib cage (supine)[mm]', data=data)

            # write clock phase time
            if vl_id in time_2s and len(time_2s[vl_id]):
                data = time_2s[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Time (supine)' in dset:
                del dset[b'Time (supine)']
            dset.create_dataset(b'Time (supine)', data=np.array(data,dtype = 'S5'))

            if vl_id in time_2p and len(time_2p[vl_id]):
                data = time_2p[vl_id]
            else:
                data = [b'n/a' for x in range(size)]
            if b'Time (prone)' in dset:
                del dset[b'Time (prone)']
            dset.create_dataset(b'Time (prone)', data=np.array(data,dtype = 'S5'))

            if perform_registration:

                # write nipple displacement from prone to supine after realignment
                if vl_id in nipple_displacement and len(nipple_displacement[vl_id]) == 2:
                    data = nipple_displacement[vl_id][0] * np.ones(size)
                else:
                    data = [b'n/a' for x in range(size)]
                if b'Left nipple displacement [mm]' in dset:
                    del dset[b'Left nipple displacement [mm]']
                dset.create_dataset(b'Left nipple displacement [mm]', data=data)

                if vl_id in nipple_displacement and len(nipple_displacement[vl_id]) == 2:
                    data = nipple_displacement[vl_id][1] * np.ones(size)
                else:
                    data = [b'n/a' for x in range(size)]
                if b'Right nipple displacement [mm]' in dset:
                    del dset[b'Right nipple displacement [mm]']
                dset.create_dataset(b'Right nipple displacement [mm]', data=data)

                # write corresponding landmarks and asociated displacements
                corresponding_data = [b'n/a' for x in range(size)]
                displacement_data = [b'n/a' for x in range(size)]

                if vl_id in cor_1_2:
                    for coresponding_indx in cor_1_2[vl_id]:
                        corresponding_data[coresponding_indx[1]] = coresponding_indx[0]+1
                        displacement_data[coresponding_indx[1]] = landmarks_displacement_1[vl_id][coresponding_indx[0]]
                if b'Corresponding landmarks' in dset:
                    del dset[b'Corresponding landmarks']
                dset.create_dataset(b'Corresponding landmarks', data=corresponding_data)

                if b'Displacements [mm]' in dset:
                    del dset[b'Displacements [mm]']
                dset.create_dataset(b'Displacements [mm]', data=displacement_data)

    results_file.close()
    print ('finish')
