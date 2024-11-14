#Standard library imports

from morphic import utils
from morphic import viewer
import bmw
# #
from tools import mask_sternum
from tools import subjectModel
from tools import sitkTools
from tools import icp

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import pandas as pd



class align_cw:
# compute the trasformation between source and target models
# target_model must include MRI and metadata structure with sternum positions
# if any evaluation ladmarks are given the error will be computed after realignement
    def __init__(self, target_model, source_model, target_mask_path, mask_width,
                 evaluation_landmarks_source, evaluation_landmarks_target, align_method ='image', debug=False):

        self.rigid_transform = np.zeros(6)
        self.rotation_center = np.zeros(6)
        self.t_model = subjectModel.SubjectModel()
        self.eval_points_error = np.zeros(2)
        self.registration_points_error = np.zeros(2)
        self.align_method = align_method
        self.vl_id = target_model.vl_id
        self.transform_path = 'C:\\Users\\kkho985\\Documents\\Masters\\Code\\female_torso\\OpenSim\\data\\sternum_transforms'

        skip = False
        skip_eval = False
        if not target_model.image_path:
                    skip = True
                    print('Subject {0} taget image is missing'.format(self.vl_id))
        if not source_model.image_path:
                    skip = True
                    print('Subject {0} source image is missing'.format(self.vl_id))
        if len(source_model.metadata.sternal_angle)==0 or len(source_model.metadata.jugular_notch)==0:
                    skip = True
                    print('Subject {0} source sternal landmarks are missing'.format(self.vl_id))
        if len(target_model.metadata.sternal_angle)==0 or len(target_model.metadata.jugular_notch)==0:
                    skip = True
                    print('Subject {0} target sternal landmarks are missing'.format(self.vl_id))


        if not  os.path.exists(target_mask_path):
            skip =True
            print('Subject {0} mask image is missing'.format(self.vl_id))


        if not skip:
            if  len(evaluation_landmarks_source)==0:
                skip_eval = True
                print('Subject {0} evaluation step is missing'.format(self.vl_id))
            if len(evaluation_landmarks_target) == 0:
                skip_eval = True
                print('Subject {0} evaluation step is missing'.format(self.vl_id))

        if not skip:
            self._apply_registration(target_model, source_model, target_mask_path, mask_width,
                                     evaluation_landmarks_source, evaluation_landmarks_target, not (skip_eval), debug)


    def _apply_registration(self,prone_model, supine_model, prone_cw_mask_path,mask_width,
                            evaluation_landmarks_prone, evaluation_landmarks_supine,
                            eval= False,debug =False):


        t_evaluation_landmark_supine = {}

        jugular_landmarks_prone = np.reshape([prone_model.metadata.sternal_angle,
                                              prone_model.metadata.jugular_notch],(2,3))

        jugular_landmarks_supine = np.reshape([supine_model.metadata.sternal_angle,
                                               supine_model.metadata.jugular_notch],(2,3))



        # ======================================================================================================
        # Align supine mesh to prone
        # ======================================================================================================
        if supine_model.vl_id == prone_model.vl_id:
            vl_id_str = 'VL{0:05d}'.format( prone_model.vl_id)
            print('Chest wall realignment {0}'.format(vl_id_str))

            if self.align_method == 'image':
                #######
                # Original Anna transform from mask
                t_supine_image = self.alignLandmarksAndImageRigiTransform(supine_model.get_scan(),
                                                            prone_model.get_scan(), prone_cw_mask_path,mask_width,
                                                            jugular_landmarks_supine, jugular_landmarks_prone,
                                                            prone_model.vl_id, debug=debug)

                # # Input transform from OpenSim model
                # alignment_transforms = pd.read_csv(os.path.join(self.transform_path, 'VL{0:05d}_t1_rigid_transform.csv'.format(self.vl_id)))
                # self.rigid_transform = alignment_transforms[:6].to_numpy().T.flatten()
                # self.rotation_center = alignment_transforms[6:].to_numpy().T.flatten()
                #######

                if debug:
                    file_name = 'realigned_supine_to_prone_{}.nii'.format(vl_id_str)
                    self.t_model.image_path = file_name
                    file_name = os.path.join('X:\\anna_data\\cw_realignment\\realigned_images',
                                             file_name)
                    t_supine_image.setRafOrientation()
                    sitkTools.writeNIFTIImage(t_supine_image, file_name)
                    t_supine_image.setAlfOrientation()



                # input =  self.rotation_center, self.rigid_transform
                t_jugular_landmark_supine = self.applySITK3DTransformToPoints(jugular_landmarks_supine)

                err_jugular, std_jugular = icp.computeLandmarkBasedError(t_jugular_landmark_supine,jugular_landmarks_prone)
                self.registration_points_error[0] = err_jugular
                self.registration_points_error[1] = std_jugular

                print ('Registration points mean error: {0}'.format(self.registration_points_error[0]))
                print ('Registration points std: {0}\n'.format( self.registration_points_error[1]))

                if supine_model.cw_surface_mesh:
                    supine_cwm_nodes = supine_model.cw_surface_mesh.get_node_ids()[0]
                    t_supine_nodes = self.applySITK3DTransformToPoints(supine_cwm_nodes)



                if eval:

                    t_evaluation_landmark_supine = self.applySITK3DTransformToPoints(evaluation_landmarks_supine)

                    self.eval_points_error[0], self.eval_points_error[1] = icp.computeLandmarkBasedError(t_evaluation_landmark_supine,
                                                                          evaluation_landmarks_prone)

                    print ('Evaluation points mean error: {0}'.format(self.eval_points_error[0]))
                    print ('Evaluation points std: {0}\n'.format(self.eval_points_error[1]))


        # =======================================================================================
        # Rigid transformation of the mesh
        # ======================================================================================

        # Align supine mesh with the prone mesh
        t_supine_model = subjectModel.SubjectModel(supine_model.mesh_path, supine_model.image_path,
                                                   supine_model.position, supine_model.vl_id)

        if os.path.exists(supine_model.mesh_path):
            t_supine_cwm = bmw.load_chest_wall_surface_mesh(supine_model.mesh_path, supine_model.position)
            t_supine_cwm = utils.convert_hermite_lagrange(t_supine_cwm, tol=1e-9)

            for node_id in range(1, len(t_supine_model.cw_surface_mesh.get_node_ids()[1]) + 1):
                t_supine_cwm.nodes[node_id].values = t_supine_nodes[node_id - 1, :]

            t_supine_model.cw_surface_mesh = t_supine_cwm


        t_supine_model.sternum_landmarks = t_jugular_landmark_supine
        t_supine_model.rigid_landmarks = t_evaluation_landmark_supine


        self.t_model = t_supine_model

        if debug:
            # supine_mask = mask_sternum.compute_sternum_mask(supine_model.get_scan(), jugular_landmarks_supine)
            # supine_sternum_image = supine_model.get_scan().copy()
            # supine_sternum_image.values = np.ma.array(data=supine_model.get_scan().values,
            #                                           mask=np.logical_not(supine_mask.values),
            #                                           fill_value=0).filled()
            #
            # # supine_sternum_fig = bmw.add_fig(viewer, label='supine_sternum image_{0}'.format(vl_id_str))
            # # bmw.view_mri(None, supine_sternum_fig, image=supine_sternum_image, axes='z_axes')

            alignment_fig=  bmw.add_fig(viewer, label='aligned_meshes')
            try:
                bmw.visualise_mesh(prone_model.cw_surface_mesh, alignment_fig, visualise=True, face_colours=(0, 0, 1), opacity=0.5)
            except:
                pass
            try:
                bmw.visualise_mesh(t_supine_model.cw_surface_mesh, alignment_fig, visualise=True, face_colours=(1, 0, 0), opacity=0.5,label = 'lab')
            except:
                pass
            bmw.plot_points(alignment_fig, 'jugular_landmark_prone',
                                    jugular_landmarks_prone, [1, 2],
                                    visualise=True, colours=(1, 0, 0),
                                    point_size=3, text_size=5)
            bmw.plot_points(alignment_fig, 't_jugular_landmark_supine',
                                    t_jugular_landmark_supine, [1, 2],
                                    visualise=True, colours=(0, 0, 1),
                                    point_size=3, text_size=5)
            try:
                bmw.plot_points(alignment_fig, 'evaluation_points_prone',
                                evaluation_landmarks_prone, [1, 2, 3, 4, 5, 6, 7, 8],
                                visualise=True, colours=(1, 0, 0),
                                point_size=3, text_size=5)
                bmw.plot_points(alignment_fig, 't_evaluation_points_supine',
                                t_evaluation_landmark_supine, [1, 2, 3, 4, 5, 6, 7, 8],
                                visualise=True, colours=(0, 0, 1),
                                 point_size=3, text_size=5)
            except:
                pass


            # bmw.view_mri(None,  alignment_fig, t_supine_image, axes='z_axes')


    def alignLandmarksAndImageRigiTransform(self,source_image, target_image, target_mask_path,mask_width, source_landmark,
                                             target_landmark,vl_id, debug=False):

        sitk_source_image = sitkTools.scanToSITK(source_image)
        sitk_target_image = sitkTools.scanToSITK(target_image)

        target_mask = mask_sternum.compute_sternum_mask(target_image,target_landmark,
                                                        target_mask_path, mask_width)

        if debug:
            prone_sternum_image = target_image.copy()
            prone_sternum_image.values = np.ma.array(data=target_image.values,
                                                     mask=np.logical_not(target_mask.values),
                                                     fill_value=0).filled()
            file_name = 'sternum_mask_{}.nii'.format(vl_id)
            target_mask.setRafOrientation()
            file_name = os.path.join('X:\\anna_data\\cw_realignment\\realigned_images',
                                     file_name)
            sitkTools.writeNIFTIImage(target_mask, file_name)
            target_mask.setAlfOrientation()
        sitk_target_mask = sitkTools.scanToSITK(target_mask)

        #
        # prone_sternum_fig = bmw.add_fig(viewer, label='prone_sternum image_{0}'.format(vl_id_str))
        # bmw.view_mri(None, prone_sternum_fig, prone_sternum_image, axes='z_axes')


        r0 = icp.alignCorrespondingLandmarksRigidRotationTranslation(source_landmark, target_landmark)

        sitkInitialTransform = sitk.Euler3DTransform()
        sitkInitialTransform.SetParameters([0,0,0, -r0.x[0], -r0.x[1], -r0.x[2]])

        rotation_center = (target_landmark[0]+target_landmark[1])/2
        rotation_center = np.append(rotation_center, [0])
        sitkInitialTransform.SetFixedParameters(rotation_center)

        sitkResampler = sitk.ResampleImageFilter()
        sitkResampler.SetReferenceImage(sitk_target_image)
        sitkResampler.SetInterpolator(sitk.sitkLinear)


        if debug:
            sitkResampler.SetTransform(sitkInitialTransform)
            sitk_source_resampled = sitk.Cast(sitkResampler.Execute(sitk_source_image), sitk.sitkInt16)

            vl_id_str = 'VL{0:05d}'.format(vl_id)
            sitkTools.display_images_with_alpha('after_landmarks_y_{0}'.format(vl_id_str), 'y', 100,
                                            sitk_target_image, sitk_source_resampled)
            sitkTools.display_images_with_alpha('after_landmarks_x_{0}'.format(vl_id_str), 'x', 100,
                                            sitk_target_image, sitk_source_resampled)


        imageRegistrationMethodstep1 = sitk.ImageRegistrationMethod()

        imageRegistrationMethodstep1.SetMetricAsCorrelation()
        imageRegistrationMethodstep1.SetMetricFixedMask(sitk.Cast(sitk_target_mask, sitk.sitkUInt8))
        imageRegistrationMethodstep1.SetInterpolator(sitk.sitkLinear)

        imageRegistrationMethodstep1.SetInterpolator(sitk.sitkLinear)
        imageRegistrationMethodstep1.SetShrinkFactorsPerLevel([2,1])
        imageRegistrationMethodstep1.SetSmoothingSigmasPerLevel([0.1,0.001])

        imageRegistrationMethodstep1.SetOptimizerAsGradientDescentLineSearch(learningRate=0.01,
                                                                             numberOfIterations=200,
                                                                             convergenceMinimumValue=1e-5,
                                                                             lineSearchLowerLimit=1e-3,
                                                                             lineSearchUpperLimit=1,
                                                                             lineSearchEpsilon=1e-5,
                                                                             estimateLearningRate=imageRegistrationMethodstep1.EachIteration)

        imageRegistrationMethodstep1.SetOptimizerScalesFromPhysicalShift(5)
        imageRegistrationMethodstep1.SetOptimizerWeights([1, 1, 1, 0.2, 0.2, 0.2])
        imageRegistrationMethodstep1.SetInitialTransform(sitkInitialTransform)

        if debug:
            imageRegistrationMethodstep1.AddCommand(sitk.sitkStartEvent, self.start_plot)
            imageRegistrationMethodstep1.AddCommand(sitk.sitkEndEvent, self.end_plot)
            imageRegistrationMethodstep1.AddCommand(sitk.sitkIterationEvent,
                                                    lambda: self.plot_values(imageRegistrationMethodstep1, vl_id_str))
            imageRegistrationMethodstep1.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                                    self.update_multires_iterations)

        optTransform = imageRegistrationMethodstep1.Execute(sitk.Cast(sitk_target_image, sitk.sitkFloat32),
                                                            sitk.Cast(sitk_source_image, sitk.sitkFloat32))

        print('Step 1:')
        print('    Final metric value: {0}'.format(imageRegistrationMethodstep1.GetMetricValue()))
        print('    Optimizer\'s stopping condition {0}\n'.format(
            imageRegistrationMethodstep1.GetOptimizerStopConditionDescription()))


        sitkResampler = sitk.ResampleImageFilter()
        sitkResampler.SetReferenceImage(sitk_target_image)
        sitkResampler.SetInterpolator(sitk.sitkLinear)
        sitkResampler.SetTransform(optTransform)
        sitk_source_resampled = sitk.Cast(sitkResampler.Execute(sitk_source_image), sitk.sitkInt16)

        if debug:
            supine_image = sitkTools.SITKToScan(sitk_source_resampled,target_image.orientation)
            supine_sternum_image = supine_image.copy()
            supine_sternum_image.values = np.ma.array(data=supine_image.values,
                                                      mask=np.logical_not(target_mask.values),
                                                      fill_value=0).filled()
            sitk_supine_sternum_image = sitkTools.scanToSITK(supine_sternum_image)

            target_sternum_image = target_image.copy()
            target_sternum_image.values = np.ma.array(data=target_image.values,
                                                      mask=np.logical_not(target_mask.values),
                                                      fill_value=0).filled()
            sitk_target_sternum_image = sitkTools.scanToSITK(target_sternum_image)

            sitkTools.display_images_with_alpha('final_y_axis{0}'.format(vl_id_str), 'y', 100,
                                            sitk_target_sternum_image, sitk_supine_sternum_image)
            sitkTools.display_images_with_alpha('final_x_axis{0}'.format(vl_id_str), 'x', 100,
                                            sitk_target_sternum_image, sitk_supine_sternum_image)


        self.rotation_center = np.array( [rotation_center[0], rotation_center[1], rotation_center[2], 0, 0, 0])

        self.rigid_transform = np.array( [-optTransform.GetParameters()[3], -optTransform.GetParameters()[4], -optTransform.GetParameters()[5],
                -optTransform.GetParameters()[0], -optTransform.GetParameters()[1], -optTransform.GetParameters()[2]])

        outputScanImage = sitkTools.SITKToScan(sitk_source_resampled,target_image.orientation)
        return outputScanImage

    # Callback invoked when the StartEvent happens, sets up our new data.
    @staticmethod
    def start_plot():
        global metric_values, multires_iterations, learning_rate, opt_parameters

        metric_values = []
        multires_iterations = []
        learning_rate = []
        opt_parameters = []

    # Callback invoked when the EndEvent happens, do cleanup of data and figure.
    @staticmethod
    def end_plot():
        global metric_values, multires_iterations, learning_rate, opt_parameters

        del metric_values
        del multires_iterations
        del learning_rate
        del opt_parameters
        # Close figure, we don't want to get a duplicate of the plot latter on.
        plt.close()

    # Callback invoked when the IterationEvent happens, update data and display new figure.
    @staticmethod
    def plot_values(registration_method, vl_id_str):
        global metric_values, multires_iterations, learning_rate, opt_parameters

        metric_values.append(registration_method.GetMetricValue())
        learning_rate.append(registration_method.GetOptimizerLearningRate())
        opt_parameters.append(registration_method.GetInitialTransform().GetParameters())

        print ('iteration number:{0}'.format(registration_method.GetOptimizerIteration()))
        print(registration_method.GetInitialTransform().GetParameters())

        # Clear the output area (wait=True, to reduce flickering), and plot current data
        # clear_output(wait=True)
        # Plot the similarity metric values

        # the path shoul be changed to rlative paths
        plt.figure('Metric_{0}'.format(vl_id_str))
        plt.plot(metric_values, 'r')
        plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.savefig('/hpc/amir309/data/optimiser_metric_{0}.png'.format(vl_id_str))
        # plt.show()
        plt.close()

        plt.figure('Learning rate {0}'.format(vl_id_str))
        plt.plot(learning_rate, 'g')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Learning rate', fontsize=12)
        plt.savefig('/hpc/amir309/data/optimiser_learning_rate_{0}.png'.format(vl_id_str))
        plt.close()

        plt.figure('Translation {0}'.format(vl_id_str))
        plt.plot([opt_parameters[i][3] for i in range(len(opt_parameters))], 'r', label='X Translation')
        plt.plot([opt_parameters[i][4] for i in range(len(opt_parameters))], 'g', label='Y Translation')
        plt.plot([opt_parameters[i][5] for i in range(len(opt_parameters))], 'b', label='Z Translation')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Translation', fontsize=12)
        plt.savefig('/hpc/amir309/data/optimiser_translation_{0}.png'.format(vl_id_str))
        plt.close()

        plt.figure('Rotation {0}'.format(vl_id_str))
        plt.plot([opt_parameters[i][0] for i in range(len(opt_parameters))], 'r', label='X Rotation')
        plt.plot([opt_parameters[i][1] for i in range(len(opt_parameters))], 'g', label='Y Rotation')
        plt.plot([opt_parameters[i][2] for i in range(len(opt_parameters))], 'b', label='Z Rotation')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Rotation', fontsize=12)
        plt.savefig('/hpc/amir309/data/optimiser_Rotation_{0}.png'.format(vl_id_str))
        plt.close()

    # Callback invoked when the sitkMultiResolutionIterationEvent occurs, update the index into the
    # metric_values list.
    @staticmethod
    def update_multires_iterations():
        global metric_values, multires_iterations
        multires_iterations.append(len(metric_values))

    def applySITK3DTransformToPoints(self, points):


        rotation = [0, 0, 0, self.rigid_transform[3], self.rigid_transform[4], self.rigid_transform[5]]
        translation = [self.rigid_transform[0], self.rigid_transform[1], self.rigid_transform[2], 0, 0, 0]

        t_points = icp.transformRigid3D(points, -1*np.array(self.rotation_center))
        t_points = icp.transformRigid3D(t_points,scipy.array(translation))
        t_points = icp.transformRigid3D(t_points,scipy.array(rotation))
        t_points = icp.transformRigid3D(t_points, scipy.array(self.rotation_center))

        return t_points



def alignModels(target_models, source_models,prone_cw_masks_path, target_eval_points, source_eval_points, method='image',debug =False):

    aligned_model ={}
    skip = {}
    skip_eval = {}

    for sub_id in target_models:
        sub_target_model = {}
        sub_source_model = {}
        vl_id_str = 'VL{0:05d}'.format(sub_id)

        skip[sub_id] = False
            # Load evaluation landmarks

        try:
            sub_target_model = target_models[sub_id]
            sub_source_model = source_models[sub_id]
            target_mask_path = os.path.join(prone_cw_masks_path,sub_target_model.position, 'rib_cage\\rib_cage_{0}.nii'.format(vl_id_str))
            if not sub_target_model.image_path:
                    skip[sub_id] = True
                    print('Subject {0} taget image is missing'.format(sub_id))
            if not sub_source_model.image_path:
                    skip[sub_id] = True
                    print('Subject {0} source image is missing'.format(sub_id))
        except:
            skip[sub_id] = True

        if not os.path.exists(target_mask_path):
            skip[sub_id] =True
            print('Subject {0} mask image is missing'.format(sub_id))


        sub_target_eval_points = {}
        sub_source_eval_points = {}
        if not skip[sub_id]:
            try:
                sub_target_eval_points = np.array(target_eval_points.landmarks[sub_id])

                sub_source_eval_points = np.array(source_eval_points.landmarks[sub_id])
            except:
                pass

        if not skip[sub_id]:
            aligned_model[sub_id] = align_cw(sub_target_model, sub_source_model,target_mask_path, 15,
                                             sub_target_eval_points, sub_source_eval_points, method, debug=debug)

    return aligned_model


def applyTransformToModelsLandmarks(t_models, landmarks):

    t_landmarks = landmarks.copy()
    for vl_id in landmarks.landmarks:
            if vl_id in t_models:
                model = t_models[vl_id]
                sub_landmarks = np.array(landmarks.landmarks[vl_id])
                if len(sub_landmarks.shape) == 1:
                    sub_landmarks =sub_landmarks.reshape((1,3))
                if t_models[vl_id].align_method == 'image':
                    t_landmarks.landmarks[vl_id] = model.applySITK3DTransformToPoints(sub_landmarks)
                else:
                    t_landmarks.landmarks[vl_id] = icp.transformRigid3D(sub_landmarks,model.rigid_transform)

    return t_landmarks



