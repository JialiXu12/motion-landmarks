# Standard library imports
import os
import json
import numpy as np
import math
import pydicom
from scipy.spatial import cKDTree
# Related third party imports
#from bmw import mesh_generation
from bmw.mesh_generation import check_mesh_quality

from tools import sitkTools
import pyvista


class Metadata(object):
    # this class contains all the information about the subject and the MRI images
    # (spacing, orientation, origin) also
    # the position of some landmarks what might be used to define the saubject specific
    # biomechanical model

    # The metadata and specific landmarks are loaded by default
    # in model coordinate system (ALF)

    def __init__(self, vl_id=None, position=None, dicome_path=None, nipple_path=None,
                 sternum_path=None, spine_path=None):
        self.age = None  # subject age, weight and height
        self.weight = None
        self.height = None
        self.image_position = []
        self.image_shape = []
        self.pixel_spacing = []
        self.left_nipple = []
        self.right_nipple = []
        self.jugular_notch = []
        self.sternal_angle = []
        self.spinal_cord = []
        self.vl_id = vl_id
        self.position = position

        vl_id_str = 'VL{0:05d}'.format(vl_id)

        self.metadata_path = os.path.join(dicome_path, vl_id_str, position)

        try:
            self.load_metadata_files(self.metadata_path)
        except:
            print('Subject {0} metadata not loaded '.format(vl_id_str))
            pass

        try:
            vl_niple_path = os.path.join(nipple_path, vl_id_str)
            self.load_nipple_landmarks(vl_niple_path)
        except:
            print('Subject {0} nipple landmarks not found '.format(vl_id_str))
            pass

        try:
            vl_sternum_path = os.path.join(sternum_path, vl_id_str)
            self.load_sternum_landmarks(vl_sternum_path)
        except:
            print('Subject {0} sternum landmarks not found '.format(vl_id_str))
            pass

        try:
            vl_spine_path = os.path.join(spine_path, vl_id_str)
            self.load_spinal_cord(vl_spine_path)
        except:
            print('Subject {0} spinal cord landmarks not found '.format(vl_id_str))
            pass

    def load_metadata_files(self, dicom_path):

        vl_str = 'VL{0:05d}'.format(self.vl_id)
        dicom_file = os.path.join(dicom_path, os.listdir(dicom_path)[0])
        try:
            dicom_image = pydicom.dcmread(dicom_file)
        except pydicom.errors.InvalidDicomError:
            print(f"The file {dicom_file} is not a valid DICOM file.")
            return False

        try:
            self.age = int(dicom_image.PatientAge[0:3])
        except:
            print('Subject {0} age not found ').format(vl_str)
            pass
        try:
            self.weight = float(dicom_image.PatientWeight)
        except:
            print('Subject {0} weight not found ').format(vl_str)
            pass
        try:
            self.height = float(dicom_image.PatientSize)
        except:
            print('Subject {0} height not found ').format(vl_str)
            pass

        try:
            self.pixel_spacing = np.zeros(3)
            spacing = [float(x) for x in dicom_image.PixelSpacing]
            spacing_z = float(dicom_image.SliceThickness)
            self.pixel_spacing = np.concatenate((spacing[::-1], [spacing_z]))
        except:
            print('Subject {0} pixel spacing not found ').format(vl_str)
            pass

        slices_position = []
        for dicom_file in os.listdir(dicom_path):
            try:
                dicom_slice = pydicom.dcmread(os.path.join(dicom_path, dicom_file))
                valid_dicom = True
            except pydicom.errors.InvalidDicomError:
                print(f"The file {dicom_file} is not a valid DICOM file.")
                valid_dicom = False
            if valid_dicom:
                try:
                    slices_position.append([float(v) for v in dicom_slice.ImagePositionPatient])
                except:
                    print('Subject {0} slice location not found ').format(vl_str) + dicom_file
                    pass

        if len(slices_position) > 0:
            slices_position = np.array(slices_position)
            self.image_position = np.array(
                [slices_position.max(0)[1], slices_position.min(0)[0], slices_position.min(0)[2]])

        else:
            print('Subject {0} image location not found').format(vl_str) + dicom_file
            pass

        try:
            self.image_shape = np.zeros(3)
            self.image_shape[1] = int(dicom_image.Rows)
            self.image_shape[0] = int(dicom_image.Columns)
            self.image_shape[2] = int(len(os.listdir(dicom_path)))
        except:
            print('Image shape not found') + dicom_file
            pass

    def load_sternum_landmarks(self, sternum_path):
        file_name = os.path.join(sternum_path, 'point.001.json')
        if os.path.exists(file_name):
            sternum_position = json.loads(open(file_name).read())
            if len(sternum_position):
                self.jugular_notch = np.zeros(3)
                pt = sternum_position[self.position + '_point']['point']
                self.jugular_notch = np.array([pt['y'] - self.image_position[0],
                                               self.pixel_spacing[1] * self.image_shape[1] - pt['x'] +
                                               self.image_position[1],
                                               pt['z'] - self.image_position[2]])

                # MRI 3d location (not image positions)
                # self.jugular_notch = np.array((pt['x'], pt['y'], pt['z']))

        file_name = os.path.join(sternum_path, 'point.002.json')
        if os.path.exists(file_name):
            sternum_position = json.loads(open(file_name).read())
            if len(sternum_position):
                self.sternal_angle = np.zeros(3)
                pt = sternum_position[self.position + '_point']['point']
                self.sternal_angle = np.array([pt['y'] - self.image_position[0],
                                               -pt['x'] + self.image_position[1] +
                                               self.pixel_spacing[1] * self.image_shape[1],
                                               pt['z'] - self.image_position[2]])
                # self.sternal_angle = np.array((pt['x'], pt['y'], pt['z']))

    def load_nipple_landmarks(self, nipple_path):
        if os.path.exists(nipple_path) and len(os.listdir(nipple_path)) == 2:
            # Load in nipple positions
            left_nipple_path = os.path.join(nipple_path, 'point.002.json')
            right_nipple_path = os.path.join(nipple_path, 'point.001.json')
            left_nipple_position = json.loads(open(left_nipple_path).read())
            right_nipple_position = json.loads(open(right_nipple_path).read())

            if len(left_nipple_position):
                self.left_nipple = np.zeros(3)
                pt = left_nipple_position[self.position + '_point']['point']
                self.left_nipple = np.array([pt['y'] - self.image_position[0],
                                             -pt['x'] + self.image_position[1] +
                                             self.pixel_spacing[1] * self.image_shape[1],
                                             pt['z'] - self.image_position[2]])
                # self.left_nipple = np.array((pt['x'], pt['y'], pt['z']))  # position in MRI 3d location
            if len(right_nipple_position):
                self.right_nipple = np.zeros(3)
                pt = right_nipple_position[self.position + '_point']['point']
                self.right_nipple = np.array([pt['y'] - self.image_position[0],
                                              -pt['x'] + self.image_position[1] + self.pixel_spacing[1] *
                                              self.image_shape[1],
                                              pt['z'] - self.image_position[2]])
                # self.right_nipple = np.array((pt['x'], pt['y'], pt['z']))

    def load_spinal_cord(self, spinal_cord_path):

        if os.path.exists(spinal_cord_path) and len(os.listdir(spinal_cord_path)) == 1:
            # Load in nipple positions
            file_name = os.path.join(spinal_cord_path, 'point.001.json')
            spinal_cord_position = json.loads(open(file_name).read())

            if len(spinal_cord_position):
                self.spinal_cord = np.zeros(3)
                pt = spinal_cord_position[self.position + '_point']['point']
                self.spinal_cord = np.array([pt['y'] - self.image_position[0],
                                             -pt['x'] + self.image_position[1] +
                                             self.pixel_spacing[1] * self.image_shape[1],
                                             pt['z'] - self.image_position[2]])
                self.right_nipple = np.array((pt['x'], pt['y'], pt['z']))


def read_metadata(vl_ids, position, root_path_mri, nipple_path=None, sternum_path=None, spine_path=None):
    # Read metadata for a list of subject given a list of subjects ids
    # vl_ids = list of id identifying the volunteer number
    # root_path = path to the folders with MRI images
    # return a dictionary of metadata, the kys are defined by the volunteers ids
    vl_metadata = {}
    for vl_id in vl_ids:
        vl_metadata[vl_id] = Metadata(vl_id, position, root_path_mri, nipple_path, sternum_path, spine_path)

    return vl_metadata


class Landmarks:
    """Class for storing segmented landmark data from each registrar
    """

    # todo the class structure needs to be improved
    # separate the landmark information for each volunteer
    # this class should return the landmarks of a single volunteer insteas of an dictionary.

    def __init__(self, user_id, volunteer_ids, position, root_data_folder=None):
        self.user_id = user_id  # the corresponding id of the volunteer from were the landmarks have been extracted

        self.position = position  # volunteer position prone or supine
        self.landmarks = {}  # dictionary of landmarks
        # the key corespond to the volunteer Id
        self.landmark_types = {}  # types defined by the registrars {lymth, fibroadenoma..}
        self.vl_ids = []  # list of volunteers ids  already existing the dictionary
        self.root_data_folder = root_data_folder  # the landmarks file location
        self.quadrants = {}  # disactionary with the location of each landmarks identified by quadrants
        # the keys coresponds to the volunteer id
        # the landmarks orders corresponds with the landmarks dictionaty

        if not root_data_folder == None:
            # load the landmarks for each volunteer in the list
            for vl_indx in range(len(volunteer_ids)):
                vl_folder_path = os.path.join(
                    self.root_data_folder, '{0}\\VL{1:05d}'.format(self.user_id, volunteer_ids[vl_indx]))
                if os.path.exists(vl_folder_path):
                    self.load_landmarks(vl_folder_path)

            if volunteer_ids[vl_indx] in self.landmarks:
                self.landmarks[volunteer_ids[vl_indx]] = np.array(self.landmarks[volunteer_ids[vl_indx]])

    def load_landmarks(self, vl_folder_path):
        # load the all landmarks existing in the folder vl_folder_path
        file_nb = len(os.listdir(vl_folder_path))

        for file in range(file_nb):
            path = os.path.join(vl_folder_path, 'point.{0:03d}.json'.format(file + 1))
            with open(path) as _landmarkFile:
                self.add_landmark(json.load(_landmarkFile))

    def add_landmark(self, landmark):

        vl_id = int(landmark['subject'][-5:])
        if vl_id not in self.landmarks:
            self.landmarks[vl_id] = []
            self.landmark_types[vl_id] = []
            self.vl_ids.append(vl_id)

        point = []
        point.append(landmark['{0}_point'.format(self.position)]['point']['x'])
        point.append(landmark['{0}_point'.format(self.position)]['point']['y'])
        point.append(landmark['{0}_point'.format(self.position)]['point']['z'])
        self.landmarks[vl_id].append(point)
        self.landmark_types[vl_id].append(landmark['type'])

    def copy(self):
        # make a copy of the object
        new_landmarks = Landmarks(self.user_id,
                                  self.vl_ids,
                                  self.position)
        new_landmarks.landmarks = self.landmarks.copy()
        try:
            new_landmarks.quadrants = self.quadrants.copy()
        except:
            pass
        new_landmarks.root_data_folder = self.root_data_folder
        new_landmarks.landmark_types = self.landmark_types
        new_landmarks.vl_ids = self.vl_ids

        return new_landmarks

    def find_quadrants(self, metadata):
        # Compute the quadrant were the landmarks are located given the nipple position and sternum position
        # The nipple and sternum position (MRI 3d locations) are read from metadata object

        # LB/RB = left breast/right breast
        # U/L = upper/lower
        # I/O = inner/outer

        for vl_id in self.landmarks:
            self.quadrants[vl_id] = []
            skip = not (vl_id in metadata)
            if not skip:
                skip = (len(metadata[vl_id].jugular_notch) != 3) \
                       or (len(metadata[vl_id].left_nipple) != 3) \
                       or (len(metadata[vl_id].right_nipple) != 3)

            if not skip:
                for landmark in self.landmarks[vl_id]:
                    if landmark[1] < metadata[vl_id].jugular_notch[1]:  # LHS
                        if landmark[1] < metadata[vl_id].left_nipple[1]:  # O
                            if landmark[2] > metadata[vl_id].left_nipple[2]:  # UO
                                self.quadrants[vl_id].append('LB_UO')
                            else:  # LO
                                self.quadrants[vl_id].append('LB_LO')
                        else:  # I
                            if landmark[2] > metadata[vl_id].left_nipple[2]:  # UI
                                self.quadrants[vl_id].append('LB_UI')
                            else:
                                self.quadrants[vl_id].append('LB_LI')  # LI

                    else:  # RHS
                        if landmark[1] < metadata[vl_id].right_nipple[1]:  # I
                            if landmark[2] > metadata[vl_id].right_nipple[2]:  # UI
                                self.quadrants[vl_id].append('RB_UI')
                            else:
                                self.quadrants[vl_id].append('RB_LI')  # LI
                        else:  # O
                            if landmark[2] > metadata[vl_id].right_nipple[2]:  # UO
                                self.quadrants[vl_id].append('RB_UO')
                            else:
                                self.quadrants[vl_id].append('RB_LO') # LO

    def getModellandmarks(self, metadata):
        # Transform points in RAF coordinates system (medical imaging) to model coordinates system
        new_landmarks = self.copy()

        new_landmarks.vl_ids = self.vl_ids
        for vl_id in self.landmarks:
            new_landmarks.landmarks[vl_id] = []

            vl_str = 'VL{0:05d}'.format(vl_id)
            print('Subject {0} transform landmarks to model reference system'.format(vl_str))

            try:
                if isinstance(metadata, dict):
                    x0 = metadata[vl_id].image_position
                    sp = metadata[vl_id].pixel_spacing
                    shape = metadata[vl_id].image_shape
                else:
                    x0 = metadata.image_position
                    sp = metadata.pixel_spacing
                    shape = metadata.image_shape
            except:
                pass
            else:
                for pt in self.landmarks[vl_id]:
                    x = np.array([pt[1] - x0[0],
                                  -pt[0] + x0[1] + sp[1] * shape[1],
                                  pt[2] - x0[2]])
                    new_landmarks.landmarks[vl_id].append(x)


        return new_landmarks

    def getRAFCoordinates(self, metadata):
        # transform the point coordinates from model coordinates
        # system to the RAF coordinates system (medical image)

        new_landmarks = self.copy()

        new_landmarks.vl_ids = self.vl_ids
        for vl_id in self.landmarks:
            new_landmarks.landmarks[vl_id] = []

            vl_str = 'VL{0:05d}'.format(vl_id)
            print('Subject {0} transform landmarks to RAF reference system'.format(vl_str))

            try:
                img_params = metadata[vl_id].position
            except:
                pass
            else:
                x0 = metadata[vl_id].image_position
                sp = metadata[vl_id].pixel_spacing
                shape = metadata[vl_id].image_shape

                for pt in self.landmarks[vl_id]:
                    x = np.array([sp[0] * shape[0] - pt[1] + x0[1],
                                  pt[0] + x0[0],
                                  pt[2] + x0[2]])
                    new_landmarks.landmarks[vl_id].append(x)

        return new_landmarks


def writeLandmarks(proneLd, supineLd, filePath):
    # write landmark to files
    # the files formats correspond to the files by picker web application

    ld_type_1 = proneLd.position
    ld_type_2 = supineLd.position
    user_id = int(proneLd.user_id[-3:])

    for vl_id in proneLd.vl_ids:
        vl_str = 'VL{0:05d}'.format(vl_id)

        sub_filePath = os.path.join(filePath, '{0}/VL{1:05d}'.format(proneLd.user_id, vl_id))
        if not os.path.exists(sub_filePath):
            os.makedirs(sub_filePath)

        if vl_id in proneLd.landmarks:
            supinePoints = False
            if vl_id in supineLd.landmarks:
                if len(proneLd.landmarks[vl_id]) == len(supineLd.landmarks[vl_id]):
                    supinePoints = True

            for indx in range(len(proneLd.landmarks[vl_id])):
                landmark = proneLd.landmarks[vl_id][indx]
                ld_type = proneLd.landmark_types[vl_id][indx]
                fileName = os.path.join(sub_filePath, 'point.{0:03d}.json').format(indx + 1)

                data = {"{0}_point".format(ld_type_1): {"time": 0.0, "point": {"x": landmark[0], "y": landmark[1],
                                                                               "z": landmark[2]}}, "user": user_id,
                        "time": 0.0, "type": ld_type, "id": indx + 1, "subject": vl_str}
                if supinePoints:
                    landmark = supineLd.landmarks[vl_id][indx]
                    data["{0}_point".format(ld_type_2)] = {"time": 0.0, "point": {"x": landmark[0], "y": landmark[1],
                                                                                  "z": landmark[2]}}
                with open(fileName, 'w') as fp:
                    json.dump(data, fp)


def get_distance_to_surfaces(landmarks, sub_model):
    """
    Calculate closest points to a mesh surface
    """
    cwm_dist = get_closest_point_on_surface(landmarks, sub_model.cw_surface_mesh)
    sm_left_dist = get_closest_point_on_surface(landmarks, sub_model.lskin_surface_mesh)
    sm_right_dist = get_closest_point_on_surface(landmarks, sub_model.rskin_surface_mesh)
    # Generate points on chest wall and skin surfaces

    closest_sm_dist = np.minimum(sm_left_dist, sm_right_dist)
    return cwm_dist, closest_sm_dist


def get_closest_point_on_surface(points, surface):
    surface_points = mesh_generation.generate_points_on_face(surface, None, None, num_points=50, dim=2)
    surface_tree = cKDTree(surface_points)
    dist, point_id = surface_tree.query(points)
    return dist, surface_points[point_id]


def shortest_distances(metadata, cw_path, landmarks, output_dir):
    skin = {}
    closest_points_skin = {}
    closest_points_cw = {}
    rib = {}

    debug = False

    for vl_id in landmarks.vl_ids:
        print('Distance estimation for subject {0}'.format(vl_id))
        skip_cw = False
        skip_skin = False

        cw_data_path = os.path.join(cw_path, metadata[vl_id].position, 'rib_cage\\rib_cage_VL{0:05d}.nii'.format(vl_id))
        skin_data_path = os.path.join(cw_path, metadata[vl_id].position, 'body\\body_VL{0:05d}.nii'.format(vl_id))
        skin[vl_id] = []
        closest_points_skin[vl_id] = []
        rib[vl_id] = []
        closest_points_cw[vl_id] = []
        if debug:
            from mayavi import mlab
            import bmw
            from morphic import viewer
            fig = bmw.add_fig(viewer, label='DISTANCE PLOT')

        if os.path.exists(skin_data_path):
            skin_mask = sitkTools.readNIFTIImage(skin_data_path)
            skin_mask.setAlfOrientation()
            skin_mask.set_origin([0, 0, 0])
            # pdb.set_trace()
            skin_points = sitkTools.extract_contour_points(skin_mask, 100000)
            # pdb.set_trace()
            skin_tree = cKDTree(skin_points)
            if debug:
                fig.plot_points('skin_surface', skin_points, color=(1, 1, 1), size=0.5)
        else:
            print('skin surface  is missing')
            skip_skin = True

        if os.path.exists(cw_data_path):
            cw_mask = sitkTools.readNIFTIImage(cw_data_path)
            cw_mask.setAlfOrientation()
            cw_mask.set_origin([0, 0, 0])
            cw_points = sitkTools.extract_contour_points(cw_mask, 100000)
            cw_tree = cKDTree(cw_points)
            if debug:
                fig.plot_points('cw_surface', cw_points, color=(0, 0, 1), size=0.5)

        else:
            print('chest wall is missing')
            skip_cw = True

        for indx, points in enumerate(landmarks.landmarks[vl_id]):
            # print(points)
            if not skip_skin:
                closest_skin_dist, closest_skin_point_id = skin_tree.query(points)
                skin[vl_id].append(closest_skin_dist)
                closest_points_skin[vl_id].append(skin_points[closest_skin_point_id])
                if debug:
                    # import pdb; pdb.set_trace()

                    fig.plot_points('skin_{0}'.format(indx), [skin_points[closest_skin_point_id]],
                                    color=(1, 0, 0), size=3)
                    fig.plot_points('landmark_{0}'.format(indx), [points],
                                    color=(1, 0, 0), size=3)
                    line = np.array([points, skin_points[closest_skin_point_id]])
                    line = np.reshape(line, (2, 3))
                    fig.plots['line_skin_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
                                                                          color=(1, 0, 1), tube_radius=1)
            if not skip_cw:
                closest_cwm_dist, closest_cw_point_id = cw_tree.query(points)
                rib[vl_id].append(closest_cwm_dist)
                closest_points_cw[vl_id].append(cw_points[closest_cw_point_id])
                if debug:
                    fig.plot_points('cw_{0}'.format(indx), [cw_points[closest_cw_point_id]],
                                    color=(0, 1, 0), size=3)
                    fig.plot_points('landmark_{0}'.format(indx), [points],
                                    color=(1, 0, 0), size=3)
                    line = np.array([points, cw_points[closest_cw_point_id]])
                    line = np.reshape(line, (2, 3))
                    fig.plots['line_cw_{0}'.format(indx)] = mlab.plot3d(line[:, 0], line[:, 1], line[:, 2],
                                                                        color=(1, 1, 0), tube_radius=1)
        if debug:
            # pdb.set_trace()
            export_figure(fig, '{1}_{0}_minimal_distance'.format(metadata[vl_id].position, vl_id), output_dir)
            fig.clear()

    return skin, closest_points_skin, rib, closest_points_cw


def get_valid_landmarks_id(prone_landmarks, supine_landmarks,
                           prone_metadata, supine_metadata):
    # Check for which landmarks all the necessary information is available
    corresponding = {}
    for vl_id in prone_landmarks.landmarks:
        use = []
        if vl_id in supine_landmarks.landmarks:
            if vl_id in prone_metadata:
                if vl_id in supine_metadata:
                    use.append('True')
                else:
                    use.append('False')
            else:
                use.append('False')
        corresponding[vl_id] = use
    return corresponding


def get_valid_subject_id(prone_landmarks, supine_landmarks,
                         prone_metadata, supine_metadata):
    # Checks for which volunteers all the necessary information is available
    corresponding = []
    for vl_indx in range(len(prone_landmarks.vl_ids)):
        vl_id = prone_landmarks.vl_ids[vl_indx]
        if vl_id in prone_landmarks.landmarks:
            if vl_id in supine_landmarks.landmarks:
                if vl_id in prone_metadata:
                    if vl_id in supine_metadata:
                        corresponding.append(vl_id)
    return corresponding


def corresponding_landmarks_between_registars(registrar_a_prone, registrar_b_prone, registrar_a_supine,
                                              registrar_b_supine):
    # Checks whether the same landmark is registered by both the registrars by checking whether the landmarks are
    # positioned within a 3 mm radius of each other in the prone and supine position and of there are no other landmarks
    # within 1 cm of the landmark.
    # Returns a dictionary and per volunteer the corresponding local landmark numbers are giving in a list.
    # The first number on cor_1_2 reffers to the the local landmark number as identified by the first registrar and
    # the second by the second registrar.
    corre = {}

    for vl_id in registrar_a_prone.landmarks:
        skip = False

        corre[vl_id] = []
        skip = not ((vl_id in registrar_b_prone.landmarks) and (vl_id in registrar_b_supine.landmarks) \
                    and (vl_id in registrar_a_prone.landmarks) and (vl_id in registrar_a_supine.landmarks))

        if not skip:
            for indx_a, landmark_a in enumerate(registrar_a_prone.landmarks[vl_id]):

                prone_count = 0
                supine_count = 0
                corresponding_indx_prone = 0
                corresponding_indx_supine = 0

                for indx_b, landmark_b in enumerate(registrar_b_prone.landmarks[vl_id]):

                    dis_prone = math.sqrt((landmark_a[0] - landmark_b[0]) ** 2 +
                                          (landmark_a[1] - landmark_b[1]) ** 2 +
                                          (landmark_a[2] - landmark_b[2]) ** 2)
                    # Check wether this distance is equal or less than 3 mm
                    if dis_prone <= 3:
                        prone_count += 1
                        corresponding_indx_prone = indx_b

                if prone_count == 1:
                    landmark_a_supine = registrar_a_supine.landmarks[vl_id][indx_a]

                    for indx_b, landmark_b in enumerate(registrar_b_supine.landmarks[vl_id]):
                        landmark_b_supine = registrar_b_supine.landmarks[vl_id][indx_b]
                        dis_supine = math.sqrt((landmark_a_supine[0] - landmark_b_supine[0]) ** 2 +
                                               (landmark_a_supine[1] - landmark_b_supine[1]) ** 2 +
                                               (landmark_a_supine[2] - landmark_b_supine[2]) ** 2)

                        # Check wether this distance is equal or less than 3 mm
                        if dis_supine <= 3:
                            supine_count += 1
                            corresponding_indx_supine = indx_b

                    if supine_count == 1 and corresponding_indx_prone == corresponding_indx_supine:
                        corre[vl_id].append([indx_a, corresponding_indx_supine])

            # check for point to point correspondance
            # List that contains the indices of duplicates (which should be deleted)
            deleteIndices = []

            for i in range(0, len(corre[vl_id])):  # Loop through entire array
                indices = list(range(0, len(corre[vl_id])))  # All indices in array
                del indices[i]  # All indices in array, except the i'th element currently being checked

                for j in indices:  # Loop through every other element in array, except the i'th element, currently being checked
                    if (corre[vl_id][i][1] == corre[vl_id][j][1]):  # Check if ind_b of element being checked is equal to the ind_b of j'th element
                        deleteIndices.append(j)  # If i'th and j'th element are equal, j is appended to deleteIndices[]

            # Sort deleteIndices in ascending order and delete duplicates index:
            if deleteIndices:
                deleteIndices.sort()
                deleteIndices = np.unique(deleteIndices)

                # Delete duplicates
                corre[vl_id] = np.delete(corre[vl_id], deleteIndices, axis=0)
    # delete empty lists
    corre = {k: v for k, v in corre.items() if len(v)}

    return corre


def displacement(landmarks_a, landmarks_b):
    # Displacement of the landmarks from prone to supine
    # landmarks_a and b - disctionary contining selected landmarks for each volunteer
    # in supine and prone position respectively

    landmarks_displacement = {}

    for vl_id in landmarks_a.landmarks:
        skip = not ((vl_id in landmarks_b.landmarks) and (vl_id in landmarks_a.landmarks))

        if not skip:
            sub_landmark_a = np.asarray(landmarks_a.landmarks[vl_id])
            sub_landmark_b = np.asarray(landmarks_b.landmarks[vl_id])

            if (len(sub_landmark_a.shape) == 1) and (sub_landmark_a.size != 0):
                sub_landmark_a = sub_landmark_a.reshape((1, 3))
            if (len(sub_landmark_b.shape) == 1) and (sub_landmark_b.size != 0):
                sub_landmark_b = sub_landmark_b.reshape((1, 3))

            skip = len(sub_landmark_b) != len(sub_landmark_a)
            if not skip:

                landmark_displacement = []
                for indx, point_a in enumerate(sub_landmark_a):
                    point_b = sub_landmark_b[indx]

                    # compute euclidian distance between the corresponding landmarks
                    displace = math.sqrt((point_a[0] - point_b[0]) ** 2 +
                                         (point_a[1] - point_b[1]) ** 2 +
                                         (point_a[2] - point_b[2]) ** 2)

                    landmark_displacement.append(displace)
                landmarks_displacement[vl_id] = landmark_displacement

    return landmarks_displacement


# Clock face coordinates
def clock(registrar, metadata):
    # Time, distance to the nipple and volunteer numbers as arrays

    time = {}
    dist_landmark_nipple = {}

    for vl_id in registrar.landmarks:
        dist_landmark_nipple[vl_id] = []
        time[vl_id] = []
        skip = not (vl_id in metadata)
        if not skip:
            skip = (len(metadata[vl_id].jugular_notch) != 3) \
                   or (len(metadata[vl_id].left_nipple) != 3) \
                   or (len(metadata[vl_id].right_nipple) != 3)

        if not skip:
            for landmark in registrar.landmarks[vl_id]:
                if landmark[1] < metadata[vl_id].jugular_notch[1]:  # LHS
                    nipple = metadata[vl_id].left_nipple
                else:
                    nipple = metadata[vl_id].right_nipple  # RHS

                x = landmark[0] - nipple[0]
                y = -(landmark[1] - nipple[1])
                z = landmark[2] - nipple[2]
                dist_to_nipple = math.sqrt(x ** 2 + y ** 2 + z ** 2)  # Calculate distance of landmark to nipple
                dist_landmark_nipple[vl_id].append(dist_to_nipple)
                n = np.sqrt(y ** 2 + z ** 2)
                t = np.arctan2(y, z)
                if n <= 10:
                    clock = 'central'  # When the landmark is too close to the nipple to accurately determine the time, the time will be 'central'
                else:
                    hour = 6 * t / math.pi
                    if hour < 0: hour = 12 + hour  # The angle is between - pi and pi, so negative angles reffer to hours 6 to 12
                    whole_hour = math.floor(hour)
                    min = hour - whole_hour
                    min = min * 60.  # Convert fractions of an hour to minutes
                    if whole_hour == 0: whole_hour = 12
                    if min < 15:  # Give the time in half hours
                        min = 0
                        clock = str(int(whole_hour)) + ':00'
                    elif min < 45:
                        min = 30
                        clock = str(int(whole_hour)) + ':30'
                    else:
                        min = 0,
                        whole_hour += 1
                        if whole_hour == 13: whole_hour = 1
                        clock = str(int(whole_hour)) + ':00'
                time[vl_id].append(clock)

    return time, dist_landmark_nipple


def getALFWorldCoordinates(rafPoints, image):
    # conver a list of points from RAF to ALF reference systems
    # using mri metadata from scan

    scan_orientation = image.orientation
    scan = image.copy()
    alfPoints = np.zeros(rafPoints.shape)

    if scan_orientation == 'ALF':
        scan.setRafOrientation()
        alfPoints = rafPoints - scan.origin
        alfPoints = np.transpose([alfPoints[:, 1],
                                  scan.spacing[0] * scan.shape[0] - alfPoints[:, 0],
                                  alfPoints[:, 2]])
        scan.setAlfOrientation
        alfPoints = alfPoints + image.origin

    if scan_orientation == 'RAF':
        alfPoints = rafPoints - scan.origin
        alfPoints = np.transpose([alfPoints[:, 1],
                                  scan.spacing[0] * scan.shape[0] - alfPoints[:, 0],
                                  alfPoints[:, 2]])
        scan.setAlfOrientation()
        alfPoints = alfPoints + scan.origin
        scan.setRafOrientation()

    return alfPoints


def getRAFWorldCoordinates(alfPoints, image):
    # converts a list of points from ALF to RAf reference system
    # using the metadata from scan object
    scan = image.copy()
    rafPoints = np.zeros(3)

    if image.orientation == 'RAF':
        scan.setAlfOrientation()
        rafPoints = alfPoints - scan.origin
        rafPoints = np.transpose([scan.spacing[0] * scan.shape[0] - rafPoints[:, 1],
                                  rafPoints[:, 0],
                                  rafPoints[:, 2]])
        scan.setRafOrientation(0)
        rafPoints = rafPoints + image.origin

    if image.orientation == 'ALF':
        rafPoints = alfPoints - scan.origin
        rafPoints = np.transpose([scan.spacing[0] * scan.shape[0] - rafPoints[:, 1],
                                  rafPoints[:, 0],
                                  rafPoints[:, 2]])
        scan.setRafOrientation()
        rafPoints = rafPoints + scan.origin
        scan.setAlfOrientation()

    return rafPoints


def export_figure(fig, label, folder):
    views = {

        'x_minus': (
            (180.0, 90.0, 937.35168032166928, np.array([208.69893612, 164.74016101, 80.77018314])), 90.0),

        'x_plus': (
            (0.0, 90.0, 937.35168032166928, np.array([208.69893612, 164.74016101, 80.77018314])), -90.0),

        'y_minus': (
            (-90.0, 90.0, 937.35168032166928, np.array([208.69893612, 164.74016101, 80.77018314])), 90.0),

        'y_plus': (
            (90.0, 90.0, 937.35168032166928, np.array([208.69893612, 164.74016101, 80.77018314])), 90.0),

        'z_minus': (
            (0.0, 180.0, 937.35168032166928, np.array([208.69893612, 164.74016101, 80.77018314])), -0.0),

        'z_plus': ((0.0, 0.0, 937.35168032166928, np.array([208.69893612, 164.74016101, 80.77018314])), 0.0),

        '3d_tilt': ((175.72585429154287, 137.17197695449596, 937.35168032164654,
                     np.array([221.79771286, 142.95230897, 75.11460734])), 88.48891815067519)}

    for key, value in views.items():
        fig.set_camera(value)
        fig.figure.scene.magnification = 4
        fig.figure.scene.save(os.path.join(folder, '{0}_{1}.jpg'.format(label, key)))