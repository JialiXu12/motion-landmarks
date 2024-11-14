# Standard library imports
import os
import json

# Related third party imports
import numpy as np
import scipy
from scipy.spatial import cKDTree
from mayavi import mlab
import morphic
import bmw
from morphic import utils
import icp
import h5py
import math
import google_apis
import matplotlib.pyplot as plt


class Landmarks:
    """Class for storing segmented landmark data from each registrar
    """

    def __init__(self, registrar_name, user_id, position, root_data_folder):
        self.registrar_name = registrar_name
        self.user_id = user_id
        self.position = position
        self.landmarks = {}  # landmarks segmented from the picker webgl gui
        self.model_landmarks = {}  # landmark location in model coordinate system
        self.landmark_types = {}
        self.vl_ids = {}
        self.max_landmarks = 15
        self.max_volunteer_id = 90
        self.root_data_folder = root_data_folder

        # Load MRI image properties
        with open('vl_t2_image_properties.json') as json_data:
            self.img_prop = json.load(json_data)

        self.load_landmarks()

    def load_landmarks(self):
        for vl_id in range(self.max_volunteer_id):
            vl_folder = os.path.join(
                self.root_data_folder,'picker/points/{0}/VL{1:05d}'.format(self.user_id, vl_id))
            for point_id in range(self.max_landmarks):
                path = os.path.join(vl_folder, 'point.{:03d}.json'.format(point_id))
                if os.path.exists(path):
                    with open(path) as _landmarkFile:
                        self.add_landmark(json.load(_landmarkFile), vl_id)
            # Convert landmark list to numpy array
            if vl_id in self.landmarks:
                self.landmarks[vl_id] = np.array(self.landmarks[vl_id])

    def add_landmark(self, landmark, vl_id):
        if vl_id not in self.landmarks:
            self.landmarks[vl_id] = []
            self.landmark_types[vl_id] = []
        point = []
        point.append(landmark['{0}_point'.format(self.position)]['point']['x'])
        point.append(landmark['{0}_point'.format(self.position)]['point']['y'])
        point.append(landmark['{0}_point'.format(self.position)]['point']['z'])
        self.landmarks[vl_id].append(point)
        self.landmark_types[vl_id].append(landmark['type'])

        if vl_id not in self.model_landmarks:
            self.model_landmarks[vl_id] = []
        vl_str = 'VL{0:05d}'.format(vl_id)
        print vl_str
        print self.img_prop[vl_str]
        try:
            img_params = self.img_prop[vl_str][self.position]
        except:
            pass
        else:
            x0 = img_params['image_position']
            sp = img_params['pixel_spacing']
            shape = img_params['shape']
            pt = landmark['{0}_point'.format(self.position)]['point']
            x = np.array([pt['y'] - x0[1],
                          -pt['x'] + x0[0] + sp[0] * shape[0],
                          pt['z'] - x0[2]])
            self.model_landmarks[vl_id].append(x)

    def find_quadrants(self, metadata):
        self.quadrants = Quadrants(self, metadata)

class Metadata():

    def __init__(self, model_path, position):
        self.metadata_path = os.path.join(model_path, 'segmented_data')
        self.ages = {}
        self.weights = {}
        self.heights = {}
        self.left_nipples = {}
        self.right_nipples = {}
        self.sternal_notches = {}

        for vl_id in range(90):
            vl_str = 'VL{0:05d}'.format(vl_id)
            data_path = os.path.join(self.metadata_path, vl_str, 'nipple_pts.data')
            if os.path.exists(data_path):
                data = morphic.Data(data_path)
                self.ages[vl_id] = data.metadata['subject']['age'].replace('Y', '')
                self.weights[vl_id] = data.metadata['subject']['weight']
                self.heights[vl_id] = data.metadata['subject']['height']
                # If you'll use the nipple positions from data, you'll use the automatically generated positions
                # (which are often incorrect) instead of the manually chosen positions
                # self.left_nipples[vl_id] =data.values[0,:]
                # self.right_nipples[vl_id] =data.values[1,:]
                self.sternal_notches[vl_id] =data.values[2,:]

                nipple_left = []
                nipple_right = []

                path = '/home/psam012/opt/breast-data/t2_nipple/picker/points/user004'

                vl_id_str = 'VL{0:05d}'.format(vl_id)
                nipple_path = os.path.join(path, vl_id_str)
                if os.path.exists(nipple_path):
                    # Load in nipple positions
                    left_nipple_path = os.path.join(nipple_path,'point.002.json')
                    right_nipple_path = os.path.join(nipple_path,'point.001.json')
                    left_nipple_position = json.loads(open(left_nipple_path).read())
                    right_nipple_position = json.loads(open(right_nipple_path).read())

                    # Nipple positions in image (pixel coordinates)
                    # (At the moment these arrays are not used or saved, so it could probably be deleted)
                    nipple_right.append(left_nipple_position[position +'_point']['point']['x'])
                    nipple_right.append(left_nipple_position[position +'_point']['point']['y'])
                    nipple_right.append(left_nipple_position[position +'_point']['point']['z'])

                    nipple_left.append(right_nipple_position[position +'_point']['point']['x'])
                    nipple_left.append(right_nipple_position[position +'_point']['point']['y'])
                    nipple_left.append(right_nipple_position[position +'_point']['point']['z'])

                    # Nipple positions in image (xyz coordinates)
                    with open('vl_t2_image_properties.json') as json_data:
                        img_prop = json.load(json_data)
                    try:
                        img_params = img_prop[vl_id_str][position]
                    except:
                        pass
                    else:
                        x0 = img_params['image_position']
                        sp = img_params['pixel_spacing']
                        shape = img_params['shape']
                        pt = left_nipple_position[position +'_point']['point']
                        x = np.array([pt['y'] - x0[1],
                            -pt['x'] + x0[0] + sp[0] * shape[0],
                            pt['z'] - x0[2]])
                        self.left_nipples[vl_id] = x

                        pt = right_nipple_position[position +'_point']['point']
                        x = np.array([pt['y'] - x0[1],
                            -pt['x'] + x0[0] + sp[0] * shape[0],
                            pt['z'] - x0[2]])
                        self.right_nipples[vl_id] = x

            else:
                self.ages[vl_id] = 'n/a'
                self.weights[vl_id] = 'n/a'
                self.heights[vl_id] = 'n/a'



class Quadrants:

    def __init__(self, registrar, metadata):
        #LB/RB = left breast/right breast
        #U/L = upper/lower
        #I/O = inner/outer
        self.landmark_quadrants = {}
        self.LB_UO = {}
        self.LB_LO = {}
        self.LB_UI = {}
        self.LB_LI = {}
        self.RB_UO = {}
        self.RB_LO = {}
        self.RB_UI = {}
        self.RB_LI = {}
        for vl_id in range(90):
            if vl_id not in self.landmark_quadrants:
                self.landmark_quadrants[vl_id] = []
            self.LB_UO[vl_id] = 0
            self.LB_LO[vl_id] = 0
            self.LB_UI[vl_id] = 0
            self.LB_LI[vl_id] = 0
            self.RB_UO[vl_id] = 0
            self.RB_LO[vl_id] = 0
            self.RB_UI[vl_id] = 0
            self.RB_LI[vl_id] = 0
            #if vl_id in registrar.model_landmarks:
            if vl_id in metadata.left_nipples:
                if vl_id in registrar.model_landmarks:
                    for landmark in registrar.model_landmarks[vl_id]:
                        if vl_id in metadata.sternal_notches:
                            if landmark[1] < metadata.sternal_notches[vl_id][1]: #LHS
                                if landmark[1] < metadata.left_nipples[vl_id][1]: # UO or LO
                                    if landmark[2] > metadata.left_nipples[vl_id][2]:  # UO
                                        self.LB_UO[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('LB_UO')
                                    elif landmark[2] < metadata.left_nipples[vl_id][2]:  # LO
                                        self.LB_LO[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('LB_LO')
                                else:# UI or LI
                                    if landmark[2] > metadata.right_nipples[vl_id][2]:  # UI
                                        self.LB_UI[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('LB_UI')
                                    elif landmark[2] < metadata.right_nipples[vl_id][2]:  # LI
                                        self.LB_LI[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('LB_LI')

                            else: #RHS
                                if landmark[1] < metadata.right_nipples[vl_id][1]: # UI or LI
                                    if landmark[2] > metadata.right_nipples[vl_id][2]:  # UI
                                        self.RB_UI[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('RB_UI')
                                    elif landmark[2] < metadata.right_nipples[vl_id][2]:  # LI
                                        self.RB_LI[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('RB_LI')
                                else:# UO or LO
                                    if landmark[2] > metadata.right_nipples[vl_id][2]:  # UO
                                        self.RB_UO[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('RB_UO')
                                    elif landmark[2] < metadata.right_nipples[vl_id][2]:  # LO
                                        self.RB_LO[vl_id] +=1
                                        self.landmark_quadrants[vl_id].append('RB_LO')
                        else:
                            self.landmark_quadrants[vl_id].append('n/a')

def corresponding_landmarks_between_registars(registrar_a_prone, registrar_b_prone, registrar_a_supine, registrar_b_supine, corresponding_landmarks_a, corresponding_landmarks_b):
    # Checks whether the same landmark is registered by both the registrars by checking whether the landmarks are
    # positioned within a 3 mm radius of each other in the prone and supine position and of there are no other landmarks
    # within 1 cm of the landmark.
    # Returns a dictionary and per volunteer the corresponding local landmark numbers are giving in a list.
    # The first number on cor_1_2 reffers to the the local landmark number as identified by the first registrar and
    # the second by the second registrar.
    cor = {}
    for vl_id in range(90):
        if vl_id in registrar_a_prone.model_landmarks:
            corre =[]
            for x in range(len(registrar_a_prone.model_landmarks[vl_id])):
                xx = str(x+1)
                if corresponding_landmarks_a[vl_id][x] == 'True':
                    landmark_a = registrar_a_prone.model_landmarks[vl_id][x]
                    if vl_id in registrar_b_prone.model_landmarks:
                        for y in range(len(registrar_b_prone.model_landmarks[vl_id])):
                            if corresponding_landmarks_b[vl_id][y] == 'True':
                                landmark_b = registrar_b_prone.model_landmarks[vl_id][y]
                                # Distance between the the landmarks in the prone position
                                dis_prone = math.sqrt((landmark_a[0]-landmark_b[0])**2 +
                                             (landmark_a[1]-landmark_b[1])**2 +
                                             (landmark_a[2]-landmark_b[2])**2)
                                # Check wether this distance is equal or less than 3 mm
                                if dis_prone <= 3:

                                    # Distance between landmarks in the supine position
                                    landmark_a_supine = registrar_a_supine.model_landmarks[vl_id][x]
                                    landmark_b_supine = registrar_b_supine.model_landmarks[vl_id][y]
                                    dis_supine = math.sqrt((landmark_a_supine[0]-landmark_b_supine[0])**2 +
                                             (landmark_a_supine[1]-landmark_b_supine[1])**2 +
                                             (landmark_a_supine[2]-landmark_b_supine[2])**2)
                                    # Check wether this distance is equal or less than 3 mm
                                    if dis_supine <= 3:
                                        corresponding = 'True'
                                        while corresponding =='True':
                                            for landmark in registrar_a_prone.model_landmarks[vl_id]:
                                                # Check whether no other landmarks are identified within a 1 cm radius
                                                # of the corresponding landmarks in the prone position
                                                if landmark[0] == landmark_a[0] and landmark[1] == landmark_a[1] \
                                                        and landmark[2] == landmark_a[2]:
                                                    corresponding = 'True'
                                                elif math.sqrt((landmark_a[0]-landmark[0])**2 +
                                                                (landmark_a[1]-landmark[1])**2 +
                                                                (landmark_a[2]-landmark[2])**2) > 10:
                                                    corresponding = 'True'
                                                else: corresponding = 'False'
                                            for landmark in registrar_b_prone.model_landmarks[vl_id]:
                                                if landmark[0] == landmark_b[0] and landmark[1] == landmark_b[1] and landmark[2] == landmark_b[2]:
                                                    corresponding = 'True'
                                                elif math.sqrt((landmark_b[0]-landmark[0])**2 +
                                                                (landmark_b[1]-landmark[1])**2 +
                                                                (landmark_b[2]-landmark[2])**2) > 10:
                                                    corresponding = 'True'
                                                else: corresponding = 'False'
                                            for landmark in registrar_a_supine.model_landmarks[vl_id]:
                                                # Check whether no other landmarks are identified within a 1 cm radius
                                                # of the corresponding landmarks in the supine position
                                                if landmark[0] == landmark_a_supine[0] and landmark[1] == landmark_a_supine[1] and landmark[2] == landmark_a_supine[2]:
                                                    corresponding = 'True'
                                                elif math.sqrt((landmark_a_supine[0]-landmark[0])**2 +
                                                                (landmark_a_supine[1]-landmark[1])**2 +
                                                                (landmark_a_supine[2]-landmark[2])**2) > 10:
                                                    corresponding = 'True'
                                                else: corresponding = 'False'
                                            for landmark in registrar_b_supine.model_landmarks[vl_id]:
                                                if landmark[0] == landmark_b_supine[0] and landmark[1] == landmark_b_supine[1] and landmark[2] == landmark_b_supine[2]:
                                                    corresponding = 'True'
                                                elif math.sqrt((landmark_b_supine[0]-landmark[0])**2 +
                                                                (landmark_b_supine[1]-landmark[1])**2 +
                                                                (landmark_b_supine[2]-landmark[2])**2) > 10:
                                                    corresponding = 'True'
                                                else: corresponding = 'False'
                                            yy = str(y+1)
                                            corresponding = 'False'
                                            corre.append([xx , yy])
            cor[vl_id] = corre
    return cor

def displacement(registrar_prone, t_supine, corresponding_landmarks):
    # Displacement of the landmarks from prone to supine
    landmark_displacement = []
    Landmark_Displacement = {}
    for vl_id in range(90):
        if vl_id in registrar_prone.model_landmarks:
            for x in range(len(registrar_prone.model_landmarks[vl_id])):
                if corresponding_landmarks[vl_id][x] == 'True':
                    if vl_id in t_supine:
                        landmark_prone = registrar_prone.model_landmarks[vl_id][x]
                        landmark_supine = t_supine[vl_id][x]
                        displace = math.sqrt((landmark_prone[0]-landmark_supine[0])**2 +
                                                 (landmark_prone[1]-landmark_supine[1])**2 +
                                                 (landmark_prone[2]-landmark_supine[2])**2)
                        landmark_displacement.append(displace)
                        Landmark_Displacement[vl_id] = displace
    return landmark_displacement, Landmark_Displacement

def displacement_nipple(metadata_prone,  t_left_nipple, t_right_nipple, registrar_prone, corresponding_landmarks):
    # Displacement of the nipples from prone to supine
    left_nipple_displacement = []
    right_nipple_displacement = []
    for vl_id in range(90):
        if vl_id in registrar_prone.model_landmarks:
            for x in range(len(registrar_prone.model_landmarks[vl_id])):
                if corresponding_landmarks[vl_id][x] == 'True':
                    if vl_id in t_left_nipple:
                        nipple_prone = metadata_prone.left_nipples[vl_id]
                        nipple_supine = t_left_nipple[vl_id]
                        displace = math.sqrt((nipple_prone[0]-nipple_supine[0])**2 +
                                                 (nipple_prone[1]-nipple_supine[1])**2 +
                                                 (nipple_prone[2]-nipple_supine[2])**2)
                        left_nipple_displacement.append(displace)
                        nipple_prone = metadata_prone.right_nipples[vl_id]
                        nipple_supine = t_right_nipple[vl_id]
                        displace = math.sqrt((nipple_prone[0]-nipple_supine[0])**2 +
                                                 (nipple_prone[1]-nipple_supine[1])**2 +
                                                 (nipple_prone[2]-nipple_supine[2])**2)
                        right_nipple_displacement.append(displace)
    return left_nipple_displacement, right_nipple_displacement

#Clock face coordinates
def clock(registrar, metadata, corresponding_landmarks):
    # Time, distance to the nipple and volunteer numbers as arrays
    volu = []
    time = []
    dist_landmark_nipple=[]
    dist_between_nipples = []

    for vl_id in range(90):
        if vl_id in registrar.model_landmarks:
            number=0
            for landmark in registrar.model_landmarks[vl_id]:
                if corresponding_landmarks[vl_id][number] == 'True':
                    volu.append(vl_id)
                    if landmark[1] < metadata.sternal_notches[vl_id][1]: #LHS
                        nipple = metadata.left_nipples[vl_id]
                    else:
                        nipple = metadata.right_nipples[vl_id] #RHS

                    x = landmark[0] - nipple[0]
                    y = -(landmark[1] - nipple[1])
                    z= landmark[2] - nipple[2]
                    dist_to_nipple = math.sqrt(x**2+y**2+z**2) # Calculate distance of landmark to nipple
                    dist_landmark_nipple.append(dist_to_nipple)
                    n = np.sqrt(y**2+z**2)
                    t= np.arctan2(y,z)
                    if n <= 10:
                        clock = 'central' # When the landmark is too close to the nipple to accurately determine the time, the time will be 'central'
                    else:
                        hour = 6*t/math.pi
                        if hour < 0: hour = 12+hour # The angle is between - pi and pi, so negative angles reffer to hours 6 to 12
                        whole_hour = math.floor(hour)
                        min = hour- whole_hour
                        min = min*60.   #Convert fractions of an hour to minutes
                        if whole_hour ==0: whole_hour=12
                        if min < 15:    # Give the time in half hours
                            min = 0
                            clock = str(int(whole_hour)) +':00'
                        elif min < 45:
                            min =30
                            clock = str(int(whole_hour)) + ':30'
                        else:
                            min = 0,
                            whole_hour += 1
                            if whole_hour == 13 : whole_hour=1
                            clock = str(int(whole_hour)) + ':00'
                    time.append(clock)
                number += 1
    return time, volu, dist_landmark_nipple

class closest_points():
    """
    Calculate closest points to a mesh surface
    """

    def __init__(self, cwm, sm_left, sm_right, label='closest_points'):
        self.label = label
        # Generate points on chest wall and skin surfaces
        self.cwm_points = bmw.generate_points_on_face(
            cwm, None, None, num_points=50, dim=2)
        self.sm_left_points = bmw.generate_points_on_face(
            sm_left, None, None, num_points=50, dim=2)
        self.sm_right_points = bmw.generate_points_on_face(
            sm_right, None, None, num_points=50, dim=2)
        self.cwm_tree = cKDTree(self.cwm_points)
        self.sm_left_tree = cKDTree(self.sm_left_points)
        self.sm_right_tree = cKDTree(self.sm_right_points)

    def query(self, landmarks):
        closest_cwm_dist, _ = self.cwm_tree.query(landmarks)
        sm_left_dist, _ = self.sm_left_tree.query(landmarks)
        sm_right_dist, _ = self.sm_right_tree.query(landmarks)
        closest_sm_dist = np.minimum(sm_left_dist, sm_right_dist)
        return closest_cwm_dist, closest_sm_dist

    def visualise_points(self, fig):
        bmw.plot_points(fig, self.label + 'cwm_points',
                        self.cwm_points, range(1, len(self.cwm_points) + 1),
                        visualise=True, colours=(0, 1, 0), point_size=2,
                        text_size=5)

class align_meshes:
    def __init__(self, prone_model_path, supine_model_path,
                 jugular_landmarks_prone, jugular_landmarks_supine,
                 prone_metadata, supine_metadata,
                 root_data_folder, registrar_prone, registrar_supine, skip_vl=[], debug=True):

        self.t_supine1_landmark = {}
        self.t_supine2_landmark = {}
        self.t_left_nipple = {}
        self.t_right_nipple ={}
        self.t_cwm_supine = {}
        self.t_sm_left_supine = {}
        self.t_sm_right_supine ={}
        self.t_jl_supine = {}
        self.jl_prone = {}
        self.rotation_prone_supine = {}

        # I added the skip_vl part:
        self.vl_id_range = []
        for vl_id in range(1, 90):
            if not vl_id in skip_vl:
                self.vl_id_range.append(vl_id)

        if debug:
            self.vl_id_range = [14, 15, 18, 24, 27, 30, 31, 32, 34, 38, 40, 41, 42, 43, 44, 47, 49, 50, 52, 56, 57, 58, 61, 62, 65, 68, 69, 74, 75, 76, 77, 78, 82, 83] #[38, 50, 56, 62, 74, 76, 77, 83]

        visualise = False
        if visualise:
            onscreen = True
            if onscreen:
                from morphic import viewer
            else:
                fig = None
                viewer = None

        self.path_mesh_prone={}
        self.path_mesh_supine={}
        self.closest_points_prone ={}
        self.closest_points_supine ={}

        for vl_id in self.vl_id_range:
            skip = False
            vl_id_str = 'VL{0:05d}'.format(vl_id)
            prone_mesh_path = os.path.join(prone_model_path, 'volunteer_meshes', vl_id_str)
            supine_mesh_path = os.path.join(supine_model_path, 'volunteer_meshes',  vl_id_str)
            self.path_mesh_prone[vl_id]=prone_mesh_path
            self.path_mesh_supine[vl_id]=supine_mesh_path

            # Load jugular landmarks
            if vl_id in jugular_landmarks_prone.model_landmarks:
                jugular_landmark_prone = np.array(
                    jugular_landmarks_prone.model_landmarks[vl_id])
            else:
                skip = True
            if vl_id in jugular_landmarks_supine.model_landmarks:
                jugular_landmark_supine = np.array(
                    jugular_landmarks_supine.model_landmarks[vl_id])
            else:
                skip = True

            if not os.path.exists(supine_mesh_path):
                skip = True
            if not os.path.exists(prone_mesh_path):
                skip = True

            if not skip:
                # Load prone meshes
                prone_cwm = bmw.load_chest_wall_surface_mesh(prone_mesh_path, 'prone')
                prone_lm = bmw.load_lung_surface_mesh(prone_mesh_path, 'prone')
                prone_sm_left = bmw.load_skin_surface_mesh(prone_mesh_path, 'prone','left')
                prone_sm_right = bmw.load_skin_surface_mesh(prone_mesh_path, 'prone','right')
                # Load supine meshes
                supine_cwm = bmw.load_chest_wall_surface_mesh(supine_mesh_path, 'supine')
                supine_lm = bmw.load_lung_surface_mesh(supine_mesh_path, 'supine')
                supine_sm_left = bmw.load_skin_surface_mesh(supine_mesh_path, 'supine','left')
                supine_sm_right = bmw.load_skin_surface_mesh(supine_mesh_path, 'supine','right')

                # Calculate closest points to a mesh surface
                prone_closest_points = closest_points(prone_cwm, prone_sm_left,
                                                  prone_sm_right, label='prone')
                supine_closest_points = closest_points(supine_cwm, supine_sm_left,
                                                   supine_sm_right, label='supine')
                self.closest_points_prone[vl_id] = prone_closest_points
                self.closest_points_supine[vl_id] = supine_closest_points
                closest_cwm_dist, closest_sm_dist = prone_closest_points.query(jugular_landmark_prone)

                if visualise:
                    # Prone
                    prone_fig = bmw.add_fig(viewer, label='prone_meshes')
                    bmw.visualise_mesh(prone_cwm, prone_fig, visualise=True, face_colours=(1, 0, 0),opacity=0.5)
                    #bmw.visualise_mesh(prone_lm, prone_fig, visualise=True, face_colours=(0, 1, 0),opacity=0.75)
                    bmw.visualise_mesh(prone_sm_left, prone_fig, visualise=True, face_colours=(0, 0, 1),opacity=0.5)
                    bmw.visualise_mesh(prone_sm_right, prone_fig, visualise=True, face_colours=(0, 0, 1),opacity=0.5)
                    bmw.plot_points(prone_fig, 'prone_left_nipple',
                                    prone_metadata.left_nipples[vl_id], [1],
                                    visualise=True, colours=(0,1,0), point_size=3,
                                    text_size=5)
                    bmw.plot_points(prone_fig, 'prone_right_nipple',
                                    prone_metadata.right_nipples[vl_id], [1],
                                    visualise=True, colours=(0, 1, 0), point_size=3,
                                    text_size=5)
                    bmw.plot_points(prone_fig, 'prone_sternal_notch',
                                    prone_metadata.sternal_notches[vl_id], [1],
                                    visualise=True, colours=(0, 1, 0), point_size=3,
                                    text_size=5)
                    if vl_id in registrar_prone.model_landmarks:
                        bb = registrar_prone.model_landmarks[vl_id]
                        bmw.plot_points(prone_fig, 'landmark', bb, range(1, len(bb) + 1), visualise = True,
                                        colours=(0,1,0), point_size=3, text_size=5)


                    prone_image_path = os.path.join(root_data_folder,
                                                    'images/vl/mri_t2/{0}/prone'.format(
                                                        vl_id_str))

                    bmw.view_mri(prone_image_path, prone_fig, axes='z_axes')

                    # Supine
                    supine_fig = bmw.add_fig(viewer, label='supine_meshes')
                    bmw.visualise_mesh(supine_cwm, supine_fig, visualise=True, face_colours=(0, 1, 0))
                    #bmw.visualise_mesh(supine_lm, supine_fig, visualise=True, face_colours=(0, 1, 0),opacity=0.75)
                    bmw.visualise_mesh(supine_sm_left, supine_fig, visualise=True, face_colours=(0, 0, 1),opacity=0.5)
                    bmw.visualise_mesh(supine_sm_right, supine_fig, visualise=True, face_colours=(0, 0, 1),opacity=0.5)
                    bmw.plot_points(supine_fig, 'supine_left_nipple',
                                    supine_metadata.left_nipples[vl_id], [1],
                                    visualise=True, colours=(0,1,0), point_size=3,
                                    text_size=5)
                    bmw.plot_points(supine_fig, 'supine_right_nipple',
                                    supine_metadata.right_nipples[vl_id], [1],
                                    visualise=True, colours=(0, 1, 0), point_size=3,
                                    text_size=5)
                    bmw.plot_points(supine_fig, 'supine_sternal_notch',
                                    supine_metadata.sternal_notches[vl_id], [1],
                                    visualise=True, colours=(0, 1, 0), point_size=3,
                                     text_size=5)
                    if vl_id in registrar_supine.model_landmarks:
                        bmw.plot_points(supine_fig, 'landmark', registrar_supine.model_landmarks[vl_id][0], [1],
                                        visualise = True, colours=(0,1,0), point_size=3, text_size=5)
                    supine_image_path = os.path.join(root_data_folder,
                                                        'images/vl/mri_t2/{0}/supine'.format(
                                                            vl_id_str))
                    bmw.view_mri(supine_image_path, supine_fig, axes='z_axes')

                #Convert hermite to lagrange
                supine_cwm = utils.convert_hermite_lagrange(supine_cwm, tol=1e-9)
                prone_cwm = utils.convert_hermite_lagrange(prone_cwm, tol=1e-9)


                supine_cwm_nodes = supine_cwm.get_node_ids()[0]
                prone_cwm_nodes = prone_cwm.get_node_ids()[0]
                supine_sm_left_nodes = supine_sm_left.get_node_ids()[0]
                supine_sm_right_nodes = supine_sm_right.get_node_ids()[0]

                supine_sm_left_nodes_n = []
                supine_sm_right_nodes_n = []

                for x in supine_sm_left_nodes:
                    if len(x) == 3:
                        supine_sm_left_nodes_n.append(np.array(x))

                for x in supine_sm_right_nodes:
                    if len(x) == 3:
                        supine_sm_right_nodes_n.append(np.array(x))

                rOpt = icp.alignCorrespondingDataAndLandmarksRigidRotationTranslation(
                    supine_cwm_nodes, prone_cwm_nodes,
                    jugular_landmark_supine, jugular_landmark_prone,weighting=100)
                rotation[vl_id]=rOpt.x

                rOpt_prone_supine = icp.alignCorrespondingDataAndLandmarksRigidRotationTranslation(
                     prone_cwm_nodes, supine_cwm_nodes,
                     jugular_landmark_prone, jugular_landmark_supine, weighting=100)
                self.rotation_prone_supine[vl_id] = rOpt_prone_supine.x


                #if debug:
                #    rOpt.x = scipy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])



                t_supine_nodes = icp.transformRigid3D(supine_cwm_nodes, scipy.array(rOpt.x))
                t_jugular_landmark_supine = icp.transformRigid3D(jugular_landmark_supine, scipy.array(rOpt.x))

                # Rigid transformation of the landmarks and nipples in supine position
                if vl_id in registrar1_supine.model_landmarks:
                    t_landmark_supine1 = icp.transformRigid3D(np.array(
                        registrar1_supine.model_landmarks[vl_id]), scipy.array(rOpt.x))
                    self.t_supine1_landmark[vl_id] = t_landmark_supine1
                if vl_id in registrar2_supine.model_landmarks:
                    t_landmark_supine2 = icp.transformRigid3D(np.array(
                        registrar2_supine.model_landmarks[vl_id]), scipy.array(rOpt.x))
                    self.t_supine2_landmark[vl_id] = t_landmark_supine2
                if vl_id in supine_metadata.left_nipples:
                    nipple = [supine_metadata.left_nipples[vl_id], supine_metadata.right_nipples[vl_id]]
                    t_nipple = icp.transformRigid3D(np.array(nipple), scipy.array(rOpt.x))
                    self.t_jl_supine[vl_id]=t_jugular_landmark_supine
                    self.jl_prone[vl_id]=jugular_landmark_prone


                    self.t_left_nipple[vl_id] = t_nipple[0]
                    self.t_right_nipple[vl_id] = t_nipple[1]

                # Align supine mesh with the prone mesh
                t_supine_cwm = bmw.load_chest_wall_surface_mesh(supine_mesh_path, 'supine')
                t_supine_cwm = utils.convert_hermite_lagrange(t_supine_cwm, tol=1e-9)
                t_supine_cwm.label = 't_supine'
                for node_id in range(1, len(supine_cwm.get_node_ids()[1])+1):
                    t_supine_cwm.nodes[node_id].values = t_supine_nodes[node_id-1, :]

                if visualise:
                    #bmw.visualise_mesh(t_supine_cwm, prone_fig, visualise=True, face_colours=(0, 0, 1), opacity=0.1)
                    bmw.plot_points(prone_fig, 'jugular_landmark_prone',
                                    jugular_landmark_prone, [1,2],
                                    visualise=True, colours=(1,0,0),
                                    point_size=3, text_size=5)
                    bmw.plot_points(prone_fig, 't_jugular_landmark_supine',
                                    t_jugular_landmark_supine, [1,2],
                                    visualise=True, colours=(0,0,1),
                                    point_size=3, text_size=5)
                    #set camera
                    prone_fig.set_camera(((-144.12706005056552,
                                      46.242345546042742,
                                      1306.0100810673252,
                                      np.array([ 224.4140625 ,  224.4140625 ,  114.74999848])),
                                      64.14121105938933))

                    prone_fig.set_camera(((-144.12706005056552,
                                     46.242345546042742,
                                      1306.0100810673252,
                                      np.array([ 224.4140625 ,  224.4140625 ,  114.74999848])),
                                      64.14121105938933))

                    mlab.savefig(filename='Test VL{0}.png'.format(vl_id))
                    prone_fig.clear()
                    supine_fig.clear()

    def shortest_distances(self, registrar, position, corresponding_landmarks):
        skin = []
        rib = []

        for vl_id in self.vl_id_range:
            if vl_id in registrar.model_landmarks:
                skip = False
                vl_id_str = 'VL{0:05d}'.format(vl_id)
                prone_mesh_path = self.path_mesh_prone[vl_id]
                supine_mesh_path = self.path_mesh_prone[vl_id]
                if not os.path.exists(supine_mesh_path):
                    skip = True
                if not os.path.exists(prone_mesh_path):
                    skip = True

                if not skip == True:
                    number = 0
                    for landmark in registrar.model_landmarks[vl_id]:
                        if corresponding_landmarks[vl_id][number] == 'True':
                            if position == 'prone':
                                closest_point = self.closest_points_prone[vl_id]
                            else:
                                closest_point = self.closest_points_supine[vl_id]
                            closest_cwm_dist, closest_sm_dist = closest_point.query(landmark) # Calculate closest points to a mesh surface
                            skin.append(closest_sm_dist)
                            rib.append(closest_cwm_dist)
                        number += 1
        return skin, rib
    def transposed(self):
        return self.t_supine1_landmark, self.t_supine2_landmark, self.t_left_nipple, self.t_right_nipple, \
               self.t_cwm_supine, self.t_sm_left_supine, self.t_sm_right_supine, self.t_jl_supine, self.jl_prone

def muscle_attachment_positions(position,volunteer_range):
    left_major = {}
    right_major = {}
    left_minor = {}
    right_minor = {}

    path = '/hpc/ilau122/opt/picker/points/bone_movement/'
    for vl_id in volunteer_range:
        major_left = []
        major_right = []
        minor_right = []
        minor_left = []
        vl_id_str = 'VL{0:05d}'.format(vl_id)
        movement_path = os.path.join(path, vl_id_str)
        if os.path.exists(movement_path):
            # Load in muscle attachment positions
            left_major_path = os.path.join(movement_path,'major_left.json')
            right_major_path = os.path.join(movement_path,'major_right.json')
            left_minor_path = os.path.join(movement_path,'minor_left.json')
            right_minor_path = os.path.join(movement_path,'minor_right.json')
            if os.path.isfile(left_major_path):
                left_major_position = json.loads(open(left_major_path).read())

                # Muscle attachment positions
                major_left.append(left_major_position[position +'_point']['point']['x'])
                major_left.append(left_major_position[position +'_point']['point']['y'])
                major_left.append(left_major_position[position +'_point']['point']['z'])

                with open('vl_t2_image_properties.json') as json_data:
                    img_prop = json.load(json_data)
                try:
                    img_params = img_prop[vl_id_str][position]
                except:
                    pass
                else:
                    x0 = img_params['image_position']
                    sp = img_params['pixel_spacing']
                    shape = img_params['shape']
                    pt = left_major_position[position +'_point']['point']
                    x = np.array([pt['y'] - x0[1],
                            -pt['x'] + x0[0] + sp[0] * shape[0],
                            pt['z'] - x0[2]])
                    left_major[vl_id] = x
            if os.path.isfile(right_major_path):
                right_major_position = json.loads(open(right_major_path).read())

                major_right.append(right_major_position[position +'_point']['point']['x'])
                major_right.append(right_major_position[position +'_point']['point']['y'])
                major_right.append(right_major_position[position +'_point']['point']['z'])

                with open('vl_t2_image_properties.json') as json_data:
                    img_prop = json.load(json_data)
                try:
                    img_params = img_prop[vl_id_str][position]
                except:
                    pass
                else:
                    pt = right_major_position[position +'_point']['point']
                    x = np.array([pt['y'] - x0[1],
                            -pt['x'] + x0[0] + sp[0] * shape[0],
                            pt['z'] - x0[2]])
                    right_major[vl_id] = x
            if os.path.isfile(left_minor_path):
                left_minor_position = json.loads(open(left_minor_path).read())
                minor_left.append(left_minor_position[position +'_point']['point']['x'])
                minor_left.append(left_minor_position[position +'_point']['point']['y'])
                minor_left.append(left_minor_position[position +'_point']['point']['z'])

                with open('vl_t2_image_properties.json') as json_data:
                    img_prop = json.load(json_data)
                try:
                    img_params = img_prop[vl_id_str][position]
                except:
                    pass
                else:
                    pt = left_minor_position[position +'_point']['point']
                    x = np.array([pt['y'] - x0[1],
                            -pt['x'] + x0[0] + sp[0] * shape[0],
                            pt['z'] - x0[2]])
                    left_minor[vl_id] = x
            if os.path.isfile(right_minor_path):
                right_minor_position = json.loads(open(right_minor_path).read())

                minor_right.append(right_minor_position[position +'_point']['point']['x'])
                minor_right.append(right_minor_position[position +'_point']['point']['y'])
                minor_right.append(right_minor_position[position +'_point']['point']['z'])

                with open('vl_t2_image_properties.json') as json_data:
                    img_prop = json.load(json_data)
                try:
                    img_params = img_prop[vl_id_str][position]
                except:
                    pass
                else:
                    pt = right_minor_position[position +'_point']['point']
                    x = np.array([pt['y'] - x0[1],
                            -pt['x'] + x0[0] + sp[0] * shape[0],
                            pt['z'] - x0[2]])
                    right_minor[vl_id] = x
    return left_major, right_major, left_minor, right_minor

class Muscle_movement:
    def __init__(self, rotation):
        # In this function the
        self.volunteer_range = [38] #[38,50, 56, 62, 74, 76, 77, 83]
        self.prone_left_major, self.prone_right_major, self.prone_left_minor, self.prone_right_minor = \
        muscle_attachment_positions('prone', self.volunteer_range)
        self.supine_left_major, self.supine_right_major, self.supine_left_minor, self.supine_right_minor = \
        muscle_attachment_positions('supine', self.volunteer_range)
        self.t_supine_left_major = {}
        self.t_supine_right_major = {}
        self.t_supine_left_minor = {}
        self.t_supine_right_minor = {}

        for vl_id in self.supine_left_major:
            [self.t_supine_left_major[vl_id], self.t_supine_right_major[vl_id]] = icp.transformRigid3D(
                np.array([self.supine_left_major[vl_id], self.supine_right_major[vl_id]]), scipy.array(rotation[vl_id]))

        for vl_id in self.supine_left_minor:
            [self.t_supine_left_minor[vl_id], self.t_supine_right_minor[vl_id]] = icp.transformRigid3D(
                np.array([self.supine_left_minor[vl_id],self.supine_right_minor[vl_id]]) , scipy.array(rotation[vl_id]))

    def minor_movement(self):
        self.movement_minor = {}
        self.direction_minor = {}
        for vl_id in self.volunteer_range:
            # Calculate magnitude of the displacement vector from prone to supine where te minor is attached
            if vl_id in self.prone_left_minor:
                movement_left = np.sqrt((self.t_supine_left_minor[vl_id][0] - self.prone_left_minor[vl_id][0])**2 +
                               (self.t_supine_left_minor[vl_id][1] - self.prone_left_minor[vl_id][1])**2 +
                               (self.t_supine_left_minor[vl_id][2] - self.prone_left_minor[vl_id][2])**2)
                movement_right = np.sqrt((self.t_supine_right_minor[vl_id][0] - self.prone_right_minor[vl_id][0])**2 +
                               (self.t_supine_right_minor[vl_id][1] - self.prone_right_minor[vl_id][1])**2 +
                               (self.t_supine_right_minor[vl_id][2] - self.prone_right_minor[vl_id][2])**2)

                # Calculate displacement vector from prone to supine
                vector_left = [(self.t_supine_left_minor[vl_id][0] - self.prone_left_minor[vl_id][0]),
                               (self.t_supine_left_minor[vl_id][1] - self.prone_left_minor[vl_id][1]),
                               (self.t_supine_left_minor[vl_id][2] - self.prone_left_minor[vl_id][2])]
                vector_right = [(self.t_supine_right_minor[vl_id][0] - self.prone_right_minor[vl_id][0]),
                               (self.t_supine_right_minor[vl_id][1] - self.prone_right_minor[vl_id][1]),
                               (self.t_supine_right_minor[vl_id][2] - self.prone_right_minor[vl_id][2])]

                # Store data
                self.movement_minor[vl_id] = [movement_left, movement_right]
                self.direction_minor[vl_id] = [vector_left, vector_right]

    def major_movement(self):
        self.movement_major = {}
        self.direction_major = {}
        for vl_id in self.volunteer_range:
            if vl_id in self.prone_left_major:
                # Calculate magnitude of the displacement from prone vector to supine for the groove where the major is attached
                movement_left = np.sqrt((self.t_supine_left_major[vl_id][0] - self.prone_left_major[vl_id][0])**2 +
                               (self.t_supine_left_major[vl_id][1] - self.prone_left_major[vl_id][1])**2 +
                               (self.t_supine_left_major[vl_id][2] - self.prone_left_major[vl_id][2])**2)
                movement_right = np.sqrt((self.t_supine_right_major[vl_id][0] - self.prone_right_major[vl_id][0])**2 +
                               (self.t_supine_right_major[vl_id][1] - self.prone_right_major[vl_id][1])**2 +
                               (self.t_supine_right_major[vl_id][2] - self.prone_right_major[vl_id][2])**2)

                # Calculate displacement vector from prone to supine
                vector_left = [(self.t_supine_left_major[vl_id][0] - self.prone_left_major[vl_id][0]),
                               (self.t_supine_left_major[vl_id][1] - self.prone_left_major[vl_id][1]),
                               (self.t_supine_left_major[vl_id][2] - self.prone_left_major[vl_id][2])]
                vector_right = [(self.t_supine_right_major[vl_id][0] - self.prone_right_major[vl_id][0]),
                               (self.t_supine_right_major[vl_id][1] - self.prone_right_major[vl_id][1]),
                               (self.t_supine_right_major[vl_id][2] - self.prone_right_major[vl_id][2])]

                # Store data
                self.movement_major[vl_id] = [movement_left, movement_right]
                self.direction_major[vl_id] = [vector_left, vector_right]

    def movement_difference(self):
        self.magnitude_left = {}
        self.magnitude_right = {}
        self.angle_left = {}
        self.angle_right = {}
        for vl_id in self.volunteer_range:
            if vl_id in self.movement_major:
                if vl_id in self.movement_minor:
                    # Difference in magnitude of the displacement
                    self.magnitude_left[vl_id] = self.movement_minor[vl_id][0] - self.movement_major[vl_id][0]
                    self.magnitude_right[vl_id] = self.movement_minor[vl_id][1] - self.movement_major[vl_id][1]

                    # Calculating the dotproduct and lengths of the displacements vectors
                    dot_product_left = 0
                    dot_product_right = 0
                    length_minor_left = 0
                    length_major_left = 0
                    length_minor_right = 0
                    length_major_right = 0

                    for x in range(len(self.direction_minor[vl_id][0])):
                        dot_product_left += self.direction_minor[vl_id][0][x] * self.direction_major[vl_id][0][x]
                        length_minor_left += self.direction_minor[vl_id][0][x]**2
                        length_major_left += self.direction_major[vl_id][0][x]**2
                    for x in range(len(self.direction_minor[vl_id][1])):
                        dot_product_right += self.direction_minor[vl_id][1][x] * self.direction_major[vl_id][1][x]
                        length_minor_right += self.direction_minor[vl_id][1][x]**2
                        length_major_right += self.direction_major[vl_id][1][x]**2
                    length_minor_left = np.sqrt(length_minor_left)
                    length_major_left = np.sqrt(length_major_left)
                    length_minor_right = np.sqrt(length_minor_right)
                    length_major_right = np.sqrt(length_major_right)

                    # Difference in direction of the displacement (angle between displacement vectors)
                    self.angle_left[vl_id] = math.acos(dot_product_left/(length_minor_left*length_major_left))
                    self.angle_right[vl_id] = math.acos(dot_product_right/(length_minor_right*length_major_right))

    def plotting(self):
        volunteers = []
        minor = []
        major = []
        angle = []
        for vl_id in self.volunteer_range:
            if vl_id in self.movement_minor:
                if vl_id in self.movement_major:
                    volunteers.append(vl_id)
                    volunteers.append(vl_id)
                    minor.append(self.movement_minor[vl_id][0])
                    minor.append(self.movement_minor[vl_id][1])
                    major.append(self.movement_major[vl_id][0])
                    major.append(self.movement_major[vl_id][1])
                    angle.append(self.angle_left[vl_id]/math.pi*180)
                    angle.append(self.angle_right[vl_id]/math.pi*180)

        plt.figure(1)
        plt.scatter(volunteers, minor, color='red', s=20)
        plt.ylabel('Coracoid process movement [mm]')
        plt.xlabel('Volunteer number')
        plt.title('Coracoid process movement')
        axes = plt.gca()
        axes.set_ylim([0, 80])
        plt.tight_layout()

        plt.figure(2)
        plt.scatter(volunteers, major, color='red', s=20)
        plt.ylabel('Pectoralis major groove movement [mm]')
        plt.xlabel('Volunteer number')
        plt.title('Pectoralis major groove movement')
        axes = plt.gca()
        axes.set_ylim([0, 80])
        plt.tight_layout()

        plt.figure(3)
        plt.scatter(volunteers, angle, color='red', s=20)
        plt.ylabel('Difference in direction of movement between coracoid process and pectoralis major groove [degree]')
        plt.xlabel('Volunteer number')
        plt.title('Difference in direction of movement between coracoid process and pectoralis major groove')
        plt.show()



def find_corresponding_landmarks(prone_landmarks, supine_landmarks,
                                 prone_metadata, supine_metadata):
    # Check for which landmarks all the necessary information is available
    corresponding ={}
    for vl_id in range(90):
        use =[]
        if vl_id in prone_landmarks.model_landmarks:
            if vl_id in supine_landmarks.model_landmarks:
                for landmark in prone_landmarks.model_landmarks[vl_id]:
                    if vl_id in prone_metadata.sternal_notches:
                        if vl_id in supine_metadata.sternal_notches:
                            use.append('True')
                        else: use.append('False')
                    else: use.append('False')
        corresponding[vl_id]=use
    return corresponding

def find_corresponding_volunteers(prone_landmarks, supine_landmarks,
                                 prone_metadata, supine_metadata):
    # Checks for which volunteers all the necessary information is available
    corresponding =[]
    for vl_id in range(90):
        if vl_id in prone_landmarks.model_landmarks:
            if vl_id in supine_landmarks.model_landmarks:
                if vl_id in prone_metadata.sternal_notches:
                    if vl_id in supine_metadata.sternal_notches:
                        corresponding.append(vl_id)
    return corresponding


class modeled_landmarks:
    def __init__(self, registrar_prone, t_supine_landmark, metadata_prone, t_left_nipple, t_right_nipple, prone_model_path, supine_model_path):
        self.prone_errors = {}
        self.prone_mesh_landmarks = {}
        self.supine_errors = {}
        self.y_supine_errors = {}
        self.supine_mesh_landmarks = {}
        self.nipple_errors = {}
        self.supine_mesh_nipples ={}
        self.types = {}

        self.volunteers = [38, 50, 56, 62, 74, 76, 77, 83]
        for vl_id in self.volunteers: #range(90):
            coordinate_prone = []
            displacement_prone = []

            vl_id_str = 'VL{0:05d}'.format(vl_id)
            prone_mesh_path = os.path.join(prone_model_path, 'mechanics_meshes', vl_id_str)
            if os.path.exists(prone_mesh_path):
                prone_m = bmw.load_volume_mesh(prone_mesh_path, 'prone')

                points, xi, elements = bmw.generate_points_in_elements(prone_m, num_points=15)

                tree = cKDTree(points)
                landmarks_prone = registrar_prone.model_landmarks[vl_id]
                landmarks_supine = t_supine_landmark[vl_id]
                left_nipple_prone = metadata_prone.left_nipples[vl_id]
                right_nipple_prone = metadata_prone.right_nipples[vl_id]
                left_nipple_supine = t_left_nipple[vl_id]
                right_nipple_supine = t_right_nipple[vl_id]

                #Determine nipple location in mesh
                dist_ln, idx_ln = tree.query(left_nipple_prone)
                element_ln = prone_m.elements[int(elements[idx_ln])]
                #left_nipple_in_mesh= element_ln.evaluate(xi[idx_ln])

                dist_rn, idx_rn = tree.query(right_nipple_prone)
                element_rn = prone_m.elements[int(elements[idx_rn])]
                #right_nipple_in_mesh= element_rn.evaluate(xi[idx_rn])


                for x in range(len(landmarks_prone)):
                    type = registrar_prone.landmark_types[vl_id][x]
                    breast = ''
                    landmark_prone = np.array(landmarks_prone[x])
                    if landmark_prone[1] > prone_metadata.sternal_notches[vl_id][1]:
                        breast = 'right'
                    else: breast = 'left'
                    dist, idx = tree.query(landmark_prone)
                    element = prone_m.elements[int(elements[idx])]
                    landmark_in_mesh= element.evaluate(xi[idx])
                    coordinate_prone.append(np.array([landmark_in_mesh]))
                    displacement_prone.append([dist])

                    landmark_supine = np.array(landmarks_supine[x])

                    coordinate_supine = []
                    displacement_supine = []
                    y_displacement_supine = []
                    supine_mesh_path = os.path.join(supine_model_path, 'mechanics_meshes', vl_id_str)
                    for par in range(4):
                        parameter_set = 'prone_parameter_set_' + str(par) # + '.mesh'
                        file = supine_mesh_path + '/' + parameter_set +'.mesh'
                        if os.path.isfile(file):
                            supine_m = bmw.load_volume_mesh(supine_mesh_path, parameter_set)
                            points_s, xi_s, elements_s = bmw.generate_points_in_elements(supine_m, num_points=15)
                            tree_s = cKDTree(points_s)

                            # dist_ln_supine, idx_ln_supine = tree_s.query(left_nipple_supine)
                            # element_ln_supine = supine_m.elements[int(elements[idx_ln_supine])]
                            # left_nipple_in_mesh= element_ln_supine.evaluate(xi[idx_ln_supine])
                            #
                            # dist_rn_supine, idx_rn_supine = tree_s.query(right_nipple_supine)
                            # element_rn_supine = supine_m.elements[int(elements[idx_rn_supine])]
                            # right_nipple_in_mesh= element_rn_supine.evaluate(xi[idx_rn_supine])

                            element_ln_supine = supine_m.elements[int(elements[idx_ln])]
                            left_nipple_in_mesh_supine = element_ln_supine.evaluate(xi[idx_ln])

                            element_rn_supine = supine_m.elements[int(elements[idx_rn])]
                            right_nipple_in_mesh_supine = element_rn_supine.evaluate(xi[idx_rn])


                            dist_ln = np.sqrt((left_nipple_supine[0] - left_nipple_in_mesh_supine[0])**2 +
                                    (left_nipple_supine[1] - left_nipple_in_mesh_supine[1])**2 +
                                    (left_nipple_supine[2] - left_nipple_in_mesh_supine[2])**2)
                            overpredict_ln = (left_nipple_supine[1] - left_nipple_in_mesh_supine[1])
                            if (left_nipple_supine[1] - left_nipple_in_mesh_supine[1]) < 0:
                                dist_ln = -dist_ln

                            dist_rn = np.sqrt((right_nipple_supine[0] - right_nipple_in_mesh_supine[0])**2 +
                                    (right_nipple_supine[1] - right_nipple_in_mesh_supine[1])**2 +
                                    (right_nipple_supine[2] - right_nipple_in_mesh_supine[2])**2)
                            over_predict_rn = (right_nipple_supine[1] - right_nipple_in_mesh_supine[1])
                            if (right_nipple_supine[1] - right_nipple_in_mesh_supine[1]) > 0:
                                dist_rn = -dist_rn
                            vol_par = str(vl_id) + '0' + str(par)
                            vol_par =int(vol_par)
                            self.nipple_errors[vol_par] = [dist_ln, dist_rn]
                            self.supine_mesh_nipples[vol_par] = [left_nipple_in_mesh_supine, right_nipple_in_mesh_supine]

                            element = supine_m.elements[int(elements[idx])]
                            landmark_in_mesh = element.evaluate(xi[idx])
                            dist = np.sqrt((landmark_in_mesh[0] - landmark_supine[0])**2 +
                                    (landmark_in_mesh[1] - landmark_supine[1])**2 +
                                    (landmark_in_mesh[2] - landmark_supine[2])**2)
                            y_dist = landmark_in_mesh[1] - landmark_supine[1]
                            if breast == 'right':
                                if landmark_in_mesh[1] - landmark_supine[1] > 0:
                                    dist = -dist
                                    y_dist = -y_dist
                            elif breast == 'left':
                                if landmark_in_mesh[1] - landmark_supine[1] < 0:
                                    dist = -dist
                                    y_dist = -y_dist
                            else: dist = 'error'

                            coordinate_supine.append(np.array([landmark_in_mesh]))
                            displacement_supine.append([dist])
                            y_displacement_supine.append([y_dist])
                            volunteer_landmark = str(vl_id) + '0' + str(x+1)
                            volunteer_landmark =int(volunteer_landmark)
                        volunteer_landmark = str(vl_id) + '0' + str(x+1)
                        volunteer_landmark =int(volunteer_landmark)
                        self.supine_errors[volunteer_landmark] = displacement_supine
                        self.supine_mesh_landmarks[volunteer_landmark] = coordinate_supine
                        self.types[volunteer_landmark] = type
                        self.y_supine_errors[volunteer_landmark] = y_displacement_supine

                self.prone_errors[vl_id] = displacement_prone
                self.prone_mesh_landmarks[vl_id] = coordinate_prone

    def determine_errors(self):
        return self.prone_mesh_landmarks, self.prone_errors, self.supine_mesh_landmarks, self.supine_errors, self.supine_mesh_nipples, self.nipple_errors
    def plot_nipples(self):
        stiffnesses = [0.3, 0.2925, 0.285, 0.2775]
        stiffness=[]
        error = []
        for vl_id in self.volunteers:
            for par in range(4):
                vol_par = str(vl_id) + '0' + str(par)
                vol_par =int(vol_par)
                if vol_par in self.nipple_errors:
                    if self.nipple_errors[vol_par][0] < 0: self.nipple_errors[vol_par][0]= -self.nipple_errors[vol_par][0]
                    if self.nipple_errors[vol_par][1] < 0: self.nipple_errors[vol_par][1]= -self.nipple_errors[vol_par][1]
                    error.append(self.nipple_errors[vol_par][0])
                    error.append(self.nipple_errors[vol_par][1])

                    stiffness.append(stiffnesses[par])
                    stiffness.append(stiffnesses[par])
        plt.figure()
        plt.scatter(stiffness, error, color='red', s=20)
        plt.ylabel('Error in nipple position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Error in nipple position')

    def plot_landmarks(self, registrar_prone):
        stiffnesses = [0.3, 0.2925, 0.285, 0.2775]
        stiffness_cyst = []
        stiffness_lymph_node = []
        stiffness_other =[]
        cyst = []
        lymph_node =[]
        other= []
        for vl_id in self.volunteers:
            landmarks_prone = registrar_prone.model_landmarks[vl_id]
            for x in range(len(landmarks_prone)):
                volunteer_landmark = str(vl_id) + '0' + str(x+1)
                volunteer_landmark =int(volunteer_landmark)
                if self.types[volunteer_landmark] == 'cyst':
                    for landmark in range(len(self.supine_errors[volunteer_landmark])):
                        cyst.append(self.supine_errors[volunteer_landmark][landmark])
                        stiffness_cyst.append(stiffnesses[landmark])
                if self.types[volunteer_landmark] == 'lymph node':
                    for landmark in range(len(self.supine_errors[volunteer_landmark])):
                        lymph_node.append(self.supine_errors[volunteer_landmark][landmark])
                        stiffness_lymph_node.append(stiffnesses[landmark])
                if self.types[volunteer_landmark] == 'other':
                    for landmark in range(len(self.supine_errors[volunteer_landmark])):
                        other.append(self.supine_errors[volunteer_landmark][landmark])
                        stiffness_other.append(stiffnesses[landmark])

        plt.figure(1)
        plt.scatter(stiffness_cyst, cyst, color='red', s=20)
        plt.ylabel('Error in cyst position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Error in cyst position')

        plt.figure(2)
        plt.scatter(stiffness_lymph_node, lymph_node, color='red', s=20)
        plt.ylabel('Error in lymph node position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Error in lymph node position')

        plt.figure(3)
        plt.scatter(stiffness_other, other, color='red', s=20)
        plt.ylabel('Error in undefined landmark position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Error in undefined landmark position')
        plt.show()

    def y_plot_landmarks(self, registrar_prone):
        stiffnesses = [0.3, 0.2925, 0.285, 0.2775]
        stiffness_cyst = []
        stiffness_lymph_node = []
        stiffness_other =[]
        cyst = []
        lymph_node =[]
        other= []
        for vl_id in self.volunteers:
            landmarks_prone = registrar_prone.model_landmarks[vl_id]
            for x in range(len(landmarks_prone)):
                volunteer_landmark = str(vl_id) + '0' + str(x+1)
                volunteer_landmark =int(volunteer_landmark)
                if self.types[volunteer_landmark] == 'cyst':
                    for landmark in range(len(self.y_supine_errors[volunteer_landmark])):
                        cyst.append(self.y_supine_errors[volunteer_landmark][landmark])
                        stiffness_cyst.append(stiffnesses[landmark])
                if self.types[volunteer_landmark] == 'lymph node':
                    for landmark in range(len(self.y_supine_errors[volunteer_landmark])):
                        lymph_node.append(self.y_supine_errors[volunteer_landmark][landmark])
                        stiffness_lymph_node.append(stiffnesses[landmark])
                if self.types[volunteer_landmark] == 'other':
                    for landmark in range(len(self.y_supine_errors[volunteer_landmark])):
                        other.append(self.y_supine_errors[volunteer_landmark][landmark])
                        stiffness_other.append(stiffnesses[landmark])

        plt.figure(1)
        plt.scatter(stiffness_cyst, cyst, color='red', s=20)
        plt.ylabel('Y Error in cyst position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Y Error in cyst position')

        plt.figure(2)
        plt.scatter(stiffness_lymph_node, lymph_node, color='red', s=20)
        plt.ylabel('Y Error in lymph node position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Y Error in lymph node position')

        plt.figure(3)
        plt.scatter(stiffness_other, other, color='red', s=20)
        plt.ylabel('Y Error in undefined landmark position [mm]')
        plt.xlabel('Stiffness [kPa]')
        plt.title('Y Error in undefined landmark position')
        plt.show()

    def plot_nipples_in_mesh(self):
        for vl_id in self.volunteers:
            for par in range(4):
                vl_id_str = 'VL{0:05d}'.format(vl_id)
                supine_model_path = '/hpc_ntot/psam012/opt/breast-modelling/data/prone_to_supine_t2/2018-05-24'
                supine_mesh_path = os.path.join(supine_model_path, 'mechanics_meshes', vl_id_str)
                parameter_set = 'prone_parameter_set_' + str(par) # + '.mesh'
                file = supine_mesh_path + '/' + parameter_set +'.mesh'
                if os.path.isfile(file):
                    vol_par = str(vl_id) + '0' + str(par)
                    vol_par =int(vol_par)
                    # Load mechanical model mesh
                    supine_mesh_path = os.path.join(supine_model_path, 'mechanics_meshes', vl_id_str)
                    supine_m = bmw.load_volume_mesh(supine_mesh_path, parameter_set)
                    from morphic import viewer
                    supine_nipple_fig = bmw.add_fig(viewer, label='supine_nipples')
                    bmw.visualise_mesh(supine_m, supine_nipple_fig, visualise=True, face_colours=(1, 0, 0))
                    bmw.plot_points(supine_nipple_fig, 'supine_left_nipple',
                                    t_left_nipple[vl_id], [1],
                                    visualise=True, colours=(0,1,0), point_size=3,
                                    text_size=5)
                    bmw.plot_points(supine_nipple_fig, 'supine_right_nipple',
                                    t_right_nipple[vl_id], [1],
                                    visualise=True, colours=(0,1,0), point_size=3,
                                    text_size=5)
                    bmw.plot_points(supine_nipple_fig, 'supine_nipple_model',
                                    supine_mesh_nipples[vol_par], [2],
                                    visualise=True, colours=(0,0,1), point_size=3,
                                    text_size=5)
                    supine_nipple_fig.clear()


    def plot_skin_mesh(self, rotation_prone_supine):
       for vl_id in self.volunteers:
            for par in range(4):
                vl_id_str = 'VL{0:05d}'.format(vl_id)
                supine_model_path = '/hpc_ntot/psam012/opt/breast-modelling/data/prone_to_supine_t2/2018-05-24'
                supine_mesh_path = os.path.join(supine_model_path, 'mechanics_meshes', vl_id_str)
                parameter_set = 'prone_parameter_set_' + str(par) # + '.mesh'
                file = supine_mesh_path + '/' + parameter_set +'.mesh'
                if os.path.isfile(file):
                    # Load mechanical model mesh
                    supine_mesh_path = os.path.join(supine_model_path, 'mechanics_meshes', vl_id_str)
                    supine_m = bmw.load_volume_mesh(supine_mesh_path, parameter_set)
                    supine_m_nodes = supine_m.get_node_ids()[0]

                    # Preform rigid transformation on mechanical mesh
                    t_supine_nodes = icp.transformRigid3D(supine_m_nodes, scipy.array(rotation_prone_supine[vl_id]))
                    #t_supine_m = utils.convert_hermite_lagrange(supine_m, tol=1e-9)
                    supine_m.label = 't_supine'
                    for node_idx, node_id in enumerate(supine_m.get_node_ids()[1]):
                        supine_m.nodes[node_id].values = t_supine_nodes[node_idx, :]

                    # Load MRI supine mesh
                    supine_model_path_MRI = '/hpc_ntot/psam012/opt/breast-modelling/data/supine_to_prone_t2/2017-07-31/'
                    supine_mesh_path_MRI = os.path.join(supine_model_path_MRI, 'volunteer_meshes',  vl_id_str)
                    supine_sm_left = bmw.load_skin_surface_mesh(supine_mesh_path_MRI, 'supine','left')
                    supine_sm_right = bmw.load_skin_surface_mesh(supine_mesh_path_MRI, 'supine','right')

                    # Plot figure
                    from morphic import viewer
                    supine_fig = bmw.add_fig(viewer, label='supine_meshes')
                    bmw.visualise_mesh(supine_m, supine_fig, visualise=True, face_colours=(0, 1, 0))
                    bmw.visualise_mesh(supine_sm_left, supine_fig, visualise=True, face_colours=(1, 0, 0))
                    bmw.visualise_mesh(supine_sm_right, supine_fig, visualise=True, face_colours=(1, 0, 0))
                    supine_fig.clear()


class Jugular_Landmark_Distances:
    def __init__(self, prone_data_path, supine_data_path, rotation,
                 jugular_landmarks_prone, jugular_landmarks_supine,
                 t_jl_supine):
        # Make empty arrays and dictionairies to store data in
        self.vol = []
        self.distance_sp1 = []
        self.distance_sp2 = []
        self.distance_sp_s1 = []
        self.distance_sp_s2 = []
        self.distance_pp = []
        self.diff_dist = []
        self.volun = []

        r = range(1, 90)
        self.movement_alignment = {} # Displacement of the jugular landmarks due to alignment
        self.d = {}  # Distance between the jugular landmarks in the prone and supine position before alignment
        self.t_d = {}  # Distance between the jugular landmarks in the prone and supine position after alignment
        self.prone_d = {}  # Distance between the jungular landmarks in the prone position
        self.supine_d = {}
        self.s_d = {}  # Scaled distance between the jugular landmarks in the prone and supine position before alignment
        self.s_t_d = {}  # Scaled distance between the jugular landmarks in the prone and supine position after alignment
        self.length = {}  # Different is distance bewteen the two different sternal landmarks in the prone and supine position

        # Create input for the attributes of the HDF5 files
        unit_s = "The data is dimensionless [-]"
        unit = "The data is expressed in millimeters [mm]"
        source = prone_data_path, ' and ', supine_data_path

        # Create HDF5 file and folders and add attributes
        jugular_landmarks = h5py.File('Jugular_Landmarks', 'w')

        data_rotation = jugular_landmarks.create_group('Data/Rotation')

        data_displacement_alignment = jugular_landmarks.create_group('Data/Displacement_due_to_alignment')
        data_displacement_alignment.attrs['Source'] = supine_data_path
        data_displacement_alignment.attrs['Unit'] = unit
        data_displacement_alignment.attrs['Description'] = "The displacement of the prone jugular landmarks due to alignment"

        data_ba_s = jugular_landmarks.create_group(
            'Data/Dist_prone_supine/Dist_before_alignment/Scaled')
        data_ba_s.attrs['Source'] = source
        data_ba_s.attrs['Unit'] = unit_s
        data_ba_s.attrs[
            'Description'] = "The distance between to corresponding jugular landmarks in the original prone and supine positions, which is scaled of the distance between the two different jugular landmarks in the prone position."

        data_ba = jugular_landmarks.create_group(
            'Data/Dist_prone_supine/Dist_before_alignment/Not_scaled')
        data_ba.attrs['Source'] = source
        data_ba.attrs['Unit'] = unit
        data_ba.attrs[
            'Description'] = "The distance between to corresponding jugular landmarks in the original prone and supine positions per volunteer."

        data_aa_s = jugular_landmarks.create_group(
            'Data/Dist_prone_supine/Dist_after_alignment/Scaled')
        data_aa_s['Source'] = source
        data_aa_s.attrs['Unit'] = unit_s
        data_aa_s.attrs[
            'Description'] = "The distance between to corresponding jugular landmarks in the prone and transposed supine-to-prone positions per volunteer, which are scaled over the distance between the two different jugular landmarks in the prone position."

        data_aa = jugular_landmarks.create_group(
            'Data/Dist_prone_supine/Dist_after_alignment/Not_scaled')
        data_aa['Source'] = source
        data_aa.attrs['Unit'] = unit
        data_aa.attrs[
            'Description'] = "The distance between to corresponding jugular landmarks in the prone and transposed supine-to-prone positions per volunteer."

        data_prone = jugular_landmarks.create_group('Data/Dist_prone')
        data_prone['Source'] = prone_data_path
        data_prone.attrs['Unit'] = unit
        data_prone.attrs[
            'Description'] = "The distance between the two different jugular landmarks in the prone position per volunteer"

        data_transformed = jugular_landmarks.create_group(
            'Data/Position_after_supine_to_prone_alignment')
        data_transformed['Source'] = source
        data_transformed.attrs['Unit'] = unit
        data_transformed.attrs[
            'Description'] = "The positions of the jugular landmarks after transformation from the supine to prone position"

        # Calculate distances between jugular landmarks and add data to HDF5 file
        for x in r:

            if x in t_jl_supine:
                volunteer = 'Volunteer ' + str(x)
                self.vol.append(x)
                self.volun.append(volunteer)

                data_rotation.create_dataset(volunteer, data=rotation[x])

                # Distance between the jungular landmarks in the prone position
                prone_lm = np.asarray(
                    jugular_landmarks_prone.model_landmarks[x])
                dist_prone = np.linalg.norm(prone_lm[1:2] - prone_lm[0:1])
                self.prone_d[x] = dist_prone

                self.distance_pp.append(dist_prone)

                data_prone.create_dataset(volunteer, data=self.prone_d[x])

                # Distance between the jungular landmarks in the supine position
                supine_lm = np.asarray(
                    jugular_landmarks_supine.model_landmarks[x])
                # dist_supine= np.linalg.norm(supine_lm[1:2]-supine_lm[0:1])
                dist_supine = np.sqrt(
                    (supine_lm[0][0] - supine_lm[1][0]) ** 2 + (
                    supine_lm[0][1] - supine_lm[1][1]) ** 2 + (
                    supine_lm[0][2] - supine_lm[1][2]) ** 2)
                self.supine_d[x] = dist_supine

                # Difference in distance between the jugular landmarks in the supine and prone position
                dist_sternum = dist_prone - dist_supine
                self.length[x] = dist_sternum
                self.diff_dist.append(dist_sternum)

                # Distance between the jugular landmarks in the prone and supine position before alignment
                supine_lm = np.asarray(
                    jugular_landmarks_supine.model_landmarks[x])
                prone_lm_ba = np.asarray(
                    jugular_landmarks_prone.model_landmarks[x])
                dist_lm_1 = np.linalg.norm(supine_lm[0:1] - prone_lm_ba[0:1])
                dist_lm_2 = np.linalg.norm(supine_lm[1:2] - prone_lm_ba[1:2])
                self.d[x] = [dist_lm_1, dist_lm_2]
                self.s_d[x] = [dist_lm_1 / self.prone_d[x],
                               dist_lm_2 / self.prone_d[x]]


                data_ba_s.create_dataset(volunteer, data=self.s_d[x])
                data_ba.create_dataset(volunteer, data=self.d[x])

                # Distance between the jugular landmarks in the prone and supine position after alignment
                t_supine_lm = np.asarray(t_jl_supine[x])
                dist_lt_1 = np.linalg.norm(
                    t_supine_lm[0:1] - prone_lm[0:1])
                dist_lt_2 = np.linalg.norm(
                    t_supine_lm[1:2] - prone_lm[1:2])
                self.t_d[x] = [dist_lt_1, dist_lt_2]
                self.s_t_d[x] = [dist_lt_1 / self.prone_d[x],
                                 dist_lt_2 / self.prone_d[x]]

                # Displacement of the jugular landmarks due to alignment
                dist_lm_1 = np.linalg.norm(supine_lm[0:1]-t_supine_lm[0:1])
                dist_lm_2 = np.linalg.norm((supine_lm[1:2] - t_supine_lm[1:2]))
                self.movement_alignment[x]= [dist_lm_1, dist_lm_2]

                self.distance_sp1.append(dist_lt_1)
                self.distance_sp2.append(dist_lt_2)
                self.distance_sp_s1.append(dist_lt_1 / self.prone_d[x])
                self.distance_sp_s2.append(dist_lt_2 / self.prone_d[x])

                data_transformed.create_dataset(volunteer,
                                                data=t_supine_lm)
                data_aa.create_dataset(volunteer, data=self.t_d[x])
                data_aa_s.create_dataset(volunteer, data=self.s_t_d[x])
                data_displacement_alignment.create_dataset(volunteer, data=self.movement_alignment[x])


        jugular_landmarks.close()

    def plotting(self):
        # Plot data
        # Graph of the distances between different jugular landmarks in the prone position
        plt.figure(1)
        plt.scatter(self.vol, self.distance_pp, color='black', s=20)
        plt.ylabel('Distance [mm]')
        plt.xlabel('Volunteer')
        plt.title('Distances between different jugular landmarks')
        axes = plt.gca()
        axes.set_ylim([0, 60])

        # Graph of the scaled distances between corresponding jugular landmarks after transformation
        plt.figure(2)
        plt.scatter(self.vol, self.distance_sp_s1, color='red', s=20,
                    label='First jugular landmarks')
        plt.scatter(self.vol, self.distance_sp_s2, color='blue', s=20,
                    label='Second jugular landmarks')
        plt.ylabel('Distance [-]')
        plt.xlabel('Volunteer')
        plt.title(
            'Scaled distances between corresponding jugular landmarks after alignment')
        axes = plt.gca()
        axes.set_ylim([0, 16])
        plt.legend(numpoints=1)

        # Graph of the distances between corresponding jugular landmarks after transformation
        plt.figure(3)
        plt.scatter(self.vol, self.distance_sp1, color='red', s=20,
                    label='First jugular landmarks')
        plt.scatter(self.vol, self.distance_sp2, color='blue', s=20,
                    label='Second jugular landmarks')

        plt.ylabel('Distance [mm]')
        plt.xlabel('Volunteer')
        plt.title(
            'Distances between corresponding jugular landmarks after alignment')
        plt.legend(numpoints=1)

        # Graph of the distances between corresponding jugular landmarks after transformation without outliner (Volunteer 24)
        plt.figure(4)
        plt.scatter(self.vol, self.distance_sp_s1, color='red', s=20,
                    label='First jugular landmarks')
        plt.scatter(self.vol, self.distance_sp_s2, color='blue', s=20,
                    label='Second jugular landmarks')
        plt.ylabel('Distance [-]')
        plt.xlabel('Volunteer')
        plt.title(
            'Scaled distances between corresponding jugular landmarks after alignment')
        axes = plt.gca()
        axes.set_ylim([0, 1.5])
        plt.tight_layout()
        plt.legend(numpoints=1)

        # Histogram of difference in length between the sternal landmarks
        plt.figure(5)
        n, bins, patches = plt.hist(self.diff_dist)
        plt.ylabel('Frequency')
        plt.xlabel('Difference in length manubrium (prone - supine) [mm]')
        plt.title('Difference in length manubrium (prone - supine) [mm]')
        plt.grid(True)
        plt.show()

# Push data
class Push_Data:
    def __init__(self, write_spreadsheet_id):

        scope = '/docs.google.com/spreadsheets/'
        self.write_spreadsheet_id = '11H4hDDwGQoAz7jZC--jpJCLxDpxnUl_O4qtQd_OyUHk'
        self.write_sheet_name = 'Sheet1'
        self.write_range_name = 'C2:B'
        self.client_secret_path = '/hpc/ilau122/opt/API/'
        self.credentials_path = '/hpc/ilau122/opt/API/'
        self.write_sheets_service = google_apis.get_sheets_service(
            credentials_path=self.credentials_path,
            client_secret_path=self.client_secret_path,
            scopes=scope)

        self.alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z']

        #google_apis.clear_sheet(self.write_sheets_service, self.write_spreadsheet_id, self.write_sheet_name)

    def push_volunteer_number(self, volu_1p, volu_2p):
        rows = [['Volunteer number']]
        #rows = []
        for x in volu_1p:
            rows.append([x])
        for y in volu_2p:
            rows.append([y])

        # Select range of cells to write to
        num_rows = len(rows)
        num_cols = 1
        start_row = 1
        start_col = 3
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_registrars(self, volu_1p, volu_2p):
        rows = [['Registrar']]
        for x in range(len(volu_1p)):
            rows.append([1])
        for y in range(len(volu_2p)):
            rows.append([2])
        num_rows = len(rows)
        num_cols = 1
        start_row = 1
        start_col = 4
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_quadrant(self, registrar1_prone, registrar1_supine,
                      registrar2_prone, registrar2_supine,
                      corresponding_volunteers1, corresponding_volunteers2):
        rows = [['Quadrant (prone)', 'Quadrant (supine)']]
        for x in corresponding_volunteers1:
            for y in range(len(registrar1_prone.quadrants.landmark_quadrants[x])):
                rows.append([registrar1_prone.quadrants.landmark_quadrants[x][y],
                             registrar1_supine.quadrants.landmark_quadrants[x][y]])
        for a in corresponding_volunteers2:
            for b in range(len(registrar2_prone.quadrants.landmark_quadrants[a])):
                rows.append([registrar2_prone.quadrants.landmark_quadrants[a][b],
                             registrar2_supine.quadrants.landmark_quadrants[a][b]])

        num_rows = len(rows)
        num_cols = len(rows[0])
        start_row = 1
        start_col = 5
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_local_landmark_numbers(self, registrar1_prone, registrar2_prone,
                                    corresponding_volunteers1, corresponding_volunteers2):
        rows = [['Local landmark numbers']]
        for x in corresponding_volunteers1:
            for y in range(len(registrar1_prone.quadrants.landmark_quadrants[x])):
                rows.append([y+1])
        for a in corresponding_volunteers2:
            for b in range(len(registrar2_prone.quadrants.landmark_quadrants[a])):
                rows.append([b+1])
        num_rows = len(rows)
        num_cols = 1
        start_row = 1
        start_col = 2
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_volunteer_info(self, pm,
                            corresponding_volunteers1, corresponding_volunteers2,
                            corresponding_landmarks1, corresponding_landmarks2):
        rows = [['Age', 'Height [m]', 'Weight [kg]']]
        for x in corresponding_volunteers1:
            for y in range(len(corresponding_landmarks1[x])):
                    rows.append([pm.ages[x], pm.heights[x], pm.weights[x]])
        for x in corresponding_volunteers2:
            for y in range(len(corresponding_landmarks2[x])):
                if corresponding_landmarks2[x][y] =='True':
                    rows.append([pm.ages[x], pm.heights[x], pm.weights[x]])
        num_rows = len(rows)
        num_cols = len(rows[0])
        start_row = 1
        start_col = 8
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_landmark_type(self, registrar1_prone, registrar2_prone,
                           corresponding_volunteers1, corresponding_volunteers2,
                           corresponding_landmarks1, corresponding_landmarks2):
        rows = [['Landmark type']]
        for x in corresponding_volunteers1:
            for z in range(len(registrar1_prone.landmark_types[x])):
                rows.append([registrar1_prone.landmark_types[x][z]])
        for y in corresponding_volunteers2:
            for w in range(len(registrar2_prone.landmark_types[y])):
                rows.append([registrar2_prone.landmark_types[y][w]])
        num_rows = len(rows)
        num_cols = 1
        start_row = 1
        start_col = 7
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_landmark_displacement(self, landmark_displacement_1, landmark_displacement_2):
        rows = [['Displacement']]
        for x in landmark_displacement_1:
            rows.append([x])
        for y in landmark_displacement_2:
            rows.append([y])
        num_rows = len(rows)
        num_cols = 1
        start_row = 1
        start_col = 11
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_nipple_movement(self, left_nipple_registrar_1, right_nipple_registrar_1,
                             left_nipple_registrar_2, right_nipple_registrar_2):
        rows = [['Left nipple displacement [mm]', 'Right nipple displacement [mm]']]
        for x in range(len(left_nipple_registrar_1)):
            rows.append([left_nipple_registrar_1[x], right_nipple_registrar_1[x]])
        for y in range(len(left_nipple_registrar_2)):
            rows.append(([left_nipple_registrar_2[y], right_nipple_registrar_2[y]]))
        num_rows = len(rows)
        num_cols = len(rows[0])
        start_row = 1
        start_col = 13
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_time(self, time_1p, time_1s, time_2p, time_2s):
        rows = [['Time (prone)', 'Time (supine)']]
        for x in range(len(time_1p)):
            rows.append([time_1p[x],time_1s[x]])
        for y in range(len(time_2p)):
            rows.append([time_2p[y], time_2s[y]])
        num_rows = len(rows)
        num_cols = len(rows[0])
        start_row = 1
        start_col = 15
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_dist_landmark_nipple(self, dist_landmark_nipple_1p, dist_landmark_nipple_1s,
                                  dist_landmark_nipple_2p, dist_landmark_nipple_2s):
        rows = [['Distance to nipple (prone) [mm]', 'Distance to nipple (supine) [mm]']]
        for x in range(len(dist_landmark_nipple_1p)):
            rows.append([dist_landmark_nipple_1p[x], dist_landmark_nipple_1s[x]])
        for y in range(len(dist_landmark_nipple_2p)):
            rows.append([dist_landmark_nipple_2p[y], dist_landmark_nipple_2s[y]])
        num_rows = len(rows)
        num_cols = len(rows[0])
        start_row = 1
        start_col = 17
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_dist_landmark_mesh(self, dist_landmark_cw_1p,dist_landmark_cw_1s,
                                dist_landmark_cw_2p,dist_landmark_cw_2s,
                                dist_landmark_skin_1p, dist_landmark_skin_1s,
                                dist_landmark_skin_2p, dist_landmark_skin_2s):
        rows = [['Distance to skin (prone) [mm]', 'Distance to skin (supine)[mm]', 'Distance to rib cage (prone) [mm]', 'Distance to rib cage (supine) [mm]']]
        for x in range(len(dist_landmark_skin_1p)):
            rows.append([dist_landmark_skin_1p[x], dist_landmark_skin_1s[x],
                         dist_landmark_cw_1p[x], dist_landmark_cw_1s[x]])
        for y in range(len(dist_landmark_skin_2p)):
            rows.append([dist_landmark_skin_2p[y], dist_landmark_skin_2s[y],
                         dist_landmark_cw_2p[y], dist_landmark_cw_2s[y]])
        num_rows = len(rows)
        num_cols = len(rows[0])
        start_row = 1
        start_col = 19
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
            self.write_sheet_name,
            self.alph[start_col-1], start_row,
            self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service, self.write_spreadsheet_id,
                            write_range_name, rows)

    def push_corresponding_landmarks(self, cor_1_2, volu_1p, volu_2p, corresponding_volunteers1):
        rows = [['Corresponding landmarks']]
        for z in range(len(volu_1p)+len(volu_2p)):
            rows.append(['n/a'])
        for x in corresponding_volunteers1:
            if cor_1_2[x] is not []:
                for y in cor_1_2[x]:
                    reg1 = int(volu_1p.index(x))+int(y[0])
                    reg2 = int(len(volu_1p)+volu_2p.index(x)) +int(y[1])
                    rows[reg1] = [y[1]]
                    rows[reg2] = [y[0]]
        num_rows = len(rows)
        num_cols = 1
        start_row = 1
        start_col = 12
        write_range_name = '{0}!{1}{2}:{3}{4}'.format(
                self.write_sheet_name,
                self.alph[start_col-1], start_row,
                self.alph[start_col + num_cols-2], '')
        google_apis.write_rows(self.write_sheets_service,
                               self.write_spreadsheet_id, write_range_name, rows)


if __name__ == '__main__':
    # Define dictionary in which the transformed jugular_landmarks are stored


    rotation = {}

    root_path_MRI = '/home/psam012/opt/breast-data/'
    root_path = '/home/data/'
    prone_data_path = '/home/psam012/opt/breast-data/prone_to_supine_t2/2017_09_06/'
    supine_data_path = '/home/psam012/opt/breast-data/supine_to_prone_t2/2017_06_25/'
    prone_model_path = '/home/psam012/opt/breast-data/prone_to_supine_t2/2017_09_06/'
    supine_model_path = '/home/psam012/opt/breast-data/supine_to_prone_t2/2017-07-31/'


    # root_path = os.environ['BREAST_T2_LANDMARK_ANALYSIS_ROOT']
    # prone_data_path = os.environ['BREAST_PRONE_DATA_PATH']
    # supine_data_path = os.environ['BREAST_SUPINE_DATA_PATH']
    # prone_model_path = os.environ['BREAST_PRONE_MODEL_PATH']
    # supine_model_path = os.environ['BREAST_SUPINE_MODEL_PATH']

    position = 'prone'
    registrar1_prone = Landmarks('Ben', 'user008', position, root_path)
    registrar2_prone = Landmarks('Clarke', 'user007', position, root_path)

    position = 'supine'
    registrar1_supine = Landmarks('Ben', 'user008', position, root_path)
    registrar2_supine = Landmarks('Clarke', 'user007', position, root_path)

    position = 'prone'
    jugular_landmarks_prone = Landmarks('jugular_landmarks', 'user001',
                                        position, root_path)
    position = 'supine'
    jugular_landmarks_supine = Landmarks('jugular_landmarks', 'user001',
                                         position, root_path)

    prone_metadata = Metadata(prone_data_path, 'prone')
    registrar1_prone.find_quadrants(prone_metadata)
    registrar2_prone.find_quadrants(prone_metadata)

    supine_metadata = Metadata(supine_data_path, 'supine')
    registrar1_supine.find_quadrants(supine_metadata)
    registrar2_supine.find_quadrants(supine_metadata)

    # Find corresponding landmarks
    corresponding_landmarks1 = find_corresponding_landmarks(
        registrar1_prone, registrar1_supine,
        prone_metadata, supine_metadata)
    corresponding_landmarks2 = find_corresponding_landmarks(
        registrar2_prone, registrar2_supine,
        prone_metadata, supine_metadata)

    # Find corresponding volunteers
    corresponding_volunteers1 = find_corresponding_volunteers(
        registrar1_prone, registrar1_supine,
        prone_metadata, supine_metadata)
    corresponding_volunteers2 = find_corresponding_volunteers(
        registrar2_prone, registrar2_supine,
        prone_metadata, supine_metadata)


    # Align prone and supine meshes
    aligned_meshes= align_meshes(prone_model_path, supine_model_path,
                                jugular_landmarks_prone, jugular_landmarks_supine,
                                prone_metadata, supine_metadata,
                                root_path_MRI, registrar1_prone, registrar1_supine,
                                skip_vl=[9, 10, 17, 20, 22, 25, 64, 70])


    t_supine1_landmark, t_supine2_landmark, t_left_nipple, t_right_nipple, t_cwm_supine, t_sm_left_supine, t_sm_right_supine, t_jl_supine, jl_prone = aligned_meshes.transposed()


    #Closest distances from landmark to skin and chest wall
    dist_landmark_skin_1p, dist_landmark_cw_1p = aligned_meshes.shortest_distances(
        registrar1_prone, 'prone', corresponding_landmarks1)
    dist_landmark_skin_2p, dist_landmark_cw_2p = aligned_meshes.shortest_distances(
        registrar2_prone, 'prone', corresponding_landmarks2)
    dist_landmark_skin_1s, dist_landmark_cw_1s = aligned_meshes.shortest_distances(
        registrar1_supine, 'supine', corresponding_landmarks1)
    dist_landmark_skin_2s, dist_landmark_cw_2s = aligned_meshes.shortest_distances(
        registrar2_supine, 'supine', corresponding_landmarks2)


    # Landmark positions in time coordinates, distance to the nipple, and volunteer numbers

    time_1p, volu_1p, dist_landmark_nipple_1p = clock(registrar1_prone, prone_metadata, corresponding_landmarks1)
    time_2p, volu_2p, dist_landmark_nipple_2p = clock(registrar2_prone, prone_metadata, corresponding_landmarks2)
    time_1s, volu_1s, dist_landmark_nipple_1s = clock(registrar1_supine, supine_metadata, corresponding_landmarks1)
    time_2s, volu_2s, dist_landmark_nipple_2s = clock(registrar2_supine, supine_metadata, corresponding_landmarks2)

    # Landmark displacements between prone and supine
    landmark_displacement_1, Landmark_Displacement_1 = displacement(
        registrar1_prone, t_supine1_landmark, corresponding_landmarks1)
    landmark_displacement_2, Landmark_Displacement_1 = displacement(
        registrar2_prone, t_supine2_landmark, corresponding_landmarks2)

    # Corresponding landmarks between registrars
    # gives a dictionary and per volunteer the corresponding local landmark numbers are giving in a list.
    # The first number on cor_1_2 reffers to the the local landmark number as identified by the first registrar and the second by
    # by the second registrar.
    cor_1_2 = corresponding_landmarks_between_registars(
        registrar1_prone, registrar2_prone,
        registrar1_supine, registrar2_supine,
        corresponding_landmarks1, corresponding_landmarks2)
    #cor_2_1 = corresponding_landmarks_between_registars(registrar2_prone, registrar1_prone, corresponding_landmarks2, corresponding_landmarks1)

    # Nipple movement from prone to supine
    left_nipple_registrar_1, right_nipple_registrar_1 = displacement_nipple(
        prone_metadata, t_left_nipple, t_right_nipple, registrar1_prone, corresponding_landmarks1)
    left_nipple_registrar_2, right_nipple_registrar_2 = displacement_nipple(
        prone_metadata, t_left_nipple, t_right_nipple, registrar2_prone, corresponding_landmarks2)


    # Pushing the data
    push_data = False
    if push_data:
        write_spreadsheet_id = '11H4hDDwGQoAz7jZC--jpJCLxDpxnUl_O4qtQd_OyUHk'
        push_data = Push_Data(write_spreadsheet_id)
        push_data.push_volunteer_number(volu_1p, volu_2p)
        push_data.push_corresponding_landmarks(cor_1_2, volu_1p, volu_2p, corresponding_volunteers1)
        push_data.push_registrars(volu_1p, volu_2p)
        push_data.push_volunteer_info(prone_metadata, corresponding_volunteers1, corresponding_volunteers2, corresponding_landmarks1, corresponding_landmarks2)
        push_data.push_landmark_type(registrar1_prone, registrar2_prone, corresponding_volunteers1, corresponding_volunteers2, corresponding_landmarks1, corresponding_landmarks2)
        push_data.push_landmark_displacement(landmark_displacement_1, landmark_displacement_2)
        push_data.push_nipple_movement(left_nipple_registrar_1, right_nipple_registrar_1,
                                     left_nipple_registrar_2, right_nipple_registrar_2)
        push_data.push_time(time_1p, time_1s, time_2p, time_2s)
        push_data.push_dist_landmark_nipple(dist_landmark_nipple_1p, dist_landmark_nipple_1s,
                                          dist_landmark_nipple_2p, dist_landmark_nipple_2s)
        push_data.push_dist_landmark_mesh(dist_landmark_cw_1p,dist_landmark_cw_1s,dist_landmark_cw_2p,dist_landmark_cw_2s,
                                        dist_landmark_skin_1p, dist_landmark_skin_1s, dist_landmark_skin_2p, dist_landmark_skin_2s)
        push_data.push_quadrant(registrar1_prone, registrar1_supine, registrar2_prone, registrar2_supine, corresponding_volunteers1, corresponding_volunteers2)
        push_data.push_local_landmark_numbers(
            registrar1_prone, registrar2_prone,
            corresponding_volunteers1, corresponding_volunteers2)

    # Jugular landmark analysis
    jugular_landmark_distances= Jugular_Landmark_Distances(
        prone_data_path,supine_data_path, rotation,
        jugular_landmarks_prone, jugular_landmarks_supine, t_jl_supine)
    jugular_landmark_distances.plotting()


    modelled_supine_breast_tissue = False
    if modelled_supine_breast_tissue:
        # Comparing the data with the modelled data
        prone_model_path_model = '/hpc_ntot/psam012/opt/breast-modelling/data/prone_to_supine_t2/2018-05-24'
        supine_model_path_model = '/hpc_ntot/psam012/opt/breast-modelling/data/prone_to_supine_t2/2018-05-24'

        modeled_landmarks = modeled_landmarks(registrar1_prone, t_supine1_landmark, prone_metadata, t_left_nipple,
                                             t_right_nipple, prone_model_path_model, supine_model_path_model)
        prone_mesh_landmarks, prone_errors, supine_mesh_landmarks, supine_errors, supine_mesh_nipples, nipple_errors \
            = modeled_landmarks.determine_errors()
        modeled_landmarks.plot_nipples()
        modeled_landmarks.plot_skin_mesh(aligned_meshes.rotation_prone_supine)
        modeled_landmarks.plot_nipples_in_mesh()

    muscle_movement_analysis = False
    if muscle_movement_analysis:
        muscle_movement = Muscle_movement(rotation)
        muscle_movement.major_movement()
        muscle_movement.minor_movement()
        muscle_movement.movement_difference()
        muscle_movement.plotting()



# xx = []
# yy = []
# zz = []
# xx_major = []
# yy_major = []
# zz_major = []
# xx_minor = []
# yy_minor = []
# zz_minor = []
# for vl_id in [38]: #[38, 50, 56, 74, 77, 83]:
#     movement_left = np.sqrt((muscle_movement.t_supine_left_major[vl_id][0] - muscle_movement.prone_left_major[vl_id][0])**2)
#     movement_right = np.sqrt((muscle_movement.t_supine_right_major[vl_id][0] - muscle_movement.prone_right_major[vl_id][0])**2)
#     movement_x2 = np.sqrt((muscle_movement.t_supine_right_minor[vl_id][0] - muscle_movement.prone_right_minor[vl_id][0])**2)
#     movement_x1 = np.sqrt((muscle_movement.t_supine_left_minor[vl_id][0] - muscle_movement.prone_left_minor[vl_id][0])**2)
#     error_xx1 = np.sqrt((aligned_meshes.t_jl_supine[vl_id][0][0]-jugular_landmarks_prone.model_landmarks[vl_id][0][0])**2)
#     error_xx2 = np.sqrt((aligned_meshes.t_jl_supine[vl_id][1][0]-jugular_landmarks_prone.model_landmarks[vl_id][1][0])**2)
#
#     xx_minor.append(movement_x1)
#     xx_minor.append(movement_x2)
#     xx_major.append(movement_left)
#     xx_major.append(movement_right)
#     xx.append(error_xx1)
#     xx.append(error_xx2)
#
#     movement_left = np.sqrt((muscle_movement.t_supine_left_major[vl_id][1] - muscle_movement.prone_left_major[vl_id][1])**2)
#     movement_right = np.sqrt((muscle_movement.t_supine_right_major[vl_id][1] - muscle_movement.prone_right_major[vl_id][1])**2)
#     movement_x2 = np.sqrt((muscle_movement.t_supine_right_minor[vl_id][1] - muscle_movement.prone_right_minor[vl_id][1])**2)
#     movement_x1 = np.sqrt((muscle_movement.t_supine_left_minor[vl_id][1] - muscle_movement.prone_left_minor[vl_id][1])**2)
#     error_xx1 = np.sqrt((aligned_meshes.t_jl_supine[vl_id][0][1]-jugular_landmarks_prone.model_landmarks[vl_id][0][1])**2)
#     error_xx2 = np.sqrt((aligned_meshes.t_jl_supine[vl_id][1][1]-jugular_landmarks_prone.model_landmarks[vl_id][1][1])**2)
#
#     yy_minor.append(movement_x1)
#     yy_minor.append(movement_x2)
#     yy_major.append(movement_left)
#     yy_major.append(movement_right)
#     yy.append(error_xx1)
#     yy.append(error_xx2)
#
#     movement_left = np.sqrt((muscle_movement.t_supine_left_major[vl_id][2] - muscle_movement.prone_left_major[vl_id][2])**2)
#     movement_right = np.sqrt((muscle_movement.t_supine_right_major[vl_id][2] - muscle_movement.prone_right_major[vl_id][2])**2)
#     movement_x2 = np.sqrt((muscle_movement.t_supine_right_minor[vl_id][2] - muscle_movement.prone_right_minor[vl_id][2])**2)
#     movement_x1 = np.sqrt((muscle_movement.t_supine_left_minor[vl_id][2] - muscle_movement.prone_left_minor[vl_id][2])**2)
#     error_xx1 = np.sqrt((aligned_meshes.t_jl_supine[vl_id][0][2]-jugular_landmarks_prone.model_landmarks[vl_id][0][2])**2)
#     error_xx2 = np.sqrt((aligned_meshes.t_jl_supine[vl_id][1][2]-jugular_landmarks_prone.model_landmarks[vl_id][1][2])**2)
#
#     zz_minor.append(movement_x1)
#     zz_minor.append(movement_x2)
#     zz_major.append(movement_left)
#     zz_major.append(movement_right)
#     zz.append(error_xx1)
#     zz.append(error_xx2)
#
#
#
# plt.figure(1)
# plt.scatter(volunteers, xx_minor, color='red', s=20)
# plt.scatter(volunteers, xx, color='blue', s=20)
# plt.ylabel('Coracoid process movement [mm]')
# plt.xlabel('Volunteer number')
# plt.title('Coracoid process movement in x-direction')
#
# plt.figure(2)
# plt.scatter(volunteers, yy_minor, color='red', s=20)
# plt.scatter(volunteers, yy, color='blue', s=20)
# plt.ylabel('Coracoid process movement [mm]')
# plt.xlabel('Volunteer number')
# plt.title('Coracoid process movement in y-direction')
#
# plt.figure(3)
# plt.scatter(volunteers, zz_minor, color='red', s=20)
# plt.scatter(volunteers, zz, color='blue', s=20)
# plt.ylabel('Coracoid process movement [mm]')
# plt.xlabel('Volunteer number')
# plt.title('Coracoid process movement in z-direction')
# plt.show
#
#
# plt.figure(1)
# plt.scatter(volunteers, xx_major, color='red', s=20)
# plt.scatter(volunteers, xx, color='blue', s=20)
# plt.ylabel('Pectoralis major groove movement [mm]')
# plt.xlabel('Volunteer number')
# plt.title('Pectoralis major groove movement x-direction')
#
# plt.figure(2)
# plt.scatter(volunteers, yy_major, color='red', s=20)
# plt.scatter(volunteers, yy, color='blue', s=20)
# plt.ylabel('Pectoralis major groove movement [mm]')
# plt.xlabel('Volunteer number')
# plt.title('Pectoralis major groove movement y-direction')
#
# plt.figure(3)
# plt.scatter(volunteers, zz_major, color='red', s=20)
# plt.scatter(volunteers, zz, color='blue', s=20)
# plt.ylabel('Pectoralis major groove movement [mm]')
# plt.xlabel('Volunteer number')
# plt.title('Pectoralis major groove movement z-direction')
# plt.show
