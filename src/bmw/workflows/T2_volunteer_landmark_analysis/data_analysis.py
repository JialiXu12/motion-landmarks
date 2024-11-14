import os
import httplib2
import time
import json
import morphic
from datetime import datetime
import morphic
from scipy import spatial
import numpy as np
from mayavi import mlab
from apiclient import discovery
from gsuites_api_access import get_drive_credentials, \
    get_sheets_credentials, \
    download_file

def write_rows(sheets_service, spreadsheet_id, write_range_name, submissions):
    # Write values to target sheet
    body = {
        "majorDimension": "ROWS",
        'values': submissions
    }
    result = sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=write_range_name,
        valueInputOption='USER_ENTERED', body=body).execute()

def send_to_spreadsheet(registrar1, registrar2, position, ages, weights, heights, LB_UO, LB_LO, LB_UI, LB_LI, RB_UO, RB_LO, RB_UI, RB_LI , LB_UO1, LB_LO1, LB_UI1, LB_LI1, RB_UO1, RB_LO1, RB_UI1, RB_LI1, LB_UO2, LB_LO2, LB_UI2, LB_LI2, RB_UO2, RB_LO2, RB_UI2, RB_LI2):
    """ Sort ABI research forum posters.

    """
    #drive_credentials = get_drive_credentials()
    #drive_http = drive_credentials.authorize(httplib2.Http())
    #drive_service = discovery.build('drive', 'v2', http=drive_http)

    sheets_credentials = get_sheets_credentials()
    sheets_http = sheets_credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                    'version=v4')
    sheets_service = discovery.build('sheets', 'v4', http=sheets_http,
                                     discoveryServiceUrl=discoveryUrl)

    spreadsheet_id = '1TsKbpk1Vs7pF9R1kCpNQ3SLuArQuU26rtcaleNtHcbU'

    closestPtRadius = [1, 2, 3, 4, 5, 6]


    submissions = [['VL', 'age', 'weight', 'height', 'Number of landmarks (Ben)', 'LB_UO', 'LB_LO', 'LB_UI', 'LB_LI', 'RB_UO', 'RB_LO', 'RB_UI', 'RB_LI', 'Number of landmarks (Clarke)']]
    for radius in closestPtRadius:
        submissions[0].append('Number of corresponding landmarks ({0} mm radius)'.format(radius))
    for vl_id in range(1,90):
        pointsExist1 = False
        pointsExist2 = False

        if vl_id in registrar1.landmarks:
            points1 = registrar1.landmarks[vl_id]
            pointsExist1 = True
            numPoints1 = len(points1)
        else:
            numPoints1 = 0
        if vl_id in registrar2.landmarks:
            points2 = registrar2.landmarks[vl_id]
            pointsExist2 = True
            numPoints2 = len(points2)
        else:
            numPoints2 = 0

        numCorrespondingPoints = []
        for radius in [1, 2, 3, 4, 5, 6]:
            if pointsExist1 and pointsExist2:
                tree = spatial.KDTree(points1)
                corresponding_points = tree.query_ball_point(points2, radius)
                corresponding_points = [item for sublist in corresponding_points for item in sublist]
                numCorrespondingPoints.append(len(corresponding_points))
            else:
                numCorrespondingPoints.append(0)

        submissions.append(['VL{0:05d}'.format(vl_id), ages[vl_id], weights[vl_id], heights[vl_id], numPoints1, LB_UO[vl_id], LB_LO[vl_id], LB_UI[vl_id], LB_LI[vl_id], RB_UO[vl_id], RB_LO[vl_id], RB_UI[vl_id], RB_LI[vl_id], numPoints2]+numCorrespondingPoints)

    write_range_name = '{0}!A1:T{1}'.format(position, len(submissions))
    write_rows(sheets_service, spreadsheet_id, write_range_name,
               submissions)
    a =1
class Landmarks:

    def __init__(self, registrar_name, user_id, position):
        self.registrar_name = registrar_name
        self.user_id = user_id
        self.position = position
        self.landmarks = {}
        self.landmark_types = {}
        self.model_landmarks = {}
        self.vl_ids = {}

        with open('vl_t2_image_properties.json') as json_data:
            self.img_prop = json.load(json_data)

        self.load_landmarks()
        results = {}


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


    def load_dicom_header(self, vl_image_folder):
        import dcmstack
        from glob import glob
        src_paths = glob(os.path.join(vl_image_folder, '*'))
        stacks = dcmstack.parse_and_stack(src_paths)
        stack = stacks[stacks.keys()[0]]
        header = stack.to_nifti_wrapper()
        return header

    def load_dicom_spacing(self, vl_image_folder):
        header = self.load_dicom_header(vl_image_folder)
        spacing = header.get_meta('PixelSpacing')
        spacing.append(header.get_meta('SliceThickness'))
        return spacing

    def load_landmarks(self):
        for vl_id in range(90):
            vl_folder = '/home/data/picker/points/{0}/VL{1:05d}'.format(self.user_id, vl_id)
            vl_image_folder = '/home/data/images/vl/mri_t2/VL{0:05d}/prone/'.format(vl_id)
            #if os.path.exists(vl_image_folder):
            #    spacing = self.load_dicom_spacing(vl_image_folder)
            for point_id in range (15):
                path = os.path.join(vl_folder, 'point.{:03d}.json'.format(point_id))
                if os.path.exists(path):
                    with open(path) as _landmarkFile:
                        self.add_landmark(json.load(_landmarkFile), vl_id)
            if vl_id in self.landmarks:
                self.landmarks[vl_id] = np.array(self.landmarks[vl_id])

    def find_quadrants(self):
        self.quadrants = Quadrants(self)

    def add_metadata(self, path):
        self.metadata = Metadata(path)


class Metadata():

    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.ages = {}
        self.weights = {}
        self.heights = {}
        self.left_nipples = {}
        self.right_nipples = {}
        self.sternal_notches = {}

        for vl_id in range(90):
            vl_str = 'VL{0:05d}'.format(vl_id)
            data_path = os.path.join(metadata_path, vl_str, 'nipple_pts.data')
            if os.path.exists(data_path):
                data = morphic.Data(data_path)
                self.ages[vl_id] = data.metadata['subject']['age'].replace('Y', '')
                self.weights[vl_id] = data.metadata['subject']['weight']
                self.heights[vl_id] = data.metadata['subject']['height']
                self.left_nipples[vl_id] =data.values[0,:]
                self.right_nipples[vl_id] =data.values[1,:]
                self.sternal_notches[vl_id] =data.values[2,:]

            else:
                self.ages[vl_id] = 'n/a'
                self.weights[vl_id] = 'n/a'
                self.heights[vl_id] = 'n/a'

class Quadrants:

    def __init__(self, registrar):
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
            if vl_id in registrar.model_landmarks:
                for landmark in registrar.model_landmarks[vl_id]:
                    if vl_id in registrar.metadata.sternal_notches:
                        if landmark[1] < registrar.metadata.sternal_notches[vl_id][1]: #LHS
                            if landmark[1] < registrar.metadata.left_nipples[vl_id][1]: # UO or LO
                                if landmark[2] > registrar.metadata.left_nipples[vl_id][2]:  # UO
                                    self.LB_UO[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('LB_UO')
                                elif landmark[2] < registrar.metadata.left_nipples[vl_id][2]:  # LO
                                    self.LB_LO[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('LB_LO')
                            else:# UI or LI
                                if landmark[2] > registrar.metadata.right_nipples[vl_id][2]:  # UI
                                    self.LB_UI[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('LB_UI')
                                elif landmark[2] < registrar.metadata.right_nipples[vl_id][2]:  # LI
                                    self.LB_LI[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('LB_LI')

                        else: #RHS
                            if landmark[1] < registrar.metadata.right_nipples[vl_id][1]: # UI or LI
                                if landmark[2] > registrar.metadata.right_nipples[vl_id][2]:  # UI
                                    self.RB_UI[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('RB_UI')
                                elif landmark[2] < registrar.metadata.right_nipples[vl_id][2]:  # LI
                                    self.RB_LI[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('RB_LI')
                            else:# UO or LO
                                if landmark[2] > registrar.metadata.right_nipples[vl_id][2]:  # UO
                                    self.RB_UO[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('RB_UO')
                                elif landmark[2] < registrar.metadata.right_nipples[vl_id][2]:  # LO
                                    self.RB_LO[vl_id] +=1
                                    self.landmark_quadrants[vl_id].append('RB_LO')
                    else:
                        self.landmark_quadrants[vl_id].append('n/a')


def load_jugular_notch():
    landmark_path = "/home/data/picker/points/user001/"
    for vl_id in range(90):
        vl_str = 'VL{0:05d}'.format(vl_id)
        data_path = os.path.join(landmark_path, vl_str, 'point.005.json')
        if os.path.exists(path):
            with open(path) as _landmarkFile:
                self.add_landmark(json.load(_landmarkFile), vl_id)

class Jugular_landmarks:

    def __init__(self, registrar_name, user_id, position):
        self.registrar_name = registrar_name
        self.user_id = user_id
        self.position = position
        self.landmarks = {}
        self.model_landmarks = {}
        self.vl_ids = {}

        with open('vl_t2_image_properties.json') as json_data:
            self.img_prop = json.load(json_data)


        self.load_landmarks()
        results = {}

    def add_landmark(self, landmark, vl_id):
        if vl_id not in self.landmarks:
            self.landmarks[vl_id] = []
        point = []
        point.append(landmark['{0}_point'.format(self.position)]['point']['x'])
        point.append(landmark['{0}_point'.format(self.position)]['point']['y'])
        point.append(landmark['{0}_point'.format(self.position)]['point']['z'])
        self.landmarks[vl_id].append(point)


        if vl_id not in self.model_landmarks:
            self.model_landmarks[vl_id] = []
        vl_str = 'VL{0:05d}'.format(vl_id)
        img_params = self.img_prop[vl_str][self.position]
        x0 = img_params['image_position']
        sp = img_params['pixel_spacing']
        shape = img_params['shape']
        pt = landmark['{0}_point'.format(self.position)]['point']
        x = np.array([pt['y'] - x0[1],
                      -pt['x'] + x0[0] + sp[0] * shape[0],
                      pt['z'] - x0[2]])
        self.model_landmarks[vl_id].append(x)

    def load_dicom_header(self, vl_image_folder):
        import dcmstack
        from glob import glob
        src_paths = glob(os.path.join(vl_image_folder, '*'))
        stacks = dcmstack.parse_and_stack(src_paths)
        stack = stacks[stacks.keys()[0]]
        header = stack.to_nifti_wrapper()
        return header

    def load_dicom_spacing(self, vl_image_folder):
        header = self.load_dicom_header(vl_image_folder)
        spacing = header.get_meta('PixelSpacing')
        spacing.append(header.get_meta('SliceThickness'))
        return spacing

    def load_landmarks(self):
        for vl_id in range(90):
            vl_folder = '/home/data/picker/points/{0}/VL{1:05d}'.format(self.user_id, vl_id)
            vl_image_folder = '/home/data/images/vl/mri_t2/VL{0:05d}/prone/'.format(vl_id)
            #if os.path.exists(vl_image_folder):
            #    spacing = self.load_dicom_spacing(vl_image_folder)
            if vl_id == 40:
                a=1
            for point_id in [1,2]: #################### updated
                path = os.path.join(vl_folder, 'point.{:03d}.json'.format(point_id))
                if os.path.exists(path):
                    with open(path) as _landmarkFile:
                        self.add_landmark(json.load(_landmarkFile), vl_id)
            if vl_id in self.landmarks:
                self.landmarks[vl_id] = np.array(self.landmarks[vl_id])

def find_corresponding_points(registrar1, registrar2):
    for vl_id in registrar1.landmarks:
        if vl_id in registrar2.landmarks:
            points1 = registrar1.landmarks[vl_id]
            tree = spatial.KDTree(points1)
            points2 = registrar2.landmarks[vl_id]
            corresponding_points = tree.query_ball_point(points2, 1.)
            corresponding_points = [item for sublist in corresponding_points for item in sublist]


def load_chest_wall_surface_mesh(mesh_path, base_mesh):
    # Load fitted chest wall surface (cwm)
    cwm_fname = os.path.join(mesh_path, 'ribcage_{0}.mesh'.format(base_mesh))
    if os.path.exists(cwm_fname):
        cwm = morphic.Mesh(cwm_fname)
        cwm.label = base_mesh+'_ribcage'
        return cwm
    else:
        raise ValueError('ribcage mesh not found')

def load_lung_surface_mesh(mesh_path, base_mesh):
    # Load fitted lung surface mesh (lung)
    lm_fname = os.path.join(mesh_path, 'lungs_{0}.mesh'.format(base_mesh))
    if os.path.exists(lm_fname):
        lm = morphic.Mesh(lm_fname)
        lm.label = base_mesh+'_lung'
        return lm
    else:
        raise ValueError('lung surface mesh not found')

def load_skin_surface_mesh(mesh_path, base_mesh,side):
    # Load fitted skin surface mesh (skin)
    sm_fname = os.path.join(mesh_path, 'skin_'+side+'_{0}.mesh'.format(base_mesh))
    if os.path.exists(sm_fname):
        sm = morphic.Mesh(sm_fname)
        sm.label = base_mesh+'_skin_'+side
        return sm
    else:
        raise ValueError('skin_'+side + ' surface mesh not found')

def create_Lagrange_chest_wall_mesh(fig, cwm):
    # Create new chestwall surface mesh
    Xe = [[0,1,2,3,4,5,54,55,42,43,44],
        [6,7,8,9,10,11,56,57,45,46,47],
        [12, 13, 14, 15, 16, 17, 36, 37, 48, 49, 50],
        [18, 19, 20, 21, 22, 23, 38, 37, 48, 49, 50],
        [24, 25, 26, 27, 28, 29, 39, 40, 51, 52, 53],
        [30, 31, 32, 33, 34, 35, 41, 40, 51, 52, 53]]
    Xe_rhs = scipy.array([[0, 1, 2, 3],
                          [ 8,  9, 10, 11],
                          [16, 17, 18, 19]])
    offset = 0
    return bmw.reposition_nodes(fig, cwm, new_bm_rhs, offset, side='rhs', xi1_Xe=Xe_rhs, elem_shape=scipy.array(Xe).shape[::-1], debug=False)

def generate_image_coordinates(image_shape, spacing,origin=[0,0,0]):
    import scipy
    x, y, z = scipy.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    x = x*spacing[0]+origin[0]
    y = y*spacing[1]+origin[1]
    z = z*spacing[2]+origin[2]
    image_coor = scipy.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z

def align_chest_wall_meshes(registrars, jugular_landmarks_prone, jugular_landmarks_supine):
    import bmw
    from morphic import utils
    # Initialise landmark counter
    lnd_mark_id = 0

    sheets_credentials = get_sheets_credentials()
    sheets_http = sheets_credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                    'version=v4')
    sheets_service = discovery.build('sheets', 'v4', http=sheets_http,
                                     discoveryServiceUrl=discoveryUrl)

    spreadsheet_id = '1TsKbpk1Vs7pF9R1kCpNQ3SLuArQuU26rtcaleNtHcbU'

    sheet_name = 'displacements'

    submissions = [['landmark', 'registrar', 'vl', 'quadrant (prone)', 'quadrant (supine)', 'type', 'displacement', 'age', 'height', 'weight']]
    alignmentErrors = {}
    new_alignment = True
    for registrar in registrars:
        prone_position = registrar[0]
        supine_position = registrar[1]

        for vl_id in range(0,90): #[43,44]: #[40,41,42,47]:#,43,44range(46,90): [47
            if vl_id in [9,22,49,64]:
                skip = True
            else:
                skip = False
            if vl_id in prone_position.model_landmarks:
                vl_id_str = 'VL{0:05d}'.format(vl_id)

                try:
                    jugular_landmark_prone = np.array(jugular_landmarks_prone.model_landmarks[vl_id])
                except:
                    skip = True
                try:
                    jugular_landmark_supine = np.array(jugular_landmarks_supine.model_landmarks[vl_id])
                except:
                    skip = True

                if not skip:
                    supine_mesh_path = os.path.join(
                        '/home/psam012/opt/breast-data/supine_to_prone_t2/2017-07-31/volunteer_meshes', vl_id_str)
                    prone_mesh_path = os.path.join(
                        '/home/psam012/opt/breast-data/prone_to_supine_t2/2017_09_06/volunteer_meshes', vl_id_str)

                    if os.path.exists(supine_mesh_path) and os.path.exists(prone_mesh_path):

                        # Load meshes
                        prone_cwm = load_chest_wall_surface_mesh(prone_mesh_path, 'prone')
                        prone_lm = load_lung_surface_mesh(prone_mesh_path, 'prone')
                        #prone_sm = load_skin_surface_mesh(prone_mesh_path, 'prone')
                        supine_cwm = load_chest_wall_surface_mesh(supine_mesh_path, 'supine')
                        #supine_lm = load_lung_surface_mesh(supine_mesh_path, 'supine')
                        supine_sm_left = load_skin_surface_mesh(supine_mesh_path, 'supine','left')
                        supine_sm_right = load_skin_surface_mesh(supine_mesh_path, 'supine','right')


                        visualise = False
                        if visualise:
                            onscreen = True
                            if onscreen:
                                from morphic import viewer
                            else:
                                fig = None
                                viewer = None
                            fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen
                            bmw.visualise_mesh(prone_cwm, fig, visualise=True, face_colours=(1, 0, 0),opacity=0.1)
                            #bmw.visualise_mesh(supine_cwm, fig, visualise=True, face_colours=(0, 1, 0))
                            #bmw.visualise_mesh(prone_lm, fig, visualise=True, face_colours=(0, 1, 0),opacity=0.75)
                            #bmw.visualise_mesh(supine_sm_left, fig, visualise=True, face_colours=(0, 0, 1),opacity=0.75)
                            #bmw.visualise_mesh(supine_sm_right, fig, visualise=True, face_colours=(0, 0, 1),opacity=0.75)

                        supine_cwm = utils.convert_hermite_lagrange(supine_cwm, tol=1e-9)
                        prone_cwm = utils.convert_hermite_lagrange(prone_cwm, tol=1e-9)

                        import scipy
                        supine_nodes = scipy.zeros((240, 3))
                        prone_nodes = scipy.zeros((240, 3))
                        for node_id in range(1, 241):
                            supine_nodes[node_id-1, :] = supine_cwm.get_nodes(node_id)
                            prone_nodes[node_id-1, :] = prone_cwm.get_nodes(node_id)

                        from bmw import transformations

                        #supine_rib_points[10,:] = [169.9226, 274.21992, 161.098926]
                        transform_supine = True

                        try:
                            prone_landmarks = np.array(prone_position.model_landmarks[vl_id])
                        except:
                            a=1
                        supine_landmarks = np.array(supine_position.model_landmarks[vl_id])

                        if vl_id in [43,44]:
                            weighting = 100.
                        else:
                            weighting = 100.

                        import icp
                        rOpt = icp.alignCorrespondingDataAndLandmarksRigidRotationTranslation(
                            supine_nodes, prone_nodes, jugular_landmark_supine, jugular_landmark_prone,weighting=100)
                        #rOpt.x = scipy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                        R_supine_to_prone, t_supine_to_prone, rmse_supine_to_prone = transformations.rigid_transform_3D(
                            supine_nodes, prone_nodes)
                        alignmentErrors[vl_id] = rOpt.fun
                        if transform_supine:

                            if new_alignment:
                                t_supine_nodes = icp.transformRigid3D(supine_nodes, scipy.array(rOpt.x))
                                t_jugular_landmark_supine = icp.transformRigid3D(jugular_landmark_supine, scipy.array(rOpt.x))
                                t_supine_landmarks = icp.transformRigid3D(supine_landmarks, scipy.array(rOpt.x))
                            else:
                                t_supine_nodes = transformations.transform_points(supine_nodes, R_supine_to_prone, t_supine_to_prone)
                                t_jugular_landmark_supine = transformations.transform_points(jugular_landmark_supine, R_supine_to_prone, t_supine_to_prone)
                                t_supine_landmarks = transformations.transform_points(supine_landmarks, R_supine_to_prone, t_supine_to_prone)
                        else:
                            t_supine_nodes = supine_nodes

                        t_supine_cwm = load_chest_wall_surface_mesh(supine_mesh_path, 'supine')
                        t_supine_cwm = utils.convert_hermite_lagrange(t_supine_cwm, tol=1e-9)
                        t_supine_cwm.label = 't_supine'
                        for node_id in range(1, 241):
                            t_supine_cwm.nodes[node_id].values = t_supine_nodes[node_id-1, :]
                        if visualise:
                            bmw.visualise_mesh(t_supine_cwm, fig, visualise=True, face_colours=(0, 0, 1), opacity=0.1)

                        landmarks_displacements = []
                        num_landmarks = len(prone_landmarks)
                        for landmark in range(num_landmarks):
                            landmarks_displacements.append(np.linalg.norm(prone_landmarks[landmark,:]-t_supine_landmarks[landmark,:]))

                        print landmarks_displacements

                        if visualise:
                            bmw.plot_points(fig, 'prone_landmarks', prone_landmarks, range(len(prone_landmarks)), visualise=True, colours=(1,0,0), point_size=10, text_size=5)
                            bmw.plot_points(fig, 't_supine_landmarks', t_supine_landmarks, range(len(t_supine_landmarks)), visualise=True, colours=(0,0,1), point_size=10, text_size=5)

                            bmw.plot_points(fig, 'jugular_landmark_prone', jugular_landmark_prone, [1,2], visualise=True, colours=(1,0,0), point_size=3, text_size=5)
                            bmw.plot_points(fig, 't_jugular_landmark_supine', t_jugular_landmark_supine, [1,2], visualise=True, colours=(0,0,1), point_size=3, text_size=5)

                        if visualise:
                            supine_data_path = os.path.join(
                                '/home/psam012/opt/breast-data/supine_to_prone_t2/2017-07-31/segmented_data',
                                vl_id_str)
                            prone_data_path = os.path.join(
                                '/home/psam012/opt/breast-data/prone_to_supine_t2/2017_09_06/segmented_data',
                                vl_id_str)
                            prone_data_skin = morphic.Data(os.path.join(prone_data_path,'skin_pts_kmeans.data'))
                            prone_data_lung = morphic.Data(os.path.join(prone_data_path,'lungs_pts_kmeans.data'))

                            sample_idxs1 = np.random.randint(0, prone_data_skin.values.shape[0], 10000)
                            sample_idxs2 = np.random.randint(0, prone_data_lung.values.shape[0], 10000)
                            prone_skin_sampled = prone_data_skin.values[sample_idxs1,:]
                            prone_lung_sampled = prone_data_lung.values[sample_idxs2,:]

                            #bmw.plot_points(fig, 'prone_skin_sampled', prone_skin_sampled, range(prone_skin_sampled.shape[0]), visualise=True, colours=(0,1,0), point_size=3, text_size=5)
                            #bmw.plot_points(fig, 'prone_lung_sampled', prone_lung_sampled, range(prone_lung_sampled.shape[0]), visualise=True, colours=(0,1,0), point_size=3, text_size=5)
                            #bmw.plot_points(fig, 'supine_data', t_supine_landmarks, range(len(t_supine_landmarks)), visualise=True, colours=(0,0,1), point_size=3, text_size=5)

                            from bmw import imaging_functions, image_warping
                            import automesh
                            supine_image = automesh.Scan('/home/data/images/vl/mri_t2/{0}/supine'.format(vl_id_str))
                            supine_image.set_origin([0, 0, 0])
                            prone_image = automesh.Scan('/home/data/images/vl/mri_t2/{0}/prone'.format(vl_id_str))
                            prone_image.set_origin([0, 0, 0])

                            #prone_image.set_origin([0, 0, 0])
                            #fig.plot_dicoms('supine', prone_image)
                            #fig.plot_dicoms('prone', prone_image)
                            #fig.plot_dicoms('supine', supine_image)

                            coor, x, y, z = generate_image_coordinates(prone_image.shape,
                                                                       prone_image.spacing)

                            src = mlab.pipeline.scalar_field(x,
                                                             y,
                                                             z,
                                                             prone_image.values.astype(
                                                                 scipy.int16))
                            outline = mlab.pipeline.outline(src, figure=fig.figure)
                            prone_ipw_z = mlab.pipeline.image_plane_widget(
                                outline,
                                plane_orientation='y_axes',
                                slice_index=int(0.5 * prone_image.num_slices),
                               colormap='black-white')

                            #t_coor = icp.transformRigid3D(coor, scipy.array(rOpt.x))
                            ##transformations.transform_points(coor, R_supine_to_prone, t_supine_to_prone) -t_jugular_landmark_supine1 +jugular_landmark_prone

                            # src2 = mlab.pipeline.scalar_field(t_coor[:,0].reshape(x.shape),
                            #                                  t_coor[:, 1].reshape(
                            #                                      y.shape),
                            #                                  t_coor[:, 2].reshape(
                            #                                      z.shape),
                            #                                  supine_image.values.astype(
                            #                                      scipy.int16))
                            # outline2 = mlab.pipeline.outline(src2, figure=fig.figure)

                            # #prone_ipw_x = mlab.pipeline.image_plane_widget(
                            # #    outline,
                            # #    plane_orientation='x_axes',
                            # #    slice_index=int(0.5 * prone_image.header.get_n_slices()),
                            # #    colormap='black-white')
                            # #prone_ipw_y = mlab.pipeline.image_plane_widget(
                            # #    outline,
                            # #    plane_orientation='y_axes',
                            # #    slice_index=int(0.5 * prone_image.header.get_n_slices()),
                            # #    colormap='black-white')
                            # prone_ipw_z = mlab.pipeline.image_plane_widget(
                            #     outline2,
                            #     plane_orientation='y_axes',
                            #     slice_index=int(0.5 * supine_image.num_slices),
                            #    colormap='black-white')
                            # a = 1

                            #import pickle
                            #prone_rib = pickle.load( open( "/home/psam012/Downloads/VL00042/rib_VL00042_points.pkl", "rb" ) )
                            #if visualise:
                            #    fig2 = bmw.add_fig(viewer, label='mesh2') # returns empty array if offscreen
                            #    bmw.plot_points(fig2, 'prone_data', prone_rib, range(len(prone_rib)), visualise=True, colours=(0,1,0), point_size=10, text_size=5)

                        for landmark_idx, displacement in enumerate(landmarks_displacements):
                            lnd_mark_id +=1
                            try:
                                submission = [
                                    lnd_mark_id,
                                    prone_position.registrar_name,
                                    vl_id_str,
                                    prone_position.quadrants.landmark_quadrants[vl_id][landmark_idx],
                                    supine_position.quadrants.landmark_quadrants[vl_id][landmark_idx],
                                    supine_position.landmark_types[vl_id][landmark_idx],
                                    displacement,
                                    supine_position.metadata.ages[vl_id],
                                    supine_position.metadata.heights[vl_id],
                                    supine_position.metadata.weights[vl_id],
                                    alignmentErrors[vl_id]]
                            except:
                                a=1
                            print submission
                            submissions.append(submission)
                        if visualise:
                            mlab.savefig(filename='VL{0}_COBYLA.png'.format(vl_id))
                            fig.clear()
    a=1
    write_range_name = '{0}!A1:U{1}'.format(sheet_name, len(submissions))
    write_rows(sheets_service, spreadsheet_id, write_range_name,
              submissions)

    return R_supine_to_prone, t_supine_to_prone


# def save_displacements_to_spreadsheet():
#     volunteer_id = 49
#     R_supine_to_prone, t_supine_to_prone = align_chest_wall_meshes(volunteer_id)
#
#     registrar1.model_landmarks[volunteer_id]
#     num_landmarks = len()

if __name__ == '__main__':
    position = 'supine'
    registrar1_supine = Landmarks('Ben', 'user008', position)
    registrar2_supine = Landmarks('Clarke', 'user007', position)

    position = 'prone'
    registrar1_prone = Landmarks('Ben', 'user008', position)
    registrar2_prone = Landmarks('Clarke', 'user007', position)

    jugular_landmarks_prone = Jugular_landmarks('jugular_landmarks','user001', 'prone')
    jugular_landmarks_supine = Jugular_landmarks('jugular_landmarks','user001', 'supine')

    registrar1_supine.add_metadata('/home/psam012/opt/breast-data/supine_to_prone_pipeline_vl_results_2017_06_25/segmented_data')
    registrar2_supine.add_metadata('/home/psam012/opt/breast-data/supine_to_prone_pipeline_vl_results_2017_06_25/segmented_data')

    registrar1_supine.find_quadrants()
    registrar2_supine.find_quadrants()

    registrar1_prone.add_metadata('/home/psam012/opt/breast-data/prone_to_supine_t2/2017_09_06/segmented_data')
    registrar2_prone.add_metadata('/home/psam012/opt/breast-data/prone_to_supine_t2/2017_09_06/segmented_data')

    registrar1_prone.find_quadrants()
    registrar2_prone.find_quadrants()

    align_chest_wall_meshes([[registrar1_prone, registrar1_supine], [registrar2_prone, registrar2_supine]], jugular_landmarks_prone, jugular_landmarks_supine)







    #seg_points = np.array([[166.13567122,   41.80553208,   43.19677465],
    # [156.08859404,  102.28935018,  116.09133159],
    #[135.45695821,    342.14141386,    90.89321314]])

    #save_displacements_to_spreadsheet(registrar1_prone, registrar1_supine)

    #find_corresponding_points(registrar1_supine, registrar2_supine)
    #send_to_spreadsheet(registrar1_supine, registrar2_supine, position, ages, weights, heights, LB_UO1, LB_LO1, LB_UI1, LB_LI1, RB_UO1, RB_LO1, RB_UI1, RB_LI1, LB_UO2, LB_LO2, LB_UI2, LB_LI2, RB_UO2, RB_LO2, RB_UI2, RB_LI2)
    a = 1
    #send_to_spreadsheet()

