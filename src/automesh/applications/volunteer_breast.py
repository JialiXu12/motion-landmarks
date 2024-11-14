import os
import time

import pickle

import automesh
import morphic
import numpy as np

from scipy.interpolate import UnivariateSpline
from sklearn.cross_decomposition import PLSRegression


def get_x_bounds(X):
    minmax = np.array([X.min(0), X.max(0)])
    xcount = []
    for x in np.arange(minmax[0, 0], minmax[1, 0] + 5, 5):
        ii1 = X[:, 0] > x
        ii2 = X[:, 0] < x + 5
        ii = ii1 * ii2

        xcount.append([x + 2.5, ii.sum()])
    xcount = np.array(xcount)

    return np.array([minmax[0, 0], minmax[1, 0]])  # xcount


def get_y_bounds(X):
    minmax = np.array([X.min(0), X.max(0)])
    ycount = []
    for y in np.arange(minmax[0, 1], minmax[1, 1] + 5, 5):
        ii1 = X[:, 1] > y
        ii2 = X[:, 1] < y + 5
        ii = ii1 * ii2
        ycount.append([y + 2.5, ii.sum()])
    ycount = np.array(ycount)

    ii = ycount[:, 1] < 50
    ycount[ii, 1] = 0
    ymin, xmax = None, None
    for yc in ycount:
        if yc[1] > 0 and ymin is None:
            ymin = yc[0]
        elif yc[1] > 0:
            ymax = yc[0]

    # Fit spline and find the minimum count in the mid region of the cloud
    spl = UnivariateSpline(ycount[:, 0], ycount[:, 1])
    spl.set_smoothing_factor(100)
    xs = np.linspace(X.min(0)[1], X.max(0)[1], 1000)
    ys = spl(xs)
    idx_mid = 250 + ys[250:750].argmin()
    xmid = xs[idx_mid]
    ymid = ys[idx_mid]

    # return np.array([ycount[0, 1], ymid, ycount[-1, 1]])
    return np.array([ymin, xmid, ymax])


def get_z_bounds(X, ybounds):
    minmax = np.array([X.min(0), X.max(0)])
    Xr = X[X[:, 1] < ybounds[1], :]
    Xl = X[X[:, 1] > ybounds[1], :]
    zmin, zmax = None, None
    xcount = []
    for x in np.arange(minmax[0, 2], minmax[1, 2] + 5, 5):
        ii1 = X[:, 2] > x
        ii2 = X[:, 2] < x + 5
        ii = ii1 * ii2
        count = ii.sum()

        if count > 40 and zmin is None:
            zmin = X[ii, 2].mean()
        elif count > 20:
            zmax = X[ii, 2].mean()

        xcount.append([x + 2.5, ii.sum()])
    xcount = np.array(xcount)

    minmax = np.array([Xl.min(0), Xl.max(0)])
    for x in np.arange(minmax[0, 2], minmax[1, 2] + 5, 5):
        ii1 = Xl[:, 2] > x
        ii2 = Xl[:, 2] < x + 5
        ii = ii1 * ii2
        count = ii.sum()
        if count > 40:
            zmax_left = Xl[ii, :].mean(0)

    minmax = np.array([Xr.min(0), Xr.max(0)])
    for x in np.arange(minmax[0, 2], minmax[1, 2] + 5, 5):
        ii1 = Xr[:, 2] > x
        ii2 = Xr[:, 2] < x + 5
        ii = ii1 * ii2
        count = ii.sum()
        if count > 40:
            zmax_right = Xr[ii, :].mean(0)

    # # Fit spline and find the minimum count in the mid region of the cloud
    # spl = UnivariateSpline(y[:, 0], y[:, 1])
    # spl.set_smoothing_factor(100)
    # xs = np.linspace(X.min(0)[1], X.max(0)[1], 1000)
    # ys = spl(xs)
    # idx_mid = 250 + ys[250:750].argmin()
    # xmid = xs[idx_mid]
    # ymid = ys[idx_mid]

    return np.array([zmin, zmax]), np.array([zmax_left, zmax_right]), xcount


def get_bounds(X):
    bounds = {}
    bounds['x'] = get_x_bounds(X)
    bounds['y'] = get_y_bounds(X)
    bounds['z'] = get_z_bounds(X)
    return bounds


def split_skin_points(skin_pts, xbounds, ybounds):
    spts = skin_pts[np.where(np.logical_and(skin_pts[:, 1] > ybounds[0], skin_pts[:, 1] < ybounds[2])), :][0]
    xmid = 0.5 * (xbounds[0] + xbounds[1])
    ii = spts[:, 0] < xmid
    skin_front = spts[ii, :]
    skin_back = spts[np.logical_not(ii), :]
    return skin_front, skin_back


def fit_meshes(meshes, skin_pts, lungs_pts):

    # ybounds = get_y_bounds(lung_pts)
    # xbounds = get_x_bounds(lung_pts)
    # zbounds = get_z_bounds(lung_pts, ybounds)
    # skin_front_pts, skin_back_pts = split_skin_points(skin_pts, xbounds, ybounds)

    for mesh in meshes.itervalues():
        if '_dx' not in mesh.nodes.keys():
            mesh.add_stdnode('_dx', [0, 0, 0], group='_dx')

    Xi = meshes['skin_left'].grid(6, method='center')
    dx = lungs_pts.mean(0) - np.mean([meshes['lungs'].nodes[i].values[:, 0] for i in range(2, 26)], 0)
    lung_elems = [1, 2, 5, 6, 9, 10, 13, 14]
    limit = 5
    w = np.zeros(24)

    data = {
        'skin': {'data': skin_pts},
        'lungs': {'data': lungs_pts},
        'dx': {'data': dx},
        'weights': {'data': w},
    }

    dofs = [
        {'mesh': 'skin_left', 'nodes': 'all', 'fix': 'all'},
        {'mesh': 'skin_right', 'nodes': 'all', 'fix': 'all'},
        {'mesh': 'lungs', 'nodes': 'all', 'fix': 'all'},
        {'extra': 'dx', 'data': 'dx', 'add_node': 'dx', 'group': '_translation'},
        {'extra': 'weights', 'data': 'weights'},
    ]

    fits = [
        {'type': 'update_weights', 'mesh': 'all', 'data': 'weights'},
        {'type': 'update_pca', 'mesh': 'all', 'translation_data': 'dx'},
        {'type': 'update_dep_nodes', 'mesh': ['skin_left', 'skin_right']},
        {'type': 'closest_data', 'data': 'skin', 'mesh': 'skin_left', 'elements': 'all', 'xi': Xi, 'limit': limit},
        {'type': 'closest_data', 'data': 'skin', 'mesh': 'skin_right', 'elements': 'all', 'xi': Xi, 'limit': limit},
        {'type': 'closest_data', 'data': 'lungs', 'mesh': 'lungs', 'elements': lung_elems, 'xi': Xi, 'limit': limit},
    ]

    meshes = automesh.fit.fit_mesh(meshes, data, fits, dofs=dofs, ftol=1e-4, xtol=1e-4, maxiter=5000, dt=10, output=True)

    return meshes



def fit_skin_mesh(mesh, X, region='all'):
    Xi_top = mesh.grid(8, method='center')[:8, :]
    Xi_bottom = mesh.grid(8, method='center')[-8:, :]
    Xi_data = mesh.grid(5, method='center')
    Xi_deriv = mesh.grid(3, method='center')

    all_nodes = [nid for nid in range(0, 72) if nid not in [27, 41]]
    all_elements = range(52)
    front_elements = range(36)
    front_nodes = [nid for nid in range(49) if nid in all_nodes]
    back_elements = range(36, 52)
    back_nodes = [nid for nid in range(49, 72) if nid in all_nodes]

    side_elements = [42, 43, 47, 48]
    front_back_elements = [eid for eid in range(0, 52) if eid not in side_elements]

    torso_elements = [0, 1, 2, 3, 4, 5, 6, 12, 18, 24]
    base_elements = [30, 31, 32, 33, 34, 35]
    breast_elements = [eid for eid in front_elements if eid not in torso_elements]

    fix_nodes = None
    fit_elements = all_elements
    if region == 'front':
        fit_elements = front_elements
        fix_nodes = back_nodes
    elif region == 'back':
        fit_elements = back_elements
        fix_nodes = front_nodes

    alpha = 5e-4
    beta_torso = 2e-3
    beta_breast = 3e-4
    beta_base = 8e-3
    beta_back = 1e-3

    data = {
        'X': {'data': X},
        'Xi_data': {'data': Xi_data}, 'Xi_deriv': {'data': Xi_deriv},
        'Xi_top': {'data': Xi_top}, 'Xi_bottom': {'data': Xi_bottom},
    }
    fits = [
        {'type': 'closest_data', 'data': 'X', 'elements': fit_elements, 'xi': Xi_data, 'out': 3, 'k': 1},
        {'type': 'penalise_derivatives', 'elements': breast_elements, 'xi': Xi_deriv, 'deriv': [2, 0],
         'weight': beta_breast},
        {'type': 'penalise_derivatives', 'elements': breast_elements, 'xi': Xi_deriv, 'deriv': [0, 2],
         'weight': beta_breast},
        {'type': 'penalise_derivatives', 'elements': breast_elements, 'xi': Xi_deriv, 'deriv': [1, 1],
         'weight': 5 * beta_breast},

        {'type': 'penalise_derivatives', 'elements': torso_elements, 'xi': Xi_deriv, 'deriv': [2, 0],
         'weight': beta_torso},
        {'type': 'penalise_derivatives', 'elements': torso_elements, 'xi': Xi_deriv, 'deriv': [0, 2],
         'weight': beta_torso},
        {'type': 'penalise_derivatives', 'elements': torso_elements, 'xi': Xi_deriv, 'deriv': [1, 1],
         'weight': 5 * beta_torso},

        {'type': 'penalise_derivatives', 'elements': base_elements, 'xi': Xi_deriv, 'deriv': [2, 0],
         'weight': beta_base},
        {'type': 'penalise_derivatives', 'elements': base_elements, 'xi': Xi_deriv, 'deriv': [0, 2],
         'weight': beta_base},
        {'type': 'penalise_derivatives', 'elements': base_elements, 'xi': Xi_deriv, 'deriv': [1, 1],
         'weight': 5 * beta_base},

        {'type': 'penalise_derivatives', 'elements': back_elements, 'xi': Xi_deriv, 'deriv': [2, 0],
         'weight': beta_back},
        {'type': 'penalise_derivatives', 'elements': back_elements, 'xi': Xi_deriv, 'deriv': [0, 2],
         'weight': 1 * beta_back},
        {'type': 'penalise_derivatives', 'elements': back_elements, 'xi': Xi_deriv, 'deriv': [1, 1],
         'weight': 0.5 * beta_back},
    ]

    dof_y = [4]
    dof_z = [8]
    dof_z_dz1 = [8, 9, 11]
    dof_y_dy2 = [4, 6, 7]
    dof_x_dx2 = [0, 2, 3]
    dofs = [
        {'nodes': 'all', 'fix': 'all'},
        {'nodes': all_nodes, 'var': range(0, 12)},
        {'nodes': range(0, 7), 'fix': dof_z_dz1},
        {'nodes': range(0, 7), 'fix': dof_y},
        {'nodes': range(49, 53), 'fix': dof_z_dz1},
        {'nodes': range(49, 53), 'fix': dof_y},
        {'nodes': range(42, 49), 'fix': dof_z_dz1},
        {'nodes': range(42, 49), 'fix': dof_y},
        {'nodes': range(67, 72), 'fix': dof_z_dz1},
        {'nodes': range(67, 72), 'fix': dof_y},
        {'nodes': [0, 7, 14, 21, 28, 35, 42], 'fix': dof_y_dy2},
        {'nodes': [0, 7, 14, 21, 28, 35, 42], 'fix': dof_z},
        {'nodes': [52, 56, 61, 66, 71], 'fix': dof_y_dy2},
        # {'nodes': range(0, 7), 'fix': dof_y},
        # {'nodes': range(42, 49), 'fix': dof_y},
        {'nodes': [34], 'fix': dof_z_dz1},
        {'nodes': [57], 'fix': dof_z_dz1},
        {'nodes': [20, 57, 58], 'fix': [0, 4, 8]},
        {'nodes': [6, 13], 'fix': [4, 6, 8]},
        # {'nodes': [49, 53], 'fix':  [4, 6, 8]},
        # {'nodes': [57, 62, 67], 'fix':  dof_x_dx2},
        {'nodes': [6, 13, 20], 'fix': dof_y_dy2},
        # {'nodes': range(8, 20), 'fix': dof_z},
    ]

    if fix_nodes is not None:
        print 'Fixing Nodes: ', fix_nodes
        dofs.append({'nodes': fix_nodes, 'fix': range(12)})

    mesh = automesh.fit.fit_mesh(mesh, data, fits, dofs=dofs, ftol=1e-6, xtol=1e-6, maxiter=10000, dt=100, output=True)

    return mesh


def generate_plsr_model(data, exclude_subject, modes=6):
    plsr = PLSRegression(n_components=modes)
    X, Y = data['X'], data['Y']
    if exclude_subject is None:
        plsr.fit(X, Y)
    else:
        ii = range(len(data['subjects']))
        subject_idx = data['subjects'].index(exclude_subject)
        if subject_idx >= 0:
            ii.pop(subject_idx)
        plsr.fit(X[ii, :], Y[ii, :])
    return plsr




root = '/home/data/breast_project/VL'
# scan_path = os.path.join(root, 'proneT1', subject)
# segment_dir = os.path.join(root, 'seg')
# segment_params_path = os.path.join(root, 'params', 'volunteer_filter_v2.pkl')
skin_fit_params_path = ''
skin_pca_mesh_path = ''
skin_mesh_path = ''

lungs_fit_params_path = ''
lungs_pca_mesh_path = os.path.join(root, )
lungs_mesh_path = ''

ribcage_plsr_model_path = ''
ribcage_mesh_path = ''

p = {
    'crop_ranges': [[30, -30], [30, -30], [20, -20]],
    'label_ordering': 'mean_intensity',
    'small_structure_threshold': 20.,
    'denoise': {'sigma_range': 3, 'sigma_spatial': 15, 'win_size': 5},
    'kmeans': {'num_clusters': 3, 'max_iterations': 20, 'num_jobs': 1, 'entropy': {'size': 9}},
    'image_to_points': {
        'smooth_image': {'run': False, 'radius': 1., 'sd': 1.},
        'contour': {'iso_value': 0.5},
        'smooth_polydata': {'run': True, 'smooth_feature_edges': True, 'iterations': 2000},
        'decimate_polydata': {'run': True, 'ratio': 0.8, 'preserve_topology': False},
        'clean': {'run': True, 'merge_points': True, 'tolerance': 0.},
        'normals': {'calculate': True},
        'curvature': {'calculate': False},
    }}
params = automesh.Params(p)

actions = automesh.Params({'segment': True,
                           'fit_pca': True,
                           'fit_left': True,
                           'fit_right': True,
                           'predict_ribcage': True})

gen_dst = 'gen8'

subject_id = 45  #47
t0 = time.time()
for subject_id in [24]: #range(20, 81):

    subject = 'VL%05d' % subject_id
    print 'Segmenting %s' % subject

    root = '/home/data/breast_project/VL'
    scan_path = os.path.join(root, 'proneT1', subject)
    segment_dir = os.path.join(root, 'seg')

    try:
        if actions.segment:
            scan = automesh.Scan(scan_path)
            scan.set_origin([0, 0, 0])

            # Segment the scan into 3 clusters (air, muscle and lungs)
            cropped_image = automesh.crop(scan, params.crop_ranges)
            cropped_image = automesh.denoise_bilateral(cropped_image, params.denoise)
            kmeans_image = automesh.kmeans_segmentation(cropped_image, params.kmeans)

            # Extract the air image based on the minimum mean pixel intensity. This will include air in the lungs.
            ordered_labels = automesh.order_labels(kmeans_image, cropped_image, params.label_ordering)
            air_image = automesh.extract_label(kmeans_image, ordered_labels[0])
            air_image = automesh.replace_image_edge(air_image, 1, 3, axes=[0, 1])

            # Extract the lung image from the air image and convert to a point cloud
            lungs_image = automesh.extract_lungs_image(air_image)
            lungs_image = automesh.remove_small_structures(lungs_image, params.small_structure_threshold)
            lungs_pts = automesh.convert_image_to_points(lungs_image, params.image_to_points)
            np.save(os.path.join(segment_dir, 'lungs', 'kmeans', 'lungs_pts_kmeans_%s.npy' % subject), lungs_pts)

            # Extract the skin image from the air image and convert to a point cloud
            skin_image = automesh.extract_skin_image(air_image)
            skin_image = automesh.remove_small_structures(skin_image, params.small_structure_threshold)
            skin_pts = automesh.convert_image_to_points(skin_image, params.image_to_points)
            np.save(os.path.join(segment_dir, 'skin', 'kmeans', 'skin_pts_kmeans_%s.npy' % subject), skin_pts)

            automesh.pretty_time('Segment time:', t0)

        else:
            lungs_pts = np.load(os.path.join(segment_dir, 'lungs', 'kmeans', 'lungs_pts_kmeans_%s.npy' % subject))
            skin_pts = np.load(os.path.join(segment_dir, 'skin', 'kmeans', 'skin_pts_kmeans_%s.npy' % subject))

        if actions.fit_pca:
            path_formats = [
                os.path.join(root, 'meshes', 'skin_left', 'gen7', 'VL%05d_prone.mesh'),
                os.path.join(root, 'meshes', 'skin_right', 'gen3', 'VL%05d_prone.mesh'),
                os.path.join(root, 'meshes', 'lungs', 'gen1', 'VL%05d_lungs.mesh'),
            ]
            subjects = [i for i in range(20, 81) if i not in [20, 22, 26, 32, 38, 42, 43, 44, 53, 56, 60, 64, 67, 73, 76, 78]]

            if subject_id in subjects:
                subjects.remove(subject_id)  # Leave-one-out

            pca_meshes = automesh.generate_pca_meshes(path_formats, subjects, groups=['_default'], zero_on=(0, 7), modes=36)

            meshes = {'skin_left': pca_meshes[0].mesh, 'skin_right': pca_meshes[1].mesh, 'lungs': pca_meshes[2].mesh}

            meshes = fit_meshes(meshes, skin_pts, lungs_pts)

            left_pca_mesh = meshes['skin_left']
            right_pca_mesh = meshes['skin_right']
            lungs_pca_mesh = meshes['lungs']

            left_pca_mesh.save(os.path.join(root, 'meshes', 'skin_left', gen_dst, 'VL%05d_prone.mesh' % subject_id))
            right_pca_mesh.save(os.path.join(root, 'meshes', 'skin_right', gen_dst, 'VL%05d_prone.mesh' % subject_id))
            lungs_pca_mesh.save(os.path.join(root, 'meshes', 'lungs', gen_dst, 'VL%05d_prone.mesh' % subject_id))

            automesh.pretty_time('PCA Fit time:', t0)

        else:
            left_pca_mesh = morphic.Mesh(os.path.join(root, 'meshes', 'skin_left', gen_dst, 'VL%05d_prone.mesh' % subject_id))
            right_pca_mesh = morphic.Mesh(os.path.join(root, 'meshes', 'skin_right', gen_dst, 'VL%05d_prone.mesh' % subject_id))
            lungs_pca_mesh = morphic.Mesh(os.path.join(root, 'meshes', 'lungs', gen_dst, 'VL%05d_prone.mesh' % subject_id))

            for mesh in [left_pca_mesh, right_pca_mesh, lungs_pca_mesh]:
                mesh.translate('dx', ['pca'])

        if actions.fit_left:
            left_mesh = automesh.collapse_pca_mesh(left_pca_mesh)
            left_mesh = fit_skin_mesh(left_mesh, skin_pts, subject)
            left_mesh.save(os.path.join(root, 'meshes', 'skin_left', gen_dst, 'VL%05d_prone_skin_left.mesh' % subject_id))

            automesh.pretty_time('Left Fit time:', t0)
        else:
            left_mesh = morphic.Mesh(os.path.join(root, 'meshes', 'skin_left', gen_dst, 'VL%05d_prone_skin_left.mesh' % subject_id))

        if actions.fit_right:
            right_mesh = automesh.collapse_pca_mesh(right_pca_mesh)
            right_mesh = fit_skin_mesh(right_mesh, skin_pts, subject)
            right_mesh.save(os.path.join(root, 'meshes', 'skin_right', gen_dst, 'VL%05d_prone_skin_right.mesh' % subject_id))

            automesh.pretty_time('Right Fit time:', t0)
        else:
            right_mesh = morphic.Mesh(os.path.join(root, 'meshes', 'skin_right', gen_dst, 'VL%05d_prone_skin_right.mesh' % subject_id))

        if actions.predict_ribcage:
            # Predict ribcage mesh from lung mesh using a PLSR model
            plsr_data = pickle.load(open(os.path.join(root, 'pop_models', 'ribcage_plsr_data.pkl'), 'r'))
            ribcage_plsr_model = generate_plsr_model(plsr_data, subject_id)
            x_input = []
            x0 = np.mean([lungs_pca_mesh.nodes[nid].values[:, 0] for nid in range(2, 26)], 0)
            for nid in range(2, 26):
                x = lungs_pca_mesh.nodes[nid].values
                x[:, 0] -= x0
                x_input.extend(np.array(x.flatten()).tolist())
            x_input = np.array(x_input)
            y_output = ribcage_plsr_model.predict(x_input)

            ribcage_mesh = morphic.Mesh(os.path.join(root, 'meshes', 'ribcage', 'gen0', 'VL%05d_ribcage_prone.mesh' % subject_id))
            idx = 0
            for nid in range(32):
                cids = ribcage_mesh.nodes[nid].cids
                ribcage_mesh._core.P[cids] = y_output[idx:idx + len(cids)]
                ribcage_mesh.nodes[nid].values[:, 0] += x0
                idx += len(cids)

            ribcage_mesh.generate()
            ribcage_mesh.save(os.path.join(root, 'meshes', 'ribcage', 'gen8', 'VL%05d_ribcage_prone.mesh' % subject_id))

            automesh.pretty_time('Predict time:', t0)

    except KeyboardInterrupt as e:
        break

    except Exception as e:
        import traceback
        traceback.print_exc()
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', subject_id, 'FAILED!'

    automesh.pretty_time('Total time:', t0)

gui.plot('lung_pts', lungs_pts, size=0, color=(0, 1, 0))
gui.plot('skin_pts', skin_pts, size=0, color=(0, 1, 1))

gui.plot('lung_mesh', meshes['lungs'].get_surfaces())
gui.plot('left', left_mesh.get_surfaces())
gui.plot('right', meshes['skin_right'].get_surfaces())


