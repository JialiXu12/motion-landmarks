import os
import bmw
from scipy.spatial import cKDTree
import numpy as np
root_path = '/home/psam012/opt/breast-data/'
prone_model_path = '/home/psam012/opt/breast-data/prone_to_supine_t2/2018-05-23/'
vl_id = 25
vl_id_str = 'VL{0:05d}'.format(vl_id)
prone_mesh_path = os.path.join(prone_model_path, 'mechanics_meshes', vl_id_str)
prone_m = bmw.load_volume_mesh(prone_mesh_path, 'prone')
print 'a'

points, xi, elements = bmw.generate_points_in_elements(prone_m, num_points=4)

tree = cKDTree(points)
landmarks = np.array([[ 0.        ,  0.        ,  0.        ],[ 0.        ,  0.        ,  2.        ]])
dist, idx = tree.query(landmarks)


prone_m.evaluate(elements[idx],xi[idx][0])