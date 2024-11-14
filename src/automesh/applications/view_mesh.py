import morphic
import numpy as np
import os
import sys

import phaser

image_path = '/home/data/breast_project/VL/proneT1'
if 'gui' not in locals():
    gui = phaser.ProjectGUI('volunteers', image_path)
    gui.configure_traits()
else:
    gui = locals()['gui']

gen = 'gen8'
if 'subject_id' not in locals():
    subject_id = 19
else:
    subject_id = locals()['subject_id']

if len(sys.argv) == 1:
    subject_id += 1
elif len(sys.argv) == 2:
    if sys.argv[1][:3] == 'gen':
        gen = sys.argv[1]
        subject_id += 1
    else:
        subject_id = int(sys.argv[1])
elif len(sys.argv) == 3:
    gen = sys.argv[1]
    subject_id = int(sys.argv[2])

root = '/home/data/breast_project/VL'
subject = 'VL%05d' % subject_id
scan_path = os.path.join(root, 'proneT1', subject)
segment_dir = os.path.join(root, 'seg')

print 'Loading %s' % subject

lungs_pts = np.load(os.path.join(segment_dir, 'lungs', 'kmeans', 'lungs_pts_kmeans_%s.npy' % subject))
skin_pts = np.load(os.path.join(segment_dir, 'skin', 'kmeans', 'skin_pts_kmeans_%s.npy' % subject))

mesh_lungs = morphic.Mesh(os.path.join(root, 'meshes', 'lungs', gen, 'VL%05d_prone.mesh') % subject_id)
mesh_lungs.translate('dx', ['pca'], update=True)

# mesh_left = morphic.Mesh(os.path.join(root, 'meshes', 'skin_left', gen, 'VL%05d_prone.mesh') % subject_id)
# mesh_right = morphic.Mesh(os.path.join(root, 'meshes', 'skin_right', gen, 'VL%05d_prone.mesh') % subject_id)
# for mesh in [mesh_left, mesh_right]:
#     mesh.translate('dx', ['pca'], update=True)

gen_dst = 'gen8'
mesh_left = morphic.Mesh(os.path.join(root, 'meshes', 'skin_left', gen_dst, 'VL%05d_prone_skin_left.mesh' % subject_id))
mesh_right = morphic.Mesh(os.path.join(root, 'meshes', 'skin_right', gen_dst, 'VL%05d_prone_skin_right.mesh' % subject_id))
mesh_ribcage = morphic.Mesh(os.path.join(root, 'meshes', 'ribcage', gen_dst, 'VL%05d_ribcage_prone.mesh' % subject_id))


gui.plot('lung_pts', lungs_pts, size=0, color=(0, 1, 0))
gui.plot('skin_pts', skin_pts, size=0, color=(0, 1, 1))

gui.plot('lungs', mesh_lungs.get_surfaces(), color=(0, 0, 1))
gui.plot('left', mesh_left.get_surfaces())
gui.plot('right', mesh_right.get_surfaces())
gui.plot('ribcage', mesh_ribcage.get_surfaces(), color=(1, 0, 1))