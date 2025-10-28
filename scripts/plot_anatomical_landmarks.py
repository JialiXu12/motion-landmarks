import os
import numpy as np
from tools import landmarks as ld
import breast_metadata
import pyvista as pv

positions = ['prone', 'supine']
# vl_ids = [9,11,12,14,15,17,18,19,20,22,25,29,30,31,32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,
#               54,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,81,82,84,85,86,87,88,89]
vl_ids = [9,11,12]
orientation_flag = "RAI"
mri_t2_images_root_path = r'U:\projects\volunteer_camri\old_data\mri_t2'

rigid_landmarks_root_path = r'U:\sandbox\jxu759\motion_of_landmarks\anna_data\picker\points_wc_raf'
nipple_path = os.path.join(rigid_landmarks_root_path, 'user002')
sternum_path = os.path.join(rigid_landmarks_root_path, 'user001')

metadata_all = {}

for position in positions:
    metadata = ld.read_metadata(vl_ids, position, mri_t2_images_root_path, nipple_path, sternum_path)
    metadata_all[position] = metadata

for vl in vl_ids:
    for position in positions:
        mri_t2_images_path = os.path.join(mri_t2_images_root_path, 'VL{0:05d}'.format(vl), position)
        mri_t2_images = breast_metadata.Scan(mri_t2_images_path)
        mri_t2_images_grid = breast_metadata.SCANToPyvistaImageGrid(mri_t2_images, orientation_flag)

        skin_mask_path = os.path.join(r'U:\sandbox\jxu759\volunteer_seg\results',position,'body','body_VL{0:05d}.nii.gz'.format(vl))
        skin_mask = breast_metadata.readNIFTIImage(skin_mask_path, orientation_flag='RAI', swap_axes=True)
        skin_points = ld.extract_contour_points(skin_mask, 100000)

        left_nipple = metadata_all[vl][position]['left_nipple']
        right_nipple = metadata_all[vl][position]['right_nipple']
        sternum = metadata_all[vl][position]['sternum_position']


        plotter = pv.Plotter()
        opacity = np.linspace(0, 0.4, 100)
        plotter.add_volume(mri_t2_images_grid, scalars='values', cmap='gray', opacity=opacity)
        plotter.add_points(skin_points, color='grey', render_points_as_spheres=True, label='Point cloud', point_size=14)
        plotter.add_points(left_nipple, color='red', render_points_as_spheres=True,
                           label='Point cloud', point_size=12)
        plotter.add_points(sternum, color='blue', render_points_as_spheres=True,
                           label='Point cloud', point_size=12)
        legend_entries = [['Images', 'grey'], ['Point cloud', 'lightskyblue'], ['nipple', 'pink'],['sternum','blue']]
        plotter.add_legend(legend_entries, bcolor='w')
        plotter.add_text(f'Check anatomical landmarks for VL{vl:05d}', position='upper_left', font_size=10, color='black')

        plotter.show()