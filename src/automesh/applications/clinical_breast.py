import os
import automesh

subject_id = 45
subject = 'VL%05d' % subject_id

root = '/home/data/breast_project/CL'
scan_path = os.path.join(root, 'proneT1', subject)
segment_params_path = os.path.join(root, 'params', 'clinical_filter_v1.pkl')
skin_fit_params_path = ''
skin_pca_mesh_path = ''
skin_mesh_path = ''

lungs_fit_params_path = ''
lungs_pca_mesh_path = os.path.join(root, )
lungs_mesh_path = ''

ribcage_plsr_model_path = ''
ribcage_mesh_path = ''


scan = automesh.Scan(scan_path)

if scan.protocol == 'Breast Inhanced':
    segment_params_path = os.path.join(root, 'params', 'clinical_filter_breast_inhanced_v1.pkl')
else:
    segment_params_path = os.path.join(root, 'params', 'clinical_filter_v3.pkl')


params = automesh.FilterParams(segment_params_path)

cropped = automesh.crop(scan, params)
cropped = automesh.denoise_bilateral(cropped, params)
forest = automesh.random_forest(cropped, params)

skin_pts = automesh.segment_skin_points(cropped, forest, params)
lungs_pts = automesh.segment_lung_points(cropped, forest, params)