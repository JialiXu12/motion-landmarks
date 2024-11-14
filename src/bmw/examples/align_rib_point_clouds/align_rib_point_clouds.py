import os
import sys
import scipy
import bmw
import morphic
reload(morphic)
import h5py
import automesh
import point_clouds

def run(volunteer_num):
    os.environ['QT_API'] = 'pyqt'
    volunteer_id = 'VL{0:05d}'.format(volunteer_num)
    p = {
        'debug' : False,
        'offscreen': True,
        'seg_dir' : '../convert_dicom_to_nii/results/segmentations/',
        'results_dir': os.path.join('results',volunteer_id),
        'volunteer_num' : volunteer_num,
        'volunteer_id': volunteer_id}
    params = automesh.Params(p)
    if not os.path.exists(params.results_dir):
        os.makedirs(params.results_dir)

    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
    fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen

    prone_seg_path = os.path.join(params.seg_dir,'{0}_ribseg_{1}.nii'.format('prone', params.volunteer_num))
    supine_seg_path = os.path.join(params.seg_dir,'{0}_ribseg_{1}.nii'.format('supine', params.volunteer_num))

    prone_points_fname = os.path.join(params.results_dir,'prone_ribpoints_{0}.h5'.format(volunteer_num))
    if os.path.exists(prone_points_fname):
        prone_points = bmw.load_hdf5(prone_points_fname, '/points')
    else:
        prone_points = point_clouds.convert_mask_to_point_cloud(prone_seg_path)
        bmw.save_hdf5(prone_points_fname, '/points', prone_points)

    supine_points_fname = os.path.join(params.results_dir,'supine_ribpoints_{0}.h5'.format(params.volunteer_num))
    if os.path.exists(supine_points_fname):
        supine_points = bmw.load_hdf5(supine_points_fname, '/points')
    else:
        supine_points = point_clouds.convert_mask_to_point_cloud(supine_seg_path)
        bmw.save_hdf5(supine_points_fname, '/points', supine_points)

    # Subsample points
    num_subsampled_pts = 5000

    subsampled_prone_points_fname = os.path.join(params.results_dir,'prone_ribpoints_subsampled_{0}.h5'.format(params.volunteer_num))
    prone_idxs = scipy.random.randint(0, high=prone_points.shape[0], size=num_subsampled_pts)
    subsampled_prone_points = prone_points[prone_idxs]
    bmw.save_hdf5(subsampled_prone_points_fname, '/points', subsampled_prone_points)

    subsampled_supine_points_fname = os.path.join(params.results_dir,'supine_ribpoints_subsampled_{0}.h5'.format(params.volunteer_num))
    supine_idxs = scipy.random.randint(0, high=supine_points.shape[0], size=num_subsampled_pts)
    subsampled_supine_points = supine_points[supine_idxs]
    bmw.save_hdf5(subsampled_supine_points_fname, '/points', subsampled_supine_points)

    from bmw import icp
    tOpt, transformed_points = icp.fitDataRigidTranslation(subsampled_prone_points, subsampled_supine_points, xtol=1e-5, maxfev=0 )

    transformed_prone_points_fname = os.path.join(params.results_dir,'prone_ribpoints_subsampled_transformed_{0}.h5'.format(params.volunteer_num))
    bmw.save_hdf5(transformed_prone_points_fname, '/points', transformed_points)

    if not params.offscreen:
        fig.plot_points('pronePoints', subsampled_prone_points, color=(1,0,0), size=5)
        fig.plot_points('supinePoints', subsampled_supine_points, color=(0,1,0), size=5)
        fig.plot_points('transformedPoints', transformed_points, color=(0,0,1), size=5)

    fname = '{0}/mirtk_init_prone_to_supine_rigid_alignment_{1}.csh'.format(params.results_dir,params.volunteer_num)
    file_id = open(fname, 'w')
    file_id.write('#!/bin/tcsh\n')
    file_id.write('set VTK_DIR = /home/psam012/usr/vtk/VTK6.0.0/\n')
    file_id.write('set PATH = "$VTK_DIR/bin/:$PATH"\n')
    file_id.write('set LD_LIBRARY_PATH = "$VTK_DIR/lib/:$LD_LIBRARY_PATH"\n')
    #file_id.write('mirtk init-{0}/dof prone_to_supine_initial_alignment_{1} -rigid -noscaling -noshearing -tx {2} -ty {3} -tz {4}\n'.format(
    #    params.results_dir, params.volunteer_num,tOpt[0],tOpt[1],tOpt[2]))
    file_id.write('ls -lrt\n')
    file_id.close()
    import stat
    os.chmod(fname, stat.S_IRWXU)
    import ipdb; ipdb.set_trace()
    os.system(fname)

if __name__ == "__main__":
    volunteer_num = 25
    run(volunteer_num)
