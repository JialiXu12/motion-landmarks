'''
Warps a prone MRI image based on a prone to supine mechanics solution
'''
import os
import sys
import scipy
import bmw
import morphic
reload(morphic)
import h5py
import automesh
os.environ['QT_API'] = 'pyqt'
script_id = 'warp_image'
run_program = 'python'
run_script = 'warp_image.py'
run_on_pipeline = False
if run_on_pipeline:
    depends_on = ['prone_to_supine']
else:
    depends_on = []

def generate_image_coordinates(image_shape, spacing):
    #import ipdb; ipdb.set_trace()
    #print scan.values.shape
    x, y, z = scipy.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    #print x.shape, y.shape, z.shape 
    x = x*spacing[0]
    y = y*spacing[1]
    z = z*spacing[2]

    #import ipdb; ipdb.set_trace()
    image_coor = scipy.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z

def run(process):

    warping_wksp = process.workspace('warping', True)
    if run_on_pipeline:
        mechanics_wksp = process.parent.workspace('mechanics')
        volunteer_id = process.parent.metadata['subject']['id']
    else:
        os.environ['QT_API'] = 'pyqt'
        volunteer_id = 'VL00025'
        warping_wksp._path_ = os.path.join('/home/psam012/Documents/opt/bmw/examples/warp_image/results/',volunteer_id)
        mechanics_wksp = process.workspace('mechanics', True)
        mechanics_wksp._path_ = os.path.join('/home/psam012/Documents/opt/bmw/examples/warp_image/mechanics/results/',volunteer_id)
        data_dir = '/home/psam012/Documents/opt/bmw/data'
        image_type = 'prone'
        if not os.path.exists(warping_wksp.path()):
            os.makedirs(warping_wksp.path())
    p = {
        'debug' : True,
        'offscreen': True,
        'mechanics_dir': mechanics_wksp.path(),
        'results_dir': warping_wksp.path(),
        'volunteer_id': volunteer_id,
        'dicom_zip_path': os.path.join(data_dir,'volunteer/dicom/',image_type,'{0}.zip'.format(volunteer_id)),
        'extracted_dicom_dir': os.path.join(data_dir,'volunteer/extracted_dicom/',image_type,'{0}'.format(volunteer_id)),
        }
    params = automesh.Params(p)

    if not os.path.exists(params.extracted_dicom_dir):
        os.makedirs(params.extracted_dicom_dir)

    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
    fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen
    #import ipdb; ipdb.set_trace()

    # Load 3D prone meshe
    prone_fname = mechanics_wksp.path('prone.mesh')
    if os.path.exists(prone_fname):
        prone_mesh = morphic.Mesh(prone_fname)
        prone_mesh.label = 'prone'
    else:
        process.completed(False, '3D prone mesh not found')

    # Load 3D supine meshe
    supine_fname = mechanics_wksp.path('supine.mesh')
    if os.path.exists(supine_fname):
        supine_mesh = morphic.Mesh(supine_fname)
        supine_mesh.label = 'supine'
    else:
        process.completed(False, '3D supine mesh not found')
    #supine_mesh = prone_mesh


    if params.debug:
        bmw.visualise_mesh(prone_mesh, fig, visualise=True, face_colours=(1,0,0), opacity=0.75)
        bmw.visualise_mesh(supine_mesh, fig, visualise=True, face_colours=(1,1,0), opacity=0.75)

    #import ipdb; ipdb.set_trace()
    if os.listdir(params.extracted_dicom_dir) == []:
        bmw.extract_zipfile(params.dicom_zip_path, params.extracted_dicom_dir)
    from bmw import imaging_functions
    temp_nifti = imaging_functions.dcmstack_dicom_to_nifti(params.extracted_dicom_dir)
    temp_nifti = imaging_functions.nifti_zero_origin(temp_nifti)
    temp_nifti = imaging_functions.nifti_set_RAI_orientation(temp_nifti)

    # Create nifti from pixel data loaded by automesh
    image = automesh.Scan(params.extracted_dicom_dir)
    image.set_origin([0, 0, 0])
    coor, x, y, z = generate_image_coordinates(image.shape, image.spacing)
    import nibabel as nii
    nifti = nii.nifti1.Nifti1Image(image.values.astype(scipy.int16),
                                   temp_nifti.affine,
                                   temp_nifti.header)
    # Wrap the nifti image using the convinence functions provided by the imaging_functions module
    nifti = imaging_functions.MRImage(image=nifti)

    #rhs_elements = range(0,66)
    #lhs_elements = range(66,132)
    #lhs_elements = [77,78,88,89,99,100]
    #rhs_elements = [11,12,22,23,33,34]

    rhs_posterior_elems = [7, 8, 9, 10, 18, 19, 20, 21, 29, 30, 31, 32, 40, 41, 42, 43, 51, 52, 53, 54, 62, 63, 64, 65]
    rhs_anterior_elems = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 60, 61]
    lhs_posterior_elems = [73, 74, 75, 76, 84, 85, 86, 87, 95, 96, 97, 98, 106, 107, 108, 109, 117, 118, 119, 120, 128, 129, 130, 131]
    lhs_anterior_elems = [66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 88, 89, 90, 91, 92, 93, 94, 99, 100, 101, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126, 127]

    embedded_points = bmw.embed_mri(fig, prone_mesh, nifti, coor, params, element_groups=[rhs_anterior_elems+lhs_anterior_elems], num_element_pts=70,num_surface_pts=150, debug=False)
    bmw.warp_mri(fig, embedded_points, supine_mesh, nifti, coor, params, element_groups=[rhs_anterior_elems+lhs_anterior_elems], num_surface_pts=150)


if __name__ == "__main__":
    import bpm
    run(bpm.get_project_process())
