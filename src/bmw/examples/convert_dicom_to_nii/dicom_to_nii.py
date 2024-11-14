'''
Solve prone to supine mechanics
'''
import os
import sys
import scipy
import bmw
import morphic
reload(morphic)
import automesh

def run(volunteer_num, image_type):

    os.environ['QT_API'] = 'pyqt'
    volunteer_id = 'VL{0:05d}'.format(volunteer_num)
    data_dir = '/home/psam012/Documents/opt/bmw/data/'
    p = {
        'debug' : False,
        'offscreen': True,
        'seg_dir' : './results/segmentations/'
        'results_dir': os.path.join('results',volunteer_id),
        'data_dir' : data_dir,
        'dicom_zip_path': os.path.join(data_dir,'volunteer/images/dicom/',image_type,'{0}.zip'.format(volunteer_id)),
        'volunteer_num' : volunteer_num,
        'volunteer_id': volunteer_id}
    params = automesh.Params(p)
    if not os.path.exists(params.results_dir):
        os.makedirs(params.results_dir)
    extracted_dicom_dir = os.path.splitext(params.dicom_zip_path)[0]

    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
    fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen

    # Extract dicom in the data folder
    if not os.path.exists(extracted_dicom_dir):
        bmw.extract_zipfile(params.dicom_zip_path, extracted_dicom_dir)
    from bmw import imaging_functions
    temp_nifti = imaging_functions.dcmstack_dicom_to_nifti(extracted_dicom_dir)
    temp_nifti = imaging_functions.nifti_zero_origin(temp_nifti)
    temp_nifti = imaging_functions.nifti_set_RAI_orientation(temp_nifti)

    # Create nifti from pixel data loaded by automesh
    image = automesh.Scan(extracted_dicom_dir)
    image.set_origin([0, 0, 0])
    import nibabel as nii
    nifti = nii.nifti1.Nifti1Image(image.values.astype(scipy.int16),
                                   temp_nifti.affine,
                                   temp_nifti.header)
    # Wrap the nifti image using the convinence functions provided by the imaging_functions module
    nifti = imaging_functions.MRImage(image=nifti)
    bmw.export_mask('{0}_{1}_orientation_corrected'.format(volunteer_id, image_type), image.values.astype(scipy.int16), nifti, save_folder=params.results_dir)

    # Reset origin of segmentation
    mask = nii.load(os.path.join(seg_dir,'{0}_ribseg_{1}.nii'.format(image_type, params.volunteer_num)))
    mask = imaging_functions.nifti_zero_origin(mask)
    mask.to_filename(os.path.join(params.results_dir,'{0}_{1}_ribseg.nii.gz'.format(volunteer_id, image_type)))

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    image_type = 'prone'
    for volunteer in range(25,25+1):
        run(volunteer_num, image_type)

    image_type = 'supine'
    for volunteer in range(25,25+1):
        run(volunteer_num, image_type)
