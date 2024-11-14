import sys
import os
os.environ['QT_API'] = 'pyqt'
import ipdb
import scipy
import h5py
from scipy.spatial import cKDTree

import morphic
reload(morphic)
import bmw
from bmw import imaging_functions
import automesh

import time
import datetime
import pickle

def in_circle(center_x, center_y, radius, x, y):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2

# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: define the line
    p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction; does not need to be normalized.

    return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane
        return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
        )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
        )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )


def load_landmark(filepath):
    data = []
    if os.path.exists(filepath):
        d = pickle.load(open(filepath, 'r'))
        if 'default' in d['data']: 
            if type(d['data']['default']) is list:
                if len(d['data']['default']) == 3:
                    data = d['data']['default']
            else:
                if d['data']['default'].size == 3:
                    data = d['data']['default']
    return data


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

def generate_image_coordinates2D(image_shape, spacing):
    #import ipdb; ipdb.set_trace()
    #print scan.values.shape
    x, y = scipy.mgrid[0:image_shape[0],0:image_shape[1]]
    #print x.shape, y.shape, z.shape 
    x = x*spacing[0]
    y = y*spacing[1]

    #import ipdb; ipdb.set_trace()
    image_coor = scipy.vstack((x.ravel(),y.ravel())).transpose()
    return image_coor, x, y

def dcmstack_dicom_to_nifti(dicom_dir):
    '''
    Load original dicom using dcmstack and convert to nifti.
    The original dicom pixel data and the nifti generate through dcmstack
    match in itksnap 
    '''
    import dcmstack
    from glob import glob
    src_paths = glob(os.path.join(dicom_dir,'*'))
    stacks = dcmstack.parse_and_stack(src_paths)
    stack = stacks[stacks.keys()[0]]
    stack_data = stack.get_data()
    stack_affine = stack.get_affine()
    dicom_to_nifti = stack.to_nifti()
    return dicom_to_nifti


if __name__ == "__main__":

    start = time.time()

    debug = True
    offset = 0
    arguments = bmw.parse_arguments()
    if arguments.volunteer is None:
        volunteer = 'VL00025'
        offscreen = False
        parameters = scipy.array([0.2, 5., 100.,100.,2.])
        parameter_set = None
        field_export_name = 'field'
    else:
        volunteer = arguments.volunteer
        parameter_set = arguments.parameter_set
        parameter_set_values = scipy.arange(0.125,0.35,0.05)[::-1]
        parameters = scipy.array([parameter_set_values[parameter_set], 5., 100.,100.,2.])
        field_export_name = 'parameter_set_{0}_field'.format(parameter_set)
        offscreen = arguments.offscreen
    print 'Parameter set: {0}'.format(parameter_set)

    mesh_dir = './../../data/'


    # Load fitted ribcage (cwm), right skin (bm_lhs), and left skin (bm_rhs) surfaces
    updated_meshes = True
    if updated_meshes:
        results_dir = './warping_results/{0}/'.format(volunteer)
        op = bmw.volunteer_setup(mesh_dir, results_dir, volunteer, parameters, offscreen)
        cwm_fname = '{0}/2016-05-26/{1}/ribcage_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_rhs_fname = '{0}/2016-05-26/{1}/skin_right_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_lhs_fname = '{0}/2016-05-26/{1}/skin_left_prone.mesh'.format(op.mesh_dir, op.volunteer)
    else:
        results_dir = './results/{0}/'.format(volunteer)
        op = bmw.volunteer_setup(mesh_dir, results_dir, volunteer, parameters, offscreen)
        cwm_fname = '{0}/volunteer/ribcages/{1}_ribcage_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_rhs_fname = '{0}/volunteer/right_skin_meshes/{1}_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_lhs_fname = '{0}/volunteer/left_skin_meshes/{1}_prone.mesh'.format(op.mesh_dir, op.volunteer)


    fig = op.add_fig('prone')
    prone_mesh = morphic.Mesh(op.results_dir + '/prone.mesh')
    bmw.visualise_mesh(prone_mesh, fig, visualise=True, face_colours=(0,1,1),pt_size=1, opacity=0.5, line_opacity = 0.75, text=False) #text_elements=spine_elements.tolist())

    p = {
        'dicom_dir': os.path.join(op.mesh_dir,'volunteer/images/dicom/',op.volunteer),
        'results_dir': './warping_results/',
        'landmark_dir': os.path.join(op.mesh_dir,'landmarks/prone/',op.volunteer),
        }
    params = automesh.Params(p)

    export_nifti = False
    dicom_to_nifti = dcmstack_dicom_to_nifti(params.dicom_dir)
    if export_nifti:
        dicom_to_nifti.to_filename(
            os.path.join(params.results_dir,'dicom_to_nifti.nii.gz'))

    
    '''
    Load original dicom using automesh and convert to nifti. The pixel data
    no longer match with the original dicom in itksnap. This means that automesh has
    loaded the pixel data in a different way. The pixel data needs to be
    transformed in order for the automesh loaded pixel data to match the
    original dicom. 
    '''
    image = automesh.Scan(params.dicom_dir)
    image.set_origin([0, 0, 0])
    import nibabel as nii
    automesh_to_nifti = nii.nifti1.Nifti1Image(image.values,
                                        dicom_to_nifti.affine,
                                        dicom_to_nifti.header)

    test_conversion = False
    if test_conversion:
        transformed_image_values = scipy.copy(image.values).astype(scipy.int16)
        for img_slice in range(image.shape[-1]):
            transformed_image_values[:,:,img_slice] = scipy.rot90(scipy.transpose(image.values[:,:,img_slice]),2)

        import nibabel as nii
        automesh_to_nifti = nii.nifti1.Nifti1Image(transformed_image_values,
                                            dicom_to_nifti.affine,
                                            dicom_to_nifti.header)
        automesh_to_nifti.to_filename(
            os.path.join(params.results_dir,'automesh_to_nifti.nii.gz'))

        #if not scipy.allclose(transformed_image_values,stack_data):
        #    raise ValueError('DICOM and nifti pixel data do not match')

    view_dicom = True
    coor, x, y, z = generate_image_coordinates(image.shape, image.spacing)
    if view_dicom:
        from mayavi import mlab
        src = mlab.pipeline.scalar_field(x,y,z,image.values)
        plane = mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=int(0.5 * image.num_slices),
                            colormap='black-white')


    filepath = os.path.join(params.landmark_dir,'sternum.sternal_angle.point')
    sternal_angle = load_landmark(filepath)
    label = 'sternal_angle'
    bmw.plot_points(fig, label, [sternal_angle], [[label]], visualise=True, colours=(0,0,1), point_size=10, plot_text=True)

    filepath = os.path.join(params.landmark_dir,'sternum.superior.point')
    sternum_superior = load_landmark(filepath)
    label = 'sternum_superior'
    bmw.plot_points(fig, label, [sternum_superior], [[label]], visualise=True, colours=(0,0,1), point_size=10, plot_text=True)

    import ipdb; ipdb.set_trace()

    rhs_elements = range(0,66)
    lhs_elements = range(66,132)
    nifti_image = imaging_functions.MRImage(image=automesh_to_nifti)
    #import ipdb; ipdb.set_trace()
    embedded_points = bmw.embed_mri(fig, prone_mesh, nifti_image, coor, params, element_groups=[lhs_elements, rhs_elements], num_element_pts=40)#75)
    supine_mesh = morphic.Mesh(op.results_dir + '/supine.mesh')
#    #import ipdb; ipdb.set_trace()
    bmw.warp_mri(fig, embedded_points, supine_mesh, nifti_image, coor, params, element_groups=[lhs_elements, rhs_elements])




    #import ipdb; ipdb.set_trace()

        #bmw.save_vtk(image,image.values,params.results_dir,'test.vtk')
        #import ipdb; ipdb.set_trace()
        #import medpy
        #image_data, image_header = medpy.io.load('path/to/image.xxx')

        #from dicom.contrib import pydicom_series
        #[series] = pydicom_series.read_files(dicom_dir, False, True) # second to not show progress bar, third to retrieve data
        #image_data = series.get_pixel_array()
        #medpy.io.save(image.values, os.path.join(params.results_dir,'test.nii.gz'), False)


        #       


        #from mayavi import mlab
#        src1 = mlab.pipeline.scalar_field(image.values)
#        src1.origin = image.origin
#        src1.spacing = image.spacing
#        plane = mlab.pipeline.image_plane_widget(src1,
#                            plane_orientation='z_axes',
#                            slice_index=int(0.5 * image.num_slices),
#                            colormap='black-white')
        #from mayavi import mlab
        #coor, x, y, z = generate_image_coordinates(image.shape, image.spacing)
        #src = mlab.pipeline.scalar_field(x,y,z,image.values)
        #bmw.save_vtk(image,image.values,params.results_dir,'test.vtk')
#        plane = mlab.pipeline.image_plane_widget(src2,
#                            plane_orientation='z_axes',
#                            slice_index=int(0.5 * image.num_slices),
#                            colormap='black-white')

        # Seperately embed pixels in elements on the lhs and rhs side of the the mesh (for speed up)





#vtk_image = convert_numpy_array_to_vtk_image(image.values, np.int16, flipDim=True)


        #from morphic import viewer
        #fig = viewer.Figure('prone')
        #fig.plot_dicoms('prone', image)


        #vtk_image = automesh.convert_numpy_array_to_vtk_image(image.values, scipy.int16, flipDim=True)
        #image_points = automesh.convert_image_to_points(image, params.image_to_points)

#    polydata = convert_vtk_image_to_polydata(vtk_image, params)
#    vertices, triangles, normals = convert_polydata_to_triangular_mesh(polydata)








#        op.scan = bmw.load_dicom_attributes(
#            '{0}/data/volunteer/{1}/prone/dicom'.format(op.root_dir, op.volunteer))
#        op.image_coor, x, y, z = imaging_functions.generate_image_coordinates(op.scan)
#        op.src = viewer.define_scalar_field(x,y,z,op.scan)
#        #import ipdb; ipdb.set_trace()
#        #from tvtk.api import write_data
#        #write_data(op.src,'./aa.vtk')
#        from mayavi import mlab
#        #op.src = None
#        #op.image_coor = None
#        #plane = fig.visualise_dicom_plane(fig, op.scan, op.src, op)
#        plane = mlab.pipeline.image_plane_widget(op.src,
#                            plane_orientation='z_axes',
#                            slice_index=int(0.5 * op.scan.num_slices),
#                            colormap='black-white', figure=fig.figure)










#    prone_nii = bmw.MRImage(label='prone',  filename=op.image_dir +'/prone.nii.gz')

#    #bmw.mask_mri(fig, prone_mesh, prone_nii, 'prone_mask',  params)

#    # Seperately embed pixels in elements on the lhs and rhs side of the the mesh (for speed up)
#    rhs_elements = range(0,66)
#    lhs_elements = range(66,132)
#    bmw.embed_mri(fig, prone_mesh, prone_nii, params, element_groups=[lhs_elements, rhs_elements], num_element_pts=75)

#    supine_mesh = morphic.Mesh(params.results_dir + '/supine.mesh')
#    #import ipdb; ipdb.set_trace()

#    prone_nii = bmw.MRImage(label='prone',  filename=params.results_dir +'/aa.nii.gz')
#    bmw.warp_mri(fig, supine_mesh, prone_nii, params, element_groups=[lhs_elements, rhs_elements])

#    # run your code
#    end = time.time()

#    elapsed = end - start
#    print str(datetime.timedelta(seconds=elapsed))

