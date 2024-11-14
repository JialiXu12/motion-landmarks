
import os
import scipy
import numpy
import h5py
from scipy.spatial import cKDTree
from bmw import mesh_generation
from bmw import imaging_functions
from tools import sitkTools as tools
import bmw
from scipy.interpolate import griddata
import numpy as np



def generate_model_surface_points_for_masking(fig, mesh, num_pts = 110):
    debug = False
    cranial_pts = mesh_generation.generate_points_on_face(
        mesh, "xi2", 0,elements=mesh.elements.get_groups('cranial'),
        num_points=num_pts)
    caudal_pts = mesh_generation.generate_points_on_face(
        mesh, "xi2", 1, elements=mesh.elements.get_groups('caudal'),
        num_points=num_pts)
    sternum_pts = mesh_generation.generate_points_on_face(
        mesh, "xi1", 0, elements=mesh.elements.get_groups('sternum'),
        num_points=num_pts)
    spine_pts = mesh_generation.generate_points_on_face(
        mesh, "xi1", 1, elements=mesh.elements.get_groups('spine'),
        num_points=num_pts)
    chestwall_pts = mesh_generation.generate_points_on_face(
        mesh, "xi3", 0, elements=mesh.elements.get_groups('chestwall'),
        num_points=num_pts)
    skin_pts = mesh_generation.generate_points_on_face(
        mesh, "xi3", 1, elements=mesh.elements.get_groups('skin'),
        num_points=num_pts)
    #import ipdb; ipdb.set_trace()
    surface_pts = scipy.vstack([cranial_pts, caudal_pts, sternum_pts,
        spine_pts, chestwall_pts, skin_pts])
    if debug:
        fig.plot_points(
            'surface_pts', surface_pts,
            color=(0,1,0), size=2)

    return surface_pts



def automesh_pixels_to_nifti_transform(image_values, image_shape, image_dtype=scipy.int16):
    transformed_image_values = scipy.copy(image_values).astype(image_dtype)
    for img_slice in range(image_shape[-1]):
        transformed_image_values[:,:,img_slice] = scipy.rot90(image_values[:,:,img_slice],1)

    return transformed_image_values

def nifti_to_automesh_pixels_transform(image_values, image_shape, image_dtype=scipy.int16):
    transformed_image_values = scipy.copy(image_values).astype(image_dtype)
    for img_slice in range(image_shape[-1]):
        transformed_image_values[:,:,img_slice] = scipy.rot90(image_values[:,:,img_slice],-1)
        #transformed_image_values[:,:,img_slice] = scipy.rot90(scipy.transpose(image_values[:,:,img_slice]),2)
        #transformed_image_values[:,:,img_slice] = scipy.transpose(image_values[:,:,img_slice])
    return transformed_image_values

def export_mask(mask_label, mask, ref_image, save_folder='./',transform=True):
    mask_image = imaging_functions.MRImage(label=mask_label)
    if transform:
        transformed_mask =automesh_pixels_to_nifti_transform(
            mask, ref_image.image_shape, image_dtype=mask.dtype)
    else:
        transformed_mask = mask
    mask_image.create_from_existing(transformed_mask, ref_image)
    mask_image.save(folder=save_folder)
    mask_image.load_data_vector()
    mask_image.calculate_non_zero_and_zero_indices()


def gen_mask_from_points(image, points):

    image_coor, x, y, z = imaging_functions.generate_image_coordinates(image.shape, image.spacing)
    mask_img_coor = (numpy.round(points/image.spacing)).astype(scipy.int16)
    mask= scipy.zeros(image.shape, scipy.bool_)
    # Convert surface points to pixel coordinates
    mask_img_coor = mask_img_coor[scipy.where(mask_img_coor[:,0]<image.shape[0])[0],:]
    mask_img_coor = mask_img_coor[scipy.where(mask_img_coor[:,1]<image.shape[1])[0],:]
    mask_img_coor = mask_img_coor[scipy.where(mask_img_coor[:,2]<image.shape[2])[0],:]
    mask[mask_img_coor[:,0],mask_img_coor[:,1],mask_img_coor[:,2]] = 1

    # mask_non_zero_idxs = scipy.nonzero(mask)[0]
    mask = imaging_functions.imfill(mask)
    mask_non_zero_idxs = scipy.flatnonzero(mask)
    masked_pixels = image.values.ravel()[mask_non_zero_idxs]

    mask_coor = image_coor[mask_non_zero_idxs,:]

    return mask_coor, mask, masked_pixels, mask_non_zero_idxs

def mask_mri(fig, mesh, image, image_coor, mask_label,  params, debug=False):
    surface_pts = gen_element_surface_points(fig, mesh, element_ids=[], num_pts=110, debug=False)
    mask_coor, _, _ , _= gen_mask_from_points(fig, image, surface_pts, mask_label, image_coor, params)

class EmbeddedPointsElementGroup():

    def __init__(self, group_type, num):
        self.group_type = group_type
        self.num = num
        self.xi = None
        self.points = None
        self.elements = None
        self.pixel_values = None
        self.distance_err = None

    def add_to_group(self, xi, points, elements, pixel_values, distance_err, mask_non_zero_idxs):
        self.xi = xi
        self.points = points
        self.elements = elements
        self.pixel_values = pixel_values
        self.distance_err = distance_err
        self.mask_non_zero_idxs = mask_non_zero_idxs

class EmbeddedPoints():

    def __init__(self, ):
        self.global_element_group = EmbeddedPointsElementGroup(group_type='global_group', num=0)
        self.num_local_element_groups = 4
        self.local_element_groups = (
            [EmbeddedPointsElementGroup(group_type='local_group', num=1),
             EmbeddedPointsElementGroup(group_type='local_group', num=2),
             EmbeddedPointsElementGroup(group_type='local_group', num=3),
             EmbeddedPointsElementGroup(group_type='local_group', num=4)])

    def get_local_element_group(self, group):
        return self.local_element_groups[group-1]

    def set_embedded_points(self, group, xi, points, elements, pixel_values, distance_err, mask_non_zero_idxs):
        group = self.get_local_element_group(group)
        group.add_to_group(xi, points, elements, pixel_values, distance_err, mask_non_zero_idxs)
    def set_embedded_points_global(self,xi, points, elements, pixel_values,distance_err, mask_non_zero_idxs):
        group = self.global_element_group
        group.add_to_group(xi,points,elements,pixel_values,distance_err,mask_non_zero_idxs)


def embed_mri(fig, mesh, image, image_coor, params, element_groups=[],  num_element_pts=50, num_surface_pts=100):
    print ('Embedding mri')

    image_type = mesh.label

    global_xi = scipy.empty([0, 3])
    global_points = scipy.empty([0, 3])
    global_elements = scipy.empty([0])
    global_pixel_values = scipy.empty([0])
    global_distance_err = scipy.empty([0])
    global_mask= scipy.zeros(image.shape, scipy.bool_).ravel()

    embedded_points = EmbeddedPoints()

    # Loop through groups of elements and find the xi v
    for element_group_idx, element_group in enumerate(element_groups):
        print ('  Element groups: ', element_group_idx)
        print ('    Generating element group mask')
        surface_pts = gen_element_surface_points(fig, mesh, element_ids=element_group, num_pts=num_surface_pts, debug=False)
        mask_label = '{0}_mask_element_group_idx_{1}'.format(image_type, element_group_idx)
        mask_coor, local_mask, masked_pixels, mask_non_zero_idxs = gen_mask_from_points(fig, image, surface_pts, mask_label,
                                                                    image_coor, params)

        if not params.offscreen:
            sampled_idxs = numpy.random.random_integers(0,mask_coor.shape[0],200)
            sampled_mask_coor = mask_coor[sampled_idxs,:]
            # fig.plot_points('sampled_mask_coor', sampled_mask_coor,color=(0,1,0), size=2)


        dd,local_xi, local_points ,local_elements = embed_points(mask_coor, mesh, element_group, num_element_pts)
        local_pixel_values = masked_pixels
        local_distance_err = dd
        local_mask_non_zero_idxs = mask_non_zero_idxs

        embedded_points.set_embedded_points(element_group_idx+1, local_xi, local_points, local_elements, local_pixel_values, local_distance_err, local_mask_non_zero_idxs)

        global_xi = scipy.vstack([global_xi, local_xi])
        global_points = scipy.vstack([global_points, local_points])
        global_elements = scipy.hstack([global_elements, local_elements])
        global_pixel_values = scipy.hstack([global_pixel_values, local_pixel_values])
        global_distance_err = scipy.hstack([global_distance_err, local_distance_err])
        global_distance_err = scipy.hstack([global_distance_err, local_distance_err])
        global_mask[scipy.nonzero(local_mask.ravel())[0]] = True

    # Todo There will be duplicates element points defined at the boundary between element groups.
    # These can be identfiied by searching for duplicate element point coordinates. In this case
    # the element/xi that produces the minimum kdtree distance error across the duplicates should be used 

    #import ipdb; ipdb.set_trace()
    embedded_points.set_embedded_points_global(global_xi, global_points, global_elements, global_pixel_values, global_distance_err, global_mask)
    if params.debug:
        print ('    Exporting full {} mask'.format(mesh.label))
        file_name = os.path.join(params.results_dir, '{0}_mask.nii'.format(mesh.label))

        scan_mask = image.copy()
        scan_mask.values = global_mask.reshape(image.shape)
        scan_mask.setRafOrientation()
        tools.writeNIFTIImage(scan_mask, file_name)

    if not params.offscreen:
        scan_mask = image.copy()
        scan_mask.values = global_mask.reshape(image.shape)
        bmw.view_mri(None, fig, scan_mask, axes='y_axes')


    return embedded_points



def evaluate_points(points_coords,xi,elem_idx,mesh ):
    warped_points = np.zeros(points_coords.shape)
    for point_idx, point in enumerate(points_coords):
        element = mesh.elements[elem_idx[point_idx]]
        warped_points[point_idx, :] = element.evaluate(xi[point_idx, :])

    return warped_points

def embed_points(mesh,points_to_embed, num_element_pts=100):

    mesh_nodes, mesh_nodes_ids = mesh.get_node_ids()
    meshs_tree = cKDTree(mesh_nodes)

    d, nodes_indx = meshs_tree.query(points_to_embed)
    nodes_set = [mesh_nodes_ids[x] for x in nodes_indx]

    elem_group = []
    for element in mesh.elements:
        for node_indx in range(len(nodes_set)):
            if nodes_set[node_indx] in element.node_ids:
                elem_group.append(element.id)

    points, elem_xi, elem_ids = mesh_generation.generate_points_in_elements(mesh,
                                                                            element_ids=elem_group,
                                                                            num_points=num_element_pts)

    points_tree = cKDTree(points)
    d, nodes_indx = points_tree.query(points_to_embed)
    points_elem_xi = elem_xi[nodes_indx]
    points_elem_ids = elem_ids[nodes_indx]

    return  points_elem_ids, points_elem_xi



def export_prone_to_supine_displacements(fig, embedded_points, prone_mesh, supine_mesh, image, image_coor, params, element_groups=[], debug=False, prefix=''):
    print ('exporting model displacements')
    image_type = 'supine'

    #embedded_pixels_filename = '{0}/embedded_pixels.h5'.format(params.results_dir)
    #hdf5_main_grp = h5py.File(embedded_pixels_filename, 'r')
    for element_group_idx, element_group in enumerate(element_groups):
        print ('  Element groups: ', element_group_idx)
        #prefix = 'local/element_group_{0}/'.format(element_group_idx)
        embedded_points_element_group = embedded_points.get_local_element_group(element_group_idx+1)
        local_xi = embedded_points_element_group.xi
        local_elements = embedded_points_element_group.elements


        prone_points = numpy.empty_like(local_xi)
        supine_points = numpy.empty_like(local_xi)
        disp = numpy.empty_like(local_xi)
        for element_id in element_group:
            prone_element = prone_mesh.elements[element_id]
            supine_element = supine_mesh.elements[element_id]
            idxs = numpy.where(local_elements == element_id)[0]
            prone_points[idxs,:] = prone_element.evaluate(local_xi[idxs,:])
            supine_points[idxs,:] = supine_element.evaluate(local_xi[idxs,:])
        disp = supine_points - prone_points

        disp_image_pixels = scipy.zeros(image.total_number_of_pixels, numpy.float32)
        disp_image_pixels[embedded_points_element_group.mask_non_zero_idxs] = disp[:,0] # x
        label = prefix + '_model_x_disp'
        disp_image_pixels = disp_image_pixels.reshape(image.image_shape)
        disp_image = imaging_functions.MRImage(label=label)
        transformed_disp_image_pixels = automesh_pixels_to_nifti_transform(disp_image_pixels, image.image_shape, image_dtype=disp_image_pixels.dtype)
        disp_image.create_from_existing(transformed_disp_image_pixels, image)
        disp_image.save(folder=params.results_dir)

        disp_image_pixels = scipy.zeros(image.total_number_of_pixels, numpy.float32)
        disp_image_pixels[embedded_points_element_group.mask_non_zero_idxs] = disp[:,1] # x
        label = prefix + '_model_y_disp'
        disp_image_pixels = disp_image_pixels.reshape(image.image_shape)
        disp_image = imaging_functions.MRImage(label=label)
        transformed_disp_image_pixels = automesh_pixels_to_nifti_transform(disp_image_pixels, image.image_shape, image_dtype=disp_image_pixels.dtype)
        disp_image.create_from_existing(transformed_disp_image_pixels, image)
        disp_image.save(folder=params.results_dir)

        disp_image_pixels = scipy.zeros(image.total_number_of_pixels, numpy.float32)
        disp_image_pixels[embedded_points_element_group.mask_non_zero_idxs] = disp[:,2] # x
        label = prefix + '_model_z_disp'
        disp_image_pixels = disp_image_pixels.reshape(image.image_shape)
        disp_image = imaging_functions.MRImage(label=label)
        transformed_disp_image_pixels = automesh_pixels_to_nifti_transform(disp_image_pixels, image.image_shape, image_dtype=disp_image_pixels.dtype)
        disp_image.create_from_existing(transformed_disp_image_pixels, image)
        disp_image.save(folder=params.results_dir)

def warp_mri(fig, embedded_points, mesh, image, image_coor, params, element_groups=[], debug=False, num_surface_pts=100):
    print ('warping mri')
    image_type = mesh.label
    total_number_of_pixels = image.shape[0]*image.shape[1]*image.shape[2]
    warped_image_pixels = scipy.zeros(total_number_of_pixels, scipy.int16)


    for element_group_idx, element_group in enumerate(element_groups):
        print ('  Element groups: ', element_group_idx)
        embedded_points_element_group = embedded_points.get_local_element_group(element_group_idx+1)
        local_xi = embedded_points_element_group.xi
        local_elements = embedded_points_element_group.elements
        pixel_val = embedded_points_element_group.pixel_values
        # mask_non_zero_idxs = embedded_points.get_local_element_group
        #


        warped_points = numpy.empty_like(local_xi)
        for element_id in element_group:
            element = mesh.elements[element_id]
            idxs = numpy.where(local_elements == element_id)[0]
            warped_points[idxs,:] = element.evaluate(local_xi[idxs,:])


        if debug:
            sampled_idxs = numpy.random.random_integers(0,mask_coor.shape[0],200)
            sampled_mask_coor = mask_coor[sampled_idxs,:]
            fig.plot_points('sampled_mask_coor', sampled_mask_coor,color=(0,1,0), size=2)



        # Generate image mask from surface points
        print ('    Generating supine mask')
        surface_pts = gen_element_surface_points(fig, mesh, element_ids=element_group, num_pts=num_surface_pts, debug=False)
        warped_mask_label = '{0}_mask_element_group_idx_{1}'.format(image_type, element_group_idx)
        warped_mask_coor, local_warped_mask, _, _ = gen_mask_from_points(fig, image, surface_pts, warped_mask_label, image_coor,
                                                                        params)

        # coords, grid_x, grid_y, grid_z = imaging_functions.generate_image_coordinates(image.shape, image.spacing,
        #                                                                               image.origin)
        #
        # image_pixels = griddata(warped_points,pixel_val,(grid_x,grid_y,grid_z),method='linear')


        warped_mask_image_non_zero_indices = scipy.nonzero(local_warped_mask.ravel())[0]



        print ('    Creating kdTree of warped points')
        warped_point_tree = cKDTree(warped_points)
        # Note that choice of radius important, increase if gaps present
        # in the image.
        #image.load_data_vector()
        print ('    Evaluating closest warped mask coordinates')
        [distance, tree_indices] = warped_point_tree.query(warped_mask_coor,n_jobs=7)#, distance_upper_bound=2.0)
        print ('      distance shape:', distance.shape)
        print ('      max distance:{0}'.format(max(distance)))
        print ('      mix distance:{0}'.format(min(distance)))
        warped_image_pixels[warped_mask_image_non_zero_indices] = embedded_points_element_group.pixel_values[tree_indices]  #image.data_vector[prone_mask_image.non_zero_indices[tree_indices]]

    print ('    Exporting final warped supine image')
    warped_image_pixels = warped_image_pixels.reshape(image.shape)
    warped_image = image.copy()
    warped_image.values = warped_image_pixels

    if debug:
        bmw.view_mri(None, fig, image=warped_image, axes='y_axes')



    return warped_image


def invers_linear_mri_warping(fig, embedded_points, mesh, image, params, element_groups=[]):
    print ('warping mri')

    total_number_of_pixels = image.shape[0]*image.shape[1]*image.shape[2]
    warped_image_pixels = scipy.zeros(total_number_of_pixels, scipy.int16)

    for element_group_idx, element_group in enumerate(element_groups):
        print ('  Element groups: ', element_group_idx)
        embedded_points_element_group = embedded_points.get_local_element_group(element_group_idx+1)
        local_xi = embedded_points_element_group.xi
        local_elements = embedded_points_element_group.elements
        mask_non_zero_idxs = embedded_points_element_group.mask_non_zero_idxs

        warped_points = numpy.empty_like(local_xi)
        warped_pixel = numpy.empty_like(local_xi)
        pixels = numpy.zeros(len(warped_points))

        for element_id in element_group:
            element = mesh.elements[element_id]
            idxs = numpy.where(local_elements == element_id)[0]
            warped_points[idxs, :] = element.evaluate(local_xi[idxs, :])
            warped_pixel[idxs,:] = image.getPixelCoordinates(warped_points[idxs, :])
            pixels[idxs] = tools.linearImageInterpolator(warped_pixel[idxs,:], image)

        warped_image_pixels[mask_non_zero_idxs] = pixels


    print ('    Exporting final warped {0} image'.format(mesh.label))
    warped_image_pixels = warped_image_pixels.reshape(image.shape)
    warped_image = image.copy()
    warped_image.values = warped_image_pixels

    if not params.offscreen:
        bmw.view_mri(None, fig, warped_image, axes='y_axes')


    return warped_image

def warp_label():
    from bmw import imaging_functions
    label_image = imaging_functions.MRImage(folder='{0}/image_warping/'.format(op.results_dir), filename='label.nii.gz')
    label_image.load_data_vector()
    label_image.calculate_non_zero_and_zero_indices()
    label_coor = op.image_coor[label_image.non_zero_indices,:]


    compute_delaunay_edges(fig, label_coor[:,0], label_coor[:,1], label_coor[:,2], visualize=True)

    image_type = 'prone'
    mask_coor_filename = '{0}{1}_mask_geometric_coordinates_region.h5'.format(op.results_dir,image_type)
    hdf5_main_grp = h5py.File(mask_coor_filename, 'r')
    mask_coor = hdf5_main_grp['{0}_mask_geometric_coordinates'.format(image_type)][()]
    hdf5_main_grp.close()

    fig2 = viewer.Figure('supine')
    image_type = 'supine'
    supine = morphic.Mesh('{0}/supine.mesh'.format(op.results_dir))
    bmw.visualise_mesh(supine, fig2, visualise=True, face_colours=(0,1,0),pt_size=1, opacity=0.25, line_opacity = 0.75)
    warped_mask_coor_filename = '{0}{1}_mask_geometric_coordinates_region.h5'.format(op.results_dir,image_type)
    hdf5_main_grp = h5py.File(warped_mask_coor_filename, 'r')
    warped_mask_coor = hdf5_main_grp['{0}_mask_geometric_coordinates'.format(image_type)][()]
    hdf5_main_grp.close()


    warped_pixels_filename = '{0}/warped_pixels.h5'.format(op.results_dir)
    hdf5_main_grp = h5py.File(warped_pixels_filename, 'r')
    warped_points = hdf5_main_grp['warped_points'][()]
    hdf5_main_grp.close()

    tree = cKDTree(mask_coor)
    print (mask_coor.shape)
    dd, label_data_idxs = tree.query(label_coor)
    #import ipdb; ipdb.set_trace()

    #prone_mask_image = imaging_functions.MRImage('{0}/prone_mask'.format(op.results_dir))
    #prone_mask_image.load_data_vector()
    #prone_mask_image.calculate_non_zero_and_zero_indices()
    #label_data = label_image.data_vector[prone_mask_image.non_zero_indices]
    #label_data_idxs = scipy.nonzero(label_data)[0]
    #fig2.plot_points('label_points', warped_points[label_data_idxs,:], color=(1,0,0), size=5)
    compute_delaunay_edges(fig2,warped_points[label_data_idxs,0], warped_points[label_data_idxs,1], warped_points[label_data_idxs,2], visualize=True)

    warped_prone_image = imaging_functions.MRImage('{0}/warped_prone'.format(op.results_dir))
    src2 = mlab.pipeline.scalar_field(x,y,z,warped_prone_image.image.get_data())

    plane = mlab.pipeline.image_plane_widget(src2,
                        plane_orientation='z_axes',
                        slice_index=int(0.5 * op.scan.num_slices),
                        colormap='black-white', figure=fig2.figure)

    #plane = mlab.pipeline.image_plane_widget(src2, plane_orientation='z_axes', slice_index=int(0.5 * op.scan.num_slices), colormap='black-white', figure=fig2.figure)


def warp_mri_linear(fig, embedded_points, mesh, image, image_coor, params, element_groups=[], debug=False, num_surface_pts=100):
    print ('warping mri')
    image_type = mesh.label
    total_number_of_pixels = image.shape[0]*image.shape[1]*image.shape[2]
    warped_image_pixels = scipy.zeros(total_number_of_pixels, scipy.int16)


    for element_group_idx, element_group in enumerate(element_groups):
        print ('  Element groups: ', element_group_idx)
        embedded_points_element_group = embedded_points.get_local_element_group(element_group_idx+1)
        local_xi = embedded_points_element_group.xi
        local_elements = embedded_points_element_group.elements
        pixel_val = embedded_points_element_group.pixel_values
        # mask_non_zero_idxs = embedded_points.get_local_element_group
        #


        warped_points = numpy.empty_like(local_xi)
        for element_id in element_group:
            element = mesh.elements[element_id]
            idxs = numpy.where(local_elements == element_id)[0]
            warped_points[idxs,:] = element.evaluate(local_xi[idxs,:])


        # Generate image mask from surface points
        print ('    Generating supine mask')
        surface_pts = gen_element_surface_points(fig, mesh, element_ids=element_group, num_pts=num_surface_pts, debug=False)
        warped_mask_label = '{0}_mask_element_group_idx_{1}'.format(image_type, element_group_idx)
        warped_mask_coor, local_warped_mask, _, _ = gen_mask_from_points(fig, image, surface_pts, warped_mask_label, image_coor,
                                                                        params)

        coords, grid_x, grid_y, grid_z = imaging_functions.generate_image_coordinates(image.shape, image.spacing*10,
                                                                                      image.origin)

        image_pixels = griddata(warped_points,pixel_val,(100,100,100),method='linear')


        # warped_image_pixels[warped_mask_image_non_zero_indices] = embedded_points_element_group.pixel_values[tree_indices]  #image.data_vector[prone_mask_image.non_zero_indices[tree_indices]]

    print ('    Exporting final warped supine image')
    warped_image_pixels = warped_image_pixels.reshape(image.shape)
    warped_image = image.copy()
    warped_image.values = warped_image_pixels

    if debug:
        bmw.view_mri(None, fig, image=warped_image, axes='y_axes')

    return warped_image

def get_warped_image(source_mesh,target_mesh,source_image, params):

    if not os.path.exists(params.results_dir):
        os.makedirs(params.results_dir)

    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
    fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen

    coor, x, y, z = imaging_functions.generate_image_coordinates(source_image.shape, source_image.spacing)

    rhs_posterior_elems = [7, 8, 9, 10, 18, 19, 20, 21, 29, 30, 31, 32, 40, 41, 42, 43, 51, 52, 53, 54, 62, 63, 64, 65]
    rhs_anterior_elems = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36,
                          37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 60, 61]
    lhs_posterior_elems = [73, 74, 75, 76, 84, 85, 86, 87, 95, 96, 97, 98, 106, 107, 108, 109, 117, 118, 119, 120, 128,
                           129, 130, 131]
    lhs_anterior_elems = [66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 88, 89, 90, 91, 92, 93, 94, 99, 100,
                          101, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126, 127]

    anterior = True
    if anterior:
        elem_groups_to_embed = [rhs_anterior_elems + lhs_anterior_elems]
        label = '_anterior'
    # elem_groups_to_embed = [rhs_anterior_elems+lhs_anterior_elems+rhs_posterior_elems+lhs_posterior_elems]
    else:
        elem_groups_to_embed = [rhs_posterior_elems + lhs_posterior_elems]
        label = '_posterior'

    if params.debug:
             bmw.visualise_mesh(target_mesh, fig, visualise=True, face_colours=(0, 1, 0), opacity=0.3)


    embedded_points = bmw.embed_mri(fig, target_mesh, source_image, coor, params, element_groups=elem_groups_to_embed,
                                            num_element_pts=100, num_surface_pts=200)
    warped_image = bmw.invers_linear_mri_warping(fig, embedded_points, source_mesh, source_image, params,
                                                  element_groups=elem_groups_to_embed)

    return warped_image
