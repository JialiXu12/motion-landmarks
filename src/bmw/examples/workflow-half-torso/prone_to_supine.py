import sys
import os

import ipdb
import scipy
import h5py
from scipy.spatial import cKDTree

import morphic
reload(morphic)
import bmw
sys.path.insert(1, os.sep.join((os.environ['OPENCMISS_ROOT'],
                                       'cm', 'bindings','python')))
from opencmiss import CMISS
from bmw import imaging_functions

def compute_delaunay_edges(fig, x, y, z, visualize=True):
    """ Given 3-D points, returns the edges of their
        Delaunay triangulation.

        Parameters
        -----------
        x: ndarray
            x coordinates of the points
        y: ndarray
            y coordinates of the points
        z: ndarray
            z coordinates of the points
    """
    if visualize:
        vtk_source = mlab.points3d(x, y, z, opacity=1, mode='2dvertex')
        vtk_source.actor.property.point_size = 3
    else:
        vtk_source =  mlab.pipeline.scalar_scatter(x, y, z, figure=False)
    delaunay =  mlab.pipeline.delaunay3d(vtk_source)
    delaunay.filter.offset = 999    # seems more reliable than the default
    edges = mlab.pipeline.extract_edges(delaunay)
    if visualize:
        mlab.pipeline.surface(edges, opacity=0.3, line_width=3, color=(1,0,0), figure=fig.figure)

if __name__ == "__main__":
    
    volunteer = 'VL00048'
    op = bmw.volunteer_setup(volunteer)
    if not os.path.exists(op.results_dir):
        os.makedirs(op.results_dir)
    visualise = True
    
    offset = 0
    if visualise:
        from morphic import viewer
        if "fig" not in locals():
            fig = viewer.Figure('prone')
    else:
        fig = None
    visualise = False
    # Load original fitted chest wall surface (cwm)
    cwm = morphic.Mesh(
        '{0}/data/volunteer/ribcages/{1}_ribcage_prone.mesh'.format(
            op.root_dir, op.volunteer))
    cwm.label = 'cwm'
    bmw.visualise_mesh(cwm, fig, visualise=False, face_colours=(1,0,0))
    
    # Load original fitted breast surface (bm)
    bm = morphic.Mesh(
        '{0}/data/volunteer/right_skin_meshes/{1}_prone.mesh'.format(
            op.root_dir, op.volunteer))
    bm.label = 'bm'
    bmw.visualise_mesh(bm, fig, visualise=False, face_colours=(0,1,1))
    #import ipdb; ipdb.set_trace()
    # Add missing elemens to the shoulder region of the breast surface mesh
    bmw.add_shoulder_elements(bm,'rhs', adjacent_nodes=[[6,49],[13,53]], armpit_nodes=[20,57,58])
    #shoulder_elems = []
    #for element in bm.elements.get_groups('shoulder'):
    #    shoulder_elems.append(element.id)
    #import ipdb; ipdb.set_trace()
    bmw.visualise_mesh(bm, fig, visualise=False, face_colours=(0,1,0))
    bm.save('{0}/prone_closed.mesh'.format(op.results_dir))
    
    # Create new breast surface mesh
    Xe = [[0,1,2,3,4,5,52,53,36,37,38],
        [6,7,8,9,10,11,54,55,39,40,41],
        [12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46],
        [18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46],
        [24, 25, 26, 27, 28, 29, 47, 48, 49, 50, 51],
        [30, 31, 32, 33, 34, 35, 47, 48, 49, 50, 51]]
    hanging_e = [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,
        None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,
        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],
        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],
        None,  None,  None,  None,  None,  None,  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],
        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]]]
    new_bm, bm_surface_normals = bmw.create_surface_mesh(fig, visualise, 'new_bm', bm, Xe, hanging_e, offset)
    new_bm.save('{0}/new_bm_surface.mesh'.format(op.results_dir))
    #bmw.visualise_mesh(new_bm, fig, visualise=True, face_colours=(0,1,1), pt_size=3)
    bmw.visualise_mesh(new_bm, fig, visualise=False, face_colours=(0,1,1), pt_size=3, text_elements=[5,6,7,8,16,17,18,19], element_ids=True)
    new_bm = bmw.smooth_shoulder_region(new_bm, fig, smoothing=0)
    bmw.visualise_mesh(new_bm, fig, visualise=False, face_colours=(0,1,1), pt_size=3)

    # Create new chestwall surface mesh
    Xe = [[0,0,0,0,1,1,1,2,2,2,3],
        [0,0,0,0,1,1,1,2,2,2,3],
        [8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11],
        [8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11],
        [16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19],
        [16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19]]
    hanging_e = [ 
        # Elem 1========================================================================================   Elem 2=================================================================   # Elem 3====================================================================
        [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,0.75],[0., 0.5]],  [[0.75,1.],[0., 0.5]],  [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,1.],[0., 0.5]],  [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,1.],[0., 0.5]],
        [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,0.75],[0.5, 1.]],  [[0.75,1.],[0.5, 1.]],  [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,1.],[0.5, 1.]],  [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,1.],[0.5, 1.]],
        [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,0.75],[0., 0.5]],  [[0.75,1.],[0., 0.5]],  [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,1.],[0., 0.5]],  [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,1.],[0., 0.5]],
        [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,0.75],[0.5, 1.]],  [[0.75,1.],[0.5, 1.]],  [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,1.],[0.5, 1.]],  [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,1.],[0.5, 1.]],
        [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,0.75],[0., 0.5]],  [[0.75,1.],[0., 0.5]],  [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,1.],[0., 0.5]],  [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,1.],[0., 0.5]],
        [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,0.75],[0.5, 1.]],  [[0.75,1.],[0.5, 1.]],  [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,1.],[0.5, 1.]],  [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,1.],[0.5, 1.]]]
    new_cwm = bmw.create_surface_mesh(fig, visualise, 'new_cwm', cwm, Xe, hanging_e, offset)[0]
    new_cwm.save('{0}/new_cwm_surface.mesh'.format(op.results_dir))
    bmw.visualise_mesh(new_cwm, fig, visualise=False, face_colours=(1,1,0), pt_size=0.5)
    
    # Create new volume mesh
    skin = False
    mesh3D = bmw.create_volume_mesh(
        new_bm, new_cwm, offset, fig, bm_surface_normals, skin=skin,
        skin_thickness=1.45,smoothing=1)
    mesh3D.label = 'mesh3D'
    bmw.visualise_mesh(mesh3D, fig, visualise=True, face_colours=(0,1,0),pt_size=1, opacity=0.25, line_opacity = 0.75, text_elements=[0])
    import ipdb; ipdb.set_trace()
    mesh3D.save('{0}/prone.mesh'.format(op.results_dir))

    # Specify volume mesh element groups
    if skin:
        cranial_elem = range(11) + range(66,77)
        caudal_elem = range(55,66) + range(121,132)
        sternum_elem = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
        spine_elem = [10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 131]
        chestwall_elem = range(66)
        skin_elem = range(66,132)
    else:
        cranial_elem = range(11)
        caudal_elem = range(55,66)
        sternum_elem = [0,11,22,33,44,55]
        spine_elem = [10,21,32,43,54,65]
        chestwall_elem = range(66)
        skin_elem = range(66)

    for element in mesh3D.elements[cranial_elem]:
        element.add_to_group('cranial')
    for element in mesh3D.elements[caudal_elem]:
        element.add_to_group('caudal')
    for element in mesh3D.elements[sternum_elem]:
        element.add_to_group('sternum')
    for element in mesh3D.elements[spine_elem]:
        element.add_to_group('spine')
    for element in mesh3D.elements[chestwall_elem]:
        element.add_to_group('chestwall')
    for element in mesh3D.elements[skin_elem]:
        element.add_to_group('skin')

    # mesh3D.get_element_cids(elements=mesh3D.elements.get_groups('cranial'))

    # Define volume mesh node groups
    cranial_points = bmw.generate_points_on_face(mesh3D, "xi2", 0,
        elements=mesh3D.elements.get_groups('cranial'))
    caudal_points = bmw.generate_points_on_face(mesh3D, "xi2", 1,
        elements=mesh3D.elements.get_groups('caudal'))
    sternum_points = bmw.generate_points_on_face(mesh3D, "xi1", 0,
        elements=mesh3D.elements.get_groups('sternum'))
    spine_points = bmw.generate_points_on_face(mesh3D, "xi1", 1,
        elements=mesh3D.elements.get_groups('spine'))
    chestwall_points = bmw.generate_points_on_face(mesh3D, "xi3", 0,
        elements=mesh3D.elements.get_groups('chestwall'))
    skin_points = bmw.generate_points_on_face(mesh3D, "xi3", 1,
        elements=mesh3D.elements.get_groups('skin'))
    visualise = False
    if visualise:
        fig.plot_points('cranial_points', cranial_points, color=(1,0,0), size=5)
        fig.plot_points('caudal_points', caudal_points, color=(0,1,0), size=5)
        fig.plot_points('sternum_points', sternum_points, color=(0,0,1), size=5)
        fig.plot_points('spine_points', spine_points, color=(1,1,0), size=5)
        fig.plot_points('chestwall_points', chestwall_points, color=(1,0,0), size=5)
        fig.plot_points('skin_points', skin_points, color=(0,1,1), size=5)
    cranial_nodes = bmw.points_2_nodes_id(mesh3D, cranial_points)
    caudal_nodes = bmw.points_2_nodes_id(mesh3D, caudal_points)
    sternum_nodes = bmw.points_2_nodes_id(mesh3D, sternum_points)
    spine_nodes = bmw.points_2_nodes_id(mesh3D, spine_points)
    chestwall_nodes = bmw.points_2_nodes_id(mesh3D, chestwall_points)
    skin_nodes = bmw.points_2_nodes_id(mesh3D, skin_points)
    visualise = False
    if visualise:
        fig.plot_points('cranial_nodes', mesh3D.get_nodes(cranial_nodes.tolist()), color=(1,0,0), size=5)
        fig.plot_points('caudal_nodes', mesh3D.get_nodes(caudal_nodes.tolist()), color=(0,1,0), size=5)
        fig.plot_points('sternum_nodes', mesh3D.get_nodes(sternum_nodes.tolist()), color=(0,0,1), size=5)
        fig.plot_points('spine_nodes', mesh3D.get_nodes(spine_nodes.tolist()), color=(1,1,0), size=5)
        fig.plot_points('chestwall_nodes', mesh3D.get_nodes(chestwall_nodes.tolist()), color=(1,0,1), size=5)
        fig.plot_points('skin_nodes', mesh3D.get_nodes(skin_nodes.tolist()), color=(0,1,1), size=5)
    
    # Export node and element groups
    export_node_groups = True
    if export_node_groups:
        hdf5_main_grp = h5py.File('{0}/node_elem_groups.h5'.format(op.results_dir), 'w')
        hdf5_main_grp.create_dataset('/elements/cranial', data = cranial_elem)
        hdf5_main_grp.create_dataset('/elements/caudal', data = caudal_elem)
        hdf5_main_grp.create_dataset('/elements/sternum', data = sternum_elem)
        hdf5_main_grp.create_dataset('/elements/spine', data = spine_elem)
        hdf5_main_grp.create_dataset('/elements/chestwall', data = chestwall_elem)
        hdf5_main_grp.create_dataset('/elements/skin', data = skin_elem)
        hdf5_main_grp.create_dataset('/nodes/cranial', data = cranial_nodes)
        hdf5_main_grp.create_dataset('/nodes/caudal', data = caudal_nodes)
        hdf5_main_grp.create_dataset('/nodes/sternum', data = sternum_nodes)
        hdf5_main_grp.create_dataset('/nodes/spine', data = spine_nodes)
        hdf5_main_grp.create_dataset('/nodes/chestwall', data = chestwall_nodes)
        hdf5_main_grp.create_dataset('/nodes/skin', data = skin_nodes)

    view_dicom = True
    if view_dicom:
        op.scan = bmw.load_dicom_attributes(
            '{0}/data/volunteer/{1}/prone/dicom'.format(op.root_dir, op.volunteer))
        op.image_coor, x, y, z = imaging_functions.generate_image_coordinates(op.scan)
        op.src = viewer.define_scalar_field(x,y,z,op.scan)
        #import ipdb; ipdb.set_trace()
        #from tvtk.api import write_data
        #write_data(op.src,'./aa.vtk')
        from mayavi import mlab
        #op.src = None
        #op.image_coor = None
        #plane = fig.visualise_dicom_plane(fig, op.scan, op.src, op)
        plane = mlab.pipeline.image_plane_widget(op.src,
                            plane_orientation='z_axes',
                            slice_index=int(0.5 * op.scan.num_slices),
                            colormap='black-white', figure=fig.figure)

        #outline = fig.visualise_dicom_outline(op.scan, op.src)
        #volume = fig.visualise_dicom_volume(scan, src)

        #fig.plot_image_data(op.volunteer, op.scan, op.src, plane)
        #fig.plot_image_data(op.volunteer+'_outline', op.scan, op.src, outline)
        #fig.plot_image_data(op.volunteer+'_volume', op.scan, op.src, volume)

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
        print mask_coor.shape
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


        import ipdb; ipdb.set_trace()
    solve_mechanics = False
    if solve_mechanics:
        from bmw import mechanics

        parameters = scipy.array([0.1, 5.])
        [converged, dependent_field1, decomposition, region, cmesh] = mechanics.solve(
            fig, op, 0, parameters, mesh=mesh3D)
        if converged:
            offset = 1
            num_deriv = 1
            for node in mesh3D.nodes:
                for comp_idx in range(3):
                    for deriv_idx in range(num_deriv):
                        node.values[comp_idx] = dependent_field1.ParameterSetGetNodeDP(
                            CMISS.FieldVariableTypes.U,
                            CMISS.FieldParameterSetTypes.VALUES,
                            1, deriv_idx + 1, node.id + offset, comp_idx + 1)
            mesh3D.save('{0}/reference.mesh'.format(op.results_dir))

            
            [converged, dependent_field2, decomposition, region, cmesh] = mechanics.solve(
                fig, op, 1, parameters, mesh=mesh3D, cmesh=cmesh, previous_dependent_field=dependent_field1,
                decomposition=decomposition, region=region)
            if converged:
                for node in mesh3D.nodes:
                    for comp_idx in range(3):
                        for deriv_idx in range(num_deriv):
                            node.values[comp_idx] = dependent_field2.ParameterSetGetNodeDP(
                                CMISS.FieldVariableTypes.U,
                                CMISS.FieldParameterSetTypes.VALUES,
                                1, deriv_idx + 1, node.id + offset, comp_idx + 1)
                mesh3D.save('{0}/supine.mesh'.format(op.results_dir))

    print 'Program successfully completed.'
