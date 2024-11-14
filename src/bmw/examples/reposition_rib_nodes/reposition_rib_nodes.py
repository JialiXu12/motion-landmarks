'''
Solve prone to supine mechanics
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
script_id = 'prone_to_supine'
run_program = 'python'
run_script = 'prone_to_supine.py'
run_on_pipeline = False
if run_on_pipeline:
    depends_on = ['generate_mesh_vl']
else:
    depends_on = []

def run(process):

    mechanics_wksp = process.workspace('mechanics', True)
    if run_on_pipeline:
        mesh_wksp = process.parent.workspace('mesh')
        volunteer_id = process.parent.metadata['subject']['id']
    else:
        os.environ['QT_API'] = 'pyqt'
        volunteer_id = 'VL00110'
        mesh_wksp = process.workspace('mesh', True)
        mesh_wksp._path_ = os.path.join('/home/psam012/Documents/opt/bmw/data/2016-05-29/volunteer_meshes/',volunteer_id)
        mechanics_wksp._path_ = os.path.join('/home/psam012/Documents/opt/bmw/examples/reposition_rib_nodes/results/',volunteer_id)
        if not os.path.exists(mechanics_wksp.path()):
            os.makedirs(mechanics_wksp.path())
    p = {
        'debug' : True,
        'offscreen': True,
        'mesh_dir': mesh_wksp.path(),
        'results_dir': mechanics_wksp.path(),
        'volunteer_id': volunteer_id,
        'offset' : 0,
        'parameters' : scipy.array([0.3, 5., 100.,100.,2.]),
        'force_full_mechanics_solve' : True,
        'field_export_name' : 'field'}
    params = automesh.Params(p)

    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
    fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen
    #import ipdb; ipdb.set_trace()

    # Load fitted chest wall surface (cwm)
    cwm_fname = mesh_wksp.path('ribcage_prone.mesh')
    if os.path.exists(cwm_fname):
        cwm = morphic.Mesh(cwm_fname)
        cwm.label = 'cwm'
    else:
        process.completed(False, 'ribcage mesh not found')

    # Load rhs and lhs fitted breast surface (bm)
    bm_rhs_fname = mesh_wksp.path('skin_right_prone.mesh')
    if os.path.exists(bm_rhs_fname):
        bm_rhs = morphic.Mesh(bm_rhs_fname)
        bm_rhs.label = 'bm_rhs'
    else:
        process.completed(False, 'rhs skin mesh not found')

    bm_lhs_fname = mesh_wksp.path('skin_left_prone.mesh')
    if os.path.exists(bm_lhs_fname):
        bm_lhs = morphic.Mesh(bm_lhs_fname)
        bm_lhs.label = 'bm_lhs'
    else:
        process.completed(False, 'lhs skin mesh not found')

    if params.debug:
        orig_mesh_fig = bmw.add_fig(viewer, label='original_mesh') # returns empty array if offscreen
        bmw.visualise_mesh(cwm, orig_mesh_fig, visualise=False, face_colours=(1,0,0), nodes='all', node_text=True, node_size=1, element_ids=True)
        bmw.visualise_mesh(bm_rhs, orig_mesh_fig, visualise=False, face_colours=(1,1,0), opacity=0.75, nodes='all', node_text=True, node_size=1)
        #bmw.visualise_mesh(bm_lhs, fig, visualise=True, face_colours=(1,1,0), opacity=0.75)

    # Add missing elements to the shoulder region of the breast surface mesh
    bmw.add_shoulder_elements(bm_rhs,'rhs', adjacent_nodes=[[6,49],[13,53]], armpit_nodes=[20,57,58])
    #bmw.smooth_shoulder_region(bm_rhs, fig, smoothing=0)

    bmw.add_shoulder_elements(bm_lhs,'lhs', adjacent_nodes=[[6,49],[13,53]], armpit_nodes=[20,57,58])

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

    new_bm_rhs, _ = bmw.create_surface_mesh(fig, 'new_bm_rhs', bm_rhs, Xe, hanging_e, params.offset, visualise=False)
    new_bm_lhs, _ = bmw.create_surface_mesh(fig, 'new_bm_lhs', bm_lhs, Xe, hanging_e, params.offset, visualise=False)

    if params.debug:
        bmw.visualise_mesh(new_bm_rhs, fig, visualise=False, face_colours=(0,1,1), nodes='all', node_text=True, node_size=1)
        bmw.visualise_mesh(new_bm_lhs, fig, visualise=False, face_colours=(0,1,1))

    # Create new chestwall surface mesh
    Xe_rhs = scipy.array([[0, 1, 2, 3],
                          [ 8,  9, 10, 11],
                          [16, 17, 18, 19]])
    new_cwm_rhs = bmw.reposition_nodes(fig, cwm, new_bm_rhs, params.offset, side='rhs', xi1_Xe=Xe_rhs, elem_shape=scipy.array(Xe).shape[::-1], debug=False)

    Xe_lhs = scipy.array(Xe_rhs)
    temp =  scipy.array([0,8,16])
    for row in range(Xe_lhs.shape[0]):
        Xe_lhs[row,:] = 7-scipy.array(Xe_rhs[0])+temp[row]
    new_cwm_lhs = bmw.reposition_nodes(fig, cwm, new_bm_lhs,params.offset, side='lhs', xi1_Xe=Xe_lhs, elem_shape=scipy.array(Xe).shape[::-1], debug=False)

    if params.debug:
        bmw.visualise_mesh(new_cwm_rhs, fig, visualise=False, face_colours=(1,1,0), nodes='all', node_text=True, node_size=1)
        bmw.visualise_mesh(new_cwm_lhs, fig, visualise=False, face_colours=(1,1,0))

    # Create new volume mesh
    mesh3D_rhs = bmw.create_volume_mesh(
        new_bm_rhs, new_cwm_rhs, 'rhs', params.offset, fig, [], skin=False,
        skin_thickness=1.45,smoothing=1)
    mesh3D_rhs.label = 'mesh3D_rhs'

    mesh3D_lhs = bmw.create_volume_mesh(
        new_bm_lhs, new_cwm_lhs, 'lhs', params.offset, fig, [], skin=False,
        skin_thickness=1.45,smoothing=1)
    mesh3D_lhs.label = 'mesh3D_lhs'

    bmw.generate_boundary_groups(mesh3D_rhs, fig, side='rhs', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=params.results_dir)
    bmw.generate_boundary_groups(mesh3D_lhs, fig, side='lhs', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=params.results_dir)

    if params.debug:
        bmw.visualise_mesh(mesh3D_rhs, fig, visualise=False, face_colours=(0,1,0),pt_size=1, opacity=0.25, line_opacity=0.75)
        bmw.visualise_mesh(mesh3D_lhs, fig, visualise=False, face_colours=(0,1,0),pt_size=1, opacity=0.25, line_opacity=0.75)

    # Load sternum/spine node groups 
    # TODO use dictionaries to store dof groups and add to metadata
    hdf5_main_grp = h5py.File('{0}/dof_groups_{1}.h5'.format(params.results_dir,'rhs'), 'r')
    rhs_sternum_nodes = hdf5_main_grp['/nodes/sternum'][()].T
    rhs_spine_nodes = hdf5_main_grp['/nodes/spine'][()].T
    hdf5_main_grp = h5py.File('{0}/dof_groups_{1}.h5'.format(params.results_dir,'lhs'), 'r')
    lhs_sternum_nodes = hdf5_main_grp['/nodes/sternum'][()].T
    lhs_spine_nodes = hdf5_main_grp['/nodes/spine'][()].T
    lhs_Xn_offset = 10000
    lhs_Xe_offset = len(mesh3D_rhs.get_element_cids())
    torso = bmw.join_lhs_rhs_meshes(mesh3D_lhs, mesh3D_rhs, fig, 'torso', scipy.hstack([rhs_sternum_nodes, rhs_spine_nodes]), scipy.hstack([lhs_sternum_nodes, lhs_spine_nodes]), lhs_Xn_offset=lhs_Xn_offset, lhs_Xe_offset=lhs_Xe_offset)
    if params.debug:
        bmw.visualise_mesh(torso, fig, visualise=True, face_colours=(0,1,1),pt_size=1, opacity=0.75, line_opacity = 0.75, text=False)
    bmw.generate_boundary_groups(torso, fig, side='both', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=params.results_dir, lhs_Xn_offset=lhs_Xn_offset, lhs_Xe_offset=lhs_Xe_offset, debug=False)
    h5_dof_groups = h5py.File('{0}/dof_groups_{1}.h5'.format(params.results_dir, 'both'), 'r')
    stiffer_shoulder_nodes = h5_dof_groups['/nodes/stiffer_shoulder'][()].T
    fixed_shoulder_nodes = h5_dof_groups['/nodes/fixed_shoulder'][()].T
    stiffer_back_nodes = h5_dof_groups['/nodes/stiffer_back'][()].T
    transitional_nodes = h5_dof_groups['/nodes/transitional'][()].T
    bmw.plot_points(fig, 'stiffer_shoulder_nodes', torso.get_nodes(stiffer_shoulder_nodes.tolist()), stiffer_shoulder_nodes, visualise=False, colours=(0,0,1), point_size=10, text_size=5)
    bmw.plot_points(fig, 'fixed_shoulder_nodes', torso.get_nodes(fixed_shoulder_nodes.tolist()), fixed_shoulder_nodes, visualise=False, colours=(1,0,0), point_size=10, text_size=5)
    bmw.plot_points(fig, 'stiffer_back_nodes', torso.get_nodes(stiffer_back_nodes.tolist()), stiffer_back_nodes, visualise=False, colours=(0,1,0), point_size=10, text_size=5)
    bmw.plot_points(fig, 'transitional_nodes', torso.get_nodes(transitional_nodes.tolist()), transitional_nodes, visualise=False, colours=(1,1,0), point_size=10, text_size=5)

    print 'Mesh construction complete.'

    mesh_quality = bmw.check_mesh_quality(torso)

    # Save mesh quality jacobian to Morphic data
    jacobian_filepath = mechanics_wksp.path('prone_jacobian.data')
    jacobian_data = morphic.Data()
    jacobian_data.values = mesh_quality['jacobians']
    jacobian_data.save(jacobian_filepath)
    torso.metadata['prone_mesh'] = {
        'setup': p,
        'mesh_generation_parameters': [],
        'jacobian_file': jacobian_filepath}
    # Only Jacobian stored in Morphic data, remaining mesh quality data 
    # stored in generic hdf5 dataset
    bmw.export_mesh_quality_data(mesh_quality, mechanics_wksp.path('prone_mesh_quality.h5'))

    torso.save(mechanics_wksp.path('prone.mesh'))

    process.completed(status=mesh_quality['no_bad_elements'],
        message='Num bad elem in prone mesh: {0}'.format(
                mesh_quality['num_bad_gauss_points']))

    solve_mechanics = True
    if solve_mechanics:
        sys.path.insert(1, os.sep.join((os.environ['OPENCMISS_ROOT'],
                                                'cm', 'bindings','python')))
        from opencmiss import CMISS
        from bmw import mechanics
        [converged, dependent_field1, decomposition, region, cmesh] = (
            mechanics.solve(fig, params, 0, params.parameters, mesh=torso, side='both',
                field_export_name=params.field_export_name, force_resolve=params.force_full_mechanics_solve))
        if converged:
            offset = 1
            num_deriv = 1
            for node in torso.nodes:
                for comp_idx in range(3):
                    for deriv_idx in range(num_deriv):
                        node.values[comp_idx] = dependent_field1.ParameterSetGetNodeDP(
                            CMISS.FieldVariableTypes.U,
                            CMISS.FieldParameterSetTypes.VALUES,
                            1, deriv_idx + 1, node.id + offset, comp_idx + 1)
            torso.save(mechanics_wksp.path('reference.mesh'))

            [converged, dependent_field2, decomposition, region, cmesh] = (
                mechanics.solve(fig, params, 1, params.parameters, mesh=torso,
                    cmesh=cmesh, previous_dependent_field=dependent_field1,
                    decomposition=decomposition, region=region, side='both',
                    field_export_name=params.field_export_name, force_resolve=params.force_full_mechanics_solve))
            if converged:
                for node in torso.nodes:
                    for comp_idx in range(3):
                        for deriv_idx in range(num_deriv):
                            node.values[comp_idx] = dependent_field2.ParameterSetGetNodeDP(
                                CMISS.FieldVariableTypes.U,
                                CMISS.FieldParameterSetTypes.VALUES,
                                1, deriv_idx + 1, node.id + offset, comp_idx + 1)

                print 'Model solve complete.'

                mesh_quality = bmw.check_mesh_quality(torso)

                # Save mesh quality jacobian to Morphic data
                jacobian_filepath = mechanics_wksp.path('supine_jacobian.data')
                jacobian_data = morphic.Data()
                jacobian_data.values = mesh_quality['jacobians']
                jacobian_data.save(jacobian_filepath)
                torso.metadata['supine_mesh'] = {
                    'setup': p,
                    'mesh_generation_parameters': [],
                    'jacobian_file': jacobian_filepath}
                # Only Jacobian stored in Morphic data, remaining mesh quality data 
                # stored in generic hdf5 dataset
                bmw.export_mesh_quality_data(mesh_quality, mechanics_wksp.path('supine_mesh_quality.h5'))

                torso.save(mechanics_wksp.path('supine.mesh'))

                process.completed(status=mesh_quality['no_bad_elements'],
                    message='Num bad elem in supine mesh: {0}'.format(
                            mesh_quality['num_bad_gauss_points']))

        process.completed(status=converged, message='Converged: %s' % converged)

if __name__ == "__main__":
    import bpm
    run(bpm.get_project_process())
