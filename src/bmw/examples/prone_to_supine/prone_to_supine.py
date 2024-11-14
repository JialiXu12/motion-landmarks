'''
Solve prone to supine mechanics
'''
import os
import sys
import scipy
os.environ['QT_API'] = 'pyqt'
script_id = 'prone_to_supine'
run_program = 'python'
run_script = 'prone_to_supine.py'
run_on_pipeline = False
debug = True
if run_on_pipeline:
    depends_on = ['generate_mesh_vl']
else:
    depends_on = []

def run(process):
    if run_on_pipeline:
        import CMISS
    else:
        sys.path.insert(1, os.sep.join((os.environ['OPENCMISS_ROOT'],
                                                'cm', 'bindings','python')))
        from opencmiss import CMISS
    import bmw
    import morphic
    reload(morphic)
    import h5py
    import automesh

    mechanics_wksp = process.workspace('mechanics', True)
    if run_on_pipeline:
        if not debug:
            mechanics_wksp.clear()
        extract_process_metadata(process)
        metadata = process.metadata
        mesh_wksp = process.parent.workspace('mesh')
        volunteer_id = metadata['subject']['id']
    else:
        os.environ['QT_API'] = 'pyqt'
        volunteer_id = 'VL00046'
        mesh_wksp = process.workspace('mesh', True)
        mesh_wksp._path_ = os.path.join('/home/psam012/Documents/opt/bmw/data/2016-07-17/volunteer_meshes/',volunteer_id)
        mechanics_wksp._path_ = os.path.join('/home/psam012/Documents/opt/bmw/examples/prone_to_supine/results/',volunteer_id)
        if not os.path.exists(mechanics_wksp.path()):
            os.makedirs(mechanics_wksp.path())
    p = {
        'debug' : False,
        'offscreen': True,
        'mesh_dir': mesh_wksp.path(),
        'results_dir': mechanics_wksp.path(),
        'volunteer_id': volunteer_id,
        'offset' : 0,
        'breast_stiffnesses' : scipy.array([ 0.3   ,  0.2925,  0.285 ,  0.2775]),
        'boundary_stiffnesses' : scipy.array([5., 100.,100.,2.]),
        'force_full_mechanics_solve' : True,
        'field_export_name' : 'field'}
    params = automesh.Params(p)
    print ('Volunteer id: ', params.volunteer_id)
    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
    fig = bmw.add_fig(viewer, label='mesh') # returns empty array if offscreen

    # Load fitted chest wall surface (cwm)
    cwm_fname = mesh_wksp.path('ribcage_prone.mesh')
    if os.path.exists(cwm_fname):
        cwm = morphic.Mesh(cwm_fname)
        cwm.label = 'cwm'
    else:
        message = 'ribcage mesh not found'
        print (message)
        process.completed(False, message)

    # Load rhs and lhs fitted breast surface (bm)
    bm_rhs_fname = mesh_wksp.path('skin_right_prone.mesh')
    if os.path.exists(bm_rhs_fname):
        bm_rhs = morphic.Mesh(bm_rhs_fname)
        bm_rhs.label = 'bm_rhs'
    else:
        message = 'rhs skin mesh not found'
        print (message)
        process.completed(False, message)

    bm_lhs_fname = mesh_wksp.path('skin_left_prone.mesh')
    if os.path.exists(bm_lhs_fname):
        bm_lhs = morphic.Mesh(bm_lhs_fname)
        bm_lhs.label = 'bm_lhs'
    else:
        message = 'lhs skin mesh not found'
        print (message)
        process.completed(False, message)

    if params.debug:
        bmw.visualise_mesh(cwm, fig, visualise=False, face_colours=(1,0,0))
        bmw.visualise_mesh(bm_rhs, fig, visualise=False, face_colours=(1,1,0), opacity=0.75)
        bmw.visualise_mesh(bm_lhs, fig, visualise=False, face_colours=(1,1,0), opacity=0.75)

    # Add missing elements to the shoulder region of the breast surface mesh
    bmw.add_shoulder_elements(bm_rhs,'rhs', adjacent_nodes=[[6,54],[13,58]], armpit_nodes=[20,49,62])
    bmw.add_shoulder_elements(bm_lhs,'lhs', adjacent_nodes=[[6,54],[13,58]], armpit_nodes=[20,49,62])

    if params.debug:
        bmw.visualise_mesh(cwm, fig, visualise=False, face_colours=(1,0,0), nodes='all', node_size=1, node_text=True, element_ids=True)
        bmw.visualise_mesh(bm_rhs, fig, visualise=False, face_colours=(1,1,0), opacity=0.75, nodes='all', node_size=1, node_text=True, element_ids=True)
        bmw.visualise_mesh(bm_lhs, fig, visualise=False, face_colours=(1,1,0), opacity=0.75, nodes='all', node_size=1, node_text=True, element_ids=True)

    # Create new breast surface mesh
    Xe = [[0,1,2,3,4,5,54,55,42,43,44],
        [6,7,8,9,10,11,56,57,45,46,47],
        [12, 13, 14, 15, 16, 17, 36, 37, 48, 49, 50],
        [18, 19, 20, 21, 22, 23, 38, 37, 48, 49, 50],
        [24, 25, 26, 27, 28, 29, 39, 40, 51, 52, 53],
        [30, 31, 32, 33, 34, 35, 41, 40, 51, 52, 53]]
    hanging_e = [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,
        None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,
        None,  None,  None,  None,  None,  None,  None,  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],
        None,  None,  None,  None,  None,  None,  None,  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],
        None,  None,  None,  None,  None,  None,  None,  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],
        None,  None,  None,  None,  None,  None,  None,  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]]]

    new_bm_rhs, _ = bmw.create_surface_mesh(fig, 'new_bm_rhs', bm_rhs, Xe, hanging_e, params.offset, visualise=False)
    new_bm_lhs, _ = bmw.create_surface_mesh(fig, 'new_bm_lhs', bm_lhs, Xe, hanging_e, params.offset, visualise=False)

    if params.debug:
        #import ipdb; ipdb.set_trace()
        bmw.visualise_mesh(cwm, fig, visualise=False, face_colours=(1,0,0), nodes='all', node_size=1, node_text=True, element_ids=True)
        bmw.visualise_mesh(new_bm_rhs, fig, visualise=False, face_colours=(1,1,0), opacity=0.75, nodes='all', node_size=1, node_text=True, element_ids=True)
        bmw.visualise_mesh(new_bm_lhs, fig, visualise=False, face_colours=(1,1,0), opacity=0.75, nodes='all', node_size=1, node_text=True, element_ids=True)

    # Create new chestwall surface mesh
    Xe_rhs = scipy.array([[0, 1, 2, 3],
                          [ 8,  9, 10, 11],
                          [16, 17, 18, 19]])
    new_cwm_rhs = bmw.reposition_nodes(fig, cwm, new_bm_rhs, params.offset, side='rhs', xi1_Xe=Xe_rhs, elem_shape=scipy.array(Xe).shape[::-1], debug=False)

    Xe_lhs = scipy.array(Xe_rhs)
    temp =  scipy.array([0,8,16])
    for row in range(Xe_lhs.shape[0]):
        Xe_lhs[row,:] = 7-scipy.array(Xe_rhs[0])+temp[row]
    new_cwm_lhs = bmw.reposition_nodes(fig, cwm, new_bm_lhs, params.offset, side='lhs', xi1_Xe=Xe_lhs, elem_shape=scipy.array(Xe).shape[::-1], debug=False)

    if params.debug:
        bmw.visualise_mesh(new_cwm_rhs, fig, visualise=False, face_colours=(1,1,0))
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
    torso_mesh = bmw.join_lhs_rhs_meshes(mesh3D_lhs, mesh3D_rhs, fig, 'torso_mesh', scipy.hstack([rhs_sternum_nodes, rhs_spine_nodes]), scipy.hstack([lhs_sternum_nodes, lhs_spine_nodes]), lhs_Xn_offset=lhs_Xn_offset, lhs_Xe_offset=lhs_Xe_offset)
    if params.debug:
        bmw.visualise_mesh(torso_mesh, fig, visualise=False, face_colours=(0,1,1),pt_size=1, opacity=0.75, line_opacity = 0.75, text=False)
    bmw.generate_boundary_groups(torso_mesh, fig, side='both', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=params.results_dir, lhs_Xn_offset=lhs_Xn_offset, lhs_Xe_offset=lhs_Xe_offset, debug=False)
    h5_dof_groups = h5py.File('{0}/dof_groups_{1}.h5'.format(params.results_dir, 'both'), 'r')
    stiffer_shoulder_nodes = h5_dof_groups['/nodes/stiffer_shoulder'][()].T
    fixed_shoulder_nodes = h5_dof_groups['/nodes/fixed_shoulder'][()].T
    stiffer_back_nodes = h5_dof_groups['/nodes/stiffer_back'][()].T
    transitional_nodes = h5_dof_groups['/nodes/transitional'][()].T
    bmw.plot_points(fig, 'stiffer_shoulder_nodes', torso_mesh.get_nodes(stiffer_shoulder_nodes.tolist()), stiffer_shoulder_nodes, visualise=False, colours=(0,0,1), point_size=10, text_size=5)
    bmw.plot_points(fig, 'fixed_shoulder_nodes', torso_mesh.get_nodes(fixed_shoulder_nodes.tolist()), fixed_shoulder_nodes, visualise=False, colours=(1,0,0), point_size=10, text_size=5)
    bmw.plot_points(fig, 'stiffer_back_nodes', torso_mesh.get_nodes(stiffer_back_nodes.tolist()), stiffer_back_nodes, visualise=False, colours=(0,1,0), point_size=10, text_size=5)
    bmw.plot_points(fig, 'transitional_nodes', torso_mesh.get_nodes(transitional_nodes.tolist()), transitional_nodes, visualise=False, colours=(1,1,0), point_size=10, text_size=5)

    print ('Mesh construction complete.')

    mesh_quality = bmw.check_mesh_quality(torso_mesh)

    # Save mesh quality jacobian to Morphic data
    jacobian_filepath = mechanics_wksp.path('prone_jacobian.data')
    jacobian_data = morphic.Data()
    jacobian_data.values = mesh_quality['jacobians']
    jacobian_data.save(jacobian_filepath)
    torso_mesh.metadata['prone_mesh'] = {
        'setup': p,
        'mesh_generation_parameters': [],
        'jacobian_file': jacobian_filepath}
    # Only Jacobian stored in Morphic data, remaining mesh quality data 
    # stored in generic hdf5 dataset
    bmw.export_mesh_quality_data(mesh_quality, mechanics_wksp.path('prone_mesh_quality.h5'))

    torso_mesh = set_metadata(torso_mesh, process)
    torso_mesh.save(mechanics_wksp.path('prone.mesh'))

    message = 'Num bad guass points in prone mesh: {0}'.format(
        mesh_quality['num_bad_gauss_points'])
    process.set_data('message', message)

    solve_mechanics = True
    if solve_mechanics:
        from bmw import mechanics
        mechanics_setup = mechanics.setup_mechanics_problem()
        # Initialise a mechanics status list, where each entry describes
        # whether a parameter has solved (printed in green) or not (printed 
        # in red - default).
        num_parameter_sets = params.breast_stiffnesses.shape[0]
        mechanics_convergence_status = [False]*num_parameter_sets
        mechanics_convergence_status_str = ['\x1B[91m'+str(stiffness)+'\x1B[0m' for stiffness in params.breast_stiffnesses]
        # Solve mechanics for each parameter set
        force_resolve=True
        for parameter_set in range(num_parameter_sets):
            parameters = scipy.hstack((params.breast_stiffnesses[parameter_set], params.boundary_stiffnesses))
            torso_mesh.metadata['parameters'] = parameters
            print '==============='
            print 'Parameter set: ', parameter_set
            save_prefix = '_parameter_set_{0}'.format(parameter_set)
            field_name = params.field_export_name + save_prefix
            # Only force a resolve of the mechanics fo the first parameter set
            if parameter_set > 0:
                force_resolve=False
            # Solve prone to unloaded state mechanics
            [converged, dependent_field1, decomposition, region, cmesh, problem1] = (
                mechanics.solve(mechanics_setup, params, 0, parameters,
                    mesh=torso_mesh, side='both', field_export_name=field_name,
                    force_resolve=force_resolve))
            if converged:
                torso_mesh = mechanics.OpenCMISS_mesh_to_morphic(torso_mesh, dependent_field1)
                torso_mesh = set_metadata(torso_mesh, process)
                torso_mesh.save(mechanics_wksp.path('reference' + save_prefix + '.mesh'))
                # Solve unloaded state to supine mechanics
                [converged, dependent_field2, decomposition, region, cmesh, problem2] = (
                    mechanics.solve(mechanics_setup, params, 1, parameters,
                        mesh=torso_mesh, cmesh=cmesh, previous_dependent_field=dependent_field1,
                        decomposition=decomposition, region=region, side='both',
                        field_export_name=field_name, force_resolve=force_resolve))
                if converged:
                    torso_mesh = mechanics.OpenCMISS_mesh_to_morphic(torso_mesh, dependent_field2)
                    print 'Model solve complete.'

                    mesh_quality = bmw.check_mesh_quality(torso_mesh)

                    # Save mesh quality jacobian to Morphic data
                    jacobian_filepath = mechanics_wksp.path('supine_jacobian' + save_prefix + '.data')
                    jacobian_data = morphic.Data()
                    jacobian_data.values = mesh_quality['jacobians']
                    jacobian_data.save(jacobian_filepath)
                    torso_mesh = set_metadata(torso_mesh, process)
                    torso_mesh.metadata['supine_mesh'] = {
                        'setup': p,
                        'mesh_generation_parameters': [],
                        'jacobian_file': jacobian_filepath}
                    # Only Jacobian stored in Morphic data, remaining mesh quality data 
                    # stored in generic hdf5 dataset
                    bmw.export_mesh_quality_data(mesh_quality, 
                        mechanics_wksp.path('supine_mesh_quality' + save_prefix + '.h5'))

                    torso_mesh.save(mechanics_wksp.path('supine' + save_prefix + '.mesh'))

                    message = 'Num bad guass points in supine mesh: {0}'.format(
                        mesh_quality['num_bad_gauss_points'])
                    print message
                    # Update mechanics status list
                    mechanics_convergence_status_str[parameter_set] = '\x1B[92m'+str(params.breast_stiffnesses[parameter_set])+'\x1B[0m'
                    mechanics_convergence_status[parameter_set] = True
                else:
                    break
            else:
                break
            mechanics.destroy_regions([region], [problem1, problem2])
            message = 'Mechanics status: '+', '.join(mechanics_convergence_status_str)+' kPa'
            process.set_data('message', message)

        mechanics_wksp.add_data({
            'mechanics_convergence_status':  mechanics_convergence_status,
            'num_parameter_sets': num_parameter_sets,
            'breast_stiffnesses': params.breast_stiffnesses.tolist(),
            'boundary_stiffnesses': params.boundary_stiffnesses.tolist()})
        converged = scipy.any(mechanics_wksp.data['mechanics_convergence_status'])
        message = 'Mechanics status: '+', '.join(mechanics_convergence_status_str)+' kPa'
        process.completed(status=converged, message=message)
    else:
        process.completed(status=False, message='Mechanics turned off')

def extract_process_metadata(process):
    process.clear_metadata()
    parent = process.parent
    for key in parent.metadata.keys():
        process.set_metadata(key, parent.metadata[key])

    pipeline = process.parent.metadata['bpm_pipeline']
    processes = pipeline['processes']
    proc_dict = {'id': process.id, 'label': process.label, 'script': process.script.label,
                 'root': process.root.id,
                 'params': process.params, 'status': process.status, 'message': process.message,
                 'started': process.started, 'duration': process.duration,
                 'workspaces': process.data['workspaces']}
    if process.parent is not None:
        proc_dict['parent'] = process.parent.id
    else:
        proc_dict['parent'] = None
    processes.append(proc_dict)
    pipeline['processes'] = processes
    process.set_metadata('bpm_pipeline', pipeline)

def set_metadata(mesh, process):
    m = process.metadata
    for key in m.keys():
        mesh.metadata[key] = m[key]
    return mesh

if __name__ == "__main__":
    import bpm
    run(bpm.get_project_process())
