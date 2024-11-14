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

if __name__ == "__main__":
    debug = True
    offset = 0
    arguments = bmw.parse_arguments()
    if arguments.volunteer is None:
        volunteer = 'VL00040'
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
        results_dir = './results_updated_meshes/{0}/'.format(volunteer)
        op = bmw.volunteer_setup(mesh_dir, results_dir, volunteer, parameters, offscreen)
        cwm_fname = '{0}/volunteer/2016-03-30/{1}_ribcage_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_rhs_fname = '{0}/volunteer/2016-03-30/{1}_skin_right_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_lhs_fname = '{0}/volunteer/2016-03-30/{1}_skin_left_prone.mesh'.format(op.mesh_dir, op.volunteer)
    else:
        results_dir = './results/{0}/'.format(volunteer)
        op = bmw.volunteer_setup(mesh_dir, results_dir, volunteer, parameters, offscreen)
        cwm_fname = '{0}/volunteer/ribcages/{1}_ribcage_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_rhs_fname = '{0}/volunteer/right_skin_meshes/{1}_prone.mesh'.format(op.mesh_dir, op.volunteer)
        bm_lhs_fname = '{0}/volunteer/left_skin_meshes/{1}_prone.mesh'.format(op.mesh_dir, op.volunteer)

    cwm = morphic.Mesh(cwm_fname)
    bm_rhs = morphic.Mesh(bm_rhs_fname)
    bm_lhs = morphic.Mesh(bm_lhs_fname)
    cwm.label = 'cwm'
    bm_rhs.label = 'bm_rhs'
    bm_lhs.label = 'bm_lhs'

    fig = op.add_fig('prone')
    bmw.visualise_mesh(cwm, fig, visualise=True, face_colours=(1,0,0), text=True, element_ids=True, node_size=7)#, nodes='all')
    bmw.visualise_mesh(bm_rhs, fig, visualise=True, face_colours=(1,1,0), text=False, element_ids=False)
    bmw.visualise_mesh(bm_lhs, fig, visualise=True, face_colours=(1,1,0), text=False, element_ids=False)

#    # Add missing elements to the shoulder region of the breast surface mesh
#    bmw.add_shoulder_elements(bm_rhs,'rhs', adjacent_nodes=[[6,49],[13,53]], armpit_nodes=[20,57,58])
#    bmw.add_shoulder_elements(bm_lhs,'lhs', adjacent_nodes=[[6,49],[13,53]], armpit_nodes=[20,57,58])

#    bmw.visualise_mesh(bm_rhs, fig, visualise=False, face_colours=(0,1,0), text=True, element_ids=True)
#    bmw.visualise_mesh(bm_lhs, fig, visualise=False, face_colours=(0,1,0), text=True, element_ids=True)
#    bm_rhs.save('{0}/prone_closed_rhs.mesh'.format(op.results_dir))
#    bm_lhs.save('{0}/prone_closed_lhs.mesh'.format(op.results_dir))

#    # Create new breast surface mesh
#    Xe = [[0,1,2,3,4,5,52,53,36,37,38],
#        [6,7,8,9,10,11,54,55,39,40,41],
#        [12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46],
#        [18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46],
#        [24, 25, 26, 27, 28, 29, 47, 48, 49, 50, 51],
#        [30, 31, 32, 33, 34, 35, 47, 48, 49, 50, 51]]
#    hanging_e = [ None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,
#        None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,
#        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],  [[0.,1.],[0.0, 0.4]],
#        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],  [[0.,1.],[0.4, 1.0]],
#        None,  None,  None,  None,  None,  None,  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],  [[0.,1.],[0., 0.50]],
#        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]],  [[0.,1.],[0.50, 1.]]]

#    new_bm_rhs, bm_rhs_surface_normals = bmw.create_surface_mesh(fig, 'new_bm_rhs', bm_rhs, Xe, hanging_e, offset, visualise=False)
#    new_bm_rhs.save('{0}/new_bm_rhs_surface.mesh'.format(op.results_dir))
#    #bmw.visualise_mesh(new_bm, fig, visualise=True, face_colours=(0,1,1), pt_size=3)
#    bmw.visualise_mesh(new_bm_rhs, fig, visualise=False, face_colours=(0,1,1), text_elements=[5,6,7,8,16,17,18,19], element_ids=False, nodes='all')

#    new_bm_lhs, bm_lhs_surface_normals = bmw.create_surface_mesh(fig, 'new_bm_lhs', bm_lhs, Xe, hanging_e, offset, visualise=False)
#    new_bm_lhs.save('{0}/new_bm_lhs_surface.mesh'.format(op.results_dir))
#    ##bmw.visualise_mesh(new_bm, fig, visualise=True, face_colours=(0,1,1), pt_size=3)
#    bmw.visualise_mesh(new_bm_lhs, fig, visualise=False, face_colours=(0,1,1), text_elements=[5,6,7,8,16,17,18,19], element_ids=False, nodes='all')

#    Xe_rhs = scipy.array([[0, 1, 2, 3],
#           [ 8,  9, 10, 11],
#           [16, 17, 18, 19]])
#    new_cwm_rhs = bmw.reposition_nodes_along_xi1(fig, cwm, offset, side='rhs', xi1_Xe=Xe_rhs, elem_shape=scipy.array(Xe).shape[::-1], debug=True)
#    new_cwm_rhs.save('{0}/new_cwm_rhs_surface.mesh'.format(op.results_dir))
#    bmw.visualise_mesh(new_cwm_rhs, fig, visualise=False, face_colours=(1,1,0), nodes='all')

#    Xe_lhs = scipy.array(Xe_rhs)
#    temp =  scipy.array([0,8,16])
#    for row in range(Xe_lhs.shape[0]):
#        Xe_lhs[row,:] = 7-scipy.array(Xe_rhs[0])+temp[row]
#    new_cwm_lhs = bmw.reposition_nodes_along_xi1(fig, cwm, offset, side='lhs', xi1_Xe=Xe_lhs, elem_shape=scipy.array(Xe).shape[::-1], debug=True)
#    new_cwm_lhs.save('{0}/new_cwm_lhs_surface.mesh'.format(op.results_dir))
#    bmw.visualise_mesh(new_cwm_lhs, fig, visualise=False, face_colours=(1,1,0), nodes='all')

#    # Create new volume mesh
#    rhs_fig = op.add_fig('rhs')
#    skin = False
#    mesh3D_rhs = bmw.create_volume_mesh(
#        new_bm_rhs, new_cwm_rhs, 'rhs', offset, fig, bm_rhs_surface_normals, skin=skin,
#        skin_thickness=1.45,smoothing=1)
#    mesh3D_rhs.label = 'mesh3D_rhs'
#    bmw.visualise_mesh(mesh3D_rhs, rhs_fig, visualise=False, face_colours=(0,1,0),pt_size=1, opacity=0.25, line_opacity=0.75)
#    mesh3D_rhs.save('{0}/prone_rhs.mesh'.format(op.results_dir))
#    bmw.generate_boundary_groups(mesh3D_rhs, fig, side='rhs', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=op.results_dir)

#    lhs_fig = op.add_fig('lhs')
#    skin = False
#    mesh3D_lhs = bmw.create_volume_mesh(
#        new_bm_lhs, new_cwm_lhs, 'lhs', offset, fig, bm_lhs_surface_normals, skin=skin,
#        skin_thickness=1.45,smoothing=1)
#    mesh3D_lhs.label = 'mesh3D_lhs'
#    bmw.visualise_mesh(mesh3D_lhs, lhs_fig, visualise=False, face_colours=(0,1,0),pt_size=1, opacity=0.25, line_opacity=0.75)
#    mesh3D_lhs.save('{0}/prone_lhs.mesh'.format(op.results_dir))
#    bmw.generate_boundary_groups(mesh3D_lhs, fig, side='lhs', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=op.results_dir)

#    # Load sternum/spine node groups
#    hdf5_main_grp = h5py.File('{0}/dof_groups_{1}.h5'.format(op.results_dir,'rhs'), 'r')
#    rhs_sternum_nodes = hdf5_main_grp['/nodes/sternum'][()].T
#    rhs_spine_nodes = hdf5_main_grp['/nodes/spine'][()].T
#    hdf5_main_grp = h5py.File('{0}/dof_groups_{1}.h5'.format(op.results_dir,'lhs'), 'r')
#    lhs_sternum_nodes = hdf5_main_grp['/nodes/sternum'][()].T
#    lhs_spine_nodes = hdf5_main_grp['/nodes/spine'][()].T
#    lhs_Xn_offset = 10000
#    lhs_Xe_offset = len(mesh3D_rhs.get_element_cids())
#    torso = bmw.join_lhs_rhs_meshes(mesh3D_lhs, mesh3D_rhs, fig, 'torso', scipy.hstack([rhs_sternum_nodes, rhs_spine_nodes]), scipy.hstack([lhs_sternum_nodes, lhs_spine_nodes]), lhs_Xn_offset=lhs_Xn_offset, lhs_Xe_offset=lhs_Xe_offset)
#    torso.save('{0}/prone.mesh'.format(op.results_dir))
#    fig_torso = op.add_fig('torso')
#    bmw.visualise_mesh(torso, fig_torso, visualise=False, face_colours=(0,1,1),pt_size=1, opacity=1, line_opacity = 0.75, text=False) #text_elements=spine_elements.tolist())
#    bmw.visualise_mesh(torso, fig_torso, visualise=False, face_colours=(0,1,1),pt_size=1, opacity=0.25, line_opacity = 0.75, text=False) #text_elements=spine_elements.tolist())
#    bmw.generate_boundary_groups(torso, fig_torso, side='both', visualise_points=False, visualise_nodes=False, export_groups=True, export_folder=op.results_dir, lhs_Xn_offset=lhs_Xn_offset, lhs_Xe_offset=lhs_Xe_offset, debug=False)
#    h5_dof_groups = h5py.File('{0}/dof_groups_{1}.h5'.format(op.results_dir, 'both'), 'r')
#    stiffer_shoulder_nodes = h5_dof_groups['/nodes/stiffer_shoulder'][()].T
#    fixed_shoulder_nodes = h5_dof_groups['/nodes/fixed_shoulder'][()].T
#    stiffer_back_nodes = h5_dof_groups['/nodes/stiffer_back'][()].T
#    transitional_nodes = h5_dof_groups['/nodes/transitional'][()].T
#    bmw.plot_points(fig_torso, 'stiffer_shoulder_nodes', torso.get_nodes(stiffer_shoulder_nodes.tolist()), stiffer_shoulder_nodes, visualise=False, colours=(0,0,1), point_size=10, text_size=5)
#    bmw.plot_points(fig_torso, 'fixed_shoulder_nodes', torso.get_nodes(fixed_shoulder_nodes.tolist()), fixed_shoulder_nodes, visualise=False, colours=(1,0,0), point_size=10, text_size=5)
#    bmw.plot_points(fig_torso, 'stiffer_back_nodes', torso.get_nodes(stiffer_back_nodes.tolist()), stiffer_back_nodes, visualise=False, colours=(0,1,0), point_size=10, text_size=5)
#    bmw.plot_points(fig_torso, 'transitional_nodes', torso.get_nodes(transitional_nodes.tolist()), transitional_nodes, visualise=False, colours=(1,1,0), point_size=10, text_size=5)

#    print 'Mesh construction complete.'

##    optimise_mesh = False
##    if optimise_mesh:
##        fig_mesh_opti = op.add_fig('mesh_opti')
##        #bmw.optimise_internal_mesh_nodes(torso, fig_mesh_opti, metric_evaluate_element_ids, node_ids)

##        bmw.optimise_spline_pts(torso, fig_torso)


##        metric_evaluate_element_ids = [12, 13, 14,23, 24, 25,34,35,36,45,46,47]
###        interior_nodes = [ 786,  787,  788,  789,  790,  791,  792,  793,  820,  821,  822,
###            823,  824,  825,  826,  827,  854,  855,  856,  857,  858,  859,
###            860,  861,  888,  889,  890,  891,  892,  893,  894,  895,  922,
###            923,  924,  925,  926,  927,  928,  929,  956,  957,  958,  959,
###            960,  961,  962,  963,  990,  991,  992,  993,  994,  995,  996,
###            997, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1058, 1059,
###            1060, 1061, 1062, 1063, 1064, 1065, 1092, 1093, 1094, 1095, 1096,
###            1097, 1098, 1099, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133,
###            1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1466, 1467, 1468,
###            1469, 1470, 1471, 1472, 1473, 1500, 1501, 1502, 1503, 1504, 1505,
###            1506, 1507, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1568,
###            1569, 1570, 1571, 1572, 1573, 1574, 1575, 1602, 1603, 1604, 1605,
###            1606, 1607, 1608, 1609, 1636, 1637, 1638, 1639, 1640, 1641, 1642,
###            1643, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1704, 1705,
###            1706, 1707, 1708, 1709, 1710, 1711, 1738, 1739, 1740, 1741, 1742,
###            1743, 1744, 1745, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779]

##        interior_nodes = [ 820,  821,  822,  823,  824,  854,  855,  856,  857,  858,  888,
##        889,  890,  891,  892,  922,  923,  924,  925,  926,  956,  957,
##        958,  959,  960,  990,  991,  992,  993,  994, 1024, 1025, 1026,
##       1027, 1028, 1058, 1059, 1060, 1061, 1062, 1466, 1467, 1468, 1469,
##       1470, 1500, 1501, 1502, 1503, 1504, 1534, 1535, 1536, 1537, 1538,
##       1568, 1569, 1570, 1571, 1572, 1602, 1603, 1604, 1605, 1606, 1636,
##       1637, 1638, 1639, 1640, 1670, 1671, 1672, 1673, 1674, 1704, 1705,
##       1706, 1707, 1708]

###"821..825,855..859,889..893,923..927,957..961,991..995,1025..1029,1059..1063,1467..1471,1501..1505,1535..1539,1569..1573,1603..1607,1637..1641,1671..1675,1705..1709"

##        bmw.optimise_internal_mesh_nodes(torso, fig_mesh_opti , interior_nodes, metric_evaluate_element_ids)

##        torso.save('{0}/prone_optimised.mesh'.format(op.results_dir))

##    #torso = morphic.Mesh('{0}/prone_optimised.mesh'.format(op.results_dir))

#    sys.path.insert(1, os.sep.join((os.environ['OPENCMISS_ROOT'],
#                                            'cm', 'bindings','python')))
#    from opencmiss import CMISS
#    from bmw import mechanics

#    export_mesh = False
#    if export_mesh:
#        mechanics.export_OpenCMISS_mesh(fig, op, mesh=torso, field_export_name='geometry')

#    solve_mechanics = True
#    if solve_mechanics:
#        [converged, dependent_field1, decomposition, region, cmesh] = (
#            mechanics.solve(fig, op, 0, parameters, mesh=torso, side='both',
#                field_export_name=field_export_name))
#        if converged:
#            offset = 1
#            num_deriv = 1
#            for node in torso.nodes:
#                for comp_idx in range(3):
#                    for deriv_idx in range(num_deriv):
#                        node.values[comp_idx] = dependent_field1.ParameterSetGetNodeDP(
#                            CMISS.FieldVariableTypes.U,
#                            CMISS.FieldParameterSetTypes.VALUES,
#                            1, deriv_idx + 1, node.id + offset, comp_idx + 1)
#            torso.save('{0}/reference.mesh'.format(op.results_dir))

#            [converged, dependent_field2, decomposition, region, cmesh] = (
#                mechanics.solve(fig, op, 1, parameters, mesh=torso,
#                    cmesh=cmesh, previous_dependent_field=dependent_field1,
#                    decomposition=decomposition, region=region, side='both',
#                    field_export_name=field_export_name))
#            if converged:
#                for node in torso.nodes:
#                    for comp_idx in range(3):
#                        for deriv_idx in range(num_deriv):
#                            node.values[comp_idx] = dependent_field2.ParameterSetGetNodeDP(
#                                CMISS.FieldVariableTypes.U,
#                                CMISS.FieldParameterSetTypes.VALUES,
#                                1, deriv_idx + 1, node.id + offset, comp_idx + 1)
#                torso.save('{0}/supine.mesh'.format(op.results_dir))

#                print 'Model solve complete.'

#                fig_supine = op.add_fig('supine')
#                supine = morphic.Mesh('{0}/supine.mesh'.format(op.results_dir))
#                bmw.visualise_mesh(supine, fig_supine, visualise=False, face_colours=(0,1,1),pt_size=1, opacity=1, line_opacity = 0.75)
#                h5_dof_groups = h5py.File('{0}/dof_groups_{1}.h5'.format(op.results_dir, 'both'), 'r')
#                stiffer_shoulder_nodes = h5_dof_groups['/nodes/stiffer_shoulder'][()].T
#                fixed_shoulder_nodes = h5_dof_groups['/nodes/fixed_shoulder'][()].T
#                bmw.plot_points(fig_supine, 'stiffer_shoulder_nodes', supine.get_nodes(stiffer_shoulder_nodes.tolist()), stiffer_shoulder_nodes, visualise=False, colours=(0,0,1), point_size=10, text_size=5)
#                bmw.plot_points(fig_supine, 'fixed_shoulder_nodes', supine.get_nodes(fixed_shoulder_nodes.tolist()), fixed_shoulder_nodes, visualise=False, colours=(1,0,0), point_size=10, text_size=5)

#        if parameter_set is not None:
#            converged_fname = '{0}/{1}_{2}.finished'.format(op.results_dir,op.volunteer,parameter_set)
#            converged_file = open(converged_fname, 'w')
#            converged_file.write('{0}'.format(converged))
#            converged_file.close()

