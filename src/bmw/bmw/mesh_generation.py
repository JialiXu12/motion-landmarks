import copy
import scipy
from scipy.spatial import cKDTree
from scipy import interpolate
import numpy
import morphic
from importlib import reload
reload(morphic)
from bmw import visualisation
import h5py
from collections import OrderedDict
import numpy as np
import pdb


def check_mesh_quality(mesh, jacobian_threshold=0.):
    '''
    Checks mesh quality (Jacobian) and returns details of bad elements for
    visualisation purposes.
    '''

    ng_xi = mesh._core.get_gauss_points([4, 4, 4])[0]
    num_ng = ng_xi.shape[0]
    num_elements = len(mesh.get_element_cids())

    results = {'no_bad_elements'    : True,
               'num_bad_gauss_points'   : 0,
               'bad_elements'       : {'element'      : [], 
                                       'gauss_pt_num' : [], 
                                       'gauss_pt_xi'  : [],
                                       'label'        : [],
                                       'jacobian'     : [],
                                       'position'     : []},
               'jacobian_threshold' : jacobian_threshold,
               'jacobians'          : scipy.zeros((num_elements,num_ng))}

    for element_idx, element in enumerate(mesh.elements):
        Xedxdnu = scipy.zeros((num_ng,3, 3))
        Xedxdnu[:,:,0] = element.evaluate(ng_xi,deriv=[1,0,0])
        Xedxdnu[:,:,1] = element.evaluate(ng_xi,deriv=[0,1,0])
        Xedxdnu[:,:,2] = element.evaluate(ng_xi,deriv=[0,0,1])
        for ng_idx in range(num_ng):
            results['jacobians'][element_idx, ng_idx] = (
                scipy.linalg.det(Xedxdnu[ng_idx,:,:]))
            if (results['jacobians'][element_idx, ng_idx] <=
                        jacobian_threshold):
                results['bad_elements']['element'].append(element.id)
                results['bad_elements']['gauss_pt_num'].append(ng_idx)
                results['bad_elements']['gauss_pt_xi'].append(
                    ng_xi[ng_idx,:])
                results['bad_elements']['label'].append(
                    'Xe_{0}_ng_{1}'.format(element.id,ng_idx))
                results['bad_elements']['jacobian'].append(
                    results['jacobians'][element_idx, ng_idx])
                results['bad_elements']['position'].append(
                    element.evaluate(ng_xi[ng_idx,:]))
                results['num_bad_gauss_points'] += 1

    if results['num_bad_gauss_points'] > 0:
        results['no_bad_elements'] = False

    return results

def export_mesh_quality_data(results, filename):

    hdf5_main_grp = h5py.File(filename, 'w')
    hdf5_main_grp.create_dataset('/no_bad_elements', 
        data = results['no_bad_elements'])
    if not results['no_bad_elements']:
        #import ipdb; ipdb.set_trace()
        #hdf5_main_grp.create_dataset('/num_bad_gauss_points', 
        #    data = results['num_bad_elements'])
        hdf5_main_grp.create_dataset('/bad_elements/element', 
            data = results['bad_elements']['element'])
        hdf5_main_grp.create_dataset('/bad_elements/gauss_pt_num', 
            data = results['bad_elements']['gauss_pt_num'])
        hdf5_main_grp.create_dataset('/bad_elements/gauss_pt_xi', 
            data = results['bad_elements']['gauss_pt_xi'])
        hdf5_main_grp.create_dataset('/bad_elements/label', 
            data = results['bad_elements']['label'])
        hdf5_main_grp.create_dataset('/bad_elements/jacobian', 
            data = results['bad_elements']['jacobian'])
        hdf5_main_grp.create_dataset('/bad_elements/position', 
            data = results['bad_elements']['position'])
        hdf5_main_grp.create_dataset('/jacobian_threshold', 
            data = results['jacobian_threshold'])
        hdf5_main_grp.create_dataset('/jacobians', 
            data = results['jacobians'])
    hdf5_main_grp.close()

def import_mesh_quality_data(filename):
    hdf5_main_grp = h5py.File(filename, 'r')
    return hdf5_main_grp['/bad_elements/position'][()]

def visualise_bad_gausspoints(fig, mesh_quality):
    '''
    Visualise position of bad Gauss points.
    '''
    if not mesh_quality['no_bad_elements']:
        visualisation.plot_points(fig, 'bad_elements',
            mesh_quality['bad_elements']['position'],
            mesh_quality['bad_elements']['label'], visualise=True,
            colours=(1,0,0), point_size=10, text_size=4, plot_text=True)

def length(v):
    """
    Calculates the length of a vector (v)
    """
    return numpy.sqrt((v * v).sum())

def vector(x1, x2, normalize=False):
    """
    Calculates a vector from two points and normalizes if normalize=True
    """
    v = numpy.array(x2) - numpy.array(x1)
    if normalize:
        return v / length(v)
    return v

rangei = lambda start, end: range(start, end+1)
rangei_add = lambda start, increment: rangei(start, start + increment)

def integrate_length(X):
    dX = X[1:, :] - X[:-1, :]
    ln = scipy.sqrt((dX*dX).sum(1))
    return ln

def join_lhs_rhs_meshes(lhs, rhs,  rhs_sternum_spine_nodes, lhs_sternum_spine_nodes,fig =None,debug =False):

    torso = morphic.Mesh()
    torso.auto_add_faces = True

    # offset lhs element and node numbering
    #lhs_Xn_offset = 10000 #len(rhs.get_node_ids()[1])

    # Add rhs mesh nodes and elements to new torso mesh
    for node_idx in rhs.get_node_ids()[1]:
        torso.add_stdnode(node_idx, rhs.nodes[node_idx].values, group='_default')
    for element in rhs.elements:
        torso.add_element(element.id, ['L3', 'L3', 'L3'], element.node_ids,group='_default')

    Xn_offset = len(rhs.get_node_ids()[1])
    Xe_offset = max(rhs.elements.ids)
    node_idx = 1
    old_ids = []
    for node_id in lhs.get_node_ids()[1]:
        if node_id not in lhs_sternum_spine_nodes:
            torso.add_stdnode(node_idx + Xn_offset, lhs.nodes[node_id].values, group='_default')
            node_idx +=1
            old_ids.append(node_id)

    if debug and fig is not None:

        for node_idx in range(len(lhs.get_nodes(lhs_sternum_spine_nodes.tolist()))):
            fig.plot_points('sternum_node_lhs_{0}'.format(node_idx), lhs.get_nodes(lhs_sternum_spine_nodes[node_idx].tolist()), color=(0,0,1), size=5)
            fig.plot_points('sternum_node_rhs_{0}'.format(node_idx), rhs.get_nodes(lhs_sternum_spine_nodes[node_idx].tolist()), color=(1,0,0), size=5)


    for element in lhs.elements:
        element_nodes = []
        for element_node in element.node_ids:
            if element_node in lhs_sternum_spine_nodes:

                lhs_idx = numpy.where(scipy.array(lhs_sternum_spine_nodes)==element_node)[0][0]
                element_nodes.append(rhs_sternum_spine_nodes[lhs_idx])
            else:
                lhs_idx = numpy.where(scipy.array(old_ids) == element_node)[0][0]
                element_nodes.append(lhs_idx + Xn_offset+1)
        torso.add_element(element.id + Xe_offset, ['L3', 'L3', 'L3'], element_nodes)
    torso.generate()
    #import ipdb; ipdb.set_trace()
    return torso

def smooth_shoulder_region(mesh, fig, smoothing=0):

    debug = False

    shoulder_spline_nodes = numpy.array([rangei_add(15,12),
        rangei_add(49,12),
        rangei_add(83,12),
        rangei_add(117,12),
        rangei_add(151,12),
        rangei_add(185,12)])
    spline_fit_node_idxs = numpy.array([0,1,2,3,9,10,11,12])
    spline_eval_node_idxs = numpy.array([4,5,6,7,8])

    num_lines = shoulder_spline_nodes.shape[0]
    num_spline_pts = 1000
    spline_xi = scipy.linspace(0, 1, num_spline_pts)
    for line in range(num_lines):
        nodes = mesh.get_nodes(shoulder_spline_nodes[line,spline_fit_node_idxs].tolist())
        x = nodes[:,0]
        y = nodes[:,1]
        z = nodes[:,2]
        if debug:
            fig.plot_points(
                'spline_line{0}_nodes'.format(line), nodes,
                color=(0,0,1), size=2)

        tck,u = interpolate.splprep([x,y,z] ,s = smoothing, k = 3)
        xnew,ynew,znew= interpolate.splev(spline_xi, tck, der = 0)
        spline_pts = scipy.array([xnew,ynew,znew]).T
        if debug:
            fig.plot_points(
                '{0}_spline_line{0}_pts'.format(line), spline_pts,
                color=(0,1,0), size=1)

        spline_pts_KDTree = cKDTree(spline_pts)
        spline_eval_bnds = spline_pts_KDTree.query(mesh.get_nodes(
                shoulder_spline_nodes[line,spline_fit_node_idxs[[3,4]]].tolist()), k=1)[1]
        spline_new_Xn_xi = scipy.linspace(
            spline_xi[spline_eval_bnds[0]], spline_xi[spline_eval_bnds[1]],
            spline_eval_node_idxs.shape[0]+1, endpoint=False)
        xnew,ynew,znew= interpolate.splev(spline_new_Xn_xi, tck, der = 0)
        spline_eval_pts = scipy.array([xnew[1:],ynew[1:],znew[1:]]).T

        if debug:
            fig.plot_points(
                '{0}_spline_line{0}_eval_pts'.format(line), spline_eval_pts,
                color=(1,0,0), size=5)

        #import ipdb; ipdb.set_trace()
        for idx, node_num in enumerate(shoulder_spline_nodes[line,spline_eval_node_idxs]):
            mesh.nodes[node_num].values[:] = spline_eval_pts[idx,:]

    return mesh

def add_shoulder_elements(mesh, node_offset=100, adjacent_nodes=None, armpit_nodes=None):
    # Add 2 new nodes based on average position
    new_Xnid = [node_offset+1, node_offset+2]

    temp_mesh = morphic.Mesh()
    node1 = temp_mesh.add_stdnode(1, mesh.nodes[adjacent_nodes[0][0]].values)
    node2 = temp_mesh.add_stdnode(2, mesh.nodes[adjacent_nodes[0][1]].values)
    node3 = temp_mesh.add_stdnode(3, mesh.nodes[adjacent_nodes[1][0]].values)
    node4 = temp_mesh.add_stdnode(4, mesh.nodes[adjacent_nodes[1][1]].values)

    elem1 = temp_mesh.add_element(1, ['H3','H3'], [1, 2,3,4])

    temp_mesh.generate()
    xi = [0.5,0]
    new_node = elem1.evaluate(xi)
    first_deriv_x =elem1.evaluate(xi, deriv=[1, 0])
    first_deriv_y = elem1.evaluate(xi, deriv=[0, 1])
    second_deriv = elem1.evaluate(xi, deriv=[1, 1])
    mesh.add_stdnode(new_Xnid[0], numpy.stack([new_node, first_deriv_x, first_deriv_y, second_deriv]).T,
                     group = '_default')

    xi = [0.5,1]
    new_node = elem1.evaluate(xi)
    first_deriv_x =elem1.evaluate(xi, deriv=[1, 0])
    first_deriv_y = elem1.evaluate(xi, deriv=[0, 1])
    second_deriv = elem1.evaluate(xi, deriv=[1, 1])
    mesh.add_stdnode(new_Xnid[1], numpy.stack([new_node, first_deriv_x, first_deriv_y, second_deriv]).T,
                     group = '_default')



    # Add 4 new elements
    num_Xe = len(mesh.get_element_cids())
    new_Xeid = [num_Xe, num_Xe+1, num_Xe+2, num_Xe+3]
    mesh.add_element(new_Xeid[0], ['H3', 'H3'], [adjacent_nodes[0][0], new_Xnid[0], adjacent_nodes[1][0], new_Xnid[1]], group='shoulder')
    mesh.add_element(new_Xeid[1], ['H3', 'H3'], [new_Xnid[0], adjacent_nodes[0][1], new_Xnid[1], adjacent_nodes[1][1]], group='shoulder')
    mesh.add_element(new_Xeid[2], ['H3', 'H3'], [adjacent_nodes[1][0], new_Xnid[1], armpit_nodes[0], armpit_nodes[1]], group='shoulder')
    mesh.add_element(new_Xeid[3], ['H3', 'H3'], [new_Xnid[1], adjacent_nodes[1][1], armpit_nodes[1], armpit_nodes[2]], group='shoulder')

    mesh.generate()
    for node in new_Xnid:
        mesh.nodes[node].smooth_derivatives([0,1])


    return mesh

def normalise(v):
    if len(v.shape)>1:
        normalised_v = scipy.zeros(v.shape)
        for idx in range(v.shape[0]):
            pt = v[idx,:]
            normalised_v[idx,:] = pt/scipy.sqrt(scipy.sum(pt*pt))
    else:
        normalised_v = v/scipy.sqrt(scipy.sum(v*v))
    return normalised_v

def magnitude(v):
    if len(v.shape)>1:
        magnitude_v = scipy.zeros(v.shape)
        for idx in range(v.shape[0]):
            pt = v[idx,:]
            magnitude_v[idx,:] = scipy.sqrt(scipy.sum(pt*pt))
    else:
        magnitude_v = scipy.sqrt(scipy.sum(v*v))
    return magnitude_v


def generate_element_nodes(basis, num_elem_xi, extent=[]):
    A=1
    if basis == ['H3', 'H3']:
        BASIS_NUMBER_OF_NODES = 4
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [2, 2]
        NUMBER_OF_NODES_XIC.insert(0,0)
    elif basis == ['L2', 'L2']:
        BASIS_NUMBER_OF_NODES = 9
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [3, 3]
        NUMBER_OF_NODES_XIC.insert(0,0)
    elif basis == ['L3', 'L3']:
        BASIS_NUMBER_OF_NODES = 16
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [4, 4]
        NUMBER_OF_NODES_XIC.insert(0,0)
    elif basis == ['L3', 'L3','L3']:
        BASIS_NUMBER_OF_NODES = 64
        NUMBER_OF_XIC = 3
        NUMBER_OF_NODES_XIC = [4, 4, 4]
        NUMBER_OF_NODES_XIC.insert(0,0)
    elif basis == ['L5', 'L5','L5']:
        BASIS_NUMBER_OF_NODES = 6*6*6
        NUMBER_OF_XIC = 3
        NUMBER_OF_NODES_XIC = [6, 6, 6]
        NUMBER_OF_NODES_XIC.insert(0,0)

    
    NUMBER_OF_ELEMENTS_XIC = copy.deepcopy(num_elem_xi)
    NUMBER_OF_ELEMENTS_XIC.insert(0,0)
    TOTAL_NUMBER_OF_NODES_XIC = [0]*(3)
    TOTAL_NUMBER_OF_NODES_XIC.insert(0,0)
    TOTAL_NUMBER_OF_ELEMENTS_XIC = [0]*(3)
    TOTAL_NUMBER_OF_ELEMENTS_XIC.insert(0,0)

    #Calculate sizes
    TOTAL_NUMBER_OF_NODES=1
    TOTAL_NUMBER_OF_ELEMENTS=1
    for xic_idx in range(1,NUMBER_OF_XIC+A):
        TOTAL_NUMBER_OF_NODES_XIC[xic_idx]=(NUMBER_OF_NODES_XIC[xic_idx]-2)*NUMBER_OF_ELEMENTS_XIC[xic_idx]+NUMBER_OF_ELEMENTS_XIC[xic_idx]+1
        TOTAL_NUMBER_OF_ELEMENTS_XIC[xic_idx]=NUMBER_OF_ELEMENTS_XIC[xic_idx]
        TOTAL_NUMBER_OF_NODES=TOTAL_NUMBER_OF_NODES*TOTAL_NUMBER_OF_NODES_XIC[xic_idx]
        TOTAL_NUMBER_OF_ELEMENTS=TOTAL_NUMBER_OF_ELEMENTS*TOTAL_NUMBER_OF_ELEMENTS_XIC[xic_idx]

    Xe_nodes = scipy.zeros((TOTAL_NUMBER_OF_ELEMENTS,BASIS_NUMBER_OF_NODES),dtype='uint32')
    Xeid = 0
    #!Set the elements for the regular mesh
    ELEMENT_NODES = [0]*(BASIS_NUMBER_OF_NODES+A)
    #Step in the xi[3)direction
    for ne3 in range(A,TOTAL_NUMBER_OF_ELEMENTS_XIC[3]+1+A):
        for ne2 in range(A,TOTAL_NUMBER_OF_ELEMENTS_XIC[2]+1+A):
            for ne1 in range(A,TOTAL_NUMBER_OF_ELEMENTS_XIC[1]+1+A):
                if((NUMBER_OF_XIC<3) or (ne3 <=TOTAL_NUMBER_OF_ELEMENTS_XIC[3])):
                    if(NUMBER_OF_XIC<2 or ne2<=TOTAL_NUMBER_OF_ELEMENTS_XIC[2]):
                        if(ne1<=TOTAL_NUMBER_OF_ELEMENTS_XIC[1]):
                            ne=ne1
                            np=1+(ne1-1)*(NUMBER_OF_NODES_XIC[1]-1)
                            if(NUMBER_OF_XIC>1):
                                ne=ne+(ne2-1)*TOTAL_NUMBER_OF_ELEMENTS_XIC[1]
                                np=np+(ne2-1)*TOTAL_NUMBER_OF_NODES_XIC[1]*(NUMBER_OF_NODES_XIC[2]-1)
                                if(NUMBER_OF_XIC>2):
                                    ne=ne+(ne3-1)*TOTAL_NUMBER_OF_ELEMENTS_XIC[1]*TOTAL_NUMBER_OF_ELEMENTS_XIC[2]
                                    np=np+(ne3-1)*TOTAL_NUMBER_OF_NODES_XIC[1]*TOTAL_NUMBER_OF_NODES_XIC[2]*(NUMBER_OF_NODES_XIC[3]-1)
                            nn=0
                            for nn1 in range(A,NUMBER_OF_NODES_XIC[1]+A):
                                nn=nn+1
                                ELEMENT_NODES[nn]=np+(nn1-1)
                            if(NUMBER_OF_XIC>1):
                                for nn2 in range(A+1,NUMBER_OF_NODES_XIC[2]+A):
                                    for nn1 in range(A,NUMBER_OF_NODES_XIC[1]+A):
                                        nn=nn+1
                                        ELEMENT_NODES[nn]=np+(nn1-1)+(nn2-1)*TOTAL_NUMBER_OF_NODES_XIC[1]
                                if(NUMBER_OF_XIC>2):
                                    for nn3 in range(A+1,NUMBER_OF_NODES_XIC[3]+A):
                                        for nn2 in range(A,NUMBER_OF_NODES_XIC[2]+A):
                                            for nn1 in range(A,NUMBER_OF_NODES_XIC[1]+A):
                                                nn=nn+1
                                                ELEMENT_NODES[nn]=np+(nn1-1)+(nn2-1)*TOTAL_NUMBER_OF_NODES_XIC[1]+(nn3-1)*TOTAL_NUMBER_OF_NODES_XIC[1]*TOTAL_NUMBER_OF_NODES_XIC[2]
                            #print ELEMENT_NODES[1:len(ELEMENT_NODES)]
                            Xe_nodes[Xeid,:] = ELEMENT_NODES[1:len(ELEMENT_NODES)]
                            Xeid += 1

    if extent == []:
        return Xe_nodes - 1 # Generated node numbers begin from 0
    else:
        Xn = numpy.zeros((TOTAL_NUMBER_OF_NODES,NUMBER_OF_XIC))

        FIELD_NODE_USER_NUMBER = 0

        if NUMBER_OF_XIC == 3:
            VALUE = [0.0]*(3)
            for Z_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[3]):
                for Y_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[2]):
                    for X_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[1]):
                        for XIC_COORDINATE in range(NUMBER_OF_XIC):
                            Xn[FIELD_NODE_USER_NUMBER,XIC_COORDINATE] = float(VALUE[XIC_COORDINATE])
                        FIELD_NODE_USER_NUMBER += 1
                        VALUE[0] = float(VALUE[0]) + float(extent[0])/float((TOTAL_NUMBER_OF_NODES_XIC[1]-1))
                    VALUE[1] = float(VALUE[1]) + float(extent[1])/float((TOTAL_NUMBER_OF_NODES_XIC[2]-1))
                    VALUE[0] = 0.0
                VALUE[2] = float(VALUE[2]) + float(extent[2])/float((TOTAL_NUMBER_OF_NODES_XIC[3]-1))
                VALUE[1] = 0.0
        elif NUMBER_OF_XIC == 2:
            VALUE = [0]*(2)
            for Y_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[2]):
                for X_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[1]):
                    for XIC_COORDINATE in range(NUMBER_OF_XIC):
                        Xn[FIELD_NODE_USER_NUMBER,XIC_COORDINATE] = VALUE[XIC_COORDINATE]
                    FIELD_NODE_USER_NUMBER += 1
                    VALUE[0] = VALUE[0] + (extent[0]/(TOTAL_NUMBER_OF_NODES_XIC[1]-1))
                VALUE[1] = extent[1]/(TOTAL_NUMBER_OF_NODES_XIC[2]-1)
                VALUE[0] = 0

        return Xe_nodes - 1, Xn # Generated node numbers begin from 0

def calculate_line_lengths(points):
    # Individual lengths
    ilens = integrate_length(points)
    total_len = ilens.sum()
    # Cumulitive lengths
    clens = scipy.zeros((points.shape[0]))
    for ln in range(1, len(ilens)+1):
        clens[ln] = ilens[0:ln].sum()
    return total_len, clens, ilens

def cstr_pt_surf_proj_objfun(xi, mesh, target_elem, source_point):
    '''
    Objective function for constrained point to surface projection.
    '''
    eval_pt = mesh.elements[target_elem].evaluate(xi)
    idxs = scipy.array([1,2]) # Coordinate indices corresponding to x and z.
    return scipy.linalg.norm(eval_pt[idxs] - source_point[idxs])


def cstr_pt_surf_proj(fig, mesh, target_elem, source_point, debug=False):
    '''
    Find the xi position of a source point projected onto a target element
    on a 3D surface mesh. This is achieved using a constrained (cstr)
    optimisation procedure.
    '''
    xi0 = scipy.array([0.5,0.5]) # Initial estimate of xi
    f = (lambda xi, mesh=mesh, target_elem=target_elem, source_point=source_point:
            (cstr_pt_surf_proj_objfun(xi, mesh, target_elem, source_point)))
    res = scipy.optimize.minimize(
        f, xi0, method='SLSQP', options={'eps':1e-6}, bounds=[(0.,1.)]*len(xi0))

    if debug:
        proj_pt = mesh.elements[target_elem].evaluate(res.x)
        fig.plot_points( 'proj_pt', [proj_pt], color=(0,0,1), size=2)

    return res.x, mesh.elements[target_elem].evaluate(res.x)

def reposition_nodes(fig, mesh, skin_mesh, offset, side='right', xi1_Xe=[], elem_shape=[], debug=False):
    '''
    Reposition nodes on chest wall surface. Mesh & xi1_Xe define the elements
    in the original cubic Hermite chest wall mesh. This will be remeshed to
    define a new cubic Lagrange mesh whose shape is desfined in elem_shape.
    '''

    # Convert the number of cubic Hermite nodes along xi1 to the equivialant number of cubic Lagrange nodes
    num_tot_lagrange_nodes_xi1 = 3*elem_shape[0]+1
    hermite_normalised_target_lens_xi1 = scipy.array([0.,
                                  0.07,
                                  0.18,
                                  0.22,
                                  0.27,
                                  0.32,
                                  0.36,
                                  0.4,
                                  0.5,
                                  0.6,
                                  0.8,
                                  1.])
    num_tot_lagrange_nodes_xi2 =  3*elem_shape[1]+1
    hermite_normalised_target_lens_xi2 = scipy.array(
        [ 0.        ,
          0.275,
          0.450,
          0.59       ,
          0.66666667,
          0.83333333,
          1.        ])

    if side == 'right':
        direction = 1.
        xi2_edge = 0.0
    elif side == 'left':
        direction = -1. # Reverse x1 direction
        xi2_edge = 1.0
        #hermite_normalised_target_lens_xi2 = 1.-hermite_normalised_target_lens_xi2
    skin_caudal_pt = skin_mesh.get_node_ids()[0][-1,:]
    orig_rib_target_elem = 16 # The element to search 
    proj_xi, proj_pt = cstr_pt_surf_proj(fig, mesh, orig_rib_target_elem, skin_caudal_pt, debug=True)
    z_trim = proj_pt[-1] # Set the z position to trim the points along the xi2=0 edge
    line  = evaluate_points_on_edge(fig, mesh, edge=2, xi=[xi2_edge,None], element_ids=xi1_Xe[:,0], direction=1, num_points=1000, z_trim=z_trim,debug=debug)
    length, clens, _ = calculate_line_lengths(line['points'])

    target_lens = calc_target_lagrange_lens(hermite_normalised_target_lens_xi2*length,
                                            num_tot_lagrange_nodes_xi2, side, xi=2,debug = debug)
    target_lens = scipy.reshape(target_lens, (-1, 1)) # convert to 2D
    clens_KDTree = cKDTree(scipy.array([clens]).T)
    closest_xi_id = clens_KDTree.query(target_lens, k=1)[1]
    xi2_closest_points =  line['points'][closest_xi_id]
    xi2_closest_xi =  line['xi'][closest_xi_id]
    xi2_closest_elements =  line['elements'][closest_xi_id]
    if debug:
        fig.plot_points('repositioned_xi2_nodes', xi2_closest_points,color=(0,1,0), size=3)

    new_Xe_nodes = generate_element_nodes(['L3', 'L3'], scipy.array(elem_shape).tolist(), extent=[])
    num_elems = scipy.prod(elem_shape)
    num_nodes =  scipy.unique(new_Xe_nodes).shape[0]
    
    new_mesh = morphic.Mesh()
    new_mesh.auto_add_faces = True
    new_mesh.label = side + '_lagrange_cws_mesh'
    
    Xe_list = scipy.array(elem_shape).flatten().tolist()
    Xn_list = scipy.arange(0+offset,num_nodes+offset)

    node_idx = 0
    for line_id in range(num_tot_lagrange_nodes_xi2):
        elem_row_idx = scipy.where(xi1_Xe[:,0]==xi2_closest_elements[line_id])[0][0]
        Xe_ids = xi1_Xe[elem_row_idx,:]
        xi2 = xi2_closest_xi[line_id,1]
        #if side == 'lhs':
        #    import ipdb; ipdb.set_trace()
        line  = evaluate_points_on_edge(
            fig, mesh, edge=1, xi=[None,xi2], element_ids=Xe_ids, direction=direction, num_points=1000,debug=False)
        length, clens, _ = calculate_line_lengths(line['points'])

        target_lens = calc_target_lagrange_lens(hermite_normalised_target_lens_xi1*length, num_tot_lagrange_nodes_xi1, side, xi=1)
        target_lens = scipy.reshape(target_lens, (-1, 1)) # convert to 2D
        clens_KDTree = cKDTree(scipy.array([clens]).T)
        closest_xi_id = clens_KDTree.query(target_lens, k=1)[1]
        closest_points =  line['points'][closest_xi_id]

        add_node = False
        for point in closest_points:
            new_mesh_nodes = new_mesh.get_nodes()
            if not new_mesh_nodes.shape[0] == 0:
                new_mesh_nodes_KDTree = cKDTree(new_mesh_nodes)
                d = new_mesh_nodes_KDTree.query(scipy.reshape(point, (-1, 1)).T, k=1)[0][0]
                if not scipy.isclose(d, 0.0):
                    add_node = True
            else:
                add_node = True
            if add_node:
                new_mesh.add_stdnode(node_idx+offset, point, group='_default')
                node_idx += 1

    visualise = False
    if visualise:
        Xn = new_mesh.get_nodes(group='_default')
        fig.plot_points(
            '{0}_NewNodes'.format(new_mesh.label), Xn,
            color=(0,0,1), size=2)
        fig.plot_text(
            '{0}_Text'.format(new_mesh.label), Xn,
            Xn_list, size=5)
    
    #import ipdb; ipdb.set_trace()

    for eidx in range(num_elems):
        new_mesh.add_element(eidx, ['L3', 'L3'], new_Xe_nodes[eidx] + offset)
    new_mesh.generate()

    return new_mesh

def calc_target_lagrange_lens(hermite_target_lens,num_tot_lagrange_nodes, side, xi, debug=False):

    initialise = False
    if initialise:
        num_tot_lagrange_nodes = 12
        mu = 0.4
        sig = 0.5
        set1 = gaussian(numpy.linspace(0., mu/2., num=numpy.ceil(num_tot_lagrange_nodes/4.)), mu, sig)*mu/2.
        set1[0] = 0.0

    else:
        if debug:
            target_lens = numpy.linspace(0., 0.25, num=num_tot_lagrange_nodes)
        else:
            #import ipdb; ipdb.set_trace()
            target_lens = numpy.zeros((num_tot_lagrange_nodes))
            if side == 'right':
                node_idx = 1
                for elem_idx in range(len(hermite_target_lens)-1):
                    #print elem_idx
                    start = hermite_target_lens[elem_idx]
                    end = hermite_target_lens[elem_idx+1]
                    temp = numpy.linspace(start,end, num=4, endpoint=True)
                    target_lens[node_idx] = temp[1]
                    node_idx += 1
                    target_lens[node_idx] = temp[2]
                    node_idx += 1
                    target_lens[node_idx] = end
                    node_idx += 1
                    #print 'node idx', node_idx

            elif  side == 'left':
                #if xi==1:
                node_idx = 1
                for elem_idx in range(len(hermite_target_lens)-1):
                    start = hermite_target_lens[elem_idx]
                    end = hermite_target_lens[elem_idx+1]
                    temp = numpy.linspace(start,end, num=4, endpoint=True)
                    target_lens[node_idx] = temp[1]
                    node_idx += 1
                    target_lens[node_idx] = temp[2]
                    node_idx += 1
                    target_lens[node_idx] = end
                    node_idx += 1

    return target_lens

def gaussian(x, mu, sig):
    return numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sig, 2.)))


#            closest_elements =  line['elements'][closest_xi_id]
#            if debug and fig is not None:
#                fig.plot_points(
#                    'closest_points', line['points'][closest_xi_id],
#                    color=(0,0,1), size=2)

#            #import ipdb; ipdb.set_trace()
#            for elem_idx in range(elem_shape[0]): # Number of elements along xi1 in target mesh
#                repositioned_Xe[elem_idx, global_line_idx] = closest_elements[elem_idx]
#                repositioned_xi[global_elem_idx,0,:] = [closest_xi[elem_idx][0], closest_xi[elem_idx+1][0]] # xi1: [start, end]
#                if line_id is 1:
#                    xi2 = numpy.array([0.,0.5])
#                elif line_id is 2:
#                    xi2 = numpy.array([0.5,1.])
#                repositioned_xi[global_elem_idx,1,:] = xi2 # xi2: [start, end]
#                global_elem_idx += 1
#            global_line_idx += 1

#    return repositioned_Xe.T, repositioned_xi

def create_surface_mesh2(fig, label, old_mesh, Xe, hanging_e, offset, visualise=False):
    
    num_Xexi = [scipy.array(Xe).shape[1], scipy.array(Xe).shape[0]]

    new_Xe_nodes = generate_element_nodes(['L3', 'L3'], num_Xexi, extent=[])
    num_elems = scipy.prod(num_Xexi)
    num_nodes =  scipy.unique(new_Xe_nodes).shape[0]
    
    mesh = morphic.Mesh()
    mesh.auto_add_faces = True
    mesh.label = label
    
    Xe_list = scipy.array(Xe).flatten().tolist()
    Xn_list = scipy.arange(0+offset,num_nodes+offset)
    
    xi = scipy.linspace(0., 1., 4)
    X, Y = scipy.meshgrid(xi,xi)
    Xi2d_default = scipy.array([
        X.reshape((X.size)),
        Y.reshape((Y.size))]).T
    global_Xsn = scipy.zeros((num_elems, Xi2d_default.shape[0], 3))
    nodes = []
    for eidx in range(num_elems):
        if hanging_e[eidx] is None:
            Xi2d = Xi2d_default
        else:
            xi1 = scipy.linspace(hanging_e[eidx][0][0], hanging_e[eidx][0][1], 4)
            xi2 = scipy.linspace(hanging_e[eidx][1][0], hanging_e[eidx][1][1], 4)
            X, Y = scipy.meshgrid(xi1,xi2)
            Xi2d = scipy.array([
                X.reshape((X.size)),
                Y.reshape((Y.size))]).T
        
        Xg = old_mesh.elements[Xe_list[eidx]].evaluate(Xi2d)
        global_Xsn[eidx,:,:] = old_mesh.elements[Xe_list[eidx]].normal(Xi2d)
        for nidx, node in enumerate(new_Xe_nodes[eidx]):
            if node not in nodes:
                nodes.append(node)
                mesh.add_stdnode(node+offset, Xg[nidx,:], group='_default')
    
    visualise = False
    if visualise:
        Xn = mesh.get_nodes(group='_default')
        fig.plot_points(
            '{0}_NewNodes'.format(mesh.label), Xn,
            color=(0,0,1), size=2)
        fig.plot_text(
            '{0}_Text'.format(mesh.label), Xn,
            Xn_list, size=5)
    
    for eidx in range(num_elems):
        mesh.add_element(eidx, ['L3', 'L3'], new_Xe_nodes[eidx] + offset)
    mesh.generate()
    
    return mesh, global_Xsn

def create_surface_mesh(fig, label, old_mesh, Xe, hanging_e, offset, visualise = True):
    # creare lagrangian surface mesh by projecting the skin nodes into the chest wall surface
    num_Xexi = [scipy.array(Xe).shape[1], scipy.array(Xe).shape[0]]

    new_Xe_nodes = generate_element_nodes(['L3', 'L3'], num_Xexi, extent=[])
    num_elems = scipy.prod(num_Xexi)
    num_nodes =  scipy.unique(new_Xe_nodes).shape[0]
    
    mesh = morphic.Mesh()
    mesh.auto_add_faces = True
    mesh.label = label
    
    Xe_list = scipy.array(Xe).flatten().tolist()
    Xn_list = scipy.arange(0+offset,num_nodes+offset)
    
    xi = scipy.linspace(0., 1., 4)
    X, Y = scipy.meshgrid(xi,xi)
    Xi2d_default = scipy.array([
        X.reshape((X.size)),
        Y.reshape((Y.size))]).T
    global_Xsn = scipy.zeros((num_elems, Xi2d_default.shape[0], 3))
    nodes = []
    for eidx in range(num_elems):
        if hanging_e[eidx] is None:
            Xi2d = Xi2d_default
        else:
            xi1 = scipy.linspace(hanging_e[eidx][0][0], hanging_e[eidx][0][1], 4)
            xi2 = scipy.linspace(hanging_e[eidx][1][0], hanging_e[eidx][1][1], 4)
            X, Y = scipy.meshgrid(xi1,xi2)
            Xi2d = scipy.array([
                X.reshape((X.size)),
                Y.reshape((Y.size))]).T
        
        Xg = old_mesh.elements[Xe_list[eidx]].evaluate(Xi2d)
        global_Xsn[eidx,:,:] = old_mesh.elements[Xe_list[eidx]].normal(Xi2d)
        for nidx, node in enumerate(new_Xe_nodes[eidx]):
            if node not in nodes:
                nodes.append(node)
                #print 'adding node: ', node
                mesh.add_stdnode(node+offset, Xg[nidx,:], group='_default')

    if visualise:
        Xn = mesh.get_nodes(group='_default')
        fig.plot_points(
            '{0}_NewNodes'.format(mesh.label), Xn,
            color=(0,0,1), size=2)
        fig.plot_text(
            '{0}_Text'.format(mesh.label), Xn,
            Xn_list, size=5)
    
    for eidx in range(num_elems):
        mesh.add_element(eidx, ['L3', 'L3'], new_Xe_nodes[eidx] + offset)
    mesh.generate(True)
    
    return mesh, global_Xsn


def get_reordered_nodes(mesh):
    Xn_temp = mesh.get_nodes()
    num_layer_nodes = Xn_temp.shape[0]
    tree = cKDTree(Xn_temp)
    map_Xn = scipy.zeros((num_layer_nodes,3))
    for idx in range(num_layer_nodes):
        map_Xn[idx,:] = mesh.get_nodes(idx)
    d, node_map = tree.query(map_Xn)
    # Reorder nodes
    Xn = scipy.zeros(Xn_temp.shape)
    Xn = Xn_temp[node_map,:]
    return Xn



def create_volume_mesh(
        bm, cwm, offset, fig, bm_surface_normals,smoothing=0):
    # Add nodes between breast and chestwall surfaces
    bm_Xn = get_reordered_nodes(bm)
    cwm_Xn = get_reordered_nodes(cwm)
    num_Xe = len(bm.elements.ids)
    num_layer_nodes = cwm_Xn.shape[0]

    layer1 = -(bm_Xn-cwm_Xn)*1./3. + bm_Xn
    layer2 = -(bm_Xn-cwm_Xn)*2./3. + bm_Xn

    visualise = True
    if visualise:
        fig.plot_points(
            '{0}_layer1'.format(0), layer1,
            color=(0,0,1), size=1)
        fig.plot_points(
            '{0}_layer2'.format(0), layer2,
            color=(1,0,1), size=1)

    
    # Create new 3D mesh
    mesh3D = morphic.Mesh()
    mesh3D.auto_add_faces = True
    mesh3D.label = 'mesh3D'
    

    node_ordering = scipy.arange(0,64)
    Xn = scipy.array((cwm_Xn,layer2,layer1,bm_Xn))
    num_layer_elem_nodes = len(bm.elements[0].node_ids)
    created_node_list = []
    for eidx in range(num_Xe):
        new_elem_node_list = []
        layer_Xe_nodes = scipy.array(bm.elements[eidx].node_ids)
        for layer in range(4):
            for node in layer_Xe_nodes:
                new_node_id = node+num_layer_nodes*layer+offset
                new_elem_node_list.append(new_node_id)
                if new_node_id not in created_node_list:
                    created_node_list.append(new_node_id)
                    mesh3D.add_stdnode(new_node_id, Xn[layer,node,:], group='_default')
        #import ipdb; ipdb.set_trace()
        mesh3D.add_element(eidx+offset, ['L3', 'L3', 'L3'],
            scipy.array(new_elem_node_list)[node_ordering])
    mesh3D.generate()

    return mesh3D


def add_skin(mesh, skin_thickness, normal_factor = 1):
    # Add nodes between breast and chestwall surfaces

    cw_nodes = []
    skin_nodes = []

    for element in mesh.elements:
        cw_nodes += element.node_ids[:16]
        skin_nodes += element.node_ids[-16:]
    cw_nodes = list(OrderedDict.fromkeys(cw_nodes))
    skin_nodes = list(OrderedDict.fromkeys(skin_nodes))

    cw_nodes_position = numpy.zeros((len(cw_nodes), 3))
    skin_nodes_position = numpy.zeros((len(skin_nodes), 3))

    for node_index in range(len(cw_nodes)):
        cw_nodes_position[node_index] = mesh.get_nodes(cw_nodes[node_index])
        skin_nodes_position[node_index] = mesh.get_nodes(skin_nodes[node_index])

    skin_points, skin_xi, skin_xi_elements = generate_points_on_face(mesh, "xi3", 1, num_points=100)
    skin_tree = cKDTree(skin_points)
    _, skin_xi_index = skin_tree.query(skin_nodes_position)
    skin_nodes_xi = skin_xi[skin_xi_index]
    skin_xi_elements = skin_xi_elements[skin_xi_index]
    skin_nodes_normals = numpy.zeros((len(skin_nodes),3))

    for node_index in range(len(skin_nodes)):
        element = mesh.elements[skin_xi_elements[node_index]]
        distance_to_ribcage = numpy.linalg.norm(cw_nodes_position[node_index] -
                                                 skin_nodes_position[node_index])
        skin_nodes_normals[node_index] = surface_normal(element,skin_nodes_xi[node_index], normalise=True)
        # chech if the normal inpointing inside the volume
        new_point = skin_nodes_position[node_index]- skin_thickness*skin_nodes_normals[node_index]
        new_distance_to_ribcage = numpy.linalg.norm(cw_nodes_position[node_index]- new_point)
        if distance_to_ribcage < new_distance_to_ribcage:
            skin_nodes_normals[node_index] = -skin_nodes_normals[node_index]


    hypodermis = -skin_thickness * skin_nodes_normals + skin_nodes_position
    hypodermis[:,2] = skin_nodes_position [:,2] # keep the nodes in the same z plane
    #tissue_len = skin_nodes_position - hypodermis
    #pdb.set_trace()
    #xi3_direction = skin_nodes_position - cw_nodes_position
    #xi3_direction = xi3_direction / np.linalg.norm( xi3_direction , axis=1).reshape((len(xi3_direction),1))
    #tissue_len = np.array([np.dot(tissue_len[ind],xi3_direction[ind])*xi3_direction[ind] for ind in range(len(tissue_len))])
    #tissue_len =np.array([ x if np.linalg.norm(x) > 0.5 else x/np.linalg.norm(x) for x in tissue_len])
    #hypodermis = skin_nodes_position - tissue_len


    # tissue_len = numpy.sqrt(numpy.sum((hypodermis - cwm_Xn)**2,axis = 1))
    tissue_len = hypodermis - cw_nodes_position

    layer1 = hypodermis - 2./3.* tissue_len
    layer2 = -tissue_len * 1. / 3. + hypodermis

    tissue_len = skin_nodes_position - hypodermis
    layer4 = tissue_len * 1. / 3. + hypodermis
    layer5 = tissue_len * 2. / 3. + hypodermis

    # Create new 3D mesh
    new_mesh = morphic.Mesh()
    new_mesh.auto_add_faces = True
    new_mesh.label = 'mesh3D'


    created_node_list = []
    nodes_ids = list(range(1,len(skin_nodes)*7+1))
    nodes_ids = numpy.reshape(nodes_ids,(len(skin_nodes),7)).T

    global_eidx = 1
    new_nodes_list = []
    for elem_layer_idx, elem_layer in enumerate([1, 2]):
        if elem_layer == 1:
            layer_nodes = scipy.array((cw_nodes_position, layer1, layer2, hypodermis))
            layer_nodes_ids = nodes_ids[:4,:]
            local_index = list(range(16))
            node_list = cw_nodes
        elif elem_layer == 2:
            layer_nodes = scipy.array((hypodermis, layer4, layer5, skin_nodes_position))
            layer_nodes_ids = nodes_ids[3:,:]
            local_index = list(range(48,64))
            node_list = skin_nodes

        for elementId in mesh.get_element_ids():

            element_nodes = [mesh.elements[elementId].node_ids[x] for x in local_index]
            base_nodes_index = [node_list.index(x) for x in element_nodes]

            element_nodes_position =[]
            element_nodes_ids=[]
            for layer in range(4):
                element_nodes_position += [layer_nodes[layer][x] for x in base_nodes_index]
                element_nodes_ids += [layer_nodes_ids[layer][x] for x in base_nodes_index]

            for node_index in range(len(element_nodes_ids)):
                if element_nodes_ids[node_index] not in new_nodes_list:
                    new_nodes_list.append(element_nodes_ids[node_index])
                    new_mesh.add_stdnode(element_nodes_ids[node_index],
                                         element_nodes_position[node_index],
                                         group= '_default')
            new_mesh.add_element(global_eidx,['L3', 'L3', 'L3'],element_nodes_ids)
            global_eidx +=1
    return new_mesh
def mesh_resample_z(mesh, xi_position ,element_grid_shape, elements=[]):
    # Add nodes between breast and chestwall surfaces

    if len(elements) ==0:
        elements =mesh.elements.ids
    cw_nodes = []
    skin_nodes = []

    existing_nodes = []
    existing_elements =[]
    for element in mesh.elements:
        if element.id in elements:
            cw_nodes += element.node_ids[:16]
            skin_nodes += element.node_ids[-16:]
        else:
            existing_elements.append(element.id)
            existing_nodes = existing_nodes + element.node_ids

    existing_nodes = existing_nodes+cw_nodes
    existing_nodes = list(numpy.unique(existing_nodes))
    cw_nodes, node_index = numpy.unique(cw_nodes, return_index=True)
    skin_nodes = [skin_nodes[x] for x in node_index]


    # Create new 3D mesh
    new_mesh = morphic.Mesh()
    new_mesh.auto_add_faces = True
    new_mesh.label = 'mesh3D'

    if len(existing_nodes) == 0:
        start_new_nodes_ids = 1
    else:
        start_new_nodes_ids = numpy.max(existing_nodes)+1
    new_nb_nodes = len(skin_nodes)*6
    new_nodes_ids = list(cw_nodes)+list(range(start_new_nodes_ids,start_new_nodes_ids+new_nb_nodes))
    new_nodes_ids =  numpy.reshape(new_nodes_ids,(7,3*element_grid_shape[0]+1,3*element_grid_shape[1]+1))

    for element_id in existing_elements:
        new_mesh.add_element(element_id,['L3', 'L3', 'L3'],mesh.elements[element_id].node_ids)

    if len(existing_elements) == 0:
        start_new_elements_ids =1
    else:
        start_new_elements_ids = numpy.max(existing_elements)+1
    # create elements and nodes grid for the new mesh
    new_elements_ids = []
    for layer in [0,1]:
        starting_node_line = 0
        for element_line in range(0,element_grid_shape[0]):
            starting_node_column = 0
            for element_column in range(0,element_grid_shape[1]):
                element_id = start_new_elements_ids+element_column+element_grid_shape[1]*element_line+\
                             layer*element_grid_shape[1]*element_grid_shape[0]
                node_id_array = new_nodes_ids[layer*3:layer*3+4,starting_node_line:starting_node_line+4,
                                        starting_node_column:starting_node_column+4]
                new_mesh.add_element(int(element_id), ['L3', 'L3', 'L3'], node_id_array.flatten())
                new_elements_ids.append(element_id)

                starting_node_column +=3
            starting_node_line +=3

    for node_id in existing_nodes:
        new_mesh.add_stdnode(node_id,list(mesh.get_nodes(node_id)[0]), group='_default')


    xi_location = []
    for node_indx in range(64):
        for xi3 in range(4):
            for xi2 in range(4):
                for xi1 in range(4):
                    xi_location.append(numpy.array([xi1*(1/3),xi2*(1/3),xi3*(1/3)]))


    for element_id in new_elements_ids[:int(len(new_elements_ids)/2)]:
        elements_node_new_mesh = [new_mesh.elements[element_id].node_ids,
                             new_mesh.elements[(element_id+len(elements))].node_ids]
        for layer in range(2):
            for node_idx,node_id in enumerate(elements_node_new_mesh[layer]):
                if node_id not in existing_nodes:
                    xi = [xi_location[node_idx][0],
                          xi_location[node_idx][1],
                          xi_location[node_idx][2]*xi_position*(1-layer)+
                        (xi_position+xi_location[node_idx][2]*(1-xi_position))*layer]
                    node_position = mesh.elements[element_id].evaluate(xi)
                    existing_nodes.append(node_id)
                    new_mesh.add_stdnode(node_id,
                                         node_position,
                                         group='_default')


    new_mesh.generate()
    return new_mesh

def mesh_resample_z_linear(mesh, xi_position ,element_grid_shape, elements=[]):
    # Add nodes between breast and chestwall surfaces

    cw_nodes = []
    skin_nodes = []

    for element in mesh.elements:
        cw_nodes += element.node_ids[:16]
        skin_nodes += element.node_ids[-16:]
    cw_nodes = list(OrderedDict.fromkeys(cw_nodes))
    skin_nodes = list(OrderedDict.fromkeys(skin_nodes))

    cw_nodes_position = numpy.zeros((len(cw_nodes), 3))
    skin_nodes_position = numpy.zeros((len(skin_nodes), 3))

    for node_index in range(len(cw_nodes)):
        cw_nodes_position[node_index] = mesh.get_nodes(cw_nodes[node_index])
        skin_nodes_position[node_index] = mesh.get_nodes(skin_nodes[node_index])


    hypodermis = (skin_nodes_position-cw_nodes_position)*xi_position + cw_nodes_position

    # tissue_len = numpy.sqrt(numpy.sum((hypodermis - cwm_Xn)**2,axis = 1))
    tissue_len = hypodermis - cw_nodes_position

    layer1 = hypodermis - 2./3.* tissue_len
    layer2 = -tissue_len * 1. / 3. + hypodermis

    tissue_len = skin_nodes_position - hypodermis
    layer4 = tissue_len * 1. / 3. + hypodermis
    layer5 = tissue_len * 2. / 3. + hypodermis

    # Create new 3D mesh
    new_mesh = morphic.Mesh()
    new_mesh.auto_add_faces = True
    new_mesh.label = 'mesh3D'


    created_node_list = []
    nodes_ids = list(range(1,len(skin_nodes)*7+1))
    nodes_ids = numpy.reshape(nodes_ids,(len(skin_nodes),7)).T

    global_eidx = 1
    new_nodes_list = []
    for elem_layer_idx, elem_layer in enumerate([1, 2]):
        if elem_layer == 1:
            layer_nodes = scipy.array((cw_nodes_position, layer1, layer2, hypodermis))
            layer_nodes_ids = nodes_ids[:4,:]
            local_index = list(range(16))
            node_list = cw_nodes
        elif elem_layer == 2:
            layer_nodes = scipy.array((hypodermis, layer4, layer5, skin_nodes_position))
            layer_nodes_ids = nodes_ids[3:,:]
            local_index = list(range(48,64))
            node_list = skin_nodes

        for elementId in mesh.get_element_ids():

            element_nodes = [mesh.elements[elementId].node_ids[x] for x in local_index]
            base_nodes_index = [node_list.index(x) for x in element_nodes]

            element_nodes_position =[]
            element_nodes_ids=[]
            for layer in range(4):
                element_nodes_position += [layer_nodes[layer][x] for x in base_nodes_index]
                element_nodes_ids += [layer_nodes_ids[layer][x] for x in base_nodes_index]

            for node_index in range(len(element_nodes_ids)):
                if element_nodes_ids[node_index] not in new_nodes_list:
                    new_nodes_list.append(element_nodes_ids[node_index])
                    new_mesh.add_stdnode(element_nodes_ids[node_index],
                                         element_nodes_position[node_index],
                                         group= '_default')
            new_mesh.add_element(global_eidx,['L3', 'L3', 'L3'],element_nodes_ids)
            global_eidx +=1
    new_mesh.generate()
    return new_mesh


def extend_mesh(mesh, elements_id, element_node_index):
    # extend mesh in the cranial part
    # elements nodes index should give the node id in the element reference system
    # 2D array row representing xi1 and columns representing xi2


    face_nodes_ids = []
    for elementId in elements_id:
        face_nodes_ids +=  [mesh.elements[elementId].node_ids[x] for x in element_node_index]

    face_nodes_ids = list(OrderedDict.fromkeys(face_nodes_ids))
    nodes_position = mesh.get_nodes(face_nodes_ids)
    layer1 = nodes_position.copy()
    layer1[:,2] = layer1[:,2]-10
    layer2 = nodes_position.copy()
    layer2[:,2] = layer2[:,2]-20
    layer3 = nodes_position.copy()
    layer3[:,2] = layer3[:,2] -30
    new_nodes_position = scipy.array((nodes_position, layer1, layer2, layer3))

    # Create new 3D mesh
    new_mesh = morphic.Mesh()
    new_mesh.label = 'mesh3D'
    new_mesh.auto_add_faces = True

    for node in list(mesh.nodes):
        new_mesh.add_stdnode(node.id,node.values,group= '_default')

    existent_node_list = new_mesh.get_node_ids()[1]
    new_nodes_ids = list(range(max(existent_node_list)+1,len(face_nodes_ids)*3+max(existent_node_list)+1))
    new_nodes_ids = numpy.reshape(new_nodes_ids,(len(face_nodes_ids),3)).T
    new_nodes_ids = [numpy.array(face_nodes_ids)]+ list(new_nodes_ids)

    global_eidx = 1

    start_element = 1
    torso_sides = numpy.reshape(elements_id,(2,int(len(elements_id)/2)))
    for side_elements in torso_sides:

        for copy_elem in range(start_element, side_elements[-1] + 1):
            new_mesh.add_element(global_eidx, ['L3', 'L3', 'L3'], mesh.elements[copy_elem].node_ids)
            global_eidx += 1
        start_element = side_elements[-1] + 1

        for elementId in side_elements:
            element_nodes_position = []
            element_node_index = element_node_index.reshape((4,4))
            element_nodes_ids = []

            for row in range(4):
                row_node_ids = [mesh.elements[elementId].node_ids[x] for x in element_node_index[row]]
                layer_index = [face_nodes_ids.index(x) for x in row_node_ids]

                for layer in range(4):
                    element_nodes_ids += [new_nodes_ids[layer][x] for x in layer_index]
                    element_nodes_position += [new_nodes_position[layer][x] for x in layer_index]

            new_mesh.add_element(global_eidx, ['L3', 'L3', 'L3'], element_nodes_ids)
            global_eidx += 1
            for node_id in element_nodes_ids:
                if not (node_id in existent_node_list):
                    node_index = element_nodes_ids.index(node_id)
                    new_mesh.add_stdnode(element_nodes_ids[node_index],
                                         element_nodes_position[node_index],
                                         group='_default')




    return new_mesh



def mesh_resample_x(mesh, xi_position , element_grid_shape, elements=[]):
    # Add nodes between breast and chestwall surfaces

    if len(elements) ==0:
        elements = mesh.get_element_ids()


    # Create new 3D mesh
    new_mesh = morphic.Mesh()
    new_mesh.auto_add_faces = True
    new_mesh.label = 'mesh3D'

    new_grid_shape = [element_grid_shape[0],element_grid_shape[1]*2]
    new_nb_nodes = (new_grid_shape[0]*3+1)*(new_grid_shape[1]*3+1)*4
    new_nodes_ids = list(range(new_nb_nodes))
    new_nodes_ids = numpy.reshape(new_nodes_ids,(4,new_grid_shape[0]*3+1,new_grid_shape[1]*3+1))


    starting_node_line = 0
    for element_line in range(0,new_grid_shape[0]):
        starting_node_column = 0
        for element_column in range(0,new_grid_shape[1]):
            element_id = element_column+new_grid_shape[1]*element_line+1
            node_id_array = new_nodes_ids[:,starting_node_line:starting_node_line+4,
                                    starting_node_column:starting_node_column+4]
            new_mesh.add_element(element_id, ['L3', 'L3', 'L3'], node_id_array.flatten())

            starting_node_column +=3
        starting_node_line +=3


    xi_location = []
    for node_indx in range(64):
        for xi3 in range(4):
            for xi2 in range(4):
                for xi1 in range(4):
                    xi_location.append(numpy.array([xi1*(1/3),xi2*(1/3),xi3*(1/3)]))

    new_nodes_list = []
    for element_idx,element_id in enumerate(elements):
        elements_node_new_mesh = [new_mesh.elements[(element_idx+1)*2-1].node_ids,
                             new_mesh.elements[(element_idx+1)*2].node_ids]
        for line in range(2):
            for node_idx,node_id in enumerate(elements_node_new_mesh[line]):
                if node_id not in new_nodes_list:
                    xi = [xi_location[node_idx][0]*xi_position*(1-line)+
                        (xi_position+xi_location[node_idx][0]*(1-xi_position))*line,
                          xi_location[node_idx][1],
                          xi_location[node_idx][2]]
                    node_position = mesh.elements[element_id].evaluate(xi)
                    new_nodes_list.append(node_id)
                    new_mesh.add_stdnode(node_id,
                                         node_position,
                                         group='_default')


    new_mesh.generate()
    return new_mesh


def mesh_resample_y(mesh, xi_position , element_grid_shape, elements=[]):
    # Add nodes between breast and chestwall surfaces

    if len(elements) ==0:
        elements = mesh.get_element_ids()


    # Create new 3D mesh
    new_mesh = morphic.Mesh()
    new_mesh.auto_add_faces = True
    new_mesh.label = 'mesh3D'

    new_grid_shape = [element_grid_shape[0]*2,element_grid_shape[1],element_grid_shape[2]]
    new_nb_nodes = (new_grid_shape[0]*3+1)*(new_grid_shape[1]*3+1)*(element_grid_shape[2]*3+1)
    new_nodes_ids = list(range(new_nb_nodes))
    new_nodes_ids = numpy.reshape(new_nodes_ids,(new_grid_shape[2]*3+1,new_grid_shape[0]*3+1,new_grid_shape[1]*3+1))

    starting_layer = 0
    for layer in range(new_grid_shape[2]):
        starting_node_line = 0
        for element_line in range(0,new_grid_shape[0]):
            starting_node_column = 0
            for element_column in range(0,new_grid_shape[1]):
                element_id = layer*new_grid_shape[0]*new_grid_shape[1]+element_column+new_grid_shape[1]*element_line+1
                node_id_array = new_nodes_ids[starting_layer:starting_layer+4,starting_node_line:starting_node_line+4,
                                        starting_node_column:starting_node_column+4]
                new_mesh.add_element(element_id, ['L3', 'L3', 'L3'], node_id_array.flatten())

                starting_node_column +=3
            starting_node_line +=3
        starting_layer += 3




    xi_location = []
    for node_indx in range(64):
        for xi3 in range(4):
            for xi2 in range(4):
                for xi1 in range(4):
                    xi_location.append(numpy.array([xi1*(1/3),xi2*(1/3),xi3*(1/3)]))

    new_nodes_list = []
    for layer in range(element_grid_shape[2]):
        for element_line in range(element_grid_shape[0]):
            for element_column in range(element_grid_shape[1]):
                element_id1 = layer*new_grid_shape[0]*element_grid_shape[1] + 2*element_line*new_grid_shape[1]+element_column+1
                element_id2 = layer*new_grid_shape[0]*element_grid_shape[1] + (2*element_line+1)*new_grid_shape[1]+element_column+1
                elements_node_new_mesh = [new_mesh.elements[element_id1].node_ids,
                                 new_mesh.elements[element_id2].node_ids]

                element_id = layer*element_grid_shape[0]*element_grid_shape[1]+ element_line*element_grid_shape[1]+element_column+1
                for line in range(2):
                    for node_idx,node_id in enumerate(elements_node_new_mesh[line]):
                        if node_id not in new_nodes_list:
                            xi = [xi_location[node_idx][0],
                                  xi_location[node_idx][1] * xi_position * (1 - line) +
                                  (xi_position + xi_location[node_idx][1] * (1 - xi_position)) * line,
                                  xi_location[node_idx][2]]
                            node_position = mesh.elements[element_id].evaluate(xi)
                            new_nodes_list.append(node_id)
                            new_mesh.add_stdnode(node_id,
                                                 node_position,
                                                 group='_default')


    new_mesh.generate()
    return new_mesh

def surface_normal(element,xi, normalise=False):
    coef = numpy.zeros_like(xi)
    for idx, xi_value in enumerate(xi):
        if numpy.isclose(1.0, xi_value, rtol=1e-08):
            coef[idx] = -1.
        else:
            coef[idx] = 1.
    pt_0 = element.evaluate(xi)

    if len(xi) ==3:
        xi1 = xi + numpy.array([coef[0]*1.e-5, 0.,0])
        xi2 = xi+numpy.array([0., coef[1]*1.e-5,0])

    else:
        xi1 = xi+numpy.array([coef[0]*1.e-5, 0.])
        xi2 = xi+numpy.array([0., coef[1]*1.e-5])

    pt_x1 = element.evaluate(xi1)
    pt_x2 = element.evaluate(xi2)

    vector1 = coef[0]*vector(pt_0, pt_x1, normalize=True)
    vector2 = coef[1]*vector(pt_0, pt_x2, normalize=True)
    cross = numpy.cross(vector1, vector2)


    debug = False
    if debug:
        print ('coef: ', coef)
        print ('xi: ', xi)
        print ('xi perturb 1: ', xi+numpy.array([coef[0]*1.e-5, 0.]))
        print ('xi perturb 2: ', xi+numpy.array([0., coef[1]*1.e-5,]))
        print ('pt_o: ', pt_0)
        print ('pt_x1: ', pt_x1)
        print ('pt_x2: ', pt_x2)
        print ('Vector1: ', vector1)
        print ('Vector2: ', vector2)
        print ('normal: ', cross)

    if normalise:
        cross /= length(cross)

        cross = -cross
    return cross


def points_2_nodes_id(mesh, points):
    Xn = mesh.get_nodes(mesh.nodes.ids)
    tree = cKDTree(Xn)
    d, node_group = tree.query(points)
    return scipy.array(mesh.nodes.ids)[node_group]

def evaluate_points_on_edge(fig, mesh, edge=1, xi=[None,0.0], element_ids=[], direction=1., num_points=4, z_trim=None, debug=False):
    '''
    Evaluate points along an edge for the specified element ids. If no 
    ids are specified then point are evaluated in all elements. 

    The edge is specfied as the array [xi-direction, line-number]. In 3D, the 
    max line-number for a hexahedral element is 4 
    (following standard x1,x2,x3 numbering).

    Assumes all elements have the same basis.
    '''
    # if elem group is empty evaluate all elements
    if not isinstance(element_ids, list):
        element_ids = element_ids.tolist()
    if not element_ids:
        element_ids = mesh.elements.ids

    num_Xe = len(element_ids)
    if numpy.isclose(direction,1.):
        generic_xi = scipy.linspace(0., 1., num_points)
    elif numpy.isclose(direction,-1.):
        generic_xi = 1.0 - scipy.linspace(0., 1., num_points)
    xi_dim = mesh.elements[0].dimensions
    if xi_dim is 2:
        if edge == 1: # specify points along Xi1
            xi1 = generic_xi
            xi2 = xi[1]
        elif edge == 2: # specify points along Xi2
            xi1 = xi[0]
            xi2 = generic_xi
        X, Y = scipy.meshgrid(xi1, xi2)
        Xi = scipy.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    else:
        raise ValueError('Only 2D elements supported')
    
    total_num_points = num_Xe*num_points
    points = scipy.zeros((num_Xe, num_points, 3))
    xi = scipy.zeros((num_Xe, num_points, xi_dim))
    elements = scipy.zeros((num_Xe, num_points, 1))
    
    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx,:,:] = element.evaluate(Xi)
        xi[idx,:,:] = Xi
        elements[idx,:] = element_id

    points = scipy.reshape(points, (total_num_points,3))
    xi = scipy.reshape(xi, (total_num_points,xi_dim))
    elements = scipy.reshape(elements, (total_num_points))
    if z_trim is not None:
        valid_idxs = scipy.where(points[:,-1]>=z_trim)[0]
        points = points[valid_idxs,:]
        xi = xi[valid_idxs,:]
        elements = elements[valid_idxs]

    results = {'points'  : points,
               'xi'      : xi,
               'elements': elements}

    if debug:
        fig.plot_points(
            'points_on_edge', results['points'],
            color=(0,0,1), size=2)

    return results

def generate_points_on_face(mesh, constant_xi, value, element_ids=[],
                            num_points=4, num_points_2 = 0):
    """
    Generate a grid of points within each element
    
    Keyword arguments:
    mesh -- mesh to evaluate points in
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions    
    """
    if num_points_2 == 0:
        num_points_2 = num_points


    if constant_xi == "xi1":
        xi1 = [value]
        xi2 = scipy.linspace(0., 1., num_points)
        xi3 = scipy.linspace(0., 1., num_points_2)
    elif constant_xi == "xi2":
        xi1 = scipy.linspace(0., 1., num_points)
        xi2 = [value]
        xi3 = scipy.linspace(0., 1., num_points_2)
    elif constant_xi == "xi3":
        xi1 = scipy.linspace(0., 1., num_points)
        xi2 = scipy.linspace(0., 1., num_points_2)
        xi3 = [value]
    X, Y, Z = scipy.meshgrid(xi1, xi2, xi3)
    XiNd = scipy.array([
        X.reshape((X.size)),
        Y.reshape((Y.size)),
        Z.reshape((Z.size))]).T

    if len(element_ids) ==0:
        # if elem group is empty evaluate all elements
        element_ids = mesh.get_element_ids()

    num_Xe = len(element_ids)
    total_num_points = num_Xe*num_points*num_points_2
    points = scipy.zeros((num_Xe, num_points*num_points_2, 3))
    xi = scipy.zeros((num_Xe,  num_points * num_points_2, 3))
    elements = scipy.zeros((num_Xe, num_points * num_points_2, 1))
    
    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(XiNd)
        xi[idx, :, :] = XiNd
        elements[idx, :] = element_id

    points = scipy.reshape(points, (total_num_points, 3))
    xi = scipy.reshape(xi, (total_num_points, 3))
    elements = scipy.reshape(elements, (total_num_points))

    return points, xi, elements


def generate_points_on_line(mesh, face, value, axis, axis_value,num_points, element_ids=[]
                            ):
    """
    Generate a grid of points within each element

    Keyword arguments:
    mesh -- mesh to evaluate points in
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions
    """


    if face == "xi1":
        if axis == "xi2":
            xi1 = [value]
            xi2 = [axis_value]
            xi3 = scipy.linspace(0., 1., num_points)
        if axis == "xi3":
            xi1 = [value]
            xi2 = scipy.linspace(0., 1., num_points)
            xi3 = [axis_value]


    elif face == "xi2":
        if axis == "xi1":
            xi1 = [axis_value]
            xi2 = [value]
            xi3 = scipy.linspace(0., 1., num_points)
        if axis == "xi3":
            xi1 = scipy.linspace(0., 1., num_points)
            xi2 = [value]
            xi3 =  [axis_value]
    elif face == "xi3":
        if axis == "xi1":
            xi1 = [axis_value]
            xi2 = scipy.linspace(0., 1., num_points)
            xi3 = [value]
        if axis == "xi2":
            xi1 = scipy.linspace(0., 1., num_points)
            xi2 = [axis_value]
            xi3 = [value]

    X, Y, Z = scipy.meshgrid(xi1, xi2, xi3)
    XiNd = scipy.array([
        X.reshape((X.size)),
        Y.reshape((Y.size)),
        Z.reshape((Z.size))]).T

    if not element_ids:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_Xe = len(element_ids)
    total_num_points = num_Xe * num_points
    points = scipy.zeros((num_Xe, num_points, 3))
    xi = scipy.zeros((num_Xe, num_points, 3))
    elements = scipy.zeros((num_Xe, num_points, 1))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(XiNd)
        xi[idx, :, :] = XiNd
        elements[idx, :] = element_id

    points = scipy.reshape(points, (total_num_points, 3))
    xi = scipy.reshape(xi, (total_num_points, 3))
    elements = scipy.reshape(elements, (total_num_points))

    return points, xi, elements



def generate_xi_grid_fem(num_points=4, y_num_points=0, z_num_points=0, dim=3):
    # Generate a grid of points within each element
    if y_num_points ==0:
        y_num_point = num_points
    if z_num_points == 0:
        z_num_points = num_points
    xi1 = scipy.linspace(0., 1., num_points)
    xi2 = scipy.linspace(0., 1., y_num_points)
    if dim == 2:
        X, Y = scipy.meshgrid(xi1, xi2)
        XiNd = scipy.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    else:
        xi3 = scipy.linspace(0., 1., z_num_points)
        X, Y, Z = scipy.meshgrid(xi1, xi2, xi3)
        XiNd = scipy.array([
            Z.reshape((Z.size)),
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    return XiNd

def generate_xi_grid_regular(num_points=4, y_num_points=0 , z_num_points=0,dim = 3):
    # Note that these points do not follow standard element node xi numbering which is Z,X,Y'
    if y_num_points ==0:
        y_num_point = num_points
    if z_num_points == 0:
        z_num_points = num_points
    xi1 = scipy.linspace(0., 1., num_points)
    xi2 = scipy.linspace(0., 1., y_num_points)
    if dim == 2:
        X, Y = scipy.meshgrid(xi1, xi2)
        XiNd = scipy.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    else:
        xi3 = scipy.linspace(0., 1., z_num_points)
        X, Y, Z = scipy.meshgrid(xi1, xi2, xi3)
        XiNd = scipy.array([
            X.reshape((X.size)),
            Y.reshape((Y.size)),
            Z.reshape((Z.size))]).T
    return XiNd

def generate_points_in_elements(mesh, element_ids=[], num_points=4, y_num_points=0,z_num_points=0, fem_numbering=False):

    if y_num_points == 0:
        y_num_points =num_points
    if z_num_points == 0:
        z_num_points = num_points
    if fem_numbering:
        XiNd = generate_xi_grid_fem(num_points=num_points, y_num_points=y_num_points,z_num_points =z_num_points)
    else:
        XiNd = generate_xi_grid_regular(num_points=num_points, y_num_points = y_num_points, z_num_points =z_num_points)

    if element_ids ==[]:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_Xe = len(element_ids)
    total_num_points = num_Xe*num_points*y_num_points*z_num_points
    points = scipy.zeros((num_Xe, num_points*y_num_points*z_num_points, 3))
    xi = scipy.zeros((num_Xe, num_points*y_num_points*z_num_points, 3))
    elements = scipy.zeros((num_Xe, num_points*y_num_points*z_num_points, 1))
    
    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx,:,:] = element.evaluate(XiNd)
        xi[idx,:,:] = XiNd
        elements[idx,:] = element_id

    points = scipy.reshape(points, (total_num_points,3))
    xi = scipy.reshape(xi, (total_num_points,3))
    elements = scipy.reshape(elements, (total_num_points))

    return points, xi, elements


def generate_points_on_surface(mesh, element_ids=[], num_points= 4, num_points2=0,):
    if num_points2 == 0:
        num_points2 = num_points


    XiNd = generate_xi_grid_regular(num_points=num_points, y_num_points=num_points2, dim = 2)

    if element_ids == []:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_Xe = len(element_ids)
    total_num_points = num_Xe * num_points * num_points2
    points = scipy.zeros((num_Xe, num_points * num_points2 ,3))
    xi = scipy.zeros((num_Xe, num_points * num_points2 , 2))
    elements = scipy.zeros((num_Xe, num_points * num_points2 , 1))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :,:] = element.evaluate(XiNd)
        xi[idx, :] = XiNd
        elements[idx] = element_id

    points = scipy.reshape(points, (total_num_points, 3))
    xi = scipy.reshape(xi, (total_num_points,2))
    elements = scipy.reshape(elements, (total_num_points))

    return points, xi, elements

def gen_element_surface_points(mesh, element_ids=[], num_points=100,fig=None):

    cranial_pts = generate_points_on_face(
        mesh, "xi2", 0, element_ids=element_ids, num_points=num_points)[0]
    caudal_pts = generate_points_on_face(
        mesh, "xi2", 1, element_ids=element_ids, num_points=num_points)[0]
    sternum_pts =generate_points_on_face(
        mesh, "xi1", 0, element_ids=element_ids, num_points=num_points)[0]
    spine_pts = generate_points_on_face(
        mesh, "xi1", 1, element_ids=element_ids, num_points=num_points)[0]
    chestwall_pts = generate_points_on_face(
        mesh, "xi3", 0, element_ids=element_ids, num_points=num_points)[0]
    skin_pts = generate_points_on_face(
        mesh, "xi3", 1, element_ids=element_ids, num_points=num_points)[0]
    surface_pts = scipy.vstack([cranial_pts, caudal_pts, sternum_pts,
        spine_pts, chestwall_pts, skin_pts])
    if fig != None:
        fig.plot_points(
            'surface_pts', surface_pts,
            color=(0,1,0), size=2)

    return surface_pts

