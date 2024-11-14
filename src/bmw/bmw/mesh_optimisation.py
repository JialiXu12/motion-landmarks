import scipy
from scipy import interpolate
from scipy.spatial import cKDTree
import numpy
from bmw import visualisation
from bmw import utils
from bmw import mesh_generation


def calculate_Q(mesh, Xe, para):
    #import ipdb; ipdb.set_trace()
    objfun_type = 'jacobian_eigen_value_ratio'
    if objfun_type == 'element_jacobian_ratio':
        alpha = 1.
        offset = 0.
        ng_xi = mesh._core.get_gauss_points([4, 4, 4])[0]
        num_ng = ng_xi.shape[0]
        Q_vec = scipy.zeros((len(Xe)))
        for element_idx, element in enumerate(mesh.elements[Xe]):
            Xedxdnu = scipy.zeros((num_ng, 3, 3))
            Xedxdnu[:,:,0] = element.evaluate(ng_xi,deriv=[1,0,0])
            Xedxdnu[:,:,1] = element.evaluate(ng_xi,deriv=[0,1,0])
            Xedxdnu[:,:,2] = element.evaluate(ng_xi,deriv=[0,0,1])
            XeJ = scipy.zeros((num_ng))
            for ng in range(num_ng):
                 XeJ[ng] = scipy.linalg.det(Xedxdnu[ng,:,:])
            Q_vec[element_idx] = (min(XeJ)/max(XeJ))

    if objfun_type == 'points_jacobian_ratio':
        alpha = 1.
        offset = 1.
        xi = mesh_generation.generate_xi_grid(num_points=4)
        num_xi = xi.shape[0]
        num_elements = len(Xe)
        Q_vec = scipy.zeros((num_xi*num_elements))
        global_xi_idx = 0
        for element_idx, element in enumerate(mesh.elements[Xe]):
            Xedxdnu = scipy.zeros((num_xi,3, 3))
            Xedxdnu[:,:,0] = element.evaluate(xi,deriv=[1,0,0])
            Xedxdnu[:,:,1] = element.evaluate(xi,deriv=[0,1,0])
            Xedxdnu[:,:,2] = element.evaluate(xi,deriv=[0,0,1])
            for local_xi_idx in range(num_xi):
                det = numpy.linalg.det(Xedxdnu[local_xi_idx,:,:])
                Q_vec[global_xi_idx] = det
                global_xi_idx += 1

    elif objfun_type == 'jacobian_eigen_value_ratio':
        alpha = 1.
        offset = 1.
        xi = mesh_generation.generate_xi_grid(num_points=4)
        num_xi = xi.shape[0]
        num_elements = len(Xe)
        Q_vec = scipy.zeros((num_xi*num_elements))
        global_xi_idx = 0
        for element_idx, element in enumerate(mesh.elements[Xe]):
            Xedxdnu = scipy.zeros((num_xi,3, 3))
            Xedxdnu[:,:,0] = element.evaluate(xi,deriv=[1,0,0])
            Xedxdnu[:,:,1] = element.evaluate(xi,deriv=[0,1,0])
            Xedxdnu[:,:,2] = element.evaluate(xi,deriv=[0,0,1])
            for local_xi_idx in range(num_xi):
                eig = numpy.linalg.svd(Xedxdnu[local_xi_idx,:,:], compute_uv=False)
                #eig = numpy.linalg.eig(Xedxdnu[local_xi_idx,:,:])[0]
                Q_vec[global_xi_idx] = min(eig)/max(eig)
                global_xi_idx += 1


    Q_mean = scipy.mean(Q_vec)
    Q_std = numpy.std(Q_vec, ddof=1)
    f = alpha*sum((Q_vec-offset)**2)
    print ('f: ', f,', Q_mean: ', Q_mean, 'Q_std: ', Q_std, ', Max para: ', max(para), ', Min para: ', min(para))
    #import ipdb; ipdb.set_trace()
    return f

def mesh_opti_internal_node_objfun(para, mesh, node_ids,original_Xn, metric_evaluate_element_ids):
    debug = False
    counter = 0

    for node_idx, node in enumerate(mesh.nodes[node_ids]):
        node.values = original_Xn[node_idx]
        #if debug:
        #    print 'Get node  : ', Xn[node_idx]
        #    print 'Index node: ', node.values
        for comp in [0,1,2]:
            node.values[comp] += para[counter]
            counter += 1
    Q = calculate_Q(mesh, metric_evaluate_element_ids, para)
    return Q

def optimise_internal_mesh_nodes(mesh, fig, node_ids, metric_evaluate_element_ids):

    #metric_evaluate_element_ids = [ 12,  13,  14,  15,  16,  23,  24,  25,  26,  27,  34,  35,  36,
    #        37,  38,  45,  46,  47,  48,  49,  78,  79,  80,  81,  82,  89,
    #        90,  91,  92,  93, 100, 101, 102, 103, 104, 111, 112, 113, 114, 115]
    #node_ids = [857,858,859,891,892,893,925,926,927,959,960,961,993,994,995,1027,1028,1029,1503,1504,1505,1537,1538,1539,1571,1572,1573,1605,1606,1607,1639,1640,1641,1673,1674,1675]

    #metric_evaluate_element_ids = [12, 13, 14,23, 24, 25,34,35,36,45,46,47]
    #node_ids = [ 857,  858,  859,  860,  861,  891,  892,  893,  894,  895,  925,
    #        926,  927,  928,  929,  959,  960,  961,  962,  963,  993,  994,
    #        995,  996,  997, 1027, 1028, 1029, 1030, 1031, 1061, 1062, 1063,
    #       1064, 1065, 1503, 1504, 1505, 1506, 1507, 1537, 1538, 1539, 1540,
    #       1541, 1571, 1572, 1573, 1574, 1575, 1605, 1606, 1607, 1608, 1609,
    #       1639, 1640, 1641, 1642, 1643, 1673, 1674, 1675, 1676, 1677, 1707,
    #       1708, 1709, 1710, 1711]

    #        nodes_of_interest = []
    #        for element_id in metric_evaluate_element_ids:
    #            nodes_of_interest += torso.elements[element_id].node_ids
    #        print scipy.array(nodes_of_interest)+1
    #        import ipdb; ipdb.set_trace()

    #        node_ids = [ 854,  855,  856,  857,  858,  859,  860,  861,  888,  889,  890,
    #            891,  892,  893,  894,  895,  922,  923,  924,  925,  926,  927,
    #            928,  929,  956,  957,  958,  959,  960,  961,  962,  963,  990,
    #            991,  992,  993,  994,  995,  996,  997, 1024, 1025, 1026, 1027,
    #           1028, 1029, 1030, 1031, 1058, 1059, 1060, 1061, 1062, 1063, 1064,
    #           1065, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1534, 1535,
    #           1536, 1537, 1538, 1539, 1540, 1541, 1568, 1569, 1570, 1571, 1572,
    #           1573, 1574, 1575, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609,
    #           1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1670, 1671, 1672,
    #           1673, 1674, 1675, 1676, 1677, 1704, 1705, 1706, 1707, 1708, 1709,
    #           1710, 1711]

#    node_ids = [ 786,  787,  788,  789,  790,  791,  792,  793,  820,  821,  822,
#            823,  824,  825,  826,  827,  854,  855,  856,  857,  858,  859,
#            860,  861,  888,  889,  890,  891,  892,  893,  894,  895,  922,
#            923,  924,  925,  926,  927,  928,  929,  956,  957,  958,  959,
#            960,  961,  962,  963,  990,  991,  992,  993,  994,  995,  996,
#            997, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1058, 1059,
#            1060, 1061, 1062, 1063, 1064, 1065, 1092, 1093, 1094, 1095, 1096,
#            1097, 1098, 1099, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133,
#            1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1466, 1467, 1468,
#            1469, 1470, 1471, 1472, 1473, 1500, 1501, 1502, 1503, 1504, 1505,
#            1506, 1507, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1568,
#            1569, 1570, 1571, 1572, 1573, 1574, 1575, 1602, 1603, 1604, 1605,
#            1606, 1607, 1608, 1609, 1636, 1637, 1638, 1639, 1640, 1641, 1642,
#            1643, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1704, 1705,
#            1706, 1707, 1708, 1709, 1710, 1711, 1738, 1739, 1740, 1741, 1742,
#            1743, 1744, 1745, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779]

    visualisation.visualise_mesh(mesh, fig, visualise=True, face_colours=(0,1,1),pt_size=1, opacity=0.25, line_opacity = 0.75, text=False, label='original', elements=metric_evaluate_element_ids, nodes=node_ids, node_colours=(1,0,0), node_size=3.)
    original_Xn = mesh.get_nodes(node_ids)
    para = scipy.zeros((len(node_ids)*3))
    print ('Number of parameters: ', len(para))

    mesh_opti_internal_node_objfun(para, mesh, node_ids, original_Xn, metric_evaluate_element_ids)
    #import ipdb; ipdb.set_trace()
    f = lambda para, mesh=mesh, node_ids=node_ids, original_Xn=original_Xn, metric_evaluate_element_ids=metric_evaluate_element_ids:(mesh_opti_internal_node_objfun(para, mesh, node_ids, original_Xn, metric_evaluate_element_ids))
    res = scipy.optimize.minimize(f, para, method='SLSQP', options={'eps':0.1})#method='COBYLA',options={'rhobeg':0.01}), bounds=[(-20.,20.)]*len(para)
    print (res.message)
    mesh_opti_internal_node_objfun(res.x, mesh, node_ids, original_Xn, metric_evaluate_element_ids)

    visualisation.visualise_mesh(mesh, fig, visualise=True, face_colours=(1,1,0),pt_size=1, opacity=0.25, line_opacity = 0.75, text=False, label='final', elements=metric_evaluate_element_ids, nodes=node_ids, node_colours=(0,1,0), node_size=3.)
    import ipdb; ipdb.set_trace()



































def evaluate_spline(fig, spline_c_pts, skin_node_idx, spline_order=2, smoothing=10, debug=False):
    # Define spline control points
    x = spline_c_pts[:,0]
    y = spline_c_pts[:,1]
    z = spline_c_pts[:,2]
    tck,u = interpolate.splprep([x,y,z], s=smoothing, k=spline_order)

    # Evaluate points on spline
    num_spline_pts = 100
    spline_xi = scipy.linspace(0, 1, num_spline_pts)
    xnew,ynew,znew= interpolate.splev(spline_xi, tck, der=0)
    spline_pts = scipy.array([xnew,ynew,znew]).T
    if debug:
        fig.plot_points('spline_pts_{0}'.format(skin_node_idx), spline_pts, color=(0,1,0), size=2)

    spline_ilens = mesh_generation.integrate_length(spline_pts)
    spline_len = spline_ilens.sum()
    spline_clens = scipy.zeros((num_spline_pts))
    for ln in range(1, len(spline_ilens)+1):
        spline_clens[ln] = spline_ilens[0:ln].sum()
    target_lns = scipy.array([[spline_len*1./3.], [spline_len*2./3.]])

    spline_clens_KDTree = cKDTree(scipy.array([spline_clens]).T)
    neighbours = 1
    closest_spline_xi_id = spline_clens_KDTree.query(target_lns, k=neighbours)[1]
    closest_spline_xi =  spline_xi[closest_spline_xi_id]
    closest_spline_pts = spline_pts[closest_spline_xi_id]
    if debug:
        fig.plot_points('new_node_pts1_{0}'.format(skin_node_idx), closest_spline_pts[0], color=(1,1,0), size=5)
        fig.plot_points('new_node_pts2_{0}'.format(skin_node_idx), closest_spline_pts[1], color=(1,1,0), size=5)

    return closest_spline_pts


def optimise_spline_pts(mesh, fig):
    #skin_nodes = [2219,2253,2287]
    #skin_nodes = [2078,2079,2080,2081,2082,2083,2084,2085,2112,2113,2114,2115,2116,2117,2118,2119,2146,2147,2148,2149,2150,2151,2152,2153,2180,2181,2182,2183,2184,2185,2186,2187,2214,2215,2216,2217,2218,2219,2220,2221,2248,2249,2250,2251,2252,2253,2254,2255,2282,2283,2284,2285,2286,2287,2288,2289,2316,2317,2318,2319,2320,2321,2322,2323,2350,2351,2352,2353,2354,2355,2356,2357,2384,2385,2386,2387,2388,2389,2390,2391,2418,2419,2420,2421,2422,2423,2424,2425]
    #skin_nodes = [2078,2079,2080,2081,2082,2112,2113,2114,2115,2116,2146,2147,2148,2149,2150,2180,2181,2182,2183,2184,2214,2215,2216,2217,2218,2248,2249,2250,2251,2252,2282,2283,2284,2285,2286,2316,2317,2318,2319,2320,2350,2351,2352,2353,2354,2384,2385,2386,2387,2388,2418,2419,2420,2421,2422]
    skin_nodes = [2219]
    spline_nodes = scipy.zeros((len(skin_nodes), 4), dtype=int)
    for node_idx, skin_node in enumerate(skin_nodes):
        for layer in range(4):
            spline_nodes[node_idx,layer] = skin_node-(646*layer)

    # Find element and xi associated with each skin node
    skin_node_element_id = scipy.zeros((len(skin_nodes)), dtype=int)
    skin_node_element_xi = scipy.zeros((len(skin_nodes),3))
    xi_grid = mesh_generation.generate_xi_grid(num_points=4)

    for skin_node_idx, skin_node in enumerate(skin_nodes):
        finished = False
        while not finished:
            for element in mesh.elements:
                if skin_node in element.node_ids:
                    skin_node_element_id[skin_node_idx] = element.id
                    element_node_idx = element.node_ids.index(skin_node)
                    skin_node_element_xi[skin_node_idx,:] = (
                        xi_grid[element_node_idx,:])
                    finished = True
                    break

    debug = False
    # Define a new set (layer) of pts normal to the skin surface
    surface_normal_layer = scipy.zeros((len(skin_nodes),3))
    for skin_node_idx, skin_node in enumerate(skin_nodes):
        element_id = skin_node_element_id[skin_node_idx]
        element = mesh.elements[element_id]

        xi = skin_node_element_xi[skin_node_idx]
        pt0 = element.evaluate(xi)
        mesh.get_nodes(skin_node)
        dx1 = element.evaluate(xi,deriv=[1,0,0])
        dx2 = element.evaluate(xi,deriv=[0,1,0])
        surface_normal = mesh_generation.normalise(scipy.cross(dx1, dx2))
        pt1 = pt0 - surface_normal*10.
        pts = scipy.vstack((pt0, pt1))
        if debug:
            fig.plot_points(fig, 'skin_normal_node_{0}'.format(skin_node), pts, ['pt0', 'pt1'], visualise=True, colours=(1,0,0), point_size=10, text_size=5)
        surface_normal_layer[skin_node_idx,:] = pt1

    original_Xn = scipy.zeros((len(skin_nodes),2,3))
    for skin_node_idx, skin_node in enumerate(skin_nodes):
        original_Xn[skin_node_idx,:,:] = scipy.vstack((
            mesh.get_nodes(spline_nodes[skin_node_idx,0])[0], # skin node
            mesh.get_nodes(spline_nodes[skin_node_idx,-1])[0])) # rib node

    para = scipy.zeros((len(skin_nodes)*3))
    metric_evaluate_element_ids = [12,13,14,23,24,25,34,35,36,45,46,47]
    #metric_evaluate_element_ids = [12,13,23,24,34,35,45,46]

    mesh_opti_spline_pts_objfun(para, mesh, fig, skin_nodes, spline_nodes, original_Xn, surface_normal_layer, metric_evaluate_element_ids, debug=True)
    visualisation.visualise_mesh(mesh, fig, visualise=True, face_colours=(1,1,0),pt_size=1, opacity=0.25, line_opacity = 0.75, text=False, label='initial', elements=metric_evaluate_element_ids, node_colours=(1,0,0), node_size=3)

    import ipdb; ipdb.set_trace()

    f = lambda para, mesh=mesh, fig=fig, skin_nodes=skin_nodes, \
        spline_nodes=spline_nodes, original_Xn=original_Xn, \
        surface_normal_layer=surface_normal_layer, \
        metric_evaluate_element_ids=metric_evaluate_element_ids :(
            mesh_opti_spline_pts_objfun(para, mesh, fig, skin_nodes, spline_nodes, original_Xn, surface_normal_layer, metric_evaluate_element_ids))

    f = lambda para, mesh=mesh, fig=fig, skin_nodes=skin_nodes, spline_nodes=spline_nodes, original_Xn=original_Xn, surface_normal_layer=surface_normal_layer, metric_evaluate_element_ids=metric_evaluate_element_ids :(mesh_opti_spline_pts_objfun(para, mesh, fig, skin_nodes, spline_nodes, original_Xn, surface_normal_layer, metric_evaluate_element_ids))

    res = scipy.optimize.minimize(f, para, bounds=[(-3.,3.)]*len(para), method='Anneal')#,options={'rhobeg':1.})

    mesh_opti_spline_pts_objfun(res.x, mesh, fig, skin_nodes, spline_nodes, original_Xn, surface_normal_layer, metric_evaluate_element_ids, debug=True)
    #visualisation.visualise_mesh(mesh, fig, visualise=True, face_colours=(1,1,0),pt_size=1, opacity=0.25, line_opacity = 0.75, text=False, label='final', elements=metric_evaluate_element_ids, node_colours=(1,0,0), node_size=3)

    #aaa = scipy.array(bmw.CMISS_int_string_to_array("2079..2086,2113..2120,2147..2154,2181..2188,2215..2222,2249..2256,2283..2290,2317..2324,2351..2358,2385..2392,2419..2426"))-1

def mesh_opti_spline_pts_objfun(para, mesh, fig, skin_nodes, spline_nodes, original_Xn, surface_normal_layer, metric_evaluate_element_ids, debug=False):
    #import ipdb; ipdb.set_trace()
    counter = 0
    for skin_node_idx, skin_node in enumerate(skin_nodes):
        # Reset optimisation parameters
        mesh.nodes[spline_nodes[skin_node_idx,1]].values = original_Xn[skin_node_idx,0,:]
        mesh.nodes[spline_nodes[skin_node_idx,2]].values = original_Xn[skin_node_idx,1,:]

        updated_surface_normal_layer = scipy.zeros((3))
        for component in [0,1,2]:
            updated_surface_normal_layer[component] = surface_normal_layer[skin_node_idx,component] + para[counter]
            counter += 1

        spline_c_pts = scipy.vstack((
            mesh.get_nodes(spline_nodes[skin_node_idx,0])[0], # skin node
            updated_surface_normal_layer, # New node
            mesh.get_nodes(spline_nodes[skin_node_idx,-1])[0])) # rib node
        if debug:
            fig.plot_points('spline_c_pts_{0}'.format(skin_node_idx), spline_c_pts, color=(0,0,1), size=11)

        closest_spline_pts = evaluate_spline(fig, spline_c_pts, skin_node_idx, spline_order=2, debug=debug)

        mesh.nodes[spline_nodes[skin_node_idx,1]].values = closest_spline_pts[0]
        mesh.nodes[spline_nodes[skin_node_idx,2]].values = closest_spline_pts[1]

    # Evaluate Jacobian metric
    Q = calculate_element_jacobian2(mesh, metric_evaluate_element_ids)
    Q_mean = scipy.mean(Q)
    Q_std = numpy.std(Q, ddof=1)
    print ('Q_mean: ', Q_mean, 'Q_std: ', Q_std, ', Max para: ', max(para), ', Min para: ', min(para))
    return -Q_mean



def calculate_element_jacobian2(mesh, Xe):
    #import ipdb; ipdb.set_trace()


    objfun_type = 'points_jacobian_ratio'
    if objfun_type == 'element_jacobian_ratio':
        ng_xi = mesh._core.get_gauss_points([4, 4, 4])[0]
        num_ng = ng_xi.shape[0]
        Q = scipy.zeros((len(Xe)))
        for element_idx, element in enumerate(mesh.elements[Xe]):
            Xedxdnu = scipy.zeros((num_ng, 3, 3))
            Xedxdnu[:,:,0] = element.evaluate(ng_xi,deriv=[1,0,0])
            Xedxdnu[:,:,1] = element.evaluate(ng_xi,deriv=[0,1,0])
            Xedxdnu[:,:,2] = element.evaluate(ng_xi,deriv=[0,0,1])
            XeJ = scipy.zeros((num_ng))
            for ng in range(num_ng):
                 XeJ[ng] = scipy.linalg.det(Xedxdnu[ng,:,:])
            Q[element_idx] = (min(XeJ)/max(XeJ))**2.

    if objfun_type == 'points_jacobian_ratio':
        xi = mesh_generation.generate_xi_grid(num_points=4)
        num_xi = xi.shape[0]
        num_elements = len(Xe)
        Q = scipy.zeros((num_xi*num_elements))
        global_xi_idx = 0
        for element_idx, element in enumerate(mesh.elements[Xe]):
            for local_xi_idx in range(num_xi):
                Xedxdnu = scipy.zeros((3, 3))
                Xedxdnu[:,0] = element.evaluate(xi[local_xi_idx,:],deriv=[1,0,0])
                Xedxdnu[:,1] = element.evaluate(xi[local_xi_idx,:],deriv=[0,1,0])
                Xedxdnu[:,2] = element.evaluate(xi[local_xi_idx,:],deriv=[0,0,1])
                det = numpy.linalg.det(Xedxdnu)
                Q[global_xi_idx] = -det**2.
                global_xi_idx += 1

    elif objfun_type == 'jacobian_eigen_value_ratio':
        xi = mesh_generation.generate_xi_grid(num_points=4)
        num_xi = xi.shape[0]
        num_elements = len(Xe)
        Q = scipy.zeros((num_xi*num_elements))
        global_xi_idx = 0
        for element_idx, element in enumerate(mesh.elements[Xe]):
            for local_xi_idx in range(num_xi):
                Xedxdnu = scipy.zeros((3, 3))
                Xedxdnu[:,0] = element.evaluate(xi[local_xi_idx,:],deriv=[1,0,0])
                Xedxdnu[:,1] = element.evaluate(xi[local_xi_idx,:],deriv=[0,1,0])
                Xedxdnu[:,2] = element.evaluate(xi[local_xi_idx,:],deriv=[0,0,1])
                eig = numpy.linalg.eig(Xedxdnu)[0]
                Q[global_xi_idx] = (min(eig)/max(eig))**2.
                global_xi_idx += 1

    #import ipdb; ipdb.set_trace()

    return Q
#    Xedxdnu[:,0] = mesh.elements[0].evaluate([0.5,0.5,0.5],deriv=[1,0,0])
#    Xedxdnu[:,1] = mesh.elements[0].evaluate([0.5,0.5,0.5],deriv=[0,1,0])
#    Xedxdnu[:,2] = mesh.elements[0].evaluate([0.5,0.5,0.5],deriv=[0,0,1])

