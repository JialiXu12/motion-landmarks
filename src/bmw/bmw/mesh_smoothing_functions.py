def align_nodes_objfun(xi2,element,xi1,z):
    return abs(element.evaluate([xi1,xi2])[2]-z)

def align_nodes(mesh, fig, element_id, torso_anterior=True, debug=False):

    '''
    bmw.align_nodes(new_bm_rhs, fig, debug=True)
    '''
    nl = 4 # Number of nodes.
    nnl = 4 # Number nodes per line.
    aligned_nodes = scipy.zeros((nl, nnl, 3))
    element = mesh.elements[element_id]
    elem_line_nodes = scipy.reshape(element.node_ids,(nl, nnl))
    # Set the target node whose z position along a xi1 line needs to be aligned.
    if torso_anterior:
        target_node_idx = 3
        alignment_node_idxs = [0,1,2]
    else: # Torso_posterior
        target_node_idx = 0
        alignment_node_idxs = [1,2,3]
    for line in range(0,nl):
        target_z = mesh.get_nodes(elem_line_nodes[line,target_node_idx])[0][2]
        for node_idx, node in enumerate(elem_line_nodes[line,:]):
            if node_idx in alignment_node_idxs:
                #if torso_anterior:
                xi1=(1./3.)*node_idx
                #else: # Torso_posterior
                    #xi1=1.-((1./3.)*node_idx)
                f = lambda xi2, element=element, xi1=xi1, z=target_z:(
                        align_nodes_objfun(xi2, element, xi1, z))
                # Determine xi2 that gives produces the target_z value.
                xi2, nfeval, rc = scipy.optimize.fmin_tnc(
                        f, 0.5, bounds=[(0.,1.)], approx_grad=True)
                aligned_nodes[line, node_idx,:] = (
                        element.evaluate([xi1,xi2]))
            else: 
                # Remaining node along xi1 line (used for setting target Z,
                # so we already know it coordinates.
                aligned_nodes[line,node_idx,:] = mesh.get_nodes(node)
    if debug and fig is not None:
        aligned_node_coor = numpy.reshape(aligned_nodes,(nl*nnl,3))
        fig.plot_points(
            '{0}_aligned_nodes'.format(mesh.label), aligned_node_coor,
            color=(0,1,0), size=2)
        #import ipdb; ipdb.set_trace()
    return aligned_nodes


def smooth_mid_shoulder(mesh, fig, smoothing=0, debug=False):
    '''
    bmw.smooth_mid_shoulder(new_bm_lhs, fig, debug=True)
    bmw.smooth_mid_shoulder(new_bm_rhs, fig, debug=False)
    '''
    spline = False
    if spline: # Smooth using a spline.
        spline_fit_node_ids = [21,55,123,225]
        spline_fit_nodes = mesh.get_nodes(spline_fit_node_ids)
        spline_eval_node_ids = [191,157,89]
        spline_eval_nodes = mesh.get_nodes(spline_eval_node_ids)
        x = spline_fit_nodes[:,0]
        y = spline_fit_nodes[:,1]
        z = spline_fit_nodes[:,2]
        num_spline_pts = 1000
        spline_xi = scipy.linspace(0, 1, num_spline_pts)
        tck,u = interpolate.splprep([x,y,z] ,s = smoothing, k = 3)
        xnew,ynew,znew= interpolate.splev(spline_xi, tck, der = 0)
        spline_pts = scipy.array([xnew,ynew,znew]).T

        spline_pts_KDTree = cKDTree(spline_pts)
        spline_eval_pts_idxs = spline_pts_KDTree.query(spline_eval_nodes, k=1)[1]

        for node_idx, node_id in enumerate(spline_eval_node_ids):
            mesh.nodes[node_id].values = spline_pts[spline_eval_pts_idxs[node_idx],:]

        if debug and fig is not None:
            fig.plot_points(
                '{0}_mid_spline_pts'.format(mesh.label), spline_pts,
                color=(0,1,0), size=1)
            fig.plot_points(
                '{0}_mid_spline_eval_pts'.format(mesh.label), spline_pts[spline_eval_pts_idxs,:],
                color=(1,0,0), size=2)

    else:
        num_points = 4 # Number of cubic Lagrange nodes per xi
        xi1 = scipy.linspace(0., 1., num_points)
        xi2 = scipy.linspace(0., 1., num_points)
        node_xi1, node_xi2 = scipy.meshgrid(xi1, xi2)
        node_xi = scipy.array([
            node_xi1.reshape((node_xi1.size)),
            node_xi2.reshape((node_xi2.size))]).T
        xi = [0.5,0.5]
        normal = surface_normal(mesh.elements[0], xi)
        pt_o = mesh.elements[0].evaluate(xi)
        if debug and fig is not None:
            fig.plot_vector(pt_o, normal, scale=2, size=3)
        target_Xes = [6, 7, 17, 18]
        target_XeXns = ([
            [122, 123],
            [124, 123],
            [122, 123, 157, 191],
            [124, 123, 157, 191]])
        for target_Xe_idx, target_Xe in enumerate(target_Xes):
            for target_XeXn in target_XeXns[target_Xe_idx]:
                pass
            
        control_nodes = [190, 191, 192]
        #import ipdb; ipdb.set_trace()

def smooth_shoulder_region(mesh, fig, smoothing=0, debug=False):
    '''
    new_bm_lhs = bmw.smooth_shoulder_region(new_bm_lhs, fig, smoothing=0, debug=False)
    new_bm_rhs = bmw.smooth_shoulder_region(new_bm_rhs, fig, smoothing=0, debug=False)
    '''
    shoulder_spline_nodes = numpy.array([rangei_add(15,12),
        rangei_add(49,12),
        rangei_add(83,12),
        rangei_add(117,12),
        rangei_add(151,12),
        rangei_add(185,12)])
    spline_fit_node_idxs = numpy.array([0,1,2,3,9,10,11,12])
    spline_eval_node_idxs = numpy.array([4,5,6,7,8])

    node_alignment = True
    if node_alignment:
        for element_id in [5,16]:
            aligned_nodes = align_nodes(mesh, fig, element_id=element_id, torso_anterior=True, debug=True)
            aligned_node_coor = numpy.reshape(aligned_nodes, (16,3))
            for node_idx, node_id in enumerate(mesh.elements[element_id].node_ids):
                mesh.nodes[node_id].values = aligned_node_coor[node_idx,:]
        for element_id in [8,19]:
            aligned_nodes = align_nodes(mesh, fig, element_id=element_id, torso_anterior=False, debug=True)
            aligned_node_coor = numpy.reshape(aligned_nodes, (16,3))
            for node_idx, node_id in enumerate(mesh.elements[element_id].node_ids):
                mesh.nodes[node_id].values = aligned_node_coor[node_idx,:]

    num_lines = shoulder_spline_nodes.shape[0]
    num_spline_pts = 1000
    spline_xi = scipy.linspace(0, 1, num_spline_pts)
    for line in range(num_lines):
        nodes = mesh.get_nodes(shoulder_spline_nodes[line,spline_fit_node_idxs].tolist())
        x = nodes[:,0]
        y = nodes[:,1]
        z = nodes[:,2]
        if debug and fig is not None:
            fig.plot_points(
                '{0}_spline_line{1}_nodes'.format(mesh,line), nodes,
                color=(0,0,1), size=2)

        tck,u = interpolate.splprep([x,y,z] ,s = smoothing, k = 3)
        xnew,ynew,znew= interpolate.splev(spline_xi, tck, der = 0)
        spline_pts = scipy.array([xnew,ynew,znew]).T
        if debug and fig is not None:
            fig.plot_points(
                '{0}_spline_line{1}_pts'.format(mesh,line), spline_pts,
                color=(0,1,0), size=1)

        spline_pts_KDTree = cKDTree(spline_pts)
        spline_eval_bnds = spline_pts_KDTree.query(mesh.get_nodes(
                shoulder_spline_nodes[line,spline_fit_node_idxs[[3,4]]].tolist()), k=1)[1]
        spline_new_Xn_xi = scipy.linspace(
            spline_xi[spline_eval_bnds[0]], spline_xi[spline_eval_bnds[1]],
            spline_eval_node_idxs.shape[0]+1, endpoint=False)
        xnew,ynew,znew= interpolate.splev(spline_new_Xn_xi, tck, der = 0)
        spline_eval_pts = scipy.array([xnew[1:],ynew[1:],znew[1:]]).T

        if debug and fig is not None:
            fig.plot_points(
                '{0}_spline_line{1}_eval_pts'.format(mesh.label,line), spline_eval_pts,
                color=(1,0,0), size=5)

        #import ipdb; ipdb.set_trace()
        for idx, node_num in enumerate(shoulder_spline_nodes[line,spline_eval_node_idxs]):
            if line in [4,5]:
                mesh.nodes[node_num].values[0:2] = spline_eval_pts[idx,0:2]
            else:
                mesh.nodes[node_num].values[:] = spline_eval_pts[idx,:]

    #import ipdb; ipdb.set_trace()

    return mesh
