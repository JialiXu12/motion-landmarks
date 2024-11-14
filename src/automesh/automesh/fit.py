import time
import numpy as np
from scipy.spatial import cKDTree
import scipy.optimize


def distance(x0, x1):
    dx = x1 - x0
    return np.sqrt(np.sum(dx * dx, axis=0))


def fit_mesh(meshes, data, fits, dofs='all', ftol=1e-6, xtol=1e-6, maxiter=0, dt=None, output=False, fit_id='default'):

    def get_node_pids(mesh, nid, params):
        node = mesh.nodes[nid]
        if params == 'all':
            return node.cids
        return [node.cids[i] for i in params]

    def init_dofs(meshes, config):
        dof = config['dofs']
        for idx, dof in enumerate(dofs):
            if 'extra' in dof.keys():
                pass
            elif 'data' in dof.keys():
                pass
            else:
                mesh = meshes[dof['mesh']]
                if 'nodes' in dof.keys():
                    if dof['nodes'] == 'all':
                        node_ids = mesh.nodes.keys()
                    else:
                        node_ids = dof['nodes']
                    if 'var' in dof.keys():
                        params = dof['var']
                        for nid in node_ids:
                            pids = get_node_pids(mesh, nid, params)
                            for pid in pids:
                                if pid not in mesh.core.variable_ids:
                                    mesh.core.variable_ids.append(pid)
                    elif 'fix' in dof.keys():
                        params = dof['fix']
                        for nid in node_ids:
                            pids = get_node_pids(mesh, nid, params)
                            for pid in pids:
                                if pid in mesh.core.variable_ids:
                                    mesh.core.variable_ids.remove(pid)
                    else:
                        print ('Warning: no DoF found for', idx, ':', dof)

        update_dofs = []
        num_vars = 0
        for mesh_id, mesh in meshes.iteritems():
            x = mesh.get_variables()
            if x.size > 0:
                update_dofs.append({'mesh': mesh_id, 'range': [num_vars, num_vars + x.size]})
                num_vars += x.size

        for dof in config['dofs']:
            if 'extra' in dof.keys():
                values = config['data'][dof['data']]['data']
                update_dofs.append(
                    {'extra': dof['extra'], 'range': [num_vars, num_vars + values.size], 'data': dof['data']})
                num_vars += values.size
            elif 'data' in dof.keys():
                values = config['data'][dof['data']]['data']
                update_dofs.append(
                    {'data': dof['data'], 'range': [num_vars, num_vars + values.size], 'values': dof['data']})
                num_vars += values.size

        config['fits'].insert(0, {'type': 'update_dofs', 'dofs': update_dofs})
        config['num_vars'] = num_vars

        return meshes, config

    def init_fits(config):
        for fit in config['fits']:
            if 'weight' not in fit.keys():
                fit['weight'] = 1.0
            if 'k' not in fit.keys():
                fit['k'] = 1
            if 'feval' not in fit.keys():
                fit['feval'] = False
            if 'out' not in fit.keys():
                fit['output_error_factor'] = 0
            elif 'out' in fit.keys():
                fit['output_error_factor'] = fit['out']
            if 'mesh' in fit.keys():
                if fit['mesh'] == 'all':
                    fit['mesh'] = meshes.keys()
                elif isinstance(fit['mesh'], str):
                    fit['mesh'] = [fit['mesh']]
        return config

    def process_data(config):
        for fit in config['fits']:
            if 'mesh' in fit.keys() and isinstance(fit['mesh'], str):
                fit['mesh'] = [fit['mesh']]

            if fit['type'] == 'closest_data':
                data = config['data'][fit['data']]
                if 'tree' not in data.keys():
                    data['tree'] = cKDTree(data['data'])
                fit['tree'] = data['tree']
            elif fit['type'] == 'closest_mesh':
                fit['_data'] = config['data'][fit['data']]['data']
            elif fit['type'] == 'vector_field_closest_mesh':
                fit['_data'] = config['data'][fit['data']]['data']
            elif fit['type'] == 'closest_field':
                fit['_data'] = config['data'][fit['data']]['data']
                data = config['data'][fit['data']]
                if 'tree' not in data.keys():
                    fit['tree'] = cKDTree(fit['_data'][:, :3])

            elif fit['type'] == 'constrain':
                if 'values' in fit.keys():
                    if fit['values'] == 'self':

                        for fit2 in config['fits']:
                            if fit2['type'] == 'update_pca':
                                for mesh_id in fit2['mesh']:
                                    mesh = meshes[mesh_id]
                                    update_pca_fit(fit2, mesh, config)

                        if 'deriv' not in fit.keys():
                            fit['deriv'] = None
                        values = []
                        Xg = fit['xi']
                        NPPE = Xg.shape[0]
                        NE = len(fit['elements'])
                        for mesh_num, mesh_id in enumerate(fit['mesh']):
                            mesh = meshes[mesh_id]
                            x = np.zeros((NE * NPPE, 3))
                            for i, eid in enumerate(fit['elements']):
                                x[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                            if fit['deriv'] is not None:
                                x = np.sqrt((x * x).sum(1))
                            values.append(x)
                        fit['values'] = values

                    elif not isinstance(fit['values'], list):
                        if 'params' in fit.keys():
                            fit['values'] = fit['values'] * np.ones((len(fit['params'])))

                        else:
                            values = []
                            for mesh_id in fit['mesh']:
                                values.append(fit['values'])
                            fit['values'] = values

                else:
                    fit['values'] = []
                    for mesh_id in fit['mesh']:
                        values.append(0.)

            elif fit['type'] == 'penalise_derivatives':
                if 'sample_values' in fit.keys():
                    if fit['sample_values'] == 'self':
                        values = []
                        Xg = fit['xi']
                        NPPE = Xg.shape[0]
                        NE = len(fit['elements'])
                        for mesh_id in fit['mesh']:
                            mesh = meshes[mesh_id]
                            derivs = np.zeros((NE * NPPE, 3))
                            for i, eid in enumerate(fit['elements']):
                                derivs[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                            values.append(derivs.flatten())
                        fit['sample_values'] = values

            elif fit['type'] == 'sobolev_smooth':
                if 'sample_values' in fit.keys():
                    if fit['sample_values'] == 'self':
                        values = []
                        Xg = fit['xi']
                        NPPE = Xg.shape[0]
                        NE = len(fit['elements'])
                        for mesh_id in fit['mesh']:
                            mesh = meshes[mesh_id]
                            derivs = np.zeros((NE * NPPE, 3))
                            for i, eid in enumerate(fit['elements']):
                                derivs[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                            derivs = np.sqrt((derivs * derivs).sum(1))
                            values.append(derivs.flatten())
                        fit['sample_values'] = values
        return config

    def generate_grids(meshes, config):
        # for fit in config['fits']:
        #     if 'elements' in fit.keys():
        #         elements = fit['elements']
        #         for mesh_id in fit['mesh']:
        #             mesh = meshes[mesh_id]
        # if elements == 'all':
        #     elements = mesh.elements.keys()
        #     fit['elements'] = elements
        # if 'xi' in fit.keys():
        #     Xg = fit['xi']
        # if 'deriv' in fit.keys():
        #     deriv = fit['deriv']
        # else:
        #     deriv = None
        # if 'sample_mesh' in fit.keys():
        #     sample_mesh = config['data'][fit['sample_mesh']]['mesh']
        #     D = []
        #     iii = 0
        #     for eid in elements:
        #         elem = sample_mesh.elements[eid]
        #         D.append(elem.evaluate(Xg, deriv=deriv).flatten())
        #         iii += 1
        #     fit['sample_values'] = np.array(D).flatten()

        return config

    def on_start(meshes, config):
        meshes, config = init_dofs(meshes, config)
        config = init_fits(config)
        config = process_data(config)
        config = generate_grids(meshes, config)
        config['iter'] = 0
        if config['dt'] is None:
            config['output'] = False
        else:
            config['output'] = True
            config['t0'] = time.time()
            config['output_time'] = 0
        return meshes, config

    def update_pca_fit(fit, mesh, config):
        mesh.update_pca_nodes()
        fitkeys = fit.keys()
        if 'translation_data' in fitkeys:
            dx_id = fit['translation_data']
            dx = config['data'][dx_id]['data']
            for node in mesh.nodes.groups[fit['group']]:
                if node.values.ndim == 2:
                    node.values[:, 0] += dx
                else:
                    node.values += dx
        elif 'translation_node' in fitkeys:
            dx_id = fit['translation_node']
            dx = mesh.nodes[dx_id].values
            for node in mesh.nodes.groups[fit['group']]:
                if node.values.ndim == 2:
                    node.values[:, 0] += dx
                else:
                    node.values += dx
        mesh._core.update_dependent_nodes()

    def measure_objfn(meshes, config, x0):
        err = objfn(x0, (meshes, config))
        return err.sum() / (err.shape[0])

    def objfn(x, args):
        meshes, config = args[0], args[1]
        # mesh.set_variables(x)
        # mesh._core.update_dependent_nodes()
        config['iter'] += 1
        Err = []
        for fit in config['fits']:
            if fit['type'] == 'update_dofs':  # this is automatically added
                meshes = set_variables(meshes, config, x)
                err = []

            elif fit['type'] == 'update_pids_from_data':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    values = config['data'][fit['data']]['data']
                    mesh._core.P[fit['pids']] = values

            elif fit['type'] == 'update_node':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    values = config['data'][fit['data']]['data']
                    mesh.nodes[fit['node']].values[:values.size] = values

            elif fit['type'] == 'update_weights':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    values = config['data'][fit['data']]['data']
                    mesh.nodes['weights'].values[1:values.size + 1] = values

            elif fit['type'] == 'update_pca':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    update_pca_fit(fit, mesh, config)
                    err.extend(fit['weight'] * (0.4 * mesh.nodes['weights'].values[1:]) ** 8)

            elif fit['type'] == 'update_dep_nodes':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    mesh.core.update_dependent_nodes()

            elif fit['type'] == 'update_maps':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    mesh.core.update_maps()

            elif fit['type'] == 'translate':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    dx_id = fit['translation_node']
                    dx = mesh.nodes[dx_id].values
                    for node in mesh.nodes.groups[fit['group']]:
                        if node.values.ndim == 2:
                            node.values[:, 0] += dx
                        else:
                            node.values += dx

            elif fit['type'] == 'closest_data':
                err = []
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    elements = fit['elements']
                    if elements == 'all':
                        elements = mesh.elements.keys()
                    Xg = fit['xi']
                    NPPE = Xg.shape[0]
                    NE = len(elements)
                    Xe = np.zeros((NE * NPPE, 3))
                    for i, eid in enumerate(elements):
                        Xe[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg)
                    Xe = np.array(Xe)
                    if fit['k'] == 1:
                        dx = fit['tree'].query(list(Xe))[0]
                    else:
                        dx = np.mean(fit['tree'].query(list(Xe), k=fit['k'])[0], axis=1)
                    if 'limit' in fit.keys():
                        ii = dx > fit['limit']
                        dx[ii] = fit['limit']
                    err.extend(fit['weight'] * dx)

            elif fit['type'] == 'closest_mesh':
                Xd = fit['_data']
                Xg = fit['xi']
                NPPE = Xg.shape[0]
                total_material_points = 0
                for mesh_id in fit['mesh']:
                    total_material_points += len(fit['elements']) * NPPE
                Xe = np.zeros((total_material_points, 3))
                i0 = 0
                for mesh_id in fit['mesh']:
                    mesh = meshes[mesh_id]
                    for i, eid in enumerate(fit['elements']):
                        Xe[i0:i0 + NPPE, :] = mesh.elements[eid].evaluate(Xg)
                        i0 += NPPE
                tree = cKDTree(Xe)
                if fit['k'] == 0:
                    r, ii = fit['weight'] * tree.query(list(Xd))
                    dx = (Xd - Xe[ii, :]).flatten()
                    err = dx * dx
                elif fit['k'] == 1:
                    err = fit['weight'] * tree.query(list(Xd))[0]
                else:
                    err = fit['weight'] * np.mean(tree.query(list(Xd), k=fit['k'])[0], axis=1)

            elif fit['type'] == 'fix_node':
                err = fit['weight'] * (mesh.nodes[fit['node']].values[:, 0] - fit['data']) ** 2

            elif fit['type'] == 'fix_nodes':
                Xn = np.array([mesh.nodes[nid].values[:, 0] for nid in fit['nodes']])
                # Xn += mesh.nodes[]
                err = fit['weight'] * (Xn - fit['data']).flatten() ** 2

            elif fit['type'] == 'fix_params':
                err = fit['weight'] * (mesh.core.P[fit['pids']] - fit['data']) ** 2

            elif fit['type'] == 'equal_distance':
                dists = []
                for nids in fit['nodes']:
                    dists.append(distance(mesh.nodes[nids[0]].values[:, 0], mesh.nodes[nids[1]].values[:, 0]))
                dists = np.array(dists)
                err = fit['weight'] * (dists - dists.mean()) ** 2

            elif fit['type'] == 'even_segments':
                elements = fit['elements']
                xi = fit['xi']
                dX = []
                for eid in elements:
                    x = mesh.elements[eid].evaluate(xi)
                    dx = ((x[1:, :] - x[:-1, :]) ** 2).sum(1)
                    dX.extend(dx.tolist())
                dX = np.array(dX)
                ddX = dX - (dX.sum() / dX.size)
                err = fit['weight'] * (ddX * ddX)

            elif fit['type'] == 'join_mesh_coordinates':
                mesh0 = meshes[fit['mesh'][0]]
                mesh1 = meshes[fit['mesh'][1]]
                nodes0 = fit['nodes'][0]
                nodes1 = fit['nodes'][1]
                dx = []
                for n0, n1 in zip(nodes0, nodes1):
                    dx.append(mesh0.nodes[n0].values[:, 0] - mesh1.nodes[n1].values[:, 0])
                dx = np.array(dx)
                err = fit['weight'] * (dx * dx).sum(1)

            elif fit['type'] == 'sobolev_smooth':
                err = []
                for mesh_num, mesh_id in enumerate(fit['mesh']):
                    mesh = meshes[mesh_id]
                    Xg = fit['xi']
                    NPPE = Xg.shape[0]
                    NE = len(fit['elements'])
                    derivs = np.zeros((NE * NPPE, 3))
                    for i, eid in enumerate(fit['elements']):
                        derivs[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                    derivs = np.sqrt((derivs * derivs).sum(1))
                    derivs = derivs.flatten()
                    if 'sample_values' in fit.keys():
                        derivs -= fit['sample_values'][mesh_num]
                    err.extend(fit['weight'] * (derivs * derivs))

            elif fit['type'] == 'function':
                err = []
                for mesh_num, mesh_id in enumerate(fit['mesh']):
                    dx = fit['function'](fit, meshes[mesh_id], data)
                    err.extend(fit['weight'] * dx)

            elif fit['type'] == 'constrain':
                err = []
                if 'elements' in fit.keys():
                    for mesh_num, mesh_id in enumerate(fit['mesh']):
                        mesh = meshes[mesh_id]
                        Xg = fit['xi']
                        NPPE = Xg.shape[0]
                        NE = len(fit['elements'])
                        x = np.zeros((NE * NPPE, 3))
                        for i, eid in enumerate(fit['elements']):
                            x[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                        if fit['deriv'] is not None:
                            x = np.sqrt((x * x).sum(1))
                        if 'values' in fit.keys():
                            x -= fit['values'][mesh_num]
                        if fit['deriv'] is None:
                            x = (x * x).sum(1)
                        else:
                            x = x * x
                        err.extend(fit['weight'] * x)

                elif 'params' in fit.keys():
                    for mesh_num, mesh_id in enumerate(fit['mesh']):
                        mesh = meshes[mesh_id]
                        dp = mesh.core.P[fit['params']] - fit['values']
                        err.extend(fit['weight'] * (dp * dp))

            elif fit['type'] == 'smooth_stretch':
                err = []
                for mesh_num, mesh_id in enumerate(fit['mesh']):
                    mesh = meshes[mesh_id]
                    Xg = fit['xi']
                    NPPE = Xg.shape[0]
                    NE = len(fit['elements'])
                    derivs = np.zeros((NE * NPPE, 3))
                    for i, eid in enumerate(fit['elements']):
                        derivs[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                    derivs = np.sqrt((derivs * derivs).sum(1))
                    deriv_mean = derivs.mean()
                    derivs -= deriv_mean
                    derivs = derivs.flatten()
                    if 'sample_values' in fit.keys():
                        derivs -= fit['sample_values'][mesh_num]
                    err.extend(fit['weight'] * (derivs * derivs))

            elif fit['type'] == 'penalise_derivatives':
                err = []
                for mesh_num, mesh_id in enumerate(fit['mesh']):
                    mesh = meshes[mesh_id]
                    Xg = fit['xi']
                    NPPE = Xg.shape[0]
                    NE = len(fit['elements'])
                    derivs = np.zeros((NE * NPPE, 3))
                    for i, eid in enumerate(fit['elements']):
                        derivs[i * NPPE:(i + 1) * NPPE, :] = mesh.elements[eid].evaluate(Xg, deriv=fit['deriv'])
                    derivs = derivs.flatten()
                    if 'sample_values' in fit.keys():
                        derivs -= fit['sample_values'][mesh_num]
                    err.extend(fit['weight'] * (derivs * derivs))

            elif fit['type'] == 'penalise_dot_product':
                err = []
                for mesh_num, mesh_id in enumerate(fit['mesh']):
                    mesh = meshes[mesh_id]
                    Xg = fit['xi']
                    NPPE = Xg.shape[0]
                    NE = len(fit['elements'])
                    derivs = np.zeros((NE * NPPE, 1))
                    for i, eid in enumerate(fit['elements']):
                        v1 = mesh.elements[eid].evaluate(Xg, deriv=[1, 0])
                        v2 = mesh.elements[eid].evaluate(Xg, deriv=[0, 1])
                        for v in [v1, v2]:
                            R = np.sqrt(np.sum(v * v, axis=1))
                            for axis in range(v.shape[1]):
                                v[:, axis] /= R

                        derivs[i * NPPE:(i + 1) * NPPE, 0] = np.sum(v1 * v2, 1)
                    derivs = derivs.flatten()
                    err.extend(fit['weight'] * (derivs * derivs))

            elif fit['type'] == 'penalise_area':
                elements = fit['elements']
                areas = []
                for eid in elements:
                    areas.append(mesh.elements[eid].area())
                err = fit['weight'] * np.array(areas)
            else:
                raise Exception('Unknown fit type: ' + fit['type'])
            Err.append(err)

        if config['output']:
            t = time.time()
            if t - config['output_time'] > config['dt']:
                err_str = ''
                for i, fit in enumerate(config['fits']):
                    if fit['output_error_factor'] != 0:
                        if fit['k'] == 0:
                            err2 = Err[i]
                        else:
                            err2 = np.array(Err[i]) * np.array(Err[i])
                        eout = np.sqrt(fit['output_error_factor'] * err2.mean())
                        err_str += '  %6.5f' % eout
                # mesh.save('fit_output.mesh')
                # mesh.save('fit_outputs/fit_output_%s_%06d.mesh' % (config['id'], config['iter']))
                print ('%5d (%4.0fs) E: %s' % (config['iter'], t - config['t0'], err_str))
                config['output_time'] = t

        Err = np.concatenate(Err)

        return Err

    def get_variables(meshes, config):
        x = np.zeros(config['num_vars'])
        update_dofs = config['fits'][0]['dofs']
        for dof in update_dofs:
            i0, i1 = dof['range']
            if 'mesh' in dof.keys():
                x[i0:i1] = meshes[dof['mesh']].get_variables()
            elif 'extra' in dof.keys():
                x[i0:i1] = config['data'][dof['data']]['data']
            elif 'data' in dof.keys():
                x[i0:i1] = config['data'][dof['data']]['data']
            else:
                print ('Unknown DOF: ', dof)
        return x

    def set_variables(meshes, config, x):
        update_dofs = config['fits'][0]['dofs']
        for dof in update_dofs:
            i0, i1 = dof['range']
            if 'mesh' in dof.keys():
                meshes[dof['mesh']].set_variables(x[i0:i1])
            elif 'extra' in dof.keys():
                config['data'][dof['data']]['data'] = x[i0:i1]
            elif 'data' in dof.keys():
                config['data'][dof['data']]['data'] = x[i0:i1]
            else:
                print ('Unknown DOF: ', dof)

        return meshes

    def add_dof_nodes(meshes, config, x):
        dofs = config['dofs']
        for dof in dofs:
            if 'extra' in dof.keys() and 'add_node' in dof.keys():
                nid = dof['add_node']
                if 'group' in dof.keys():
                    group = dof['group']
                else:
                    group = '_default'
                for update_dof in config['fits'][0]['dofs']:
                    if update_dof['extra'] == dof['extra']:
                        i0, i1 = update_dof['range']
                        for mesh in meshes.itervalues():
                            if nid not in mesh.nodes.keys():
                                mesh.add_node(nid, x[i0:i1], group=group)
                            else:
                                mesh.nodes[nid].values = x[i0:i1]
                        break
        return meshes

    start_time = time.time()

    config = {'id': fit_id, 'data': data, 'fits': fits, 'dofs': dofs, 'dt': dt}

    defaulted_mesh = False
    if not isinstance(meshes, dict):
        defaulted_mesh = True
        meshes = {'default': meshes}

        for dof in config['dofs']:
            if 'mesh' not in dof.keys():
                dof['mesh'] = 'default'

        for fit in config['fits']:
            if 'mesh' not in fit.keys():
                fit['mesh'] = 'default'

    meshes, config = on_start(meshes, config)

    for mesh in meshes.itervalues():
        mesh.generate()

    x0 = get_variables(meshes, config)
    err0 = measure_objfn(meshes, config, x0)
    x, success = scipy.optimize.leastsq(objfn, x0, args=[meshes, config], ftol=ftol, xtol=xtol, maxfev=maxiter)
    err1 = measure_objfn(meshes, config, x)
    meshes = set_variables(meshes, config, x)
    meshes = add_dof_nodes(meshes, config, x)

    dt = time.time() - start_time
    print ('Errors: %f > %f   Time: %6.2f s' % (err0, err1, dt))

    if defaulted_mesh:
        return meshes['default']

    return meshes