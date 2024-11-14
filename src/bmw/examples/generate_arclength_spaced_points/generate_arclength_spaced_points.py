import os

import scipy
from scipy.spatial import cKDTree
import h5py
import morphic
reload(morphic)

def integrate_length(mesh, elem_id, Xi):
    X = mesh.elements[elem_id].evaluate(Xi)
    dX = X[1:, :] - X[:-1, :]
    ln = scipy.sqrt((dX*dX).sum(1))
    return ln

def visualise_mesh(mesh, fig, visualise, face_colours):
    # Load surface meshes
    Xn = mesh.get_nodes(group='_default')
    Xnid = mesh.get_node_ids(group='_default')
    
    mid = mesh.label
    
    if visualise:
        # View breast surface mesh
        Xs, Ts = mesh.get_surfaces(res=16)
        Xl = mesh.get_lines(res=32)
        
        fig.plot_surfaces('{0}_Faces'.format(mid), Xs, Ts, color=face_colours, opacity=0.5)
        fig.plot_points('{0}_Nodes'.format(mid), Xn, color=(1,0,1), size=5)
        fig.plot_lines('{0}_Lines'.format(mid), Xl, color=(1,1,0), size=5)
        #fig.plot_text('{0}_Text'.format(mid), Xnid[0], Xnid[1], size=5)
        
        fig.plot_element_ids('{0}_Xecid'.format(mid), mesh, size=1)

def plot_points(fig_label_postfix, data, color, data_labels):
    fig.plot_points(
        '{0}_Points'.format(fig_label_postfix), data, color=color, size=10)
    fig.plot_text(
        '{0}_Text'.format(fig_label_postfix), data, data_labels, size=5)

def generate_new_nodes(mesh, num_new_nodes, r_e, r_xi2):
    num_r_sets = len(r_e)
    new_Xn = scipy.zeros((num_new_nodes * num_r_sets,3,4))
    new_node_id = 0
    
    for r_id, e in enumerate(r_e):
        
        # Generate grid
        num_exi_evals = 100
        xi = scipy.linspace(0, 1, num_exi_evals)
        X, Y = scipy.meshgrid(xi, scipy.array([r_xi2[r_id]]))
        Xi = scipy.array([
                X.reshape((X.size)),
                Y.reshape((Y.size))]).T
        
        # Calculate length of each element along xi1
        '''
        clns = cumulitive length for all elements
        ilns = individual length for all elements
        eilns = individual lengths in an element
        tln = total length
        eids = element id
        '''
        
        num_e = len(e)
        clns = scipy.zeros(((num_exi_evals-1)*num_e))
        ilns = scipy.zeros(((num_exi_evals-1)*num_e))
        eids = scipy.zeros(((num_exi_evals-1)*num_e, 2), dtype=int)
        counter = 0
        for cid in e:
            eilns = integrate_length(mesh, cid, Xi)
            for ln in range(len(eilns)):
                ilns[counter] = eilns[ln]
                clns[counter] = ilns[0:counter].sum()
                eids[counter,0] = cid
                eids[counter,1] = ln
                counter += 1
        
        # Calculate total length
        tln = clns[-1]
        
        # Specify spacing of new nodes
        target_lns = scipy.zeros((num_new_nodes,1))
        target_lns[:,0] = scipy.linspace(0, tln, num_new_nodes)
        
        # Find xi values in existing mesh that are closest to new nodes
        XdKDTree = cKDTree(scipy.array([clns]).T)
        neighbours = 1
        closest_lnid = XdKDTree.query(target_lns, k=neighbours)[1]
        closest_eids = eids[closest_lnid,0]
        closest_Xi = Xi[eids[closest_lnid,1],:]
        closest_Xi[-1,0] = 1.0 # Force last new node to coincide with existing node
        
        # Evaluate coordinates of new nodes
        new_r_Xn = scipy.zeros((num_new_nodes,3,4))
        new_r_Xn[:,:,0] = mesh.evaluate2(
                    closest_eids.tolist(), closest_Xi.tolist(), deriv=None)
        new_r_Xn[:,:,1] = mesh.evaluate2(
                    closest_eids.tolist(), closest_Xi.tolist(), deriv=[1, 0])
        new_r_Xn[:,:,2] = mesh.evaluate2(
                    closest_eids.tolist(), closest_Xi.tolist(), deriv=[0, 1])
        new_r_Xn[:,:,3] = mesh.evaluate2(
                    closest_eids.tolist(), closest_Xi.tolist(), deriv=[1, 1])
        for Xnid in range(num_new_nodes):
            new_Xn[new_node_id,:,:] = new_r_Xn[Xnid,:,:]
            new_node_id += 1
        
    return new_Xn

if __name__ == "__main__":
    
    root_dir = os.environ['BREAST_MODELLING_DIR']
    
    visualise = True
    if visualise:
        from morphic import viewer
        if "fig" not in locals():
            fig = viewer.Figure()
    

    volunteer = 'VL00046'
    cwm = morphic.Mesh(
        '{0}/data/volunteer/{1}/fitted_surface_mesh/{1}_ribcage_prone.mesh'.format(
            root_dir, volunteer))
    cwm.label = 'cwm'
    
    visualise_mesh(cwm, fig, visualise, face_colours=(1,0,0))
    cwm.generate()
    
    r_e = [[0, 1, 2, 3],
           [0, 1, 2, 3],
           [8, 9, 10, 11],
           [16, 17, 18, 19],
           [16, 17, 18, 19]] 
    r_xi2 = [0.0, 0.5, 0.0, 0.0, 1.0]
    num_new_nodes = 10
    cwm_new_Xn = generate_new_nodes(cwm, num_new_nodes, r_e, r_xi2)
    
    visualise = True
    if visualise:
        plot_points(cwm.label, cwm_new_Xn[:,:,0], (0,0,1), range(cwm_new_Xn.shape[0]))
    
    original_mesh = False
    if original_mesh:
        bm =  morphic.Mesh(
                '{0}/data/volunteer/{1}/fitted_surface_mesh/{1}_prone.mesh'.format(
            root_dir, volunteer))

        r_e = [[12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46],
               [24, 25, 26, 27, 28, 29, 47, 48, 49, 50, 51],
               [30, 31, 32, 33, 34, 35, 47, 48, 49, 50, 51]] 
    else:
        bm =  morphic.Mesh(
                '{0}/examples2/add_elements/{1}_prone_closed.mesh'.format(
            root_dir, volunteer))
        r_e = [[0,1,2,3,4,5,52,53,36,37,38],
               [6,7,8,9,10,11,54,55,39,40,41],
               [12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46],
               [24, 25, 26, 27, 28, 29, 47, 48, 49, 50, 51],
               [30, 31, 32, 33, 34, 35, 47, 48, 49, 50, 51]] 
    bm.generate()
    visualise_mesh(bm, fig, visualise, face_colours=(0,1,0))
    
    r_xi2 = [0.0, 0.0, 0.0, 0.0, 1.0]
    num_new_nodes = 10
    bm_new_Xn = generate_new_nodes(bm, num_new_nodes, r_e, r_xi2)
    
    
    visualise = True
    if visualise:
        plot_points(bm.label, bm_new_Xn[:,:,0], (1,1,1), range(bm_new_Xn.shape[0]))
    
    # Output nodes
    fname = './new_mesh_nodes.h5'
    hdf5_main_grp = h5py.File(fname, 'w')
    hdf5_main_grp.create_dataset('/cwm_nodes', data = cwm_new_Xn)
    hdf5_main_grp.create_dataset('/bm_nodes', data = bm_new_Xn)
    hdf5_main_grp.close()


