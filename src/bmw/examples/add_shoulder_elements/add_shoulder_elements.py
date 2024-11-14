import os
import scipy
from scipy.spatial import cKDTree
import morphic
reload(morphic)

def add_elements_to_existing_surface(mesh):
    # Add 2 new nodes based on average position
    Xn = mesh.get_nodes()
    num_Xn = Xn.shape[0]
    new_Xn1_adj = [6,49]
    new_Xn2_adj = [13,53]
    new_Xnid = [num_Xn,num_Xn+1]
    
    new_Xn1 = scipy.zeros((3,4))
    new_Xn1[:,0] = scipy.mean(mesh.get_nodes(new_Xn1_adj, deriv=None),axis=0)
    new_Xn1[:,1] = scipy.mean(mesh.get_nodes(new_Xn1_adj, deriv=1),axis=0)
    new_Xn1[:,2] = scipy.mean(mesh.get_nodes(new_Xn1_adj, deriv=2),axis=0)
    new_Xn1[:,3] = scipy.mean(mesh.get_nodes(new_Xn1_adj, deriv=3),axis=0)
    
    new_Xn2 = scipy.zeros((3,4))
    new_Xn2[:,0] = scipy.mean(mesh.get_nodes(new_Xn2_adj, deriv=None),axis=0)
    new_Xn2[:,1] = scipy.mean(mesh.get_nodes(new_Xn2_adj, deriv=1),axis=0)
    new_Xn2[:,2] = scipy.mean(mesh.get_nodes(new_Xn2_adj, deriv=2),axis=0)
    new_Xn2[:,3] = scipy.mean(mesh.get_nodes(new_Xn2_adj, deriv=3),axis=0)
    
    mesh.add_stdnode(new_Xnid[0], new_Xn1)
    mesh.add_stdnode(new_Xnid[1], new_Xn2)
    
    # Add 4 new elements
    num_Xe = len(mesh.get_element_cids())
    new_Xeid = [num_Xe, num_Xe+1, num_Xe+2, num_Xe+3]
    mesh.add_element(new_Xeid[0], ['H3', 'H3'], [6, new_Xnid[0], 13, new_Xnid[1]])
    mesh.add_element(new_Xeid[1], ['H3', 'H3'], [new_Xnid[0], 49, new_Xnid[1], 53])
    mesh.add_element(new_Xeid[2], ['H3', 'H3'], [13, new_Xnid[1], 20, 57])
    mesh.add_element(new_Xeid[3], ['H3', 'H3'], [new_Xnid[1], 53, 57, 58])
    mesh.generate()
    
    new_Xn1_normal = normalise(mesh.elements[53].normal(scipy.array([[0.0, 0.0]])))[0]
    new_Xn2_normal = normalise(mesh.elements[55].normal(scipy.array([[0.0, 0.0]])))[0]
    mesh.nodes[new_Xnid[0]].values[0:2,0] += new_Xn1_normal[0:2]*20
    mesh.nodes[new_Xnid[1]].values[0:2,0] += new_Xn2_normal[0:2]*20
    mesh.generate()

    return mesh

def normalise(v):
    return v/scipy.sqrt(scipy.sum(v*v))

def visualise_mesh(mesh, fig, visualise, face_colours):
    mid = mesh.label
    Xn = mesh.get_nodes(group='_default')
    Xnid = mesh.get_node_ids(group='_default')
    
    if visualise:
        # View breast surface mesh
        Xs, Ts = mesh.get_surfaces(res=16)
        Xl = mesh.get_lines(res=32)
        
        fig.plot_surfaces('{0}_Faces'.format(mid), Xs, Ts, color=face_colours,
                          opacity=0.5)
        fig.plot_points('{0}_Nodes'.format(mid), Xn, color=(1,0,1), size=5)
        fig.plot_lines('{0}_Lines'.format(mid), Xl, color=(1,1,0), size=5)
        #fig.plot_text('{0}_Text'.format(mid), Xnid[0], Xnid[1], size=5)
        
        #fig.plot_element_ids('{0}_Xecid'.format(mid), mesh, size=1)

if __name__ == "__main__":

    root_dir = os.environ['BREAST_MODELLING_DIR']
    
    visualise = True
    if visualise:
        from morphic import viewer
        if "fig" not in locals():
            fig = viewer.Figure()
    
    volunteer = 'VL00046'
    mesh = morphic.Mesh(
        '{0}/data/volunteer/{1}/fitted_surface_mesh/{1}_prone.mesh'.format(
            root_dir, volunteer))
    mesh.label = 'bm'
    add_elements_to_existing_surface(mesh)
    visualise_mesh(mesh, fig, visualise, face_colours=(0,1,0))
    
    mesh.save('./{0}_prone_closed.mesh'.format(volunteer))
