import os

import scipy
import copy
from scipy.spatial import cKDTree
import h5py
import morphic
reload(morphic)


def generate_surface(fig, visualise, label, old_mesh, Xe, hanging_e, offset):
    
    num_Xexi = [scipy.array(Xe).shape[1], scipy.array(Xe).shape[0]]
    new_Xe_nodes = generate_element_nodes(['L3', 'L3'], num_Xexi)
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
        Xsn = old_mesh.elements[Xe_list[eidx]].normal(Xi2d)
        print Xsn
        for nidx, node in enumerate(new_Xe_nodes[eidx]):
            if node not in nodes:
                nodes.append(node)
                mesh.add_stdnode(node+offset, Xg[nidx,:])
    
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
    
    return mesh

def visualise_mesh(mesh, fig, visualise, face_colours):
    mid = mesh.label
    Xn = mesh.get_nodes(group='_default')
    Xnid = mesh.get_node_ids(group='_default')
    
    if visualise:
        # View breast surface mesh
        Xs, Ts = mesh.get_surfaces(res=32)
        Xl = mesh.get_lines(res=32)
        
        fig.plot_surfaces('{0}_Faces'.format(mid), Xs, Ts, color=face_colours, opacity=0.5)
        #fig.plot_points('{0}_Nodes'.format(mid), Xn, color=(0,0,1), size=10)
        fig.plot_lines('{0}_Lines'.format(mid), Xl, color=(1,1,0), size=5)
        #fig.plot_text('{0}_Text'.format(mid), Xnid[0], Xnid[1], size=5)
        
        #fig.plot_element_ids('{0}_Xecid'.format(mid), mesh, size=1)


def generate_element_nodes(basis, num_elem_xi):
    A=1
    if basis == ['H3', 'H3']:
        BASIS_NUMBER_OF_NODES = 4
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [2, 2]
        NUMBER_OF_NODES_XIC.insert(0,0)
    elif basis == ['L3', 'L3']:
        BASIS_NUMBER_OF_NODES = 16
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [4, 4]
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
    return Xe_nodes - 1 # Generated node numbers begin from 0


if __name__ == "__main__":
    
    root_dir = os.environ['BREAST_MODELLING_DIR']
    volunteer = 'VL00046'
    visualise = True
    offset = 0
    if visualise:
        from morphic import viewer
        if "fig" not in locals():
            fig = viewer.Figure()
    
    # Create new breast surface mesh (bm)
    bm = morphic.Mesh('{0}/examples/add_shoulder_elements/{1}_prone_closed.mesh'.format(
            root_dir, volunteer))
    bm.label = 'bm'
    visualise = False
    visualise_mesh(bm, fig, visualise, face_colours=(0,1,0))
    
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
        None,  None,  None,  None,  None,  None,  [[0.,1.],[0., 0.75]],  [[0.,1.],[0., 0.75]],  [[0.,1.],[0., 0.75]],  [[0.,1.],[0., 0.75]],  [[0.,1.],[0., 0.75]],
        None,  None,  None,  None,  None,  None,  [[0.,1.],[0.75, 1.]],  [[0.,1.],[0.75, 1.]],  [[0.,1.],[0.75, 1.]],  [[0.,1.],[0.75, 1.]],  [[0.,1.],[0.75, 1.]]]
    new_bm = generate_surface(fig, visualise, 'new_bm', bm, Xe, hanging_e, offset)
    new_bm.save('new_bm_surface.mesh')
    visualise = True
    visualise_mesh(new_bm, fig, visualise, face_colours=(0,1,1))
    
    # Create new chestwall surface mesh (cwm)
    cwm = morphic.Mesh(
        '{0}/data/volunteer/{1}/fitted_surface_mesh/{1}_ribcage_prone.mesh'.format(
            root_dir, volunteer))
    cwm.label = 'cwm'
    visualise = False
    visualise_mesh(cwm, fig, visualise, face_colours=(1,0,0))
    
    Xe = [[0,0,0,1,1,1,1,2,2,2,3],
        [0,0,0,1,1,1,1,2,2,2,3],
        [8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11],
        [8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11],
        [16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 19],
        [16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 19]]
    hanging_e = [ 
        # Elem 1====================================================================   Elem 2==========================================================================================   # Elem 3====================================================================
        [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,0.75],[0., 0.5]],  [[0.75,1.],[0., 0.5]],  [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,1.],[0., 0.5]],
        [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,0.75],[0.5, 1.]],  [[0.75,1.],[0.5, 1.]],  [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,1.],[0.5, 1.]],
        [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,0.75],[0., 0.5]],  [[0.75,1.],[0., 0.5]],  [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,1.],[0., 0.5]],
        [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,0.75],[0.5, 1.]],  [[0.75,1.],[0.5, 1.]],  [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,1.],[0.5, 1.]],
        [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,0.25],[0., 0.5]],  [[0.25, 0.5],[0., 0.5]],  [[0.5,0.75],[0., 0.5]],  [[0.75,1.],[0., 0.5]],  [[0.,1./3.],[0., 0.5]],  [[1./3., 2./3.],[0., 0.5]],  [[2./3.,1.],[0., 0.5]],  [[0.,1.],[0., 0.5]],
        [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,0.25],[0.5, 1.]],  [[0.25, 0.5],[0.5, 1.]],  [[0.5,0.75],[0.5, 1.]],  [[0.75,1.],[0.5, 1.]],  [[0.,1./3.],[0.5, 1.]],  [[1./3., 2./3.],[0.5, 1.]],  [[2./3.,1.],[0.5, 1.]],  [[0.,1.],[0.5, 1.]]]
    new_cwm = generate_surface(fig, visualise, 'new_cwm', cwm, Xe, hanging_e, offset)
    new_cwm.save('new_cwm_surface.mesh')
    
    visualise = True
    visualise_mesh(new_cwm, fig, visualise, face_colours=(1,1,0))

