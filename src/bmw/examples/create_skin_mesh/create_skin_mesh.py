import os
import copy

import scipy
from scipy.spatial import cKDTree
import h5py
import morphic
reload(morphic)

def normalise(v):
    if len(v.shape)>1:
        normalised_v = scipy.zeros(v.shape)
        for idx in range(v.shape[0]):
            pt = v[idx,:]
            normalised_v[idx,:] = pt/scipy.sqrt(scipy.sum(pt*pt))
    else:
        normalised_v = v/scipy.sqrt(scipy.sum(v*v))
    return normalised_v

def create_volume_mesh(new_bm, new_cwm):
    # Add nodes between breast and chestwall surfaces
    new_bm_Xn = new_bm.get_nodes()
    new_cwm_Xn = new_cwm.get_nodes()
    num_Xe = len(new_bm.elements.ids)
    
    skin = True
    if skin:
        thickness = 2.0
        layer1 = -(new_bm_Xn-new_cwm_Xn)*1./3. + new_bm_Xn
        layer2 = -(new_bm_Xn-new_cwm_Xn)*2./3. + new_bm_Xn
        layer3 = -(normalise(new_bm_Xn-new_cwm_Xn))*thickness + new_bm_Xn
        layer4 = -(normalise(new_bm_Xn-new_cwm_Xn))*thickness*2./3. + new_bm_Xn
        layer5 = -(normalise(new_bm_Xn-new_cwm_Xn))*thickness*1./3. + new_bm_Xn
    else:
        layer1 = -(new_bm_Xn-new_cwm_Xn)*1./3. + new_bm_Xn
        layer2 = -(new_bm_Xn-new_cwm_Xn)*2./3. + new_bm_Xn
    
    visualise = True
    if visualise:
        fig.plot_points(
            '{0}_layer1'.format(0), layer1,
            color=(0,0,1), size=2)
        fig.plot_points(
            '{0}_layer2'.format(0), layer2,
            color=(1,0,1), size=2)
        if skin:
            fig.plot_points(
                '{0}_layer3'.format(0), layer3,
                color=(0,1,0), size=3)
            fig.plot_points(
                '{0}_layer4'.format(0), layer4,
                color=(0,1,0), size=3)
            fig.plot_points(
                '{0}_layer5'.format(0), layer5,
                color=(0,1,0), size=3)
    
    # Create new 3D mesh
    mesh3D = morphic.Mesh()
    mesh3D.auto_add_faces = True
    mesh3D.label = 'mesh3D'
    
    #num_layer_Xn = new_bm_Xn.shape[0]
    tree = cKDTree(new_cwm_Xn)
    num_layer_nodes = new_cwm_Xn.shape[0]
    map_Xn = scipy.zeros((num_layer_nodes,3))
    for idx in range(num_layer_nodes):
        map_Xn[idx,:] = new_cwm.get_nodes(idx)
    d, node_map = tree.query(map_Xn)
    
    #for idx, node in enumerate(Xn):
    #    mesh3D.add_stdnode(idx, Xn[idx,:])
    
    if skin:
        num_layer_nodes = new_bm_Xn.shape[0]
        num_layer_elem_nodes = len(new_bm.elements[0].node_ids)
        created_node_list = []
        elem_layer_node_offset = 0
        global_eidx = 0
        for elem_layer in [1,2]:
            if elem_layer == 1:
                Xn = scipy.array((new_cwm_Xn,layer2,layer1,layer3))
            elif elem_layer == 2:
                Xn = scipy.array((layer3,layer4,layer5,new_bm_Xn))
            for eidx in range(num_Xe):
                new_elem_node_list = []
                layer_Xe_nodes = scipy.array(new_bm.elements[eidx].node_ids)
                for layer in range(4):
                    for node in layer_Xe_nodes:
                        new_node_id = node+num_layer_nodes*layer + elem_layer_node_offset
                        new_elem_node_list.append(new_node_id)
                        if new_node_id not in created_node_list:
                            created_node_list.append(new_node_id)
                            mesh3D.add_stdnode(new_node_id + offset, Xn[layer,node_map[node],:])
                mesh3D.add_element(global_eidx+offset, ['L3', 'L3', 'L3'], new_elem_node_list)
                global_eidx += 1
            elem_layer_node_offset += len(created_node_list)
    else:
        Xn = scipy.array((new_cwm_Xn,layer2,layer1,new_bm_Xn))
        num_layer_nodes = new_bm_Xn.shape[0]
        num_layer_elem_nodes = len(new_bm.elements[0].node_ids)
        created_node_list = []
        for eidx in range(num_Xe):
            new_elem_node_list = []
            layer_Xe_nodes = scipy.array(new_bm.elements[eidx].node_ids)
            for layer in range(4):
                for node in layer_Xe_nodes:
                    new_node_id = node+num_layer_nodes*layer
                    new_elem_node_list.append(new_node_id)
                    if new_node_id not in created_node_list:
                        created_node_list.append(new_node_id)
                        mesh3D.add_stdnode(new_node_id + offset, Xn[layer,node_map[node],:])
            mesh3D.add_element(eidx+offset, ['L3', 'L3', 'L3'], new_elem_node_list)
    mesh3D.generate()
    
    return mesh3D

def visualise_mesh(mesh, fig, visualise, face_colours):
    mid = mesh.label
    Xn = mesh.get_nodes(group='_default')
    Xnid = mesh.get_node_ids(group='_default')
    
    if visualise:
        # View breast surface mesh
        Xs, Ts = mesh.get_surfaces(res=32)
        if Xs.shape[0] == 0:
            Xs, Ts = mesh3D.get_faces(res=16)

        Xl = mesh.get_lines(res=32)
        
        fig.plot_surfaces('{0}_Faces'.format(mid), Xs, Ts, color=face_colours, opacity=0.5)
        fig.plot_points('{0}_Nodes'.format(mid), Xn, color=(1,0,1), size=2)
        fig.plot_lines('{0}_Lines'.format(mid), Xl, color=(1,1,0), size=5)
        #fig.plot_text('{0}_Text'.format(mid), Xnid[0], Xnid[1], size=2)
        
        #fig.plot_element_ids('{0}_Xecid'.format(mid), mesh, size=1)


if __name__ == "__main__":
    
    root_dir = os.environ['BREAST_MODELLING_DIR']
    volunteer = 'VL00046'
    visualise = True
    offset = 0
    if visualise:
        from morphic import viewer
        if "fig" not in locals():
            fig = viewer.Figure()
    
    # Load surface meshes
    new_bm = morphic.Mesh('../create_surface_mesh/new_bm_surface.mesh')
    new_cwm = morphic.Mesh('../create_surface_mesh/new_cwm_surface.mesh')
    visualise = False
    visualise_mesh(new_cwm, fig, visualise, face_colours=(1,1,0))
    visualise = False
    visualise_mesh(new_bm, fig, visualise, face_colours=(1,0,0))
    
    mesh3D = create_volume_mesh(new_bm, new_cwm)
    
    visualise_mesh(mesh3D, fig, visualise, face_colours=(0,1,0))
    
    mid = mesh3D.label
    Xn = mesh3D.get_nodes(group='_default')
    Xnid = mesh3D.get_node_ids(group='_default')
    
    # View breast surface mesh
    Xs, Ts = mesh3D.get_surfaces(res=32)
    Xl = mesh3D.get_lines(res=16)
    Xf, Tf = mesh3D.get_faces(res=16)
    
    fig.plot_surfaces('{0}_Faces'.format(mid), Xf, Tf, color=(0,1,0), opacity=0.5)
    fig.plot_points('{0}_Nodes'.format(mid), Xn, color=(0,0,1), size=2)
    fig.plot_lines('{0}_Lines'.format(mid), Xl, color=(1,1,0), size=5)
    #fig.plot_text('{0}_Text'.format(mid), Xnid[0], Xnid[1], size=2)
    
    temp = scipy.zeros((8,3))
    temp[0,:]= mesh3D.elements[0].evaluate([0,0,0])
    temp[1,:]= mesh3D.elements[0].evaluate([1,0,0])
    temp[2,:]= mesh3D.elements[0].evaluate([0,1,0])
    temp[3,:]= mesh3D.elements[0].evaluate([1,1,0])
    temp[4,:]= mesh3D.elements[0].evaluate([0,0,1])
    temp[5,:]= mesh3D.elements[0].evaluate([1,0,1])
    temp[6,:]= mesh3D.elements[0].evaluate([0,1,1])
    temp[7,:]= mesh3D.elements[0].evaluate([1,1,1])
    fig.plot_points('temp',temp, color=(1,0,0), size=2)
    
    fig.plot_text('temp_text', temp,range(8),  size=2)
    
    mesh3D.save('./{0}_volume.mesh'.format(volunteer))
    
    #new_bm.elements[eidx].node_ids

