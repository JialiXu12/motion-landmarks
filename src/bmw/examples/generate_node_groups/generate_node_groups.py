import scipy
import morphic
import h5py
from scipy.spatial import cKDTree
reload(morphic)

def export_datapoints_exdata(data, label, filename):
    # Shape of data should be a [num_datapoints,dim] numpy array.  
    
    field_id = open(filename + '.exdata', 'w')
    field_id.write(' Group name: {0}\n'.format(label))
    field_id.write(' #Fields=1\n')
    field_id.write(' 1) coordinates, coordinate, rectangular cartesian, #Components=3\n')
    field_id.write('  x.  Value index=1, #Derivatives=0, #Versions=1\n')
    field_id.write('  y.  Value index=2, #Derivatives=0, #Versions=1\n')
    field_id.write('  z.  Value index=3, #Derivatives=0, #Versions=1\n')
    
    for point_idx, point in enumerate(range(1,data.shape[0]+1)):
        field_id.write(' Node: {0}\n'.format(point))
        for value_idx in range(data.shape[1]):
            field_id.write(' {0:.12E}\n'.format(data[point_idx,value_idx]))
    field_id.close()

def points_2_nodes_id(mesh, points):
    Xn = mesh.get_nodes(mesh.nodes.ids)
    tree = cKDTree(Xn)
    d, node_group = tree.query(points)
    return node_group

def generate_points_on_face(mesh, face, value, elem_group = []):
    # Generate a grid of points within each element
    num_points_per_elem_xi = 4
    if face == "xi1":
        xi1 = [value]
        xi2 = scipy.linspace(0., 1., num_points_per_elem_xi)
        xi3 = scipy.linspace(0., 1., num_points_per_elem_xi)
    elif face ==  "xi2":
        xi1 = scipy.linspace(0., 1., num_points_per_elem_xi)
        xi2 = [value]
        xi3 = scipy.linspace(0., 1., num_points_per_elem_xi)
    elif face ==  "xi3":
        xi1 = scipy.linspace(0., 1., num_points_per_elem_xi)
        xi2 = scipy.linspace(0., 1., num_points_per_elem_xi)
        xi3 = [value]
    X, Y, Z = scipy.meshgrid(xi1, xi2, xi3)
    Xi3d = scipy.array([
        X.reshape((X.size)),
        Y.reshape((Y.size)),
        Z.reshape((Z.size))]).T
    
    if not elem_group:
        # if elem group is empty evaluate all elements
        elem_group = mesh.elements.ids
    num_Xe = len(elem_group)
    total_num_points = num_Xe*num_points_per_elem_xi**2
    points = scipy.zeros((num_Xe, num_points_per_elem_xi**2, 3))
    
    for idx, Xeid in enumerate(elem_group):
        points[idx,:,:] = mesh.elements[Xeid].evaluate(Xi3d)
    points = scipy.reshape(points, (total_num_points,3))
    
    return points

if __name__ == "__main__":
    visualise = True
    volunteer = 'VL00046'
    if visualise:
        from morphic import viewer
        if "fig" not in locals():
            fig = viewer.Figure()
    
    # Load surface meshes
    mesh = morphic.Mesh('../create_volume_mesh/{0}_volume.mesh'.format(volunteer))
    Xn = mesh.get_nodes(mesh.nodes.ids)
    Xnid = mesh.get_node_ids(group='_default')
    
    if visualise:
        Xf, Tf = mesh.get_faces(res=10)
        Xl = mesh.get_lines(res=10)
        
        fig.plot_surfaces('Faces', Xf, Tf, color=(0,1,0), opacity=0.25)
        #fig.plot_points('Nodes', Xn, color=(1,0,1), size=2)
        fig.plot_lines('Lines', Xl, color=(1,1,0), size=5)
        # Takes a long time to render text
        #fig.plot_text('Text', Xnid[0], Xnid[1], size=2)
    
    cranial_elem = range(11)
    caudal_elem = range(55,66)
    sternum_elem = [0,11,22,33,44,55]
    spine_elem = [10,21,32,43,54,65]
    
    cranial_points = generate_points_on_face(mesh, "xi2", 0, elem_group=cranial_elem)
    caudal_points = generate_points_on_face(mesh, "xi2", 1, elem_group=caudal_elem)
    sternum_points = generate_points_on_face(mesh, "xi1", 0, elem_group=sternum_elem)
    spine_points = generate_points_on_face(mesh, "xi1", 1, elem_group=spine_elem)
    chestwall_points = generate_points_on_face(mesh, "xi3", 0)
    skin_points = generate_points_on_face(mesh, "xi3", 1)
    
    visualise = True
    if visualise:
        fig.plot_points('cranial_points', cranial_points, color=(1,0,0), size=5)
        fig.plot_points('caudal_points', caudal_points, color=(0,1,0), size=5)
        fig.plot_points('sternum_points', sternum_points, color=(0,0,1), size=5)
        fig.plot_points('spine_points', spine_points, color=(1,1,0), size=5)
        fig.plot_points('chestwall_points', chestwall_points, color=(1,0,0), size=5)
        fig.plot_points('skin_points', skin_points, color=(0,1,1), size=5)
    
    cranial_nodes = points_2_nodes_id(mesh, cranial_points)
    caudal_nodes = points_2_nodes_id(mesh, caudal_points)
    sternum_nodes = points_2_nodes_id(mesh, sternum_points)
    spine_nodes = points_2_nodes_id(mesh, spine_points)
    chestwall_nodes = points_2_nodes_id(mesh, chestwall_points)
    skin_nodes = points_2_nodes_id(mesh, skin_points)
    
    visualise = True
    if visualise:
        fig.plot_points('cranial_nodes', mesh.get_nodes(cranial_nodes.tolist()), color=(1,0,0), size=5)
        fig.plot_points('caudal_nodes', mesh.get_nodes(caudal_nodes.tolist()), color=(0,1,0), size=5)
        fig.plot_points('sternum_nodes', mesh.get_nodes(sternum_nodes.tolist()), color=(0,0,1), size=5)
        fig.plot_points('spine_nodes', mesh.get_nodes(spine_nodes.tolist()), color=(1,1,0), size=5)
        fig.plot_points('chestwall_nodes', mesh.get_nodes(chestwall_nodes.tolist()), color=(1,0,1), size=5)
        fig.plot_points('skin_nodes', mesh.get_nodes(skin_nodes.tolist()), color=(0,1,1), size=5)
    
    hdf5_main_grp = h5py.File('node_elem_groups.h5', 'w')
    
    hdf5_main_grp.create_dataset('/elements/cranial', data = cranial_elem)
    hdf5_main_grp.create_dataset('/elements/caudal', data = caudal_elem)
    hdf5_main_grp.create_dataset('/elements/sternum', data = sternum_elem)
    hdf5_main_grp.create_dataset('/elements/spine', data = spine_elem)
    
    hdf5_main_grp.create_dataset('/nodes/cranial', data = cranial_nodes)
    hdf5_main_grp.create_dataset('/nodes/caudal', data = caudal_nodes)
    hdf5_main_grp.create_dataset('/nodes/sternum', data = sternum_nodes)
    hdf5_main_grp.create_dataset('/nodes/spine', data = spine_nodes)
    hdf5_main_grp.create_dataset('/nodes/chestwall', data = chestwall_nodes)
    hdf5_main_grp.create_dataset('/nodes/skin', data = skin_nodes)

