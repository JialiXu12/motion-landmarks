import h5py
import numpy as np
import os
import mesh_tools
from opencmiss.iron import iron

def exfile_to_morphic(nodeFilename, coordinateField,
                      dimension=2, interpolation='linear'):
    """Convert an exnode and exelem files to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    nodeFilename -- exnode filename
    elementFilename -- exelem filename
    coordinateField -- the field to read in
    dimension -- dimension of mesh to read in
    """

    # Create morphic mesh
    import morphic
    mesh = morphic.Mesh()

    # Load exfiles
    exnode = mesh_tools.Exnode(nodeFilename)
    elem_nodes = np.array([[19, 19, 142, 174],
                           [19, 13, 174, 176],
                           [13, 14, 176, 148],
                           [14, 14, 148, 123],
                           [19, 19, 142, 128],
                           [19, 13, 128, 111],
                           [13, 14, 111, 110],
                           [14, 14, 110, 123],
                           [142, 174, 401, 384],
                           [174, 176, 384, 386],
                           [176, 148, 386, 370],
                           [148, 123, 370, 323],
                           [110, 123, 303, 323],
                           [142, 128, 401, 346],
                           [128, 111, 346, 328],
                           [111, 110, 328, 303],
                           [401, 384, 650, 601],
                           [384, 386, 601, 585],
                           [386, 370, 585, 605],
                           [370, 323, 605, 536],
                           [303, 323, 538, 536],
                           [328, 303, 558, 538],
                           [346, 328, 597, 558],
                           [401, 346, 650, 597],
                           [650, 601, 768, 723],
                           [601, 585, 723, 724],
                           [585, 605, 724, 774],
                           [605, 536, 774, 715],
                           [650, 597, 768, 718],
                           [597, 558, 718, 699],
                           [558, 538, 699, 698],
                           [538, 536, 698, 715],
                           [768, 723, 837, 837],
                           [723, 724, 837, 869],
                           [724, 774, 869, 865],
                           [768, 718, 837, 837],
                           [718, 699, 837, 869],
                           [699, 698, 869, 865],
                           [698, 715, 865, 865],
                           [774, 715, 865, 865]]).astype('int32')
    # Add nodes
    if interpolation == 'hermite':
        derivatives = range(1,9)
    else:
        derivatives = [1]
    for node_num in exnode.nodeids:
        coordinates = []
        for component in range(1, 4):
            component_name = ["x", "y", "z"][component - 1]
            componentValues = []
            for derivative_idx, derivative in enumerate(derivatives):
                componentValues.append(exnode.node_value(coordinateField,
                                                     component_name, node_num,
                                                     derivative))
            coordinates.append(componentValues)

        mesh.add_stdnode(node_num, coordinates, group='_default')
        #print('Morphic node added', node_num, coordinates)

    if dimension == 2:
        if interpolation == 'linear':
            element_interpolation = ['L1', 'L1']
        if interpolation == 'quadratic':
            element_interpolation = ['L2', 'L2']
    elif dimension == 3:
        if interpolation == 'linear':
            element_interpolation = ['L1', 'L1', 'L1']
        if interpolation == 'quadratic':
            element_interpolation = ['L2', 'L2', 'L2']
        if interpolation == 'cubic':
            element_interpolation = ['L3', 'L3', 'L3']
        if interpolation == 'hermite':
            element_interpolation = ['H3', 'H3', 'H3']

    # Add elements
    for elem_idx, elem in enumerate(elem_nodes):
        elem_num = elem_idx + 1
        mesh.add_element(elem_num, element_interpolation, elem_nodes[elem_idx])
        #print('Morphic element added', elem.number)

    # Generate the mesh
    mesh.generate(True)

    return mesh

if __name__ == '__main__':

    # Parameters
    dimension = 2
    results_folder='./results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    fname = '/home/psam012/opt/cvt-data/2018-10-30 pre CT lung/PointCloudAlignedToLaserLinesPreCT.hdf5'
    data = mesh_tools.import_h5_dataset(fname, '/Aligned Point Cloud')
    mesh_tools.export_datapoints_exdata(data, 'data', os.path.join(results_folder, 'data'))

    nodeFilename = '/home/psam012/opt/cvt-data/2018-11-9 Post microCT/MeshFitting/data/left_surface.exnode'
    elementFilename = '/home/psam012/opt/cvt-data/2018-11-9 Post microCT/MeshFitting/data/left_surface.exelem'

    nodeFilename = '/home/psam012/opt/cvt-data/2018-11-9 Post microCT/MeshFitting/data/left_surface.exnode'
    coordinateField = 'coordinates'
    m_mesh = exfile_to_morphic(
        nodeFilename, coordinateField, dimension=2, interpolation='linear')

    #m_mesh = mesh_tools.renumber_mesh(
    #    m_mesh, node_offset=0, element_offset=0, label='', debug=False)


    region, basis = mesh_tools.generate_opencmiss_region(
        interpolation=iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE,
        dimension=2)

    meshUserNumber = 1
    mesh, coordinates, node_list, element_list = mesh_tools.morphic_to_OpenCMISS(
        m_mesh, region, basis, meshUserNumber, dimension=2,
        interpolation='linear',  include_derivatives=False)


    # Export mesh in ex format using OpenCMISS
    numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
    computationalNodeNumber = iron.ComputationalNodeNumberGet()
    decompositionUserNumber = 1
    geometricFieldUserNumber = 1
    decomposition = iron.Decomposition()
    decomposition.CreateStart(decompositionUserNumber, mesh)
    decomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    decomposition.CreateFinish()

    geometric_field = iron.Field()
    geometric_field.CreateStart(geometricFieldUserNumber, region)
    geometric_field.MeshDecompositionSet(decomposition)
    geometric_field.VariableLabelSet(iron.FieldVariableTypes.U, "coordinates")
    geometric_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 1, 1)
    geometric_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 2, 1)
    geometric_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 3, 1)
    geometric_field.CreateFinish()


    mesh_tools.set_field_values(
        geometric_field, node_list, coordinates, derivative=1,
         variable=iron.FieldVariableTypes.U, update_scale_factors=False)

    fields = iron.Fields()
    fields.AddField(geometric_field)
    fields.NodesExport(os.path.join(results_folder, "geometry"), "FORTRAN")
    fields.ElementsExport(os.path.join(results_folder, "geometry"), "FORTRAN")


