#!/usr/bin/env python

#> \file
#> \author Thiranja Prasad Babarenda Gamage
#> \brief 
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is Breast Modelling in Openiron
#>
#> The Initial Developer of the Original Code is:
#> Thiranja Prasad Babarenda Gamage
#> Auckland, New Zealand, 
#> All Rights Reserved.
#>
#> Contributor(s):
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. If you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. If you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>

#> Main script

# Standard library imports
import sys
import os

# Local application/library specific imports
import h5py
import numpy as np
from scipy import spatial

# Intialise Openiron
import scipy
#import iron

# sys.path.insert(1, os.sep.join((os.environ['OPENiron_ROOT'],
#                                        'cm', 'bindings','python')))
# Intialise OpenCMISS
from opencmiss.iron import iron


VARIABLE_TYPES = (
    {iron.FieldVariableTypes.U : 'u',
     iron.FieldVariableTypes.DELUDELN : 'deludeln'})

PRONE_TO_REFERENCE = 0
REFERENCE_TO_SUPINE = 1
SAVED_SOLUTION_FILENAME = 'saved_solutions'
SAVED_STIFFNESS_FIELD_FILENAME = 'saved_stiffness_field'
SAVED_SOLUTION_LABELS = (
    {PRONE_TO_REFERENCE: 'prone_to_reference',
     REFERENCE_TO_SUPINE: 'reference_to_supine'})

def initialise_dataset(hdf5_main_grp, group_label, parameters, gravity_vector):
    hdf5_field_group = hdf5_main_grp.create_group(group_label)
    number_of_saved_solutions = 0
    hdf5_field_group.create_dataset('number_of_saved_solutions', data=number_of_saved_solutions)
    hdf5_field_group.create_dataset('parameters', data=np.zeros((1,parameters.shape[0])), chunks=True, maxshape=(None, gravity_vector.shape[0]))
    hdf5_field_group.create_dataset('gravity_vectors', data=np.zeros((1, gravity_vector.shape[0])), chunks=True, maxshape=(None, gravity_vector.shape[0]))
    #hdf5_field_group.create_dataset('sliding_lengths', data=np.zeros((1,1)), chunks=True, maxshape=(None, 1))
    return number_of_saved_solutions

def save_solutions(
        volunteer, results_folder, problem_idx, parameters, gravity_vector, iron, dependent_fields, lagrange_field=None):
    # Save dependent field solution for each solve. Store dependent field from
    # prone to reference and reference to supine solves, seperately. These
    # will be used used as initial guesses to solving each of these problems.
    
    # Create and hdf5 file - will overwrite any existing file.  
    hdf5_main_grp = h5py.File('{0}/{1}.hdf5'.format(
        results_folder, SAVED_SOLUTION_FILENAME), 'a')
    group_label = SAVED_SOLUTION_LABELS[problem_idx]
    try:
        hdf5_field_group = hdf5_main_grp[group_label]
    except:
        hdf5_field_group = hdf5_main_grp.create_group(group_label)
        number_of_saved_solutions = 1
        hdf5_field_group.create_dataset('number_of_saved_solutions', data=number_of_saved_solutions)
        hdf5_field_group.create_dataset('parameters', data=np.zeros((1,parameters.shape[0])), chunks=True, maxshape=(None, parameters.shape[0]))
        hdf5_field_group.create_dataset('gravity_vectors', data=np.zeros((1,gravity_vector.shape[0])), chunks=True, maxshape=(None, gravity_vector.shape[0]))
    else:
        number_of_saved_solutions = hdf5_field_group['number_of_saved_solutions'][...]
        number_of_saved_solutions += 1
        hdf5_field_group['number_of_saved_solutions'][...] = number_of_saved_solutions
        hdf5_field_group['parameters'].resize((number_of_saved_solutions, parameters.shape[0]))
        hdf5_field_group['gravity_vectors'].resize((number_of_saved_solutions, gravity_vector.shape[0]))
    
    hdf5_field_group['parameters'][number_of_saved_solutions-1,:] = parameters
    hdf5_field_group['gravity_vectors'][number_of_saved_solutions-1,:] = gravity_vector
    hdf5_field_subgroup = hdf5_field_group.create_group('{0}'.format(number_of_saved_solutions-1))
    
    
    # Group together fields that will be output.  
    fields = []
    for field in dependent_fields:
        fields.append(field)
    if lagrange_field is not None:
        fields.append(lagrange_field)
    field_names = { 0 : 'dependent_field1',
                    1 : 'dependent_field2',
                    2 : 'lagrange_field'}

    for region_idx, field in enumerate(fields):
        hdf5_field_subsubgroup = hdf5_field_subgroup.create_group(field_names[region_idx])
        for variable_type, variable_label in VARIABLE_TYPES.iteritems() :
            parameters = field.ParameterSetDataGetDP(
                                    variable_type, iron.FieldParameterSetTypes.VALUES)
            hdf_parameters = np.empty_like(parameters)
            hdf_parameters[:] = parameters
            hdf5_field_subsubgroup.create_dataset(variable_label,
                data = parameters)
            field.ParameterSetDataRestoreDP(
                variable_type, iron.FieldParameterSetTypes.VALUES,
                hdf_parameters)
    
    hdf5_main_grp.close()

def load_solutions(volunteer, problem_idx, results_folder, parameters, gravity_vector, iron, dependent_fields, distance_upperbound=0.1, force_resolve=False):

    solved = False
    load_lagrange_field = False
    group_label = SAVED_SOLUTION_LABELS[problem_idx]
    try:
        hdf5_main_grp = h5py.File('{0}/{1}.hdf5'.format(
            results_folder,SAVED_SOLUTION_FILENAME), 'r')
    except:
        pass
    else:
        group_label = SAVED_SOLUTION_LABELS[problem_idx]
        try:
            hdf5_field_group = hdf5_main_grp[group_label]
        except:
            pass
        else:
            number_of_saved_solutions = hdf5_field_group['number_of_saved_solutions'][...]
            saved_parameters = hdf5_field_group['parameters'][...]
            saved_gravity_vectors = hdf5_field_group['gravity_vectors'][...]
            # Search in the saved parameter list for a parameter set that is
            # closest to the current parameters to be solved for.
            saved_values = np.append(saved_parameters, saved_gravity_vectors, 1)

            # Search for identical solution
            search_values = np.append(parameters, gravity_vector)
            parameter_tree = spatial.cKDTree(saved_values)
            [distance, tree_index] = parameter_tree.query(search_values, 
                distance_upper_bound=distance_upperbound)
            if np.isfinite(distance):
                closest_solution_idx = tree_index
                if abs(distance) < 1e-14:
                    print ('  Model already solved, loading saved solutions')
                    solved = True
                    if force_resolve:
                        solved = False
                #import ipdb; ipdb.set_trace()
                # Closest solution found
                print ('    Loaded parameters: ', saved_parameters[tree_index])
                print ('    Loaded gravity vector: ', saved_gravity_vectors[tree_index])
                
                hdf5_field_subgroup = hdf5_field_group['{0}'.format(tree_index)]
                
                # Group together fields that will be loaded.
                field_names = { 0 : 'dependent_field1',
                                1 : 'dependent_field2'}
                for field_idx, dependent_field in enumerate(dependent_fields):
                    hdf5_field_group = hdf5_field_subgroup[field_names[field_idx]]
                    for variable_type, variable_label in VARIABLE_TYPES.iteritems() :
                        data = hdf5_field_group[variable_label][()]
                        dependent_field.ParameterSetUpdateLocalDofsDP(
                            variable_type, iron.FieldParameterSetTypes.VALUES,
                            data)
    
        hdf5_main_grp.close()

    return solved


def save_stiffness_field(
        volunteer, results_folder, iron, material_field):
    # Save dependent field solution for each solve. Store dependent field from
    # prone to reference and reference to supine solves, seperately. These
    # will be used used as initial guesses to solving each of these problems.
    
    # Create and hdf5 file - will overwrite any existing file.  
    hdf5_main_grp = h5py.File('{0}/{1}.hdf5'.format(
        results_folder, SAVED_STIFFNESS_FIELD_FILENAME), 'w')

    # Group together fields that will be output.  
    fields = [material_field]
    field_names = { 0 : 'material_field'}
    for region_idx, field in enumerate(fields):
        hdf5_field_subsubgroup = hdf5_main_grp.create_group(field_names[region_idx])
        parameters = field.ParameterSetDataGetDP(
            iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
        hdf_parameters = np.empty_like(parameters)
        hdf_parameters[:] = parameters
        hdf5_field_subsubgroup.create_dataset('u',
            data = parameters)
        field.ParameterSetDataRestoreDP(
            iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
            hdf_parameters)
    
    hdf5_main_grp.close()

def load_stiffness_field(volunteer, results_folder, material_field):

    solved = False
    try:
        hdf5_main_grp = h5py.File('{0}/{1}.hdf5'.format(
            results_folder,SAVED_STIFFNESS_FIELD_FILENAME), 'r')
    except:
        pass
    else:
        field_names = { 0 : 'material_field'}
        try:
            hdf5_field_group = hdf5_main_grp['material_field']
        except:
            pass
        else:
            data = hdf5_field_group['u'][()]
            material_field.ParameterSetUpdateLocalDofsDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                data)
            solved = True

        hdf5_main_grp.close()

    return solved

def iron_intStringToArray(String):
    StringArray = String.split(',')
    Data = []
    for value in StringArray:
        if value.find('..')>-1:
            temp = value.split('..')
            temp2 = range(int(temp[0]),int(temp[1])+1)
            [Data.append(point) for point in temp2]
        else:
            Data.append(int(value))
    return Data


"""Solve single region breast model"""
def calculate_stiffness(mechanics_setup, op, parameters, decomposition=None, mesh=None,
    cmesh=None, region=None, side='rhs', field_export_name='Field'):
    
    print ('Calculating stiffness')

    MESH_COMPONENT1 = 1
    MESH_COMPONENT2 = 2
    linear_elem_node_idxs = [0,3,12,15,48,51,60,63] # Linear element node idx in cubic element.
    offset = 1
    converged = True
    skin = False
    # load dof groups
    h5_dof_groups = h5py.File('{0}/dof_groups_{1}.h5'.format(op.results_dir, side), 'r')
    if skin:
        skin_elem = h5_dof_groups['/elements/skin'][()].T

    geometric_mesh_component = 1
    pressure_mesh_component = 2

    #scaling_type = iron.FieldScalingTypes.UNIT
    scaling_type = iron.FieldScalingTypes.NONE
    
    region_user_num = 1
    generated_mesh_user_num = 1
    mesh_user_num = 1
    decomposition_user_num = 1
    geometric_field_user_num = 1
    dependent_field_user_num = 2
    equations_set_field_user_num = 3
    material_field_user_num = 4
    equations_set_user_num = 1
    problem_user_num = 1
    
    coordinate_system = mechanics_setup['coordinate_system']
    basis = mechanics_setup['basis']
    pressure_basis = mechanics_setup['pressure_basis']

    # Get the number of computational nodes and this computational node 
    # number.  
    number_of_computational_nodes = iron.ComputationalNumberOfNodesGet()
    computational_node_number = iron.ComputationalNodeNumberGet()    
    
    # Create a region and assign the coordinate system to the region.  
    region = iron.Region()
    region.CreateStart(region_user_num, iron.WorldRegion)
    region.LabelSet("Region")
    region.CoordinateSystemSet(coordinate_system)
    region.CreateFinish()

    #elem_nums = [11,22,33,44,55,66,77,88,99,110,121,132]
    #elem_nums = [11,22,33,44,55,66,77,88,99,110,121,132]
    # Create mesh topology
    cmesh = iron.Mesh()
    cmesh.CreateStart(mesh_user_num, region, 3)
    cmesh.NumberOfComponentsSet(2)
    cmesh.NumberOfElementsSet(mesh.elements.size())
    #cmesh.NumberOfElementsSet(len(elem_nums))
    nodes = iron.Nodes()
    nodes.CreateStart(region, mesh.nodes.size())
    all_node_nums = (scipy.array(mesh.get_node_ids()[1])+1).astype('int32')
    nodes.AllUserNumbersSet(all_node_nums)
    nodes.CreateFinish()
    
    elements = iron.MeshElements()
    elements.CreateStart(cmesh, MESH_COMPONENT1, basis)
    #elements.AllUserNumbersSet(elem_nums)
    #elements.AllUserNumbersSet(iron_intStringToArray("4..11,15..22,26..33,37..44,48..55,59..66,70..77,81..88,92..99,103..110,114..121,125..132"))
    for element in mesh.elements:
        #if (element.id+ offset) not in [1,2,3,12,13,14,23,24,25,34,35,36,45,46,47,56,57,58,67,68,69,78,79,80,89,90,91,100,101,102,111,112,113,122,123,124]:
        #if (element.id+ offset) in elem_nums:
        elements.NodesSet(element.id + offset, scipy.array(element.node_ids, dtype='int32') + offset)
    elements.CreateFinish()
    

    pressure_elements = iron.MeshElements()
    pressure_elements.CreateStart(cmesh, MESH_COMPONENT2, pressure_basis)
    #pressure_elements.AllUserNumbersSet(elem_nums)
    #pressure_elements.AllUserNumbersSet(iron_intStringToArray("4..11,15..22,26..33,37..44,48..55,59..66,70..77,81..88,92..99,103..110,114..121,125..132"))
    for element in mesh.elements:
        #if (element.id+ offset) not in [1,2,3,12,13,14,23,24,25,34,35,36,45,46,47,56,57,58,67,68,69,78,79,80,89,90,91,100,101,102,111,112,113,122,123,124]:
        #if (element.id+ offset) in elem_nums:
        elem_nodes = scipy.array(element.node_ids, dtype='int32') + offset
        pressure_elements.NodesSet(
            element.id + offset, elem_nodes[linear_elem_node_idxs])
        if skin:
            deriv = 1
            version = 2
            if element.id in skin_elem:
                for elem_node in [1, 2, 3, 4]:
                    pressure_elements.LocalElementNodeVersionSet(
                        element.id + offset, version, deriv, elem_node)
    pressure_elements.CreateFinish()


    cmesh.CreateFinish()
    
    # Create a decomposition for the mesh.  
    decomposition = iron.Decomposition()
    decomposition.CreateStart(decomposition_user_num, cmesh)
    decomposition.type = iron.DecompositionTypes.CALCULATED
    decomposition.NumberOfDomainsSet(number_of_computational_nodes)
    decomposition.CreateFinish()
    
    # Create a field for the geometry.  
    geometric_field = iron.Field()
    geometric_field.CreateStart(geometric_field_user_num, region)
    geometric_field.MeshDecompositionSet(decomposition)
    geometric_field.VariableLabelSet(iron.FieldVariableTypes.U,
                                     'Geometry')
    geometric_field.ScalingTypeSet(scaling_type)
    geometric_field.CreateFinish()
    
    num_deriv = 1
    for node in mesh.nodes:
        for comp_idx in range(3):
            for deriv_idx in range(num_deriv):
                geometric_field.ParameterSetUpdateNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, deriv_idx + 1, node.id + offset, comp_idx + 1,
                    node.values[comp_idx])

    geometric_field.ParameterSetUpdateStart(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    geometric_field.ParameterSetUpdateFinish(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)

#    solved = load_solutions(
#       op.volunteer_id, problem_idx, op.results_dir, parameters, gravity_vector, iron,
#       [dependent_field], distance_upperbound=100., force_resolve=force_resolve)

    solved = False

    if not solved:

        # Create the equations_set.  
        equations_set_field = iron.Field()
        equations_set = iron.EquationsSet()
        equations_set.CreateStart(
            equations_set_user_num, region, geometric_field,
            iron.EquationsSetClasses.CLASSICAL_FIELD,
            iron.EquationsSetTypes.LAPLACE_EQUATION,
            iron.EquationsSetSubtypes.STANDARD_LAPLACE,
            equations_set_field_user_num, equations_set_field)
        equations_set.CreateFinish()

        # Create dependent field
        dependent_field = iron.Field()
        equations_set.DependentCreateStart(dependent_field_user_num, dependent_field)
        dependent_field.DOFOrderTypeSet(iron.FieldVariableTypes.U,iron.FieldDOFOrderTypes.SEPARATED)
        dependent_field.DOFOrderTypeSet(iron.FieldVariableTypes.DELUDELN,iron.FieldDOFOrderTypes.SEPARATED)
        #for component in [1]:
        #    dependent_field.ComponentMeshComponentSet(
        #        iron.FieldVariableTypes.U, component, pressure_mesh_component)
        #    dependent_field.ComponentMeshComponentSet(
        #        iron.FieldVariableTypes.DELUDELN, component, pressure_mesh_component)
        equations_set.DependentCreateFinish()

        # Initialise dependent field
        dependent_field.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,0.0)

        # Create equations.  
        equations = iron.Equations()
        equations_set.EquationsCreateStart(equations)
        #equations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
        equations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
        equations_set.EquationsCreateFinish()
        
        # Define the problem.  
        problem = iron.Problem()
        problem.CreateStart(problem_user_num)
        problem.SpecificationSet(iron.ProblemClasses.CLASSICAL_FIELD,
                iron.ProblemTypes.LAPLACE_EQUATION,
                iron.ProblemSubTypes.STANDARD_LAPLACE)
        problem.CreateFinish()
        
        # Create control loops
        problem.ControlLoopCreateStart()
        problem.ControlLoopCreateFinish()

        # Create problem solver
        solver = iron.Solver()
        problem.SolversCreateStart()
        problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
        solver.outputType = iron.SolverOutputTypes.SOLVER
        #solver.linearType = iron.LinearSolverTypes.ITERATIVE
        #solver.linearIterativeAbsoluteTolerance = 1.0E-12
        #solver.linearIterativeRelativeTolerance = 1.0E-12
        problem.SolversCreateFinish()

        # Create solver equations and add equations set to solver equations
        solver = iron.Solver()
        solver_equations = iron.SolverEquations()
        problem.SolverEquationsCreateStart()
        problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
        solver.SolverEquationsGet(solver_equations)
        #solver_equations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
        equationsSetIndex = solver_equations.EquationsSetAdd(equations_set)
        problem.SolverEquationsCreateFinish()
        
        # Prescribe boundary conditions.  
        boundary_conditions = iron.BoundaryConditions()
        solver_equations.BoundaryConditionsCreateStart(boundary_conditions)
        
#        bcs = { 
#            'CRANNODES': {'nodes': h5_dof_groups['/nodes/cranial'][()].T, 'derivatives': [[], [], [1]]},
#            'CAUDNODES': {'nodes': h5_dof_groups['/nodes/caudal'][()].T, 'derivatives': [[], [], [1]]},
#            'SKINNODES': {'nodes': h5_dof_groups['/nodes/skin'][()].T, 'derivatives': [[], [], []]},
#            'RIBNODES': {'nodes': h5_dof_groups['/nodes/chestwall'][()].T, 'derivatives': [[1], [1], [1]]},
#            'SHOULDER': {'nodes': h5_dof_groups['/nodes/fixed_shoulder'][()].T, 'derivatives': [[1], [1], [1]]}}
#        if side != 'both':
#            bcs['STERNUMNODES'] = {'nodes': h5_dof_groups['/nodes/sternum'][()].T, 'derivatives': [[], [1], [1]]}
#            bcs['AXILLANODES'] = {'nodes': h5_dof_groups['/nodes/spine'][()].T, 'derivatives': [[], [1], [1]]}
#        #if side == 'both':
#        #    bcs['FIXEDNODES'] = {'nodes': h5_dof_groups['/nodes/fixed'][()].T, 'derivatives': [[1], [1], [1]]}
#        
#        # Loop over each node group
#        for bcid, values in bcs.iteritems():
#            # Loop over the nodes in each node group.  
#            for node in values['nodes']:
#                #import ipdb; ipdb.set_trace()
#                #node_domain = decomposition.NodeDomainGet(node + offset, 1)
#                #if node_domain == computational_node_number:
#                    # Prescribe the boundary conditions for this node's 
#                    # derivatives based on those defined for this particular 
#                    # node group.  
#                for component, componentDerivatives in enumerate(values['derivatives']):
#                    for derivative in componentDerivatives:
#                        #print 'BC applied on node {0}, component {1}, derivative {2}'.format(
#                        #    node, component+1, derivative)
#                        boundary_conditions.AddNode(
#                            dependent_field, iron.FieldVariableTypes.U, 1,
#                            derivative, node + offset, component+1,
#                            iron.BoundaryConditionsTypes.FIXED, 0.0)


        deriv = 1
        version = 1

        stiffer_shoulder_nodes = h5_dof_groups['/nodes/stiffer_shoulder'][()].T
        for node in stiffer_shoulder_nodes:
            #import ipdb; ipdb.set_trace()
            boundary_conditions.AddNode(
                dependent_field, iron.FieldVariableTypes.U,
                version, deriv, node + offset, 1,
                iron.BoundaryConditionsTypes.FIXED, parameters[2])

        if side == 'both':
            stiffer_back_nodes = h5_dof_groups['/nodes/stiffer_back'][()].T
            for node in stiffer_back_nodes:
                boundary_conditions.AddNode(
                    dependent_field, iron.FieldVariableTypes.U,
                    version, deriv, node + offset, 1,
                    iron.BoundaryConditionsTypes.FIXED, parameters[3])

            anterior_nodes = h5_dof_groups['/nodes/all_anterior_nodes'][()].T
            for node in anterior_nodes:
                try:
                    boundary_conditions.AddNode(
                        dependent_field, iron.FieldVariableTypes.U,
                        version, deriv, node + offset, 1,
                        iron.BoundaryConditionsTypes.FIXED, parameters[0])
                except:
                    pass

            transition_nodes = h5_dof_groups['/nodes/all_transition_nodes'][()].T
            for node in transition_nodes:
                try:
                    boundary_conditions.AddNode(
                        dependent_field, iron.FieldVariableTypes.U,
                        version, deriv, node + offset, 1,
                        iron.BoundaryConditionsTypes.FIXED, parameters[4])
                except:
                    pass

        solver_equations.BoundaryConditionsCreateFinish()
        
        print ('  Solving Problem')
        try:
            problem.Solve()
        except:
            converged = False
            print ('    Problem did not converge')
        else:
            print ('    Problem converged')


        #material_field_interpolation = iron.FieldInterpolationTypes.ELEMENT_BASED
        material_field_interpolation = iron.FieldInterpolationTypes.NODE_BASED
        # Create the material field.
        material_field = iron.Field()
        material_field.CreateStart(material_field_user_num, region)
        material_field.MeshDecompositionSet(decomposition)
        material_field.TypeSet(iron.FieldTypes.MATERIAL)
        material_field.GeometricFieldSet(geometric_field)
        material_field.NumberOfVariablesSet(2)
        material_field.VariableTypesSet([iron.FieldVariableTypes.U,
                                          iron.FieldVariableTypes.V])
        material_field.NumberOfComponentsSet(iron.FieldVariableTypes.U, 2)
        material_field.NumberOfComponentsSet(iron.FieldVariableTypes.V, 1)
        material_field.ComponentInterpolationSet(iron.FieldVariableTypes.U, 1,
            material_field_interpolation)
        material_field.ComponentInterpolationSet(iron.FieldVariableTypes.U, 2,
            material_field_interpolation)
        material_field.ComponentInterpolationSet(iron.FieldVariableTypes.V, 1,
            material_field_interpolation)
        if material_field_interpolation == iron.FieldInterpolationTypes.NODE_BASED:
            material_field.ComponentMeshComponentSet(
                iron.FieldVariableTypes.U, 1, pressure_mesh_component)
            material_field.ComponentMeshComponentSet(
                iron.FieldVariableTypes.U, 2, pressure_mesh_component)
            material_field.ComponentMeshComponentSet(
                iron.FieldVariableTypes.V, 1, pressure_mesh_component)
        material_field.VariableLabelSet(iron.FieldVariableTypes.U, "Material")
        material_field.VariableLabelSet(iron.FieldVariableTypes.V, "Density")
        material_field.ScalingTypeSet(scaling_type)
        material_field.CreateFinish()

        for node in all_node_nums:
            value = dependent_field.ParameterSetGetNodeDP(
                iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES,
                1, 1, int(node), 1)
            try:
                material_field.ParameterSetUpdateNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, int(node), 1,
                    value)
                #print 'node: {0}, value: {1}'.format(node, value)
            except:
                pass

        save_stiffness_field(
            op.volunteer_id, op.results_dir, iron,
            material_field)

    if converged:
        # Export results.  
        fields = iron.Fields()
        fields.CreateRegion(region)
        fields.NodesExport("{0}/{1}".format(op.results_dir, field_export_name), "FORTRAN")
        fields.ElementsExport("{0}/{1}".format(op.results_dir, field_export_name), "FORTRAN")

        print ('  Stiffness field successfully calculated.')
    if 'problem' in locals():
        problem.Destroy()
        region.Destroy()


    return [converged]#, dependent_field, decomposition, region, cmesh, problem]

"""Solve single region breast model"""
def solve(mechanics_setup, op, problem_idx, parameters, previous_dependent_field=None,
    material_field=None, source_field=None, decomposition=None, mesh=None,
    cmesh=None, region=None, side='rhs', field_export_name='Field', 
    force_resolve=False):
    
    print ('Running problem {0}'.format(problem_idx))

    MESH_COMPONENT1 = 1
    MESH_COMPONENT2 = 2
    linear_elem_node_idxs = [0,3,12,15,48,51,60,63] # Linear element node idx in cubic element.
    offset = 1
    density = 1.0E-3 # in g mm^-3
    converged = True
    skin = False
    # load dof groups
    h5_dof_groups = h5py.File('{0}/dof_groups_{1}.h5'.format(op.results_dir, side), 'r')
    if skin:
        skin_elem = h5_dof_groups['/elements/skin'][()].T
    
    if previous_dependent_field is None:
        previous_dependent_field = iron.Field()

    number_of_load_increments = 1

    simulation_equation_set_types = (
        [iron.EquationsSetSubtypes.REFERENCE_STATE_MOONEY_RIVLIN,
         iron.EquationsSetSubtypes.MOONEY_RIVLIN])
    gravity_vectors = scipy.array([[-9.81,0.0, 0.0],
                        [9.81,0.0, 0.0]]) # in m s^-2
    gravity_vector = gravity_vectors[problem_idx]
    print ('  Parameters: ', parameters)
    print ('  Gravity vector: ', gravity_vector)
    
    geometric_mesh_component = 1
    pressure_mesh_component = 1
    if mechanics_setup['use_pressure_basis']:
        pressure_mesh_component = 2

    #scaling_type = iron.FieldScalingTypes.UNIT
    scaling_type = iron.FieldScalingTypes.NONE
    
    region_user_num = 1
    generated_mesh_user_num = 1
    mesh_user_num = 1
    decomposition_user_num = 1
    geometric_field_user_nums = [1,2]
    material_field_user_nums = [3,4]
    dependent_field_user_nums = [5,6]
    source_field_user_nums = [7,8]
    equations_set_field_user_nums = [9,10]
    equations_set_user_nums = [1,2]
    problem_user_num = 1
    
    coordinate_system = mechanics_setup['coordinate_system']
    basis = mechanics_setup['basis']
    pressure_basis = mechanics_setup['pressure_basis']

    # Get the number of computational nodes and this computational node 
    # number.  
    number_of_computational_nodes = iron.ComputationalNumberOfNodesGet()
    computational_node_number = iron.ComputationalNodeNumberGet()    

    all_node_nums = (scipy.array(mesh.get_node_ids()[1])+1).astype('int32')
    
    if problem_idx == 0:
        # Create a region and assign the coordinate system to the region.  
        region = iron.Region()
        region.CreateStart(region_user_num, iron.WorldRegion)
        region.LabelSet("Region")
        region.CoordinateSystemSet(coordinate_system)
        region.CreateFinish()

        #elem_nums = [11,22,33,44,55,66,77,88,99,110,121,132]
        #elem_nums = [11,22,33,44,55,66,77,88,99,110,121,132]
        # Create mesh topology
        cmesh = iron.Mesh()
        cmesh.CreateStart(mesh_user_num, region, 3)
        cmesh.NumberOfComponentsSet(2)
        cmesh.NumberOfElementsSet(mesh.elements.size())
        #cmesh.NumberOfElementsSet(len(elem_nums))
        nodes = iron.Nodes()
        nodes.CreateStart(region, mesh.nodes.size())
        nodes.AllUserNumbersSet(all_node_nums)
        nodes.CreateFinish()
        
        elements = iron.MeshElements()
        elements.CreateStart(cmesh, MESH_COMPONENT1, basis)
        #elements.AllUserNumbersSet(elem_nums)
        #elements.AllUserNumbersSet(iron_intStringToArray("4..11,15..22,26..33,37..44,48..55,59..66,70..77,81..88,92..99,103..110,114..121,125..132"))
        for element in mesh.elements:
            #if (element.id+ offset) not in [1,2,3,12,13,14,23,24,25,34,35,36,45,46,47,56,57,58,67,68,69,78,79,80,89,90,91,100,101,102,111,112,113,122,123,124]:
            #if (element.id+ offset) in elem_nums:
            elements.NodesSet(element.id + offset, scipy.array(element.node_ids, dtype='int32') + offset)
        elements.CreateFinish()
        
        if mechanics_setup['use_pressure_basis']:
            pressure_elements = iron.MeshElements()
            pressure_elements.CreateStart(cmesh, MESH_COMPONENT2, pressure_basis)
            #pressure_elements.AllUserNumbersSet(elem_nums)
            #pressure_elements.AllUserNumbersSet(iron_intStringToArray("4..11,15..22,26..33,37..44,48..55,59..66,70..77,81..88,92..99,103..110,114..121,125..132"))
            for element in mesh.elements:
                #if (element.id+ offset) not in [1,2,3,12,13,14,23,24,25,34,35,36,45,46,47,56,57,58,67,68,69,78,79,80,89,90,91,100,101,102,111,112,113,122,123,124]:
                #if (element.id+ offset) in elem_nums:
                elem_nodes = scipy.array(element.node_ids, dtype='int32') + offset
                pressure_elements.NodesSet(
                    element.id + offset, elem_nodes[linear_elem_node_idxs])
                if skin:
                    deriv = 1
                    version = 2
                    if element.id in skin_elem:
                        for elem_node in [1, 2, 3, 4]:
                            pressure_elements.LocalElementNodeVersionSet(
                                element.id + offset, version, deriv, elem_node)
            pressure_elements.CreateFinish()
        
        cmesh.CreateFinish()
        
        # Create a decomposition for the mesh.  
        decomposition = iron.Decomposition()
        decomposition.CreateStart(decomposition_user_num, cmesh)
        decomposition.type = iron.DecompositionTypes.CALCULATED
        decomposition.NumberOfDomainsSet(number_of_computational_nodes)
        decomposition.CreateFinish()
        
    simulation_equation_set_type = (
        simulation_equation_set_types[problem_idx])
    
    # Specify the identifier for the Openiron objects.  
    geometric_field_user_num = (
        geometric_field_user_nums[problem_idx])
    dependent_field_user_num = (
        dependent_field_user_nums[problem_idx])
    material_field_user_num = (
        material_field_user_nums[problem_idx])
    source_field_user_num = (
        source_field_user_nums[problem_idx])
    equations_set_field_user_num = (
        equations_set_field_user_nums[problem_idx])
    equations_set_user_num = (
        equations_set_user_nums[problem_idx])
    
    # Create a field for the geometry.  
    geometric_field = iron.Field()
    geometric_field.CreateStart(geometric_field_user_num, region)
    geometric_field.MeshDecompositionSet(decomposition)
    geometric_field.VariableLabelSet(iron.FieldVariableTypes.U,
                                     'Geometry'+str(problem_idx))
    geometric_field.ScalingTypeSet(scaling_type)
    geometric_field.CreateFinish()
    
    num_deriv = 1
    if problem_idx == 0:
        for node in mesh.nodes:
            #try:
                #node_domain = decomposition.NodeDomainGet(node.id + offset, 1)
                #if node_domain == computational_node_number:
            for comp_idx in range(3):
                for deriv_idx in range(num_deriv):
                    geometric_field.ParameterSetUpdateNodeDP(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, deriv_idx + 1, node.id + offset, comp_idx + 1,
                        node.values[comp_idx])
            #except:
            #    pass
    else:
        # Update the geometric field from the previously solved problem's 
        # dependent field.  
        for component in [1, 2, 3]:
            iron.Field.ParametersToFieldParametersComponentCopy(
                previous_dependent_field, iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES, component,
                geometric_field, iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES, component)

    geometric_field.ParameterSetUpdateStart(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    geometric_field.ParameterSetUpdateFinish(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)

#    import ipdb; ipdb.set_trace()
#    DXDNU = scipy.zeros((3,3))
#    DXDNU[:,0] = mesh.elements[0].evaluate([0.5,0.5,0.5],deriv=[1,0,0])
#    DXDNU[:,1] = mesh.elements[0].evaluate([0.5,0.5,0.5],deriv=[0,1,0])
#    DXDNU[:,2] = mesh.elements[0].evaluate([0.5,0.5,0.5],deriv=[0,0,1])
#    J = scipy.linalg.det(DXDNU.T)

#    geometric_field.ParameterSetInterpolateSingleXiDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,1,[0.5,0.5,0.5], 3)

    # Export results.  
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("{0}/{1}".format(op.results_dir,field_export_name), "FORTRAN")
    fields.ElementsExport("{0}/{1}".format(op.results_dir,field_export_name), "FORTRAN")
    fields.Finalise()

    # Create the dependent field.  
    dependent_field = iron.Field()
    dependent_field.CreateStart(dependent_field_user_num, region)
    dependent_field.MeshDecompositionSet(decomposition)
    dependent_field.TypeSet(iron.FieldTypes.GEOMETRIC_GENERAL)
    dependent_field.DependentTypeSet(iron.FieldDependentTypes.DEPENDENT)
    dependent_field.GeometricFieldSet(geometric_field)
    dependent_field.NumberOfVariablesSet(2)
    dependent_field.NumberOfComponentsSet(
        iron.FieldVariableTypes.U, 4)
    dependent_field.NumberOfComponentsSet(
        iron.FieldVariableTypes.DELUDELN, 4)
    dependent_field.VariableLabelSet(iron.FieldVariableTypes.U,
                                     'Dependent'+str(problem_idx))
    dependent_field.VariableLabelSet(iron.FieldVariableTypes.DELUDELN,
                                     'DependentDelUDelN'+str(problem_idx))
    for component in [1, 2, 3]:
        dependent_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.U, component, geometric_mesh_component)
        dependent_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.DELUDELN, component, geometric_mesh_component)
    dependent_field.ComponentMeshComponentSet(
        iron.FieldVariableTypes.U, 4, pressure_mesh_component)
    dependent_field.ComponentMeshComponentSet(
        iron.FieldVariableTypes.DELUDELN, 4, pressure_mesh_component)
    # Set the pressure to be nodally based and use the second mesh component.  
    if mechanics_setup['use_pressure_basis']:
        dependent_field.ComponentInterpolationSet(
            iron.FieldVariableTypes.U, 4, 
            iron.FieldInterpolationTypes.NODE_BASED)
        dependent_field.ComponentInterpolationSet(
            iron.FieldVariableTypes.DELUDELN, 4, 
            iron.FieldInterpolationTypes.NODE_BASED)
    else:
        dependent_field.ComponentInterpolationSet(
            iron.FieldVariableTypes.U, 4, 
            iron.FieldInterpolationTypes.ELEMENT_BASED)
        dependent_field.ComponentInterpolationSet(
            iron.FieldVariableTypes.DELUDELN, 4, 
            iron.FieldInterpolationTypes.ELEMENT_BASED)
    dependent_field.ScalingTypeSet(scaling_type)
    dependent_field.CreateFinish()

    #material_field_interpolation = iron.FieldInterpolationTypes.ELEMENT_BASED
    material_field_interpolation = iron.FieldInterpolationTypes.NODE_BASED
    # Create the material field.
    material_field = iron.Field()
    material_field.CreateStart(material_field_user_num, region)
    material_field.MeshDecompositionSet(decomposition)
    material_field.TypeSet(iron.FieldTypes.MATERIAL)
    material_field.GeometricFieldSet(geometric_field)
    material_field.NumberOfVariablesSet(2)
    material_field.VariableTypesSet([iron.FieldVariableTypes.U,
                                      iron.FieldVariableTypes.V])
    material_field.NumberOfComponentsSet(iron.FieldVariableTypes.U, 2)
    material_field.NumberOfComponentsSet(iron.FieldVariableTypes.V, 1)
    material_field.ComponentInterpolationSet(iron.FieldVariableTypes.U, 1,
        material_field_interpolation)
    material_field.ComponentInterpolationSet(iron.FieldVariableTypes.U, 2,
        material_field_interpolation)
    material_field.ComponentInterpolationSet(iron.FieldVariableTypes.V, 1,
        material_field_interpolation)
    if material_field_interpolation == iron.FieldInterpolationTypes.NODE_BASED:
        material_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.U, 1, pressure_mesh_component)
        material_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.U, 2, pressure_mesh_component)
        material_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.V, 1, pressure_mesh_component)
    material_field.VariableLabelSet(iron.FieldVariableTypes.U, "Material")
    material_field.VariableLabelSet(iron.FieldVariableTypes.V, "Density")
    material_field.ScalingTypeSet(scaling_type)
    material_field.CreateFinish()
    
    # Set Mooney-Rivlin constants c10 and c01 respectively.  
    load_stiffness_field(op.volunteer_id, op.results_dir, material_field)

    for node in all_node_nums:
        try:
            value = material_field.ParameterSetGetNodeDP(
                iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES,
                1, 1, int(node), 1)
        except:
            pass
        else:
            material_field.ParameterSetUpdateNodeDP(
                iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES,
                1, 1, int(node), 1,
                value*parameters[0])
            #print 'node: {0}, value: {1}'.format(node, value)


    #material_field.ComponentValuesInitialiseDP(
    #    iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1,
    #    parameters[0])
    material_field.ComponentValuesInitialiseDP(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 2,
        0.0)
    material_field.ComponentValuesInitialiseDP(
        iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, 1,
        density)
#    
#    deriv = 1
#    version = 1
#    stiffer_shoulder_nodes = h5_dof_groups['/nodes/stiffer_shoulder'][()].T
#    for node in stiffer_shoulder_nodes:
#        material_field.ParameterSetUpdateNodeDP(
#            iron.FieldVariableTypes.U,
#            iron.FieldParameterSetTypes.VALUES,
#            version, deriv, node + offset, 1, parameters[2])


#    if side == 'both':
#        stiffer_back_nodes = h5_dof_groups['/nodes/stiffer_back'][()].T
#        deriv = 1
#        version = 1
#        for node in stiffer_back_nodes:
#            material_field.ParameterSetUpdateNodeDP(
#                iron.FieldVariableTypes.U,
#                iron.FieldParameterSetTypes.VALUES,
#                version, deriv, node + offset, 1, parameters[3])

#        transitional_nodes = h5_dof_groups['/nodes/transitional'][()].T
#        deriv = 1
#        version = 1
#        for node in transitional_nodes:
#            material_field.ParameterSetUpdateNodeDP(
#                iron.FieldVariableTypes.U,
#                iron.FieldParameterSetTypes.VALUES,
#                version, deriv, node + offset, 1, parameters[4])
    
    # Update skin material properties
    if skin:
        for elem in skin_elem:
            material_field.ParameterSetUpdateElementDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                elem, 1, parameters[1])
    
    material_field.ParameterSetUpdateStart(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    material_field.ParameterSetUpdateFinish(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    material_field.ParameterSetUpdateStart(
        iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES)
    material_field.ParameterSetUpdateFinish(
        iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES)

    solved = load_solutions(
       op.volunteer_id, problem_idx, op.results_dir, parameters, gravity_vector, iron,
       [dependent_field], distance_upperbound=100., force_resolve=force_resolve)

    if not solved:
        # Initialise dependent field from geometry and set hydrostatic pressure.  
        for component in [1, 2, 3]:
            iron.Field.ParametersToFieldParametersComponentCopy(
                geometric_field, iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES, component,
                dependent_field, iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES, component)
#        iron.Field.ComponentValuesInitialiseDP(
#            dependent_field, iron.FieldVariableTypes.U,
#            iron.FieldParameterSetTypes.VALUES, 4, -parameters[0])

#        deriv = 1
#        version = 1
#        for node in stiffer_shoulder_nodes:
#            dependent_field.ParameterSetUpdateNodeDP(
#                iron.FieldVariableTypes.U,
#                iron.FieldParameterSetTypes.VALUES,
#                version, deriv, node + offset, 4, -parameters[2])

#        if side == 'both':
#            deriv = 1
#            version = 1
#            for node in stiffer_back_nodes:
#                dependent_field.ParameterSetUpdateNodeDP(
#                    iron.FieldVariableTypes.U,
#                    iron.FieldParameterSetTypes.VALUES,
#                    version, deriv, node + offset, 4, -parameters[3])
#            deriv = 1
#            version = 1
#            for node in transitional_nodes:
#                dependent_field.ParameterSetUpdateNodeDP(
#                    iron.FieldVariableTypes.U,
#                    iron.FieldParameterSetTypes.VALUES,
#                    version, deriv, node + offset, 4, -parameters[4])

        if side == 'both':
            for node in all_node_nums:
                try:
                    value = material_field.ParameterSetGetNodeDP(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, 1, int(node), 1)
                except:
                    pass
                else:
                    dependent_field.ParameterSetUpdateNodeDP(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, 1, int(node), 4,
                        -value*parameters[0])

        if skin:
            # Specify skin material properties independently of the breast tissue
            deriv = 1
            for element in mesh.elements:
                if element.id in skin_elem:
                    elem_nodes = scipy.array(
                        element.node_ids, dtype='int32')[linear_elem_node_idxs]
                    version = 2
                    for node in elem_nodes[0:4]:
                        dependent_field.ParameterSetUpdateNodeDP(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES,
                            version, deriv, node + offset, 4, -parameters[1])
                    
                    version = 1
                    for node in elem_nodes[4:8]:
                        dependent_field.ParameterSetUpdateNodeDP(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES,
                            version, deriv, node + offset, 4, -parameters[1])
        
        iron.Field.ParameterSetUpdateStart(
            dependent_field, iron.FieldVariableTypes.U,
            iron.FieldParameterSetTypes.VALUES)
        iron.Field.ParameterSetUpdateFinish(
            dependent_field, iron.FieldVariableTypes.U,
            iron.FieldParameterSetTypes.VALUES)
        
        # Create the source field with the gravity vector.
        source_field = iron.Field()
        source_field.CreateStart(source_field_user_num, region)
        source_field.MeshDecompositionSet(decomposition)
        source_field.TypeSet(iron.FieldTypes.GENERAL)
        source_field.GeometricFieldSet(geometric_field)
        for component in [1, 2, 3]:
            source_field.ComponentInterpolationSet(
                iron.FieldVariableTypes.U, component,
                iron.FieldInterpolationTypes.ELEMENT_BASED)
        source_field.ScalingTypeSet(scaling_type)
        source_field.CreateFinish()
        
        # Set the gravity vector component values.
        for component in [1, 2, 3]:
            source_field.ComponentValuesInitialiseDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                component, gravity_vector[component-1])
        
        source_field.ParameterSetUpdateStart(iron.FieldVariableTypes.U,
                                             iron.FieldParameterSetTypes.VALUES)
        source_field.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES)
        
        # Create the equations_set.  
        equations_set_field = iron.Field()
        equations_set = iron.EquationsSet()
        equations_set.CreateStart(
            equations_set_user_num, region, geometric_field,
            iron.EquationsSetClasses.ELASTICITY,
            iron.EquationsSetTypes.FINITE_ELASTICITY,
            simulation_equation_set_type, equations_set_field_user_num,
            equations_set_field)
        equations_set.CreateFinish()
        
        equations_set.DependentCreateStart(dependent_field_user_num, dependent_field)
        equations_set.DependentCreateFinish()
        
        equations_set.MaterialsCreateStart(material_field_user_num, material_field)
        equations_set.MaterialsCreateFinish()
        
        equations_set.SourceCreateStart(source_field_user_num, source_field)
        equations_set.SourceCreateFinish()
        
        # Create equations.  
        equations = iron.Equations()
        equations_set.EquationsCreateStart(equations)
        equations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
        equations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
        equations_set.EquationsCreateFinish()
        
        # Define the problem.  
        problem = iron.Problem()
        problem.CreateStart(problem_idx)
        problem.SpecificationSet(iron.ProblemClasses.ELASTICITY,
                                 iron.ProblemTypes.FINITE_ELASTICITY,
                                 iron.ProblemSubTypes.NONE)
        problem.CreateFinish()
        
        # Create the problem control loop.  
        problem.ControlLoopCreateStart()
        control_loop = iron.ControlLoop()
        problem.ControlLoopGet([iron.ControlLoopIdentifiers.NODE], control_loop)
        control_loop.MaximumIterationsSet(number_of_load_increments)
        problem.ControlLoopCreateFinish()
        
        # Create problem solver.  
        nonlinear_solver = iron.Solver()
        linear_solver = iron.Solver()
        problem.SolversCreateStart()
        problem.SolverGet([iron.ControlLoopIdentifiers.NODE], 1, nonlinear_solver)
        nonlinear_solver.OutputTypeSet(iron.SolverOutputTypes.MATRIX)
        nonlinear_solver.OutputTypeSet(iron.SolverOutputTypes.PROGRESS)
        nonlinear_solver.NewtonJacobianCalculationTypeSet(
            iron.JacobianCalculationTypes.EQUATIONS)
        nonlinear_solver.NewtonLinearSolverGet(linear_solver)
        nonlinear_solver.NewtonAbsoluteToleranceSet(1.0E-10)
        nonlinear_solver.NewtonRelativeToleranceSet(1.0E-10)
        nonlinear_solver.NewtonSolutionToleranceSet(1.0E-10)
        linear_solver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
        #linear_solver.LibraryTypeSet(iron.SolverLibraries.SUPERLU)
        problem.SolversCreateFinish()
        
        # Create solver equations and add equations set to solver equations.  
        solver = iron.Solver()
        solver_equations = iron.SolverEquations()
        problem.SolverEquationsCreateStart()
        problem.SolverGet([iron.ControlLoopIdentifiers.NODE], 1, solver)
        solver.SolverEquationsGet(solver_equations)
        solver_equations.SparsityTypeSet(iron.SolverEquationsSparsityTypes.SPARSE)
        equations_set1_index = solver_equations.EquationsSetAdd(equations_set)
        problem.SolverEquationsCreateFinish()
        
        # Prescribe boundary conditions.  
        boundary_conditions = iron.BoundaryConditions()
        solver_equations.BoundaryConditionsCreateStart(boundary_conditions)
        
        bcs = { 
            'CRANNODES': {'nodes': h5_dof_groups['/nodes/cranial'][()].T, 'derivatives': [[], [], [1]]},
            'CAUDNODES': {'nodes': h5_dof_groups['/nodes/caudal'][()].T, 'derivatives': [[], [], [1]]},
            'SKINNODES': {'nodes': h5_dof_groups['/nodes/skin'][()].T, 'derivatives': [[], [], []]},
            'RIBNODES': {'nodes': h5_dof_groups['/nodes/chestwall'][()].T, 'derivatives': [[1], [1], [1]]},
            'SHOULDER': {'nodes': h5_dof_groups['/nodes/fixed_shoulder'][()].T, 'derivatives': [[1], [1], [1]]}}
        if side != 'both':
            bcs['STERNUMNODES'] = {'nodes': h5_dof_groups['/nodes/sternum'][()].T, 'derivatives': [[], [1], [1]]}
            bcs['AXILLANODES'] = {'nodes': h5_dof_groups['/nodes/spine'][()].T, 'derivatives': [[], [1], [1]]}
        #if side == 'both':
        #    bcs['FIXEDNODES'] = {'nodes': h5_dof_groups['/nodes/fixed'][()].T, 'derivatives': [[1], [1], [1]]}
        
        # Loop over each node group
        for bcid, values in bcs.iteritems():
            # Loop over the nodes in each node group.  
            for node in values['nodes']:
                #import ipdb; ipdb.set_trace()
                #node_domain = decomposition.NodeDomainGet(node + offset, 1)
                #if node_domain == computational_node_number:
                    # Prescribe the boundary conditions for this node's 
                    # derivatives based on those defined for this particular 
                    # node group.  
                for component, componentDerivatives in enumerate(values['derivatives']):
                    for derivative in componentDerivatives:
                        #print 'BC applied on node {0}, component {1}, derivative {2}'.format(
                        #    node, component+1, derivative)
                        boundary_conditions.AddNode(
                            dependent_field, iron.FieldVariableTypes.U, 1,
                            derivative, node + offset, component+1,
                            iron.BoundaryConditionsTypes.FIXED, 0.0)

        solver_equations.BoundaryConditionsCreateFinish()
        
        print ('  Solving Problem')
        try:
            problem.Solve()
        except:
            converged = False
            print ('    Problem did not converge')
        else:
            print ('    Problem converged')

        save_solutions(
            op.volunteer_id, op.results_dir, problem_idx, parameters, gravity_vector, iron,
            [dependent_field])

    if converged:
        # Export results.  
        fields = iron.Fields()
        fields.CreateRegion(region)
        fields.NodesExport("{0}/{1}_{2}".format(op.results_dir, field_export_name, problem_idx), "FORTRAN")
        fields.ElementsExport("{0}/{1}_{2}".format(op.results_dir, field_export_name, problem_idx), "FORTRAN")
        
        print ('  Problem {0} successfully solved.'.format(problem_idx))
    if 'problem' not in locals():
        problem = None

    return [converged, dependent_field, decomposition, region, cmesh, problem]

def setup_mechanics_problem():
    # Create a 3D rectangular cartesian coordinate system.  
    coordinate_system_user_num = 1
    coordinate_system = iron.CoordinateSystem()
    coordinate_system.CreateStart(coordinate_system_user_num)
    coordinate_system.DimensionSet(3)
    coordinate_system.CreateFinish()

    # Create basis functions
    number_of_xi = 3
    number_of_Gauss_xi = 4

    # Define displacement basis.  
    #interpolation_type = iron.BasisInterpolationSpecifications.CUBIC_HERMITE
    interpolation_type = iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE
    # If the dependent field's hydrostatic pressure component is required to use
    # greater than zero'th order interpolation, then a second (pressure) basis
    # needs to be defined. If not then the existing basis for interpolating the
    # geometric field can be used for interpolating the hydrostatic pressure
    # component of the dependent field but the interpolation type of the
    # pressure component will need to be set to
    # iron.FieldInterpolationTypes.ELEMENT_BASED.  
    displacement_basis_user_num = 1
    basis = iron.Basis()
    basis.CreateStart(displacement_basis_user_num)
    if interpolation_type in (1, 2, 3, 4):
        basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    elif interpolation_type in (7, 8, 9):
        basis.TypeSet(iron.BasisTypes.SIMPLEX)
    basis.NumberOfXiSet(number_of_xi)
    basis.InterpolationXiSet([interpolation_type]*number_of_xi)
    if(number_of_Gauss_xi>0):
        basis.QuadratureNumberOfGaussXiSet([number_of_Gauss_xi]*number_of_xi)
    basis.CreateFinish()
    
    use_pressure_basis = True
    if(use_pressure_basis):
        # Define pressure basis.  
        pressure_interpolation_type = (
            iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE)
        pressure_basis_user_num = 2
        pressure_basis = iron.Basis()
        pressure_basis.CreateStart(pressure_basis_user_num)
        if interpolation_type in (1, 2, 3, 4):
            pressure_basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
        elif interpolation_type in (7, 8, 9):
            pressure_basis.TypeSet(iron.BasisTypes.SIMPLEX)
        pressure_basis.NumberOfXiSet(number_of_xi)
        pressure_basis.InterpolationXiSet(
            [pressure_interpolation_type]*number_of_xi)
        if(number_of_Gauss_xi>0):
            pressure_basis.QuadratureNumberOfGaussXiSet([number_of_Gauss_xi]*
                                                          number_of_xi)
        pressure_basis.CreateFinish()

    setup = {
        'coordinate_system' : coordinate_system,
        'basis': basis,
        'pressure_basis': pressure_basis,
        'use_pressure_basis':use_pressure_basis}
    return setup

def destroy_regions(regions, problems):
    for region in regions:
        region.Destroy()
    for problem in problems:
        if problem is not None:
              problem.Destroy()

"""Export Openiron mesh"""
def export_Openiron_mesh(op, mesh=None, field_export_name='Field'):
    
    MESH_COMPONENT1 = 1
    offset = 1
    
    #interpolation_type = iron.BasisInterpolationSpecifications.CUBIC_HERMITE
    interpolation_type = iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE
    # If the dependent field's hydrostatic pressure component is required to use
    # greater than zero'th order interpolation, then a second (pressure) basis
    # needs to be defined. If not then the existing basis for interpolating the
    # geometric field can be used for interpolating the hydrostatic pressure
    # component of the dependent field but the interpolation type of the
    # pressure component will need to be set to
    # iron.FieldInterpolationTypes.ELEMENT_BASED.  
    number_of_xi = 3
    number_of_Gauss_xi = 4
    geometric_mesh_component = 1
    pressure_mesh_component = 1
    #scaling_type = iron.FieldScalingTypes.UNIT
    scaling_type = iron.FieldScalingTypes.NONE
    
    coordinate_system_user_num = 1
    region_user_num = 1
    displacement_basis_user_num = 1
    generated_mesh_user_num = 1
    mesh_user_num = 1
    decomposition_user_num = 1
    geometric_field_user_num = 1
    
    # Get the number of computational nodes and this computational node 
    # number.  
    number_of_computational_nodes = iron.ComputationalNumberOfNodesGet()
    computational_node_number = iron.ComputationalNodeNumberGet()    

    #iron.DiagnosticsSetOn(iron.DiagnosticTypes.IN,[1,2,3,4,5],"Diagnostics",["DECOMPOSITION_TOPOLOGY_LINES_CALCULATE"])

    # Create a 3D rectangular cartesian coordinate system.  
    coordinate_system = iron.CoordinateSystem()
    coordinate_system.CreateStart(coordinate_system_user_num)
    coordinate_system.DimensionSet(3)
    coordinate_system.CreateFinish()
    
    # Create a region and assign the coordinate system to the region.  
    region = iron.Region()
    region.CreateStart(region_user_num, iron.WorldRegion)
    region.LabelSet("Region")
    region.CoordinateSystemSet(coordinate_system)
    region.CreateFinish()

    # Define basis.  
    basis = iron.Basis()
    basis.CreateStart(displacement_basis_user_num)
    if interpolation_type in (1, 2, 3, 4):
        basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    elif interpolation_type in (7, 8, 9):
        basis.TypeSet(iron.BasisTypes.SIMPLEX)
    basis.NumberOfXiSet(number_of_xi)
    basis.InterpolationXiSet([interpolation_type]*number_of_xi)
    if(number_of_Gauss_xi>0):
        basis.QuadratureNumberOfGaussXiSet([number_of_Gauss_xi]*number_of_xi)
    basis.CreateFinish()
    
    # Create mesh topology
    cmesh = iron.Mesh()
    cmesh.CreateStart(mesh_user_num, region, 3)
    cmesh.NumberOfComponentsSet(1)
    cmesh.NumberOfElementsSet(mesh.elements.size())
    nodes = iron.Nodes()
    nodes.CreateStart(region, mesh.nodes.size())
    nodes.AllUserNumbersSet((scipy.array(mesh.get_node_ids()[1])+1).astype('int32'))
    nodes.CreateFinish()
    
    elements = iron.MeshElements()
    elements.CreateStart(cmesh, MESH_COMPONENT1, basis)
    for element in mesh.elements:
        elements.NodesSet(element.id + offset, scipy.array(element.node_ids, dtype='int32') + offset)
    elements.CreateFinish()
        
    cmesh.CreateFinish()
    
    # Create a decomposition for the mesh.  
    decomposition = iron.Decomposition()
    decomposition.CreateStart(decomposition_user_num, cmesh)
    decomposition.type = iron.DecompositionTypes.CALCULATED
    decomposition.NumberOfDomainsSet(number_of_computational_nodes)
    decomposition.CreateFinish()
    
    # Create a field for the geometry.  
    geometric_field = iron.Field()
    geometric_field.CreateStart(geometric_field_user_num, region)
    geometric_field.MeshDecompositionSet(decomposition)
    geometric_field.VariableLabelSet(iron.FieldVariableTypes.U,
                                     'Geometry')
    geometric_field.ScalingTypeSet(scaling_type)
    geometric_field.CreateFinish()
    
    num_deriv = 1
    for node in mesh.nodes:
        for comp_idx in range(3):
            for deriv_idx in range(num_deriv):
                geometric_field.ParameterSetUpdateNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, deriv_idx + 1, node.id + offset, comp_idx + 1,
                    node.values[comp_idx])

    geometric_field.ParameterSetUpdateStart(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    geometric_field.ParameterSetUpdateFinish(
        iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)

    # Export results.  
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("{0}/{1}".format(op.results_dir,field_export_name), "FORTRAN")
    fields.ElementsExport("{0}/{1}".format(op.results_dir,field_export_name), "FORTRAN")
    fields.Finalise()

def Openiron_mesh_to_morphic(mesh,dependent_field, offset=1, num_deriv=1):
    for node in mesh.nodes:
        for comp_idx in range(3):
            for deriv_idx in range(num_deriv):
                node.values[comp_idx] = dependent_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, deriv_idx + 1, node.id + offset, comp_idx + 1)
    return mesh
