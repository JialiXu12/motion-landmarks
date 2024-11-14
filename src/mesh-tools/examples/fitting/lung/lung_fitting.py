#!/usr/bin/env python

import sys, os
import mesh_tools
import numpy
import math

# Intialise OpenCMISS
from opencmiss.iron import iron


# defining the output file to be written in the ExDataFile
def writeExdataFile(filename, dataPointLocations, dataErrorVector,
                    dataErrorDistance, offset):
    "Writes data points to an exdata file"

    numberOfDimensions = dataPointLocations[1].shape[0]
    try:
        f = open(filename, "w")
        if numberOfDimensions == 1:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components=''' + str(
                numberOfDimensions) + '''
  x.  Value index=1, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components=''' + str(
                numberOfDimensions) + '''
  x.  Value index=2, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=3, #Derivatives=0, #Versions=1
'''
        elif numberOfDimensions == 2:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components=''' + str(
                numberOfDimensions) + '''
  x.  Value index=1, #Derivatives=0, #Versions=1
  y.  Value index=2, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components=''' + str(
                numberOfDimensions) + '''
  x.  Value index=3, #Derivatives=0, #Versions=1
  y.  Value index=4, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=5, #Derivatives=0, #Versions=1
'''
        elif numberOfDimensions == 3:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components=''' + str(
                numberOfDimensions) + '''
  x.  Value index=1, #Derivatives=0, #Versions=1
  y.  Value index=2, #Derivatives=0, #Versions=1
  x.  Value index=3, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components=''' + str(
                numberOfDimensions) + '''
  x.  Value index=4, #Derivatives=0, #Versions=1
  y.  Value index=5, #Derivatives=0, #Versions=1
  z.  Value index=6, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=7, #Derivatives=0, #Versions=1
'''
        f.write(header)

        numberOfDataPoints = len(dataPointLocations)
        for i in range(numberOfDataPoints):
            line = " Node: " + str(offset + i + 1) + '\n'
            f.write(line)
            for j in range(numberOfDimensions):
                line = ' ' + str(dataPointLocations[i, j]) + '\t'
                f.write(line)
            line = '\n'
            f.write(line)
            for j in range(numberOfDimensions):
                line = ' ' + str(dataErrorVector[i, j]) + '\t'
                f.write(line)
            line = '\n'
            f.write(line)
            line = ' ' + str(dataErrorDistance[i])
            f.write(line)
            line = '\n'
            f.write(line)
        f.close()

    except IOError:
        print('Could not open file: ' + filename)


if __name__ == '__main__':


    # Set Sobolev smoothing parameters
    tau = 0.01
    kappa = 0.005

    # iteratively fit
    iterationNumber = 1

    numberOfGaussXi = 3

    ZERO_TOLERANCE = 0.00001

    maximumProjectionDistance = 20.0

    exnodeFilename = "FittedGeometry_" + str(iterationNumber - 1) + ".part0.exnode"
    exelemFilename = "FittedGeometry_" + str(iterationNumber - 1) + ".part0.exelem"
    exdataFilename = "myo.exdata"

    # =================================================================

    (coordinateSystemUserNumber,
     regionUserNumber,
     basisUserNumber,
     generatedMeshUserNumber,
     meshUserNumber,
     decompositionUserNumber,
     geometricFieldUserNumber,
     equationsSetFieldUserNumber,
     dependentFieldUserNumber,
     independentFieldUserNumber,
     dataPointUserNumber,
     dataPointFieldUserNumber,
     materialFieldUserNumber,
     analyticFieldUserNumber,
     dependentDataFieldUserNumber,
     dataProjectionUserNumber,
     equationsSetUserNumber,
     problemUserNumber) = range(1, 19)

    import numpy as np
    node_file = open('nodes.txt', 'r')
    node_text = node_file.read().splitlines()
    node_nums = np.array(node_text[0::2], dtype=int)
    node_coords_text = node_text[1::2]
    num_nodes = len(node_nums)
    node_coords = np.zeros((num_nodes, 3))
    for node in range(num_nodes):
        node_coords[node, :] = node_coords_text[node].split()

    node_file = open('elements.txt', 'r')
    elem_text = node_file.read().splitlines()
    elem_nums = np.array(elem_text[0::2], dtype=int)
    elem_coords_text = elem_text[1::2]
    num_elems = len(elem_nums)
    elem_nodes = np.zeros((num_elems, 9))
    for elem in range(num_elems):
        elem_nodes[node, :] = elem_coords_text[node].split()


    #exnodes = exfile.ReadExnode(exnodeFilename)
    #print("Number of nodes = ", exnodes.num_nodes)
    #exelems = exfile.ReadExelem(exelemFilename)
    #print("Number of elements = ", exelems.num_elements)
    #exdata = exfile.ReadExdata(exdataFilename)
    #print("Number of data points = ", exdata.num_dataPoints)

    # Get the number of computational nodes and this computational node number
    numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
    computationalNodeNumber = iron.ComputationalNodeNumberGet()

    # Create a RC coordinate system
    print("Creating coordinate system")
    coordinateSystem = iron.CoordinateSystem()
    coordinateSystem.CreateStart(coordinateSystemUserNumber)
    coordinateSystem.DimensionSet(3)
    coordinateSystem.CreateFinish()

    # Create a region
    print("Creating region")
    region = iron.Region()
    region.CreateStart(regionUserNumber, iron.WorldRegion)
    region.LabelSet("FittingRegion")
    region.CoordinateSystemSet(coordinateSystem)
    region.CreateFinish()

    # Create a tricubic Hermite basis
    print("Creating basis functions")
    basis = iron.Basis()
    basis.CreateStart(basisUserNumber)
    basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    basis.NumberOfXiSet(3)
    basis.InterpolationXiSet(
        [iron.BasisInterpolationSpecifications.CUBIC_HERMITE] * 3)
    basis.QuadratureNumberOfGaussXiSet([numberOfGaussXi] * 3)
    basis.CreateFinish()

    # Define nodes for the mesh
    print("Creating nodes")
    nodes = iron.Nodes()
    nodes.CreateStart(region, exnodes.num_nodes)
    nodes.CreateFinish()

    # Create the mesh
    print("Creating mesh")
    mesh = iron.Mesh()
    mesh.CreateStart(meshUserNumber, region, 3)
    mesh.NumberOfComponentsSet(1)
    mesh.NumberOfElementsSet(exelems.num_elements)
    elements = iron.MeshElements()
    meshComponentNumber = 1
    elements.CreateStart(mesh, meshComponentNumber, basis)
    for element in exelems.elements:
        elements.NodesSet(element.number, element.nodes)
    elements.CreateFinish()
    mesh.CreateFinish()

    # Create a decomposition for the mesh
    print("Decomposing mesh")
    decomposition = iron.Decomposition()
    decomposition.CreateStart(decompositionUserNumber, mesh)
    decomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    decomposition.CalculateFacesSet(True)
    decomposition.CreateFinish()

    # =================================================================
    # Geometric Field
    # =================================================================

    # Create a field for the geometry
    print("Creating geometric field")
    geometricField = iron.Field()
    geometricField.CreateStart(geometricFieldUserNumber, region)
    geometricField.MeshDecompositionSet(decomposition)
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 1, 1)
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 2, 1)
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 3, 1)
    geometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
    # geometricField.ScalingTypeSet(iron.FieldScalingTypes.UNIT)
    geometricField.CreateFinish()

    # Set the geometric field dofs
    for componentIdx in range(1, 4):
        componentName = ["x", "y", "z"][componentIdx - 1]
        for nodeIdx in range(1, exnodes.num_nodes + 1):
            nodeDomain = decomposition.NodeDomainGet(nodeIdx, 1)
            if (nodeDomain == computationalNodeNumber):
                for derivativeIdx in range(1, 9):
                    if (iterationNumber == 1):
                        value = exnodes.node_value("Coordinate", componentName,
                                                   nodeIdx, derivativeIdx)
                    else:
                        value = exnodes.node_value("Dependent", componentName,
                                                   nodeIdx, derivativeIdx)
                    geometricField.ParameterSetUpdateNode(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, derivativeIdx, nodeIdx, componentIdx, value)

    # Normalise derivatives
    derivativeVector = [0.0, 0.0, 0.0, 0.0]
    for nodeIdx in range(1, exnodes.num_nodes + 1):
        nodeDomain = decomposition.NodeDomainGet(nodeIdx, 1)
        if (nodeDomain == computationalNodeNumber):
            for derivativeIdx in [iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,
                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,
                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3]:
                length = 0.0
                for componentIdx in range(1, 4):
                    derivativeVector[
                        componentIdx] = geometricField.ParameterSetGetNode(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, derivativeIdx, nodeIdx, componentIdx)
                    length = length + derivativeVector[componentIdx] * \
                             derivativeVector[componentIdx]
                if (length > ZERO_TOLERANCE):
                    length = math.sqrt(length)
                    for componentIdx in range(1, 4):
                        value = derivativeVector[componentIdx] / length
                        geometricField.ParameterSetUpdateNode(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES,
                            1, derivativeIdx, nodeIdx, componentIdx, value)
                else:
                    derivativeVector[
                        1] = -1.0 * geometricField.ParameterSetGetNode(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, 1, nodeIdx, 1)
                    derivativeVector[
                        2] = -1.0 * geometricField.ParameterSetGetNode(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, 1, nodeIdx, 2)
                    derivativeVector[3] = 0.0
                    length = derivativeVector[1] * derivativeVector[1] + \
                             derivativeVector[2] * derivativeVector[2]
                    length = math.sqrt(length)
                    for componentIdx in range(1, 4):
                        value = derivativeVector[componentIdx] / length
                        geometricField.ParameterSetUpdateNode(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES,
                            1, derivativeIdx, nodeIdx, componentIdx, value)
            if (iterationNumber == 1):
                for derivativeIdx in [
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S3,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2_S3,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3]:
                    for componentIdx in range(1, 4):
                        geometricField.ParameterSetUpdateNode(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES,
                            1, derivativeIdx, nodeIdx, componentIdx, 0.0)

    geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,
                                           iron.FieldParameterSetTypes.VALUES)
    geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,
                                            iron.FieldParameterSetTypes.VALUES)

    # uncomment if you want to get elements volume

    # ~ for elementIdx in range(1,exelems.num_elements+1):
    # ~ volume = geometricField.GeometricParametersElementVolumeGet(elementIdx)
    # ~ print('Element ID = ', elementIdx, ' Volume = ', volume)

    # ~ sys.exit()

    # =================================================================
    # Data Points
    # =================================================================

    # Create the data points
    print("Creating data points")
    dataPoints = iron.DataPoints()
    dataPoints.CreateStart(dataPointUserNumber, region, exdata.num_dataPoints)

    # Set up CMISS data points with geometric values
    dataPointLocations = numpy.zeros((exdata.num_dataPoints, 3))
    for dataPointIdx in range(1, exdata.num_dataPoints + 1):
        dataPoint = [0.0] * 3
        for componentIdx in range(1, 4):
            componentName = ["x", "y", "z"][componentIdx - 1]
            dataPoint[componentIdx - 1] = exdata.data_value("Coordinate",
                                                            componentName,
                                                            dataPointIdx, 1)
            dataPointLocations[dataPointIdx - 1][componentIdx - 1] = dataPoint[
                componentIdx - 1]
            dataPoints.PositionSet(dataPointIdx, dataPoint)

    dataPoints.CreateFinish()

    # =================================================================
    # Data Projection on Geometric Field
    # =================================================================

    print("Projecting data points onto geometric field")
    # Set up data projection
    dataProjection = iron.DataProjection()
    dataProjection.CreateStart(dataProjectionUserNumber, dataPoints,
                               geometricField, iron.FieldVariableTypes.U)
    # dataProjection.ProjectionTypeSet(iron.DataProjectionProjectionTypes.BOUNDARY_FACES)
    dataProjection.ProjectionTypeSet(
        iron.DataProjectionProjectionTypes.ALL_ELEMENTS)
    # candidateElements = [0]*exelems.num_elements
    # candidateFaces =[iron.ElementNormalXiDirections.MINUS_XI3]*exelems.num_elements
    # for elementIdx in range(1,exelems.num_elements+1):
    #    candidateElements[elementIdx-1]=elementIdx
    # dataProjection.ProjectionCandidateFacesSet(candidateElements,candidateFaces)
    # dataProjection.ProjectionCandidateElementsSet(candidateElements)
    dataProjection.CreateFinish()

    # Evaluate data projection based on geometric field
    dataProjection.DataPointsProjectionEvaluate(iron.FieldParameterSetTypes.VALUES)

    dataProjection.ResultAnalysisOutput(
        "InitialProjection_" + str(iterationNumber - 1))

    dataProjection.ProjectionCancelByDistance(
        iron.DataProjectionDistanceRelations.GREATER, maximumProjectionDistance)
    dataProjection.ProjectionCancelByExitTags(
        [iron.DataProjectionExitTags.EXIT_TAG_BOUNDS,
         iron.DataProjectionExitTags.EXIT_TAG_MAX_ITERATION,
         iron.DataProjectionExitTags.EXIT_TAG_NO_ELEMENT])

    dataProjection.ResultAnalysisOutput(
        "CancelProjection_" + str(iterationNumber - 1))

    # Create mesh topology for data projection
    mesh.TopologyDataPointsCalculateProjection(dataProjection)
    # Create decomposition data projection
    decomposition.TopologyDataProjectionCalculate()

    dataErrorVector = numpy.zeros((exdata.num_dataPoints, 3))
    dataErrorDistance = numpy.zeros(exdata.num_dataPoints)
    for elementIdx in range(1, exelems.num_elements + 1):
        numberOfProjectedDataPoints = decomposition.TopologyNumberOfElementDataPointsGet(
            elementIdx)
        for dataPointIdx in range(1, numberOfProjectedDataPoints + 1):
            dataPointNumber = decomposition.TopologyElementDataPointUserNumberGet(
                elementIdx, dataPointIdx)
            errorVector = dataProjection.ResultProjectionVectorGet(dataPointNumber,
                                                                   3)
            dataErrorVector[dataPointNumber - 1, 0] = errorVector[0]
            dataErrorVector[dataPointNumber - 1, 1] = errorVector[1]
            dataErrorVector[dataPointNumber - 1, 2] = errorVector[2]
            errorDistance = dataProjection.ResultDistanceGet(dataPointNumber)
            dataErrorDistance[dataPointNumber - 1] = errorDistance

    # write data points to exdata file for CMGUI
    offset = 0
    writeExdataFile("DataProjection_" + str(iterationNumber - 1) + ".part" + str(
        computationalNodeNumber) + ".exdata", dataPointLocations, dataErrorVector,
                    dataErrorDistance, offset)
    print("Projection complete")

    # =================================================================
    # Equations Set
    # =================================================================

    # Create vector fitting equations set
    print("Creating equations set")
    equationsSetField = iron.Field()
    equationsSet = iron.EquationsSet()
    equationsSetSpecification = [iron.EquationsSetClasses.FITTING,
                                 iron.EquationsSetTypes.DATA_FITTING_EQUATION,
                                 iron.EquationsSetSubtypes.DATA_POINT_FITTING,
                                 iron.EquationsSetFittingSmoothingTypes.SOBOLEV_DIFFERENCE]
    equationsSet.CreateStart(equationsSetUserNumber, region, geometricField,
                             equationsSetSpecification,
                             equationsSetFieldUserNumber, equationsSetField)
    equationsSet.CreateFinish()

    # =================================================================
    # Dependent Field
    # =================================================================

    # Create dependent field (will be deformed fitted values based on data point locations)
    print("Creating dependent field")
    dependentField = iron.Field()
    equationsSet.DependentCreateStart(dependentFieldUserNumber, dependentField)
    dependentField.VariableLabelSet(iron.FieldVariableTypes.U, "Dependent")
    dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.U, 3)
    dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.DELUDELN, 3)
    dependentField.ComponentLabelSet(iron.FieldVariableTypes.U, 1, "x")
    dependentField.ComponentLabelSet(iron.FieldVariableTypes.U, 2, "y")
    dependentField.ComponentLabelSet(iron.FieldVariableTypes.U, 3, "z")
    dependentField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
    equationsSet.DependentCreateFinish()

    # Initialise dependent field to undeformed geometric field
    for componentIdx in range(1, 4):
        geometricField.ParametersToFieldParametersComponentCopy(
            iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
            componentIdx, dependentField, iron.FieldVariableTypes.U,
            iron.FieldParameterSetTypes.VALUES, componentIdx)

    # =================================================================
    # Independent Field
    # =================================================================

    # Create data point field (independent field, with vector values stored at the data points)
    print("Creating independent field")
    independentField = iron.Field()
    equationsSet.IndependentCreateStart(independentFieldUserNumber,
                                        independentField)
    independentField.VariableLabelSet(iron.FieldVariableTypes.U, "DataPointVector")
    independentField.VariableLabelSet(iron.FieldVariableTypes.V, "DataPointWeight")
    independentField.DataProjectionSet(dataProjection)
    equationsSet.IndependentCreateFinish()

    # loop over each element's data points and set independent field values to data point locations on surface of the sphere
    for elementIdx in range(1, exelems.num_elements + 1):
        elementDomain = decomposition.ElementDomainGet(elementIdx)
        if (elementDomain == computationalNodeNumber):
            numberOfProjectedDataPoints = decomposition.TopologyNumberOfElementDataPointsGet(
                elementIdx)
            for dataPointIdx in range(1, numberOfProjectedDataPoints + 1):
                dataPointNumber = decomposition.TopologyElementDataPointUserNumberGet(
                    elementIdx, dataPointIdx)
                dataList = dataPoints.PositionGet(dataPointNumber, 3)
                x = dataList[0]
                y = dataList[1]
                z = dataList[2]
                # set data point field values
                independentField.ParameterSetUpdateElementDataPointDP(
                    iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                    elementIdx, dataPointIdx, 1, x)
                independentField.ParameterSetUpdateElementDataPointDP(
                    iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                    elementIdx, dataPointIdx, 2, y)
                independentField.ParameterSetUpdateElementDataPointDP(
                    iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                    elementIdx, dataPointIdx, 3, z)

    # =================================================================
    # Material Field
    # =================================================================

    # Create material field (Sobolev parameters)
    print("Creating materials field")
    materialField = iron.Field()
    equationsSet.MaterialsCreateStart(materialFieldUserNumber, materialField)
    materialField.VariableLabelSet(iron.FieldVariableTypes.U,
                                   "SmoothingParameters")
    equationsSet.MaterialsCreateFinish()

    # Set kappa and tau - Sobolev smoothing parameters
    # Order is tau_1, kappa_11, tau_2, kappa_22, kappa_12, tau_3, kappa_33, kappa_13, kappa_23
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              1, tau)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              2, kappa)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              3, tau)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              4, kappa)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              5, 2.0 * kappa)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              6, tau)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              7, kappa)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              8, 2.0 * kappa)
    materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,
                                              iron.FieldParameterSetTypes.VALUES,
                                              9, 2.0 * kappa)

    # =================================================================
    # Equations
    # =================================================================

    # Create equations
    print("Creating equations")
    equations = iron.Equations()
    equationsSet.EquationsCreateStart(equations)
    equations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
    equations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
    # equations.OutputTypeSet(iron.EquationsOutputTypes.MATRIX)
    equationsSet.EquationsCreateFinish()

    # =================================================================
    # Problem setup
    # =================================================================

    # Create fitting problem
    print("Creating problem")
    problem = iron.Problem()
    problemSpecification = [iron.ProblemClasses.FITTING,
                            iron.ProblemTypes.DATA_FITTING,
                            iron.ProblemSubtypes.STATIC_FITTING]
    problem.CreateStart(problemUserNumber, problemSpecification)
    problem.CreateFinish()

    # Create control loops
    print("Creating control loops")
    problem.ControlLoopCreateStart()
    problem.ControlLoopCreateFinish()

    # Create problem solver
    print("Creating solver")
    solver = iron.Solver()
    problem.SolversCreateStart()
    problem.SolverGet([iron.ControlLoopIdentifiers.NODE], 1, solver)
    solver.OutputTypeSet(iron.SolverOutputTypes.NONE)
    # solver.OutputTypeSet(iron.SolverOutputTypes.MATRIX)
    # solver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
    # solver.LibraryTypeSet(iron.SolverLibraries.UMFPACK) # UMFPACK/SUPERLU
    solver.LinearTypeSet(iron.LinearSolverTypes.ITERATIVE)
    solver.LinearIterativeMaximumIterationsSet(5000)
    solver.LinearIterativeAbsoluteToleranceSet(1.0E-10)
    solver.LinearIterativeRelativeToleranceSet(1.0E-05)
    problem.SolversCreateFinish()

    # Create solver equations and add equations set to solver equations
    print("Creating solver equations")
    solver = iron.Solver()
    solverEquations = iron.SolverEquations()
    problem.SolverEquationsCreateStart()
    problem.SolverGet([iron.ControlLoopIdentifiers.NODE], 1, solver)
    solver.SolverEquationsGet(solverEquations)
    solverEquations.SparsityTypeSet(iron.SolverEquationsSparsityTypes.SPARSE)
    equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
    problem.SolverEquationsCreateFinish()

    # =================================================================
    # Boundary Conditions
    # =================================================================

    # Create boundary conditions and fix inner surface
    print("Applying boundary conditions")
    boundaryConditions = iron.BoundaryConditions()
    solverEquations.BoundaryConditionsCreateStart(boundaryConditions)
    fixedNodes = [1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32, 53, 54, 55, 56]
    # fixedNodes = [1..4,25..32,53..57,59,61,63,65,67,69,71,105..112]
    fixedNodes = [1]
    for nodeIdx in range(len(fixedNodes)):
        nodeNumber = fixedNodes[nodeIdx]
        nodeDomain = decomposition.NodeDomainGet(nodeNumber, 1)
        if (nodeDomain == computationalNodeNumber):
            for componentIdx in range(1, 4):
                for derivativeIdx in [
                    iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S3,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2_S3,
                    iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3]:
                    boundaryConditions.AddNode(dependentField,
                                               iron.FieldVariableTypes.U,
                                               1, derivativeIdx, nodeNumber,
                                               componentIdx,
                                               iron.BoundaryConditionsTypes.FIXED,
                                               0.0)
    solverEquations.BoundaryConditionsCreateFinish()

    # =================================================================
    # S o l v e    a n d    E x p o r t    D a t a
    # =================================================================

    # Write RMS error
    rmsError = dataProjection.ResultRMSErrorGet()
    print("Starting RMS error for iteration " + str(
        iterationNumber - 1) + " = " + str(rmsError))

    # Solve the problem
    print("Solving fitting problem for iteration: " + str(iterationNumber))

    # solver.MumpsSetIcntl(14,5000)
    problem.Solve()

    # Normalise derivatives
    derivativeVector = [0.0, 0.0, 0.0, 0.0]
    for nodeIdx in range(1, exnodes.num_nodes + 1):
        nodeDomain = decomposition.NodeDomainGet(nodeIdx, 1)
        if (nodeDomain == computationalNodeNumber):
            for derivativeIdx in [iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,
                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,
                                  iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3]:
                length = 0.0
                for componentIdx in range(1, 4):
                    derivativeVector[
                        componentIdx] = dependentField.ParameterSetGetNode(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES,
                        1, derivativeIdx, nodeIdx, componentIdx)
                    length = length + derivativeVector[componentIdx] * \
                             derivativeVector[componentIdx]
                if (length > ZERO_TOLERANCE):
                    length = math.sqrt(length)
                    for componentIdx in range(1, 4):
                        value = derivativeVector[componentIdx] / length
                        dependentField.ParameterSetUpdateNode(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES,
                            1, derivativeIdx, nodeIdx, componentIdx, value)

    # Update dependent field
    dependentField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,
                                           iron.FieldParameterSetTypes.VALUES)
    dependentField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,
                                            iron.FieldParameterSetTypes.VALUES)

    # ~ # Export fields
    print("Writing fitted geometry for iteration " + str(iterationNumber))
    fields = iron.Fields()
    fields.AddField(geometricField)
    fields.AddField(dependentField)
    fields.NodesExport("FittedGeometry_" + str(iterationNumber), "FORTRAN")
    fields.ElementsExport("FittedGeometry_" + str(iterationNumber), "FORTRAN")
