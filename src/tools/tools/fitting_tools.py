import numpy
from bmw import mesh_generation
import scipy
from scipy.spatial import cKDTree
from scipy.optimize import leastsq
import itertools


def writeExdataFile(filename,dataPointLocations,dataErrorVector,dataErrorDistance,offset):
    "Writes data points to an exdata file"

    numberOfDimensions = dataPointLocations[1].shape[0]
    try:
        f = open(filename,"w")
        if numberOfDimensions == 1:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=1, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=2, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=3, #Derivatives=0, #Versions=1
'''
        elif numberOfDimensions == 2:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=1, #Derivatives=0, #Versions=1
  y.  Value index=2, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=3, #Derivatives=0, #Versions=1
  y.  Value index=4, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=5, #Derivatives=0, #Versions=1
'''
        elif numberOfDimensions == 3:
             header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=1, #Derivatives=0, #Versions=1
  y.  Value index=2, #Derivatives=0, #Versions=1
  x.  Value index=3, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=4, #Derivatives=0, #Versions=1
  y.  Value index=5, #Derivatives=0, #Versions=1
  z.  Value index=6, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=7, #Derivatives=0, #Versions=1
'''
        f.write(header)

        numberOfDataPoints = len(dataPointLocations)
        for i in range(numberOfDataPoints):
            line = " Node: " + str(offset+i+1) + '\n'
            f.write(line)
            for j in range (numberOfDimensions):
                line = ' ' + str(dataPointLocations[i,j]) + '\t'
                f.write(line)
            line = '\n'
            f.write(line)
            if len(dataErrorVector)>0 and len(dataErrorDistance) >0:
                for j in range (numberOfDimensions):
                    line = ' ' + str(dataErrorVector[i,j]) + '\t'
                    f.write(line)
                line = '\n'
                f.write(line)
                line = ' ' + str(dataErrorDistance[i])
                f.write(line)
                line = '\n'
                f.write(line)
        f.close()

    except IOError:
        print ('Could not open file: ' + filename)

def fit_ribcage_to_skin_length(ribcage_mesh, skin_left_mesh,skin_right_mesh = None):
    scaled_mesh = ribcage_mesh.copy_mesh()
    ribcage_nodes = ribcage_mesh.get_nodes()


    ls_nodes = []
    for node in skin_left_mesh.get_nodes():
        if len(node) == 3:
            ls_nodes.append(node)
    ls_nodes = numpy.array(ls_nodes)

    if skin_right_mesh != None:
        rs_nodes = []
        for node in skin_right_mesh.get_nodes():
            if len(node) == 3:
                rs_nodes.append(node)
        rs_nodes = numpy.array(rs_nodes)
        skin_nodes = numpy.concatenate((ls_nodes, rs_nodes), axis=0)
    else:
        skin_nodes = ls_nodes



    minz_s = numpy.min(skin_nodes[:, 2])
    maxz_s = numpy.max(skin_nodes[:, 2])
    maxz_rc = numpy.max(ribcage_nodes[:, 2])
    minz_rc = numpy.min(ribcage_nodes[:, 2])
    scaling_factor = (maxz_s-minz_s)/(maxz_rc-minz_rc)

    geom_centroid = numpy.sum(ribcage_nodes, axis=0) / len(ribcage_nodes)

    scaled_nodes = ribcage_nodes - geom_centroid
    scaled_nodes[:, 2] = scaled_nodes[:, 2] * scaling_factor
    scaled_nodes += geom_centroid
    delta_z = numpy.min(scaled_nodes[:, 2]) - numpy.min(skin_nodes[:, 2])
    scaled_nodes[:, 2] -= delta_z

    for node_inx in range(len(ribcage_nodes)):
        scaled_mesh.nodes[node_inx].values[:, 0] = scaled_nodes[node_inx, :]
        scaled_mesh.nodes[node_inx].values[2, 1:4] = scaled_mesh.nodes[node_inx].values[2, 1:4] * scaling_factor

    return scaled_mesh

def compute_nodes_displacements(source_surface, target_surface, mesh_3d, elements_ids = [],
                                element_nodes_indx = []):
    surface_points, elem_xi, elem_ids = mesh_generation.generate_points_on_surface(source_surface,
                                                                                   source_surface.get_element_cids(),
                                                                                   num_points=100)
    if len(elements_ids) == 0:
        elements_ids = mesh_3d.get_element_cids()
    if len(element_nodes_indx) == 0:
        element_nodes_indx = list(range(len(mesh_3d.elements[0].node_ids)))

    lagrange_nodes_ids = []
    for elem_id in elements_ids:
        lagrange_nodes_ids += [mesh_3d.elements[elem_id].node_ids[x] for x in element_nodes_indx]

    lagrange_nodes = mesh_3d.get_nodes(lagrange_nodes_ids)
    surface_points_tree = cKDTree(surface_points)
    d, nodes_ids = surface_points_tree.query(lagrange_nodes)
    lagrange_xi = elem_xi[nodes_ids]
    lagrange_elem = elem_ids[nodes_ids]

    target_nodes = numpy.zeros(lagrange_nodes.shape)
    lagrange_nodes_ids = numpy.array(lagrange_nodes_ids)
    for elem_id in source_surface.get_element_cids():
        point_indx = numpy.where(lagrange_elem == elem_id)[0]
        target_nodes[point_indx, :] = target_surface.elements[elem_id].evaluate(lagrange_xi[point_indx])

    return target_nodes, lagrange_nodes_ids

def add_linear_sliding(points_arclength, maximal_sliding,sliding_support, alpha):
    # compute the arclength after node sliding using linear interpolation
    # alpha factor indicate an increasing or decreasing linear interpolation
    beta = 1 / (sliding_support[1]-sliding_support[0])
    new_node_arclength = points_arclength.copy()
    if alpha == 1:
        new_node_arclength = new_node_arclength + \
                             maximal_sliding * (new_node_arclength - sliding_support[0]) * beta
    if alpha == -1:
        new_node_arclength = new_node_arclength + \
                             maximal_sliding * (1 - (new_node_arclength- sliding_support[0]) * beta)

    return  new_node_arclength

def add_poly_sliding(points_arclength, maximal_sliding,maximal_sliding_position, sliding_support, sliding_fraction):
    # compute the arclength after node sliding using polynomial interpolation
    #
    line_length = 1/abs(sliding_support[1] - sliding_support[0])
    new_node_arclength = points_arclength.copy()

    maximal_position = abs(maximal_sliding_position - sliding_support[0]) *line_length
    z_position = abs(points_arclength - sliding_support[0]) *line_length

    coeff = numpy.polyfit([0, maximal_position, 1], [sliding_fraction[0], 1, sliding_fraction[1]], 2)
    local_sliding = (coeff[0] * z_position ** 2 + z_position * coeff[1] + coeff[2])

    new_node_arclength = new_node_arclength + maximal_sliding * local_sliding

    return new_node_arclength

def add_bilinear_sliding(points_arclength, maximal_sliding,maximal_sliding_position, sliding_support, sliding_fraction):
    # compute the arclength after node sliding using polynomial interpolation
    #
    line_length = 1/abs(sliding_support[1] - sliding_support[0])
    new_node_arclength = points_arclength.copy()

    maximal_position = abs(maximal_sliding_position - sliding_support[0]) *line_length
    z_position = abs(points_arclength - sliding_support[0]) *line_length

    local_slidig = numpy.zeros(len(new_node_arclength))
    local_slidig[z_position<=maximal_position] = z_position[z_position<=maximal_position]/maximal_position
    local_slidig[z_position > maximal_position] = (z_position[z_position > maximal_position]-1)/(maximal_position-1)
    local_slidig[numpy.where(local_slidig < 0)] = 0
    new_node_arclength = new_node_arclength + maximal_sliding * local_slidig

    return new_node_arclength

def select_nodes_on_line(line_points, nodes,epsilon):
    # search just for the points at the given position xi_x

    # search the points corresponding to the mesh nodes lying on the line and compute the corresponding
    # arclenght
    tree = cKDTree(line_points)
    dist, indx = tree.query(nodes)

    closest_points_indx = []
    nodes_idx = []
    if numpy.any(dist < epsilon):
        closest_points_indx = indx[dist < epsilon]
        nodes_idx = numpy.where(dist < epsilon)[0]

    return  nodes_idx,closest_points_indx

def compute_arclegth(line_points):
    euclid_dist = numpy.zeros((len(line_points) - 1))
    for dimension in range(3):
        euclid_dist = euclid_dist + (line_points[1:, dimension] - line_points[:-1, dimension]) ** 2
    euclid_dist = numpy.sqrt(euclid_dist)

    line_arclength = numpy.zeros(len(line_points))
    line_arclength[1:] = list(itertools.accumulate(euclid_dist))

    return line_arclength

def compute_max_linear_line_sliding(points, sliding_support,maximal_sliding_position, sliding_fraction):
    # compute maximal sliding for given line using a second degree polynomial.
    # maximal sliding position define the maximum of the polynomial
    # return a coefficient between 0 and 1
    torso_length = abs(sliding_support[1]-sliding_support[0])
    maximal_position = abs(maximal_sliding_position -sliding_support[0])/torso_length
    z_position = abs((numpy.sum(points, axis=0)[2] / points.shape[0]) - sliding_support[0]) / torso_length

    coeff = numpy.polyfit([0, maximal_position, 1], [sliding_fraction[0], 1, sliding_fraction[1]], 2)
    local_sliding = (coeff[0] * z_position ** 2 + z_position * coeff[1] + coeff[2])

    if local_sliding < maximal_position:
        local_slidig= z_position/maximal_position
    else:
        local_slidig = (z_position-1)/(maximal_position-1)

    return local_sliding

def compute_maximal_line_sliding(points,points_arcl,source_line,target_line ):
    # compute maximal sliding for given line using a second degree polynomial.
    # maximal sliding position define the maximum of the polynomial
    # return a coefficient between 0 and 1
    # torso_length = abs(sliding_support[1]-sliding_support[0])

    if source_line[1,0] < target_line[1,0]:
        scaling = 1
    else :
        scaling = -1
    point_position = (numpy.sum(points, axis=0) / points.shape[0])


    source_point = closest_point_on_line(source_line[0], source_line[1], point_position)
    target_point = closest_point_on_line(target_line[0], target_line[1], source_point)

    points_tree = cKDTree(points)
    d,indx = points_tree.query(target_point)
    target_arclength = points_arcl[indx]
    d, indx = points_tree.query(source_point)
    source_arclength = points_arcl[indx]
    print(source_point)
    print(target_point)
    print(target_arclength)
    print(source_arclength)

    local_sliding = scaling*(target_arclength - source_arclength)
    upper_points = points[numpy.where(points[:, 0] < points[indx, 0]), :][0]

    sliding_position = numpy.argmin(numpy.abs((upper_points[:, 1] - (points[indx, 1] ))))

    if local_sliding > 0:
        return scaling*local_sliding, upper_points[sliding_position]
    else:
        return 0 , upper_points[sliding_position]



def compute_sliding(mesh,elements_set,sliding_model, epsilon, delta_xi ):

    last_element = max(elements_set)
    start_element = min(elements_set)
    maximal_sliding = 0
    elements_step = [11,6]
    rc_nodes_set = list(range(16))

    displacements = []
    displaced_node_ids = []

    while start_element < last_element:
        # elements Ids corresponding to the elements line
        elements_set = list(range(start_element, start_element + elements_step[0]))
        xi_y = 0

        while xi_y <= 1:
            # search just for the points at the given position xi_x
            line_points, line_xi, line_elem = mesh_generation.generate_points_on_line(mesh, "xi3", 0,
                                                                                      "xi2", xi_y,
                                                                                      element_ids=elements_set,
                                                                                      num_points=100)

            line_arclength = compute_arclegth(line_points)

            # search the nodes corresponding to the subset of elements and to the rib cage surface
            # for subset_index, elements_subset in enumerate(elements_subsets):


            max_arclength = line_arclength[(line_elem == elements_set[-1]) & \
                                           (line_xi[:, 0] == 1)]
            min_arclength = line_arclength[(line_elem == elements_set[0]) & \
                                           (line_xi[:, 0] == 0)]

            cw_nodes = []
            for temp_element in elements_set:
                cw_nodes += [mesh.elements[temp_element].node_ids[x] for x in rc_nodes_set]
            cw_nodes = numpy.unique(cw_nodes)
            # search the nodes lying on the defined line
            line_nodes_indx, closest_points_indx = select_nodes_on_line(line_points,
                                                                         mesh.get_nodes(
                                                                         cw_nodes.tolist()), epsilon)
            if (len(line_nodes_indx) > 0):

                line_nodes_ids = cw_nodes[line_nodes_indx]
                # compute the amount of node displacement proportional to the arclength
                node_arclength = line_arclength[closest_points_indx]

                maximal_local_sliding, maximal_sliding_location = compute_maximal_line_sliding(line_points,line_arclength,
                                                                     sliding_model[0],sliding_model[1])

                if (maximal_sliding_location[2] < sliding_model[0][1, 2] or maximal_sliding_location[2] < sliding_model[1][1, 2]):
                    maximal_local_sliding = maximal_sliding
                else:
                    maximal_sliding = maximal_local_sliding



                # decrease sliding in z direction then moving toward caudal edge

                median_arclength_indx,_ = numpy.where(line_points == maximal_sliding_location)
                median_arclength = line_arclength[median_arclength_indx[0]]
                new_node_arclength = add_bilinear_sliding(node_arclength, maximal_local_sliding,median_arclength,
                                                                [min_arclength[0], max_arclength[0]] , [0,0])
                # compute new node position on the rib cage surface
                displaced_node_indx = [(abs(line_arclength - new_node_arclength[x])).argmin() for x in
                                       range(len(new_node_arclength))]
                new_node_position = line_points[displaced_node_indx]
                displacements += (new_node_position - mesh.get_nodes(line_nodes_ids.tolist())).tolist()
                displaced_node_ids += line_nodes_ids.tolist()
            xi_y += delta_xi

        start_element += elements_step[0]

    displacements = numpy.array(displacements)
    displaced_node_ids = numpy.array(displaced_node_ids)
    displaced_node_ids, unique_index = numpy.unique(displaced_node_ids,return_index=True)
    displacements = displacements[unique_index]
    return displacements,displaced_node_ids


def closest_point_on_line(a, b, p):
    ap = p-a
    ab = b-a
    result = a + numpy.dot(ap,ab)/numpy.dot(ab,ab) * ab
    return result


# ======================================================================#
def fit_line_on_points(X, line,w, xtol=1e-5, maxfev=0):
    """ fit list of points X to a line defined by two points by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    X = scipy.array(X)

    def obj( t ):
        d=scipy.zeros((X.shape[0]))
        for indx,point in enumerate(X) :
            closest_point = closest_point_on_line(line[0],[t[0],t[1],line[1][2]],point)
            d[indx] = w[indx] * scipy.linalg.norm(closest_point-point)
        return d

    t0 =  scipy.array([line[1][0],line[1][1]])
    tOpt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]

    return scipy.array([line[0],[tOpt[0],tOpt[1],line[1][2]]])


