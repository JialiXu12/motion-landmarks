import numpy as np
import os
import pickle
import morphic
import time
import vtk
from vtk.util import numpy_support


class Params(object):

    def __init__(self, dictionary):
        # d = {}
        for key in dictionary.keys():
            if isinstance(dictionary[key], dict):
                self.__dict__[key] = Params(dictionary[key])
            else:
                self.__dict__[key] = dictionary[key]
        # self.__dict__ = dictionary

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.__dict__.__str__()


def pretty_time(mesg, time0):
    time1 = time.time()
    dt = time1 - time0
    if dt < 1:
        print ('%s: %dms' % (mesg, int(dt * 1000)))
    elif dt < 10:
        print ('%s: %4.2fs' % (mesg, dt))
    elif dt < 120:
        print ('%s: %4.1fs' % (mesg, dt))
    elif dt < 3600:
        print ('%s: %dm:%ds' % (mesg, int(dt / 60), int(dt % 60)))
    else:
        print ('%s: %dh:%dm' % (mesg, int(dt / 3600), int((dt % 3600) / 60)))
    return time1


def save_pickle(obj, filepath):
    pickle.dump(obj, open(filepath, 'w'))


def load_pickle(filepath):
    if os.path.exists(filepath):
        return pickle.load(open(filepath, 'r'))


def replace_image_edge(image, value, edge_size, axes=[0, 1, 2]):
    if 0 in axes:
        image.values[:edge_size, :, :] = value
        image.values[-edge_size:, :, :] = value
    if 1 in axes:
        image.values[:, :edge_size, :] = value
        image.values[:, -edge_size:, :] = value
    if 2 in axes:
        image.values[:, :edge_size, :] = value
        image.values[:, -edge_size:, :] = value
    return image


def convert_image_to_points(image, params):
    vtk_image = convert_scan_to_vtk_image(image, np.int16, flipDim=True)
    polydata = convert_vtk_image_to_polydata(vtk_image, params)
    vertices, triangles, normals = convert_polydata_to_triangular_mesh(polydata)

    vertices = vertices + [0.0, 0.0, 1.0]
    ii = np.zeros(vertices.shape)
    ii = np.float32(ii)
    ii[:, 0] = image.origin[0] + vertices[:, 0] * image.spacing[0]
    ii[:, 1] = image.origin[1] + vertices[:, 1] * image.spacing[1]
    ii[:, 2] = image.origin[2] + vertices[:, 2] * image.spacing[2]

    return vertices


# Functions for converting image volumes to point clouds
def convert_numpy_array_to_vtk_image(arrayImage, dtype, flipDim=False, retImporter=False):
    """ convert 3d numpy array into vtkImage. arrayImage.datatype should
    be either uint8 or int16.
    """
    imageImporter = vtk.vtkImageImport()
    imageString = arrayImage.astype(dtype).tostring()
    imageImporter.CopyImportVoidPointer(imageString, len(imageString))
    if dtype == np.int16:
        imageImporter.SetDataScalarTypeToShort()
    elif dtype == np.uint8:
        imageImporter.SetDataScalarTypeToUnsignedChar()
    imageImporter.SetNumberOfScalarComponents(1)
    # set imported image size
    s = arrayImage.shape
    if flipDim:
        imageImporter.SetWholeExtent(0, s[2] - 1, 0, s[1] - 1, 0, s[0] - 1)
    else:
        imageImporter.SetWholeExtent(0, s[0] - 1, 0, s[1] - 1, 0, s[2] - 1)
    imageImporter.SetDataExtentToWholeExtent()

    if retImporter:
        return imageImporter
    else:
        return imageImporter.GetOutput()

def convert_scan_to_vtk_image(scanImage, dtype, flipDim=False):
    """ convert 3d numpy array into vtkImage. arrayImage.datatype should
    be either uint8 or int16.
    """
    from scipy import ndimage
    label_im, nb_labels = ndimage.label(scanImage.values)
    sizes = ndimage.sum(scanImage.values, label_im, range(nb_labels + 1))
    mask_size = np.invert(sizes ==sizes.max())
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    label_im = ndimage.binary_fill_holes(label_im).astype(int)

    # Convert numpy array to VTK array (vtkFloatArray)
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=label_im.swapaxes(2,0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_SHORT)

    # Convert the VTK array to vtkImageData
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(scanImage.shape)
    img_vtk.SetSpacing(scanImage.spacing)
    img_vtk.SetOrigin(-1,-1,-1)
    img_vtk.GetPointData().SetScalars(vtk_data_array)


    return img_vtk


def convert_vtk_image_to_polydata(vtkImage, params):
    def _init():
        return vtkImage
    getPreviousOutput = _init

    if params.smooth_image.run:
        print ('smoothing image...')
        imageSmoother = vtk.vtkImageGaussianSmooth()
        imageSmoother.SetInputData(getPreviousOutput())
        imageSmoother.SetStandardDeviation(params.smooth_image.sd)
        imageSmoother.SetRadiusFactor(params.smooth_image.radius)
        getPreviousOutput = imageSmoother.GetOutput

    # triangulate image to create mesh
    print ("extracting contour...")
    contourExtractor = vtk.vtkMarchingCubes()
    contourExtractor.SetInputData(getPreviousOutput())
    contourExtractor.ComputeNormalsOn()
    contourExtractor.SetValue(0, params.contour.iso_value)
    contourExtractor.Update()
    getPreviousOutput = contourExtractor.GetOutput

    # triangle filter
    triFilter = vtk.vtkTriangleFilter()
    triFilter.SetInputData(getPreviousOutput())
    triFilter.Update()
    getPreviousOutput = triFilter.GetOutput

    # smooth polydata
    if params.smooth_polydata.run:
        print ("smoothing...")
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(getPreviousOutput())
        smoother.SetNumberOfIterations(params.smooth_polydata.iterations)
        smoother.SetFeatureEdgeSmoothing(params.smooth_polydata.smooth_feature_edges)
        smoother.Update()
        getPreviousOutput = smoother.GetOutput

    # decimate polydata
    if params.decimate_polydata.run:
        print ("decimating using quadric...")
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(getPreviousOutput())
        decimator.SetTargetReduction(params.decimate_polydata.ratio)
        # decimator.SetPreserveTopology( params.decimate_polydata.preserve_topology )
        # decimator.SplittingOn()
        decimator.Update()
        getPreviousOutput = decimator.GetOutput

    # clean mesh
    if params.clean.run:
        print ("cleaning...")
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(getPreviousOutput())
        cleaner.SetConvertLinesToPoints(1)
        cleaner.SetConvertStripsToPolys(1)
        cleaner.SetConvertPolysToLines(1)
        cleaner.SetPointMerging(params.clean.merge_points)
        cleaner.SetTolerance(params.clean.tolerance)
        cleaner.Update()
        getPreviousOutput = cleaner.GetOutput

    # filter normals
    if params.normals.calculate:
        print ("filtering normals...")
        normal = vtk.vtkPolyDataNormals()
        normal.SetInputData(getPreviousOutput())
        normal.SetAutoOrientNormals(1)
        normal.SetComputePointNormals(1)
        normal.SetConsistency(1)
        normal.Update()
        getPreviousOutput = normal.GetOutput

    if params.curvature.calculate:
        print ("calculating curvature...")
        curvature = vtk.vtkCurvatures()
        curvature.SetCurvatureTypeToMean()
        curvature.SetInputData(getPreviousOutput())
        curvature.Update()
        getPreviousOutput = curvature.GetOutput
    return getPreviousOutput()


def convert_polydata_to_triangular_mesh(polydata):
    if polydata.GetNumberOfPoints() == 0:
        raise (ValueError, 'no points in polydata')
    # get vertices
    vertices = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    # get triangles
    triangles = []
    for i in range(polydata.GetNumberOfCells()):
        ids = polydata.GetCell(i).GetPointIds()
        triangles.append((ids.GetId(0), ids.GetId(1), ids.GetId(2)))
    triangles = np.array(triangles, dtype=int)
    # curvature
    # normals
    polydata_normals = polydata.GetPointData().GetNormals()
    if polydata_normals is not None:
        s = polydata_normals.GetDataSize()
        normals = np.zeros(s, dtype=float)
        for i in range(s):
            normals[i] = polydata_normals.GetValue(i)
        normals = normals.reshape((int(s / 3), 3))
    else:
        normals = None
    return vertices, triangles, normals


class PCAMesh(object):

    def __init__(self, groups=None):
        self.X = []
        self.pca = None
        self.num_modes = 5
        self.input_mesh = None
        self.node_ids = []
        self.groups = groups
        self.mesh = None

    def add_mesh(self, mesh, dx=None):
        if isinstance(mesh, str):
            mesh = morphic.Mesh(mesh)

        if self.input_mesh is None:
            self.input_mesh = mesh
            self.node_ids = []
            if self.groups is None:
                for node in mesh.nodes:
                    if not isinstance(node, morphic.mesher.DepNode):
                        if node.id not in self.node_ids:
                            self.node_ids.append(node.id)
            else:
                for node in mesh.nodes:
                    if node.in_group(self.groups):
                        if node.id not in self.node_ids:
                            self.node_ids.append(node.id)

        x = []
        for nid in self.node_ids:
            x.extend(self.get_node_values(mesh.nodes[nid], dx))

        self.X.append(x)

    @staticmethod
    def get_node_values(node, dx):
        if dx is None:
            return node.values.flatten().tolist()
        else:
            xn = node.values
            if xn.ndim == 1:
                xn -= dx
            elif xn.ndim == 2:
                xn[:, 0] -= dx
            return xn.flatten().tolist()

    @staticmethod
    def merged_pca(pca_meshes, num_modes=5):
        from sklearn import decomposition
        X = None
        for mesh in pca_meshes:
            mesh.X = np.array(mesh.X)
            if X is None:
                X = np.array(mesh.X)
            else:
                X = np.concatenate((X, mesh.X), axis=1)
        pca_decomp = decomposition.PCA(n_components=num_modes)
        pca_decomp.fit(X)

        idx = 0
        for mesh in pca_meshes:
            cols = mesh.X.shape[1]
            mesh.num_modes = num_modes
            mesh.mean = pca_decomp.mean_[idx:idx + cols]
            mesh.components = pca_decomp.components_[:, idx:idx + cols].T
            mesh.variance = np.array(pca_decomp.explained_variance_)
            mesh.generate_mesh()
            idx += cols

        return pca_meshes

    def generate(self, num_modes=5):
        from sklearn import decomposition
        self.X = np.array(self.X)
        self.num_modes = num_modes
        self.pca = decomposition.PCA(n_components=num_modes)
        self.pca.fit(self.X)
        self.mean = self.pca.mean_
        self.components = self.pca.components_.T
        self.variance = self.pca.explained_variance_
        self.generate_mesh()
        return self.mesh

    def generate_mesh(self):
        ### Generate mesh from PCA results
        self.mesh = morphic.Mesh()
        weights = np.zeros(self.num_modes + 1)
        weights[0] = 1.
        self.mesh.add_stdnode('weights', weights)
        variance = np.zeros(self.num_modes + 1)
        variance[0] = 1.0
        variance[1:] = np.sqrt(self.variance)
        self.mesh.add_stdnode('variance', variance)

        idx = 0
        for nid in self.node_ids:
            node = self.input_mesh.nodes[nid]
            nsize = node.values.size
            x = self.get_pca_node_values(node, idx)
            self.mesh.add_pcanode(node.id, x, 'weights', 'variance', group='pca')
            idx += nsize

        for node in self.input_mesh.nodes:
            if node.id not in self.node_ids:
                if isinstance(node, morphic.mesher.StdNode):
                    self.mesh.add_stdnode(node.id, node.values, group=node.groups())
                elif isinstance(node, morphic.mesher.DepNode):
                    self.mesh.add_depnode(node.id, node.element, node.node, shape=node.shape, scale=node.scale,
                                          group=node.groups())
                if isinstance(node, morphic.mesher.PCANode):
                    raise Exception("Not implemented")

        for element in self.input_mesh.elements:
            self.mesh.add_element(element.id, element.basis, element.node_ids)

        self.mesh.generate()

    def get_pca_node_values(self, node, idx):
        nsize = node.values.size
        if len(node.shape) == 1:
            pca_node_shape = (node.shape[0], 1, self.num_modes)
            x = np.zeros((node.shape[0], 1, self.num_modes + 1))  # +1 to include mean
            x[:, 0, 0] = self.mean[idx:idx+nsize].reshape(node.shape)  # mean values
            x[:, :, 1:] = self.components[idx:idx+nsize, :].reshape(pca_node_shape) # mode values
            return x
        elif len(node.shape) == 2:
            pca_node_shape = (node.shape[0], node.shape[1], self.num_modes)
            x = np.zeros((node.shape[0], node.shape[1], self.num_modes + 1))  # +1 to include mean
            x[:, :, 0] = self.mean[idx:idx+nsize].reshape(node.shape)  # mean values
            x[:, :, 1:] = self.components[idx:idx+nsize, :].reshape(pca_node_shape)  # mode values
            return x
        else:
            print ('Cannot reshape this node when genrating pca mesh')


def generate_pca_meshes(path_formats, subjects, groups=None, zero_on=None, modes=5):
    # Create a PCA Mesh generator for each of the left skin, right skin and lung meshes
    pca_meshes = [PCAMesh(groups=groups) for _ in path_formats]

    # For each subject, load the left skin, right skin and lung meshes and add to the associated PCA Mesh.
    for subject in subjects:
        meshes = []
        paths = [path_format % subject for path_format in path_formats]
        for path in paths:
            if not os.path.exists(path):
                print ('Path missing: %s' % path)
                continue
            meshes.append(morphic.Mesh(path))

        # Get the origin for all meshes based on mesh = zero_on[0] and node = zero_on[1]
        dx = meshes[zero_on[0]].nodes[zero_on[1]].values[:, 0]

        # Add meshes to PCA mesh
        for pca_mesh, mesh in zip(pca_meshes, meshes):
            pca_mesh.add_mesh(mesh, dx=dx)

    # Perform a merged PCA decomposition
    pca_meshes = PCAMesh.merged_pca(pca_meshes, num_modes=modes)

    return pca_meshes


def generate_pca_meshes_v2(subject_paths, origins, groups=None, modes=5):
    # Create a PCA Mesh generator for each of the left skin, right skin and lung meshes
    subjects = subject_paths.keys()
    pca_meshes = [PCAMesh(groups=groups) for _ in subject_paths[subjects[0]]]

    # For each subject, load the left skin, right skin and lung meshes and add to the associated PCA Mesh.
    for subject in subjects:
        meshes = []
        paths = subject_paths[subject]
        for path in paths:
            if not os.path.exists(path):
                print ('Path missing: %s' % path)
                continue
            meshes.append(morphic.Mesh(path))

        # Get the origin for all meshes based on mesh = zero_on[0] and node = zero_on[1]
        dx = origins[subject]

        # Add meshes to PCA mesh
        for pca_mesh, mesh in zip(pca_meshes, meshes):
            pca_mesh.add_mesh(mesh, dx=dx)

    # Perform a merged PCA decomposition
    pca_meshes = PCAMesh.merged_pca(pca_meshes, num_modes=modes)

    return pca_meshes


def generate_pca_mesh(paths, groups=None, origin_node=None, modes=5):

    pca_mesh = PCAMesh(groups=groups)
    for path in paths:
        if not os.path.exists(path):
            print ('Path missing: %s' % path)
            continue
        mesh = morphic.Mesh(path)
        dx = mesh.nodes[origin_node].values[:, 0]
        pca_mesh.add_mesh(mesh, dx=dx)

    pca_mesh.generate(num_modes=modes)
    # pca_meshes = PCAMesh.merged_pca([pca_mesh], num_modes=modes)
    # return pca_meshes[0].mesh
    return pca_mesh.mesh


def predict_lungs(left_mesh, right_mesh, plsr_path_formats, plsr_subject_ids, modes=6):
    xnids = [nid for nid in range(73) if nid not in range(49, 54)]
    ynids = range(2, 26)
    X, Y = [], []
    for sid in subject_ids:
        paths = [path_format % sid for path_format in path_formats]
        all_paths_exist = True
        for path in paths:
            if os.path.exists(path):
                all_paths_exist = False
        if all_paths_exist:
            x = []
            for path in paths[:2]:
                mesh = morphic.Mesh(path)
                for nid in xnids:
                    x.extend(mesh.nodes[nid].values.flatten().tolist())
            X.append(x)
            y = []
            lung_mesh = morphic.Mesh(paths[2])
            for nid in ynids:
                y.extend(lung_mesh.nodes[nid].values.flatten().tolist())
            Y.append(y)
    X, Y = np.array(X), np.array(Y)
    print (X.shape, Y.shape)
    plsr = PLSRegression(copy=True, n_components=modes, scale=False)
    plsr.fit(X, Y)

    Xin = []
    for nid in xnids:
        Xin.extend(left_mesh.nodes[nid].values.flatten().tolist())
    for nid in xnids:
        Xin.extend(right_mesh.nodes[nid].values.flatten().tolist())
    Xin = np.array(Xin)

    return plsr


def predict_ribcage_mesh(ribcage_plsr_model, lungs_pca_mesh, ribcage_template):
    x_input = []
    x0 = np.mean([lungs_pca_mesh.nodes[nid].values[:, 0] for nid in range(2, 26)], 0)
    for nid in range(2, 26):
        x = lungs_pca_mesh.nodes[nid].values
        x[:, 0] -= x0
        x_input.extend(np.array(x.flatten()).tolist())
    x_input = np.array([x_input])
    y_output = ribcage_plsr_model.predict(x_input)[0]

    idx = 0
    for nid in range(32):
        cids = ribcage_template.nodes[nid].cids
        ribcage_template._core.P[cids] = y_output[idx:idx + len(cids)]
        ribcage_template.nodes[nid].values[:, 0] += x0
        idx += len(cids)

    ribcage_template.generate()

    return ribcage_template
