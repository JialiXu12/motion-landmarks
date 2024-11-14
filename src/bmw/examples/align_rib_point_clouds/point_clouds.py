import numpy as np 
from numpy import zeros, array, uint8, int16
import nibabel as nib 
from skimage.segmentation import slic, mark_boundaries
import time
import matplotlib.pyplot as plt
from skimage.filter import gabor_kernel
from scipy import ndimage as nd
import pyximport
import pickle
import sys
import os 
from scipy.ndimage.measurements import labeled_comprehension
from scipy.ndimage.measurements import center_of_mass
import scipy
from scipy.ndimage.morphology import binary_fill_holes as bh
from numpy import zeros, array, uint8, int16
import pickle
import vtk

def read_3D_images(impath):
    img= nib.load(impath)
    im = img.get_data()
    spa = np.zeros([3])
    spa[0] = img.get_header().get_zooms()[0]
    spa[1] = img.get_header().get_zooms()[1]
    spa[2] = img.get_header().get_zooms()[2]
    return im, spa
def array2vtkImage( arrayImage, dtype, flipDim=False, retImporter=False ):
    """ convert 3d numpy array into vtkImage. arrayImage.datatype should
    be either uint8 or int16.
    """
    imageImporter = vtk.vtkImageImport()
    imageString = arrayImage.astype(dtype).tostring()
    imageImporter.CopyImportVoidPointer( imageString, len( imageString ) )
    if dtype==int16:
        imageImporter.SetDataScalarTypeToShort()
    elif dtype==uint8:
        imageImporter.SetDataScalarTypeToUnsignedChar()
    imageImporter.SetNumberOfScalarComponents(1)
    # set imported image size
    s = arrayImage.shape
    if flipDim:
        imageImporter.SetWholeExtent(0, s[2]-1, 0, s[1]-1, 0, s[0]-1)
    else:
        imageImporter.SetWholeExtent(0, s[0]-1, 0, s[1]-1, 0, s[2]-1)
    imageImporter.SetDataExtentToWholeExtent()    
    
    if retImporter:
        return imageImporter
    else:
        return imageImporter.GetOutput()
        
def polyData2Tri(p):
    if p.GetNumberOfPoints()==0:
        raise ValueError, 'no points in polydata'
    # get vertices
    V = array([p.GetPoint(i) for i in xrange(p.GetNumberOfPoints())])
    # get triangles
    T = []
    for i in xrange(p.GetNumberOfCells()):
        ids = p.GetCell(i).GetPointIds()
        T.append((ids.GetId(0), ids.GetId(1), ids.GetId(2)))
    T = array(T, dtype=int)    
    # curvature
    # normals
    polydataNormals = p.GetPointData().GetNormals()
    if polydataNormals!=None:
        s = polydataNormals.GetDataSize()
        N = zeros(s, dtype=float)
        for i in xrange(s):
            N[i] = polydataNormals.GetValue(i)
        N = N.reshape((s/3,3))
    else:
        N = None
    return V, T, N

class polydataFromImageParams     (object):
    def __init__( self ):
        self.smoothImage = 1
        self.imgSmthSD = 2.0
        self.imgSmthRadius = 1.5
        self.isoValue = 200.0
        self.smoothIt = 100
        self.smoothFeatureEdge = 0
        self.deciRatio = 0.5
        self.deciPerserveTopology = 0
        self.clean = True
        self.cleanPointMerging = 1
        self.cleanTolerance = 0.0
        self.filterNormal = 1
        self.calcCurvature = 1        
    def save( self, filename ):
        f = open( filename+'.polyparams', 'w' )
        pickle.dump( self, f )
        f.close()
        
def polydataFromImage( vtkImage, params, disp=0 ):    
    def _init():
        return vtkImage
    getPreviousOutput = _init    
    # testing - gaussian smoothing to binary image
    if params.smoothImage:
        print 'smoothing image...'
        imageSmoother = vtk.vtkImageGaussianSmooth()
        imageSmoother.SetInput( getPreviousOutput() )
        imageSmoother.SetStandardDeviation( params.imgSmthSD )
        imageSmoother.SetRadiusFactor( params.imgSmthRadius )
        getPreviousOutput = imageSmoother.GetOutput
    
    # triangulate image to create mesh    
    print "extracting contour..."
    # contourExtractor = vtk.vtkContourFilter()
    contourExtractor = vtk.vtkMarchingCubes()
    contourExtractor.SetInput( getPreviousOutput() )
    contourExtractor.ComputeNormalsOn()
    contourExtractor.SetValue( 0, params.isoValue )
    contourExtractor.Update()
    getPreviousOutput = contourExtractor.GetOutput
    
    # triangle filter
    triFilter = vtk.vtkTriangleFilter()
    triFilter.SetInput( getPreviousOutput() )
    triFilter.Update()
    getPreviousOutput = triFilter.GetOutput    
    # smooth polydata
    if params.smoothIt:
        print "smoothing..."
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInput( getPreviousOutput() )
        smoother.SetNumberOfIterations( params.smoothIt )
        smoother.SetFeatureEdgeSmoothing(params.smoothFeatureEdge )
        smoother.Update()
        getPreviousOutput = smoother.GetOutput        
    # decimate polydata
    if params.deciRatio:
        # print "decimating..."
        # decimator = vtk.vtkDecimatePro()
        # decimator.SetInput( getPreviousOutput() )
        # decimator.SetTargetReduction( params.deciRatio )
        # decimator.SetPreserveTopology( params.deciPerserveTopology )
        # decimator.SplittingOn()
        # decimator.Update()
        # getPreviousOutput = decimator.GetOutput
        # if disp:
        #     RenderPolyData( decimator.GetOutput() )
        print "decimating using quadric..."
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInput( getPreviousOutput() )
        decimator.SetTargetReduction( params.deciRatio )
        # decimator.SetPreserveTopology( params.deciPerserveTopology )
        # decimator.SplittingOn()
        decimator.Update()
        getPreviousOutput = decimator.GetOutput
        if disp:
            RenderPolyData( decimator.GetOutput() )
    
    # clean mesh
    if params.clean:
        print "cleaning..."
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInput( getPreviousOutput() )
        cleaner.SetConvertLinesToPoints(1)
        cleaner.SetConvertStripsToPolys(1)
        cleaner.SetConvertPolysToLines(1)
        cleaner.SetPointMerging( params.cleanPointMerging )
        cleaner.SetTolerance( params.cleanTolerance )
        cleaner.Update()
        getPreviousOutput = cleaner.GetOutput
    # filter normals
    if params.filterNormal:
        print "filtering normals..."
        normal = vtk.vtkPolyDataNormals()
        normal.SetInput( getPreviousOutput() )
        normal.SetAutoOrientNormals(1)
        normal.SetComputePointNormals(1)
        normal.SetConsistency(1)
        normal.Update()
        getPreviousOutput = normal.GetOutput
        if disp:
            RenderPolyData( normal.GetOutput() )    
    if params.calcCurvature:
        print "calculating curvature..."
        curvature = vtk.vtkCurvatures()
        curvature.SetCurvatureTypeToMean()
        curvature.SetInput( getPreviousOutput() )
        curvature.Update()
        getPreviousOutput = curvature.GetOutput
        if disp:
            RenderPolyData( curvature.GetOutput() )
    
    return getPreviousOutput()

def image2Triangle(imageArray, isoValue, deciRatio=0.5, smoothIt=200):
    IMGDTYPE = int16
    imageArray = imageArray.astype(IMGDTYPE)
    vtkImage = array2vtkImage(imageArray, IMGDTYPE, flipDim=True)

    params = polydataFromImageParams()
    params.smoothImage = False
    params.imgSmthRadius = 1.0
    params.imgSmthSD = 1.0
    params.isoValue = isoValue
    params.smoothIt = smoothIt
    params.smoothFeatureEdge = 1
    params.deciRatio = deciRatio
    params.deciPerserveTopology = 1
    params.clean = 1
    params.cleanPointMerging = 1
    params.cleanTolerance = 0.0
    params.filterNormal = 1
    params.calcCurvature = 0
    polydata = polydataFromImage(vtkImage, params)

    V, T, N = polyData2Tri(polydata)
    V = V[:,::-1] + [0.0,0.0,1.0]

    return V, T, N


def convert_mask_to_point_cloud(img_path):
    #f = open('rib_VL00023_points.pkl', 'wb')
    #impath = '/hpc/hbal361/ribseg_23.nii'
    # Reading the segmentation 
    im, spa = read_3D_images(img_path)
    #Extracting the vertices from segmentation
    #import ipdb; ipdb.set_trace()
    vertices, triangles, normals = image2Triangle(im.astype(float), 0.5, deciRatio=0.80, smoothIt=2000)
    vertices = np.uint16(np.round(vertices))
    # Convert the points to world co-ordinates 
    ii =np.zeros(vertices.shape)
    ii = np.float32(ii)
    ii[:,0] = vertices[:,0]*spa[0]
    ii[:,1] = vertices[:,1]*spa[1]
    ii[:,2] = vertices[:,2]*spa[2]
    # Save the image as pickle 
    #pickle.dump(ii, f, pickle.HIGHEST_PROTOCOL)
    #f.close()
    return ii

