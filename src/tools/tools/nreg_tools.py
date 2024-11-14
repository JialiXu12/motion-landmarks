import os
import subprocess
from tools import sitkTools
from vtk import *
import numpy as np
from vtk.util import numpy_support
from scipy.spatial import cKDTree
import automesh
import scipy
from bmw import image_warping

def imageNReg(params):

    exePath = params.exePath
    filename3 = os.path.splitext(os.path.basename(params.params_file))[0]
    transOutFile = None

    if os.path.exists(params.target_data_file) and os.path.exists(params.source_data_file):
        os.chdir(exePath)

        exeCommand = './nreg ' + params.target_data_file +' '+params.source_data_file

        if os.path.exists(params.mask_data_file):
            exeCommand += ' -mask ' +params.mask_data_file


        if not os.path.exists(params.output_dir):
            os.mkdir(params.output_dir)


        if os.path.exists(params.params_file):
            exeCommand +=' -parin '+params.params_file
        else:

            outParFile = os.path.join(params.output_dir, 'outParams_' + filename3 + '.par')
            exeCommand+= ' -parout '+outParFile

        #exeCommand += ' -Sp '+ str(params.alpha1)+ ' -Vp '+str(params.alpha2)

        if os.path.exists(params.initil_transform_file):

            exeCommand += ' -dofin '+ params.initil_transform_file
            transOutFile = 'outTrans_'+ filename3 +'.dof'


        else:

            transOutFile = 'outTrans_'+ filename3 +'.dof'

        transOutFile = os.path.join(params.output_dir,transOutFile )
        exeCommand += ' -dofout ' + transOutFile

        p = subprocess.Popen(exeCommand, shell=True,  stderr=subprocess.PIPE,)

        while True:
            out = p.stderr.read(1)
            if out == b'' and p.poll() != None:
                break
            if out != b'':
                sys.stdout.write(out.decode('utf-8'))
                sys.stdout.flush()

    return transOutFile

def imageAReg(params):

    exePath = params.exePath
    filename3 = os.path.splitext(os.path.basename(params.params_file))[0]
    transOutFile = None

    if os.path.exists(params.target_data_file) and os.path.exists(params.source_data_file):
        os.chdir(exePath)

        exeCommand = './areg2 ' + params.target_data_file +' '+params.source_data_file

        if os.path.exists(params.mask_data_file):
            exeCommand += ' -mask ' +params.mask_data_file


        if not os.path.exists(params.output_dir):
            os.mkdir(params.output_dir)


        if os.path.exists(params.params_file):
            exeCommand +=' -parin '+params.params_file
        else:

            outParFile = os.path.join(params.output_dir, 'affineParams_' + filename3 + '.par')
            exeCommand+= ' -parout '+outParFile

        if os.path.exists(params.initil_transform_file):

            exeCommand += ' -dofin '+ params.initil_transform_file
            transOutFile = 'affineTrans_'+ filename3 +'.dof'


        else:

            transOutFile = 'affineTrans_'+ filename3 +'.dof'

        transOutFile = os.path.join(params.output_dir,transOutFile )
        exeCommand += ' -dofout ' + transOutFile

        p = subprocess.Popen(exeCommand, shell=True,  stderr=subprocess.PIPE,)

        while True:
            out = p.stderr.read(1)
            if out == b'' and p.poll() != None:
                break
            if out != b'':
                sys.stdout.write(out.decode('utf-8'))
                sys.stdout.flush()

    return transOutFile

def surfaceNReg(params):

    exePath = params.exePath
    filename1 = os.path.splitext(os.path.basename(params.source_data_file))[0]
    filename2 = os.path.splitext(os.path.basename(params.target_data_file))[0]
    transOutFile = None

    if os.path.exists(params.target_data_file) and os.path.exists(params.source_data_file):
        os.chdir(exePath)

        exeCommand = './snreg ' + params.target_data_file +' '+params.source_data_file

        exeCommand += ' -locator 2 -symmetric -iterations 2'


        if not os.path.exists(params.output_dir):
            os.mkdir(params.output_dir)

        if os.path.exists(params.initil_transform_file):

            filename3 = os.path.splitext(os.path.basename(params.initil_transform_file))[0]
            exeCommand += ' -dofin '+ params.initil_transform_file
            transOutFile = 'outTrans_'+ filename3 +'.dof'


        else:

            transOutFile = 'outTrans_'+filename1+'_'\
                           +filename2+'.dof'

        transOutFile = os.path.join(params.output_dir,transOutFile )
        exeCommand += ' -dofout ' + transOutFile
        if params.spacing > 0:
            exeCommand += ' -ds ' + str(params.spacing)

        p = subprocess.Popen(exeCommand, shell=True,  stderr=subprocess.PIPE,)

        while True:
            out = p.stderr.read(1)
            if out == b'' and p.poll() != None:
                break
            if out != b'':
                sys.stdout.write(out.decode('utf-8'))
                sys.stdout.flush()

    return transOutFile

def pointsNReg(params):

    exePath = params.exePath
    filename1 = os.path.splitext(os.path.basename(params.source_data_file))[0]
    filename2 = os.path.splitext(os.path.basename(params.target_data_file))[0]

    if os.path.exists(params.target_data_file) and os.path.exists(params.source_data_file):
        os.chdir(exePath)

        exeCommand = './pnreg ' + params.source_data_file +' '+params.target_data_file

        if not os.path.exists(params.output_dir):
            os.mkdir(params.output_dir)

        if os.path.exists(params.initil_transform_file):
            filename3 = os.path.splitext(os.path.basename(params.initil_transform_file))[0]

            exeCommand += ' -dofin '+ params.initil_transform_file
            transOutFile = filename3 +'.dof'


        else:

            transOutFile = 'outTrans_'+filename1+'_'\
                           +filename2+'.dof'

        transOutFile = os.path.join(params.output_dir,transOutFile )
        exeCommand += ' -dofout ' + transOutFile

        if params.spacing > 0:
            exeCommand += ' -ds ' + str(params.spacing)
        p = subprocess.Popen(exeCommand, shell=True,  stderr=subprocess.PIPE,)

        while True:
            out = p.stderr.read(1)
            if out == b'' and p.poll() != None:
                break
            if out != b'':
                sys.stdout.buffer.write(out)
                sys.stdout.buffer.flush()

    return transOutFile

def multiLevelPointsNReg(params, nbLevels):


    spacing = params.spacing
    initil_transform_file = params.initil_transform_file
    for level in range(nbLevels):
        p_nreg = {
            'debug': params.debug,
            'offscreen': params.offscreen,
            'source_data_file': params.source_data_file,
            'target_data_file': params.target_data_file,
            'volunteer_id': params.volunteer_id,
            'output_dir': params.output_dir,
            'initil_transform_file': initil_transform_file,
            'exePath': params.exePath,
            'spacing': spacing,
            }
        params_nreg = automesh.Params(p_nreg)
        initil_transform_file = pointsNReg(params_nreg)
        spacing = int(spacing/2)


    return initil_transform_file

def applyDOF2VTKPoints(params):
    exePath = params.exePath
    t_points = None
    filename1 = 'nreg_' + os.path.splitext(os.path.basename(params.source_data_file))[0] + '.vtk'
    outputFile = os.path.join(params.output_dir, filename1)

    if os.path.exists(outputFile):
        os.remove(outputFile)

    if os.path.exists(params.source_data_file) and os.path.exists(params.initil_transform_file):
        os.chdir(exePath)

        exeCommand = './ptransformation ' + params.source_data_file + ' ' \
                                          + outputFile + ' -dofin ' \
                                          + params.initil_transform_file
        if (params.invert):
            exeCommand += ' -invert'


        p = subprocess.Popen(exeCommand, shell=True, stderr=subprocess.PIPE)

        while True:
            out = p.stderr.read(1)
            if out == b'' and p.poll() != None:
                break
            if out != b'':
                sys.stdout.buffer.write(out)
                sys.stdout.buffer.flush()
    if os.path.exists(outputFile):
        t_points= readPoints(outputFile)
    return outputFile,t_points


def applyDOF2Scan(params):
    exePath = params.exePath
    filename1 = 'nreg_' +os.path.splitext(os.path.basename(params.source_data_file))[0]+'.nii'
    outputScan = os.path.join(params.output_dir,filename1)
    if os.path.exists(params.source_data_file) and os.path.exists(params.initil_transform_file):
        os.chdir(exePath)

        exeCommand = './transformation ' + params.source_data_file + ' ' \
                                          + outputScan + ' -dofin ' + params.initil_transform_file \
                                        + ' -'+params.interpolator

        if params.invert:
            exeCommand += ' -invert'

        p = subprocess.Popen(exeCommand, shell=True, stderr=subprocess.PIPE, )


        while True:
            out = p.stderr.read(1)
            if out == b'' and p.poll() != None:
                break
            if out != b'':
                sys.stdout.buffer.write(out)
                sys.stdout.buffer.flush()
    if os.path.exists(outputScan):
        t_scan = sitkTools.readNIFTIImage(outputScan)
    return t_scan, outputScan

def numpyAray2Polydata(array):

    polydata = vtk.vtkPolyData()
    vtkPoints=vtk.vtkPoints()
    vtkPoints.SetData(numpy_support.numpy_to_vtk(array))
    polydata.SetPoints(vtkPoints)
    polydata.Modified()
    return polydata

def polydata2NumpyArray(polydata):
    if polydata.GetPoints():
        array = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        return array
    else: return None

def writePoints(array, filename):
    polydata = numpyAray2Polydata(array)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def readPoints(filename):
    if os.path.exists(filename):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        array = polydata2NumpyArray(polydata)
        return array
    else: return None

def model2rafCoordinates(points_array, metadata):
# convert an array of points from the model coordinate system (ALF) to the image(RAF)
#coordinate system using MRI metadata

    rafPoints = np.transpose([metadata.pixel_spacing[0] * metadata.image_shape[0]-points_array[:, 1],
                            points_array[:, 0],
                            points_array[:, 2]])

    rafPoints = np.transpose([rafPoints[:, 0] + metadata.image_position[1],
                              rafPoints[:, 1] + metadata.image_position[0],
                              rafPoints[:, 2] + metadata.image_position[2]])



    return rafPoints
def raf2modelCoordinates(points_array, metadata):
# convert an array of points from the image coordinate system (RAF) to the model(ALF)
#coordinate system using MRI metadata
    modelPoints = np.transpose([points_array[:, 0] - metadata.image_position[1],
                                points_array[:, 1] - metadata.image_position[0],
                                points_array[:, 2] - metadata.image_position[2]])

    modelPoints = np.transpose([modelPoints[:, 1],
                                metadata.pixel_spacing[1] * metadata.image_shape[1] -modelPoints[:, 0] ,
                                modelPoints[:, 2]])


    return modelPoints

def raf2lpfCoordinates(points_array):
# RITK image coordinates

    lpfPoints = np.transpose([-points_array[:, 0],
                              -points_array[:, 1],
                               points_array[:, 2]])
    return lpfPoints


def lpf2rafCoordinates(points_array):
# RITK image coordinates

    lpfPoints = np.transpose([-points_array[:, 0],
                              -points_array[:, 1],
                               points_array[:, 2]])
    return lpfPoints

def cut_upper_points(points, cut_edge, axis = 'z_axes'):

    new_points = points.copy()
    if axis == 'y_axes':
        cut = np.argwhere(points[:, 1] < cut_edge)
        cut = np.reshape(cut, len(cut))
        new_points = points[cut, :]

    if axis == 'x_axes':
        cut = np.argwhere(points[:, 0] < cut_edge)
        cut = np.reshape(cut, len(cut))
        new_points = points[cut, :]

    if axis == 'z_axes':
        cut = np.argwhere(points[:, 2] < cut_edge)
        cut = np.reshape(cut, len(cut))
        new_points = points[cut, :]

    return new_points
def cut_lower_points(points, cut_edge, axis = 'z_axes'):

    new_points = points.copy()
    if axis == 'y_axes':
        cut = np.argwhere(points[:, 1] > cut_edge)
        cut = np.reshape(cut, len(cut))
        new_points = points[cut, :]

    if axis == 'x_axes':
        cut = np.argwhere(points[:, 0] > cut_edge)
        cut = np.reshape(cut, len(cut))
        new_points = points[cut, :]

    if axis == 'z_axes':
        cut = np.argwhere(points[:, 2] > cut_edge)
        cut = np.reshape(cut, len(cut))
        new_points = points[cut, :]

    return new_points

def hausdorff_distance(surface1, surface2):
    surface1_tree = cKDTree(surface1)
    surface2_tree = cKDTree(surface2)
    dist_to_surface1, point_id = surface1_tree.query(surface2)
    dist_to_surface2, point_id = surface2_tree.query(surface1)
    hd1 = np.max(dist_to_surface1)
    hd2 = np.max(dist_to_surface2)

    return max(hd1,hd2)

def modified_hausdorff_distance(surface1, surface2):
    surface1_tree = cKDTree(surface1)
    surface2_tree = cKDTree(surface2)
    dist_to_surface1, point_id = surface1_tree.query(surface2)
    dist_to_surface2, point_id = surface2_tree.query(surface1)
    hd1 = np.mean(dist_to_surface1)
    hd2 = np.mean(dist_to_surface2)
    std1 = np.std(dist_to_surface1)
    std2 = np.std(dist_to_surface2)

    return max(hd1,hd2)

def average_symmetric_distance(surface1, surface2):
    surface1_tree = cKDTree(surface1)
    surface2_tree = cKDTree(surface2)
    dist_to_surface1, point_id = surface1_tree.query(surface2)
    dist_to_surface2, point_id = surface2_tree.query(surface1)
    hd1 = np.sum(dist_to_surface1)
    hd2 = np.sum(dist_to_surface2)
    n= len(dist_to_surface1)+len(dist_to_surface2)

    return (hd2+hd1)/n
