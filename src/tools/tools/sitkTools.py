# Standard library imports

import automesh
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import itk
from skimage.measure import label
from skimage.transform import resize
from skimage import img_as_bool

from scipy.ndimage import zoom,gaussian_filter,binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
import scipy
import math




def display_images_with_alpha(title,axis, slice, fixed, moving):
    nda_fixed = sitk.GetArrayViewFromImage(fixed)
    nda_moving = sitk.GetArrayViewFromImage(moving)
    nda_fixed = nda_fixed / (nda_fixed.max() / 255.0)
    nda_moving = nda_moving / (nda_moving.max() / 255.0)


    if axis == 'x':
        img1 = nda_fixed[slice,:,:] - nda_moving[slice,:,:]
        img2= nda_fixed[slice, :, :]
        img3 = nda_moving[slice, :, :]

    elif axis == 'y':
        img1 = nda_fixed[:, slice, :] - nda_moving[:, slice, :]
        img2 = nda_fixed[:, slice, :]
        img3 = nda_moving[:, slice, :]


    else:
        img1 = nda_fixed[:, :, slice] - nda_moving[:, :,slice]
        img2 = nda_fixed[:, :, slice]
        img3 = nda_moving[:, :, slice]

    fig = plt.figure(title)
    ax = fig.add_subplot(121)
    ax.imshow(img2, cmap='Reds',norm=colors.PowerNorm(0.4))
    ax.imshow(img3, cmap='Greens',interpolation='none',
               norm=colors.PowerNorm(0.4), alpha=0.5)
    ax.axis('off')
    ax2 = fig.add_subplot(122)
    ax2.imshow(img1, cmap='gray')
    ax2.axis('off')
    plt.savefig('/hpc/amir309/data/{0}'.format(title))
    plt.close()
    # plt.show()

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).


def scanToSITK(image):

    values = np.swapaxes(image.values,0,2).copy()
    sitkImage = sitk.GetImageFromArray(values.astype(np.int16))
    sitkImage.SetSpacing(image.spacing)
    sitkImage.SetOrigin(image.origin)

    return sitkImage

def SITKToScan(image, orientationFlag):

    scanImage = automesh.Scan()
    scanImage.spacing = image.GetSpacing()
    scanImage.set_origin (image.GetOrigin())
    scanImage.num_slices = image.GetSize()[0]
    values = np.swapaxes(sitk.GetArrayFromImage(image),2,0).copy()
    scanImage.values = values
    scanImage.orientation = orientationFlag

    return scanImage


def readNIFTIImage(image_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    itk_rib_mask = reader.Execute()

    rib_mask_scan = SITKToScan(itk_rib_mask,'RAF')
    return rib_mask_scan

def imageResaple (image,imageTemplate, transform=[], rotation_center=[]):
# transform = [tanslationX, translationY, translationZ, rotationX, rotationY, rotationZ]

    sitkImage = scanToSITK(image)
    sitkTemplate = scanToSITK(imageTemplate)

    sitkResampler = sitk.ResampleImageFilter()
    sitkResampler.SetReferenceImage(sitkTemplate)

    sitkResampler.SetInterpolator(sitk.sitkLinear)

    if len(transform):
        sitkTransform = sitk.Euler3DTransform()
        sitkTransform.SetParameters([-transform[3], -transform[4], -transform[5],
                                     - transform[0], -transform[1], -transform[2]])
        if len(rotation_center):
            fixed_param = np.append(rotation_center, [0])
        else:
            fixed_param = [0, 0, 0, 0]

        sitkTransform.SetFixedParameters(fixed_param)

        sitkResampler.SetTransform(sitkTransform)

    sitk_source_resampled = sitk.Cast(sitkResampler.Execute(sitkImage), sitk.sitkUInt16)

    return  SITKToScan(sitk_source_resampled,imageTemplate.orientation)

def writeNIFTIImage (image, filename):


    # image.spacing = [image.spacing[1],image.spacing[0],image.spacing[2]]
    sitk_iamge  = scanToSITK(image)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(sitk_iamge)



def myshow(img, title=None, axis='z', slice =None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()


    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        if slice == None:
            slice = nda.shape[2] // 2

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            if axis == 'z':
                nda = nda[:, :, slice]
            elif axis == 'y':
                nda = nda[:, slice, :]
            else:
                nda = nda[slice, :, :]


    elif nda.ndim == 4:
        c = nda.shape[-1]

        # if not c in (3, 4):
            # raise Runtime("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[:, :, :, nda.shape[0] // 2]

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure( figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

    t = ax.imshow(nda, extent=extent, interpolation=None)

    # if nda.ndim == 2:
    t.set_cmap("gray")

    if (title):
        plt.title(title)



def linearImageInterpolator (points, image):

    PixelType = itk.F
    ScalarType = itk.D
    Dimension = len(image.shape)

    ImageType = itk.Image[PixelType,Dimension]


    values = np.swapaxes(image.values.astype(np.float32), 0, 2).copy()
    itkImage = itk.GetImageFromArray(values)
    itkImage.SetSpacing(image.spacing)
    itkImage.SetOrigin(image.origin)
    InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, ScalarType]
    interpolator = InterpolatorType.New()

    if itkImage:
        interpolator.SetInputImage(itkImage)
        points_val = [interpolator.EvaluateAtContinuousIndex(point) for point in points]

    return points_val


def createImage():

    image = automesh.Scan()

    values = np.zeros((10,10,10))
    for i in range(10):
        values[i] = 10*i

    image.values = values
    image.spacing = [1,1,1]
    image.origin = [0,0,0]

    return image

def computeNCC(image1, image2, mask):
    data1 = image1.values.copy()
    data2 = image2.values.copy()
    mask_data = mask.values.copy()

    data1 = data1.ravel()
    data2 = data2.ravel()
    mask_data = mask_data.ravel()

    data1 = data1[np.argwhere(mask_data == 1)]
    data2 = data2[np.argwhere(mask_data == 1)]

    data1 = np.reshape(data1 - np.mean(data1), len(data1))
    data2 = np.reshape(data2 - np.mean(data2), len(data2))

    ncc =  np.dot(data1,data2)**2/(np.dot(data1, data1)*np.dot(data2,data2))

    return -ncc


def getLargestCC(scan):
    labels = label(scan.values)
    largestCC = scan.copy()
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC.values = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def smoothBinaryMask(mask_image):

    rib_cage_mask = mask_image.copy()
    for slice in range(mask_image.shape[2]):
        rib_cage_mask.values[:, :, slice] = binary_fill_holes(rib_cage_mask.values[:, :, slice])

    rib_cage_mask.values = zoom(rib_cage_mask.values.astype(float), zoom=0.4, order=0)
    rib_cage_mask.values = gaussian_filter(rib_cage_mask.values * 255.0, 2, order=0)
    rib_cage_mask.values = resize(rib_cage_mask.values, mask_image.values.shape, anti_aliasing=True)
    offset = threshold_otsu(rib_cage_mask.values)
    rib_cage_mask.values = np.where(rib_cage_mask.values < offset, 0, 1)
    for slice in range(mask_image.shape[2]):
        rib_cage_mask.values[:, :, slice] = binary_fill_holes(rib_cage_mask.values[:, :, slice])

    return  rib_cage_mask

def generate_image_coordinates(image_shape, spacing):
    x, y, z = scipy.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    x = x*spacing[0]
    y = y*spacing[1]
    z = z*spacing[2]
    image_coor = scipy.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z

def extract_contour_points(mask, nb_points):
    labels = mask.values.copy()
    # new_shape = [round(x/resampleFactor) for x in labels.shape]
    # new_spacing = (np.array(mask.shape) * np.array( mask.spacing))/ np.array( new_shape)

    boundaries = find_boundaries(labels,mode = 'inner').astype(np.uint8)
    image_coordinates,x,y,z = generate_image_coordinates(labels.shape,mask.spacing)
    points = np.array(image_coordinates+mask.origin)
    points =points[np.array( boundaries.ravel()).astype(bool),:]
    if (nb_points < len(points)):
        step = math.trunc(len(points)/nb_points)
        indx = range(0,len(points), step)
        return points[indx,:]
    else:
        return points

def extend_mask_boundaries(mask, nb_rows_start, nb_rows_end, axis = 'z_axis'):
    # keeps the same mask for the slice index between start and end row
    # copy mask from the starting index to all slices index lower then starting index
    # copy mask from the end index to all slices index higher than end index
    mask_array = mask.values.copy()

    if axis == 'z_axis':
        for z_ind in range(nb_rows_start):
            mask_array[:,:,z_ind] = mask_array[:,:,nb_rows_start]
        for z_ind in range(nb_rows_end, mask.shape[2]):
            mask_array[:,:,z_ind] = mask_array[:,:,nb_rows_end]

    if axis == 'y_axis':
        for y_ind in range(nb_rows_start):
            mask_array[:,y_ind,:] = mask_array[:,nb_rows_start,:]
        for y_ind in range(nb_rows_end, mask.shape[1]):
            mask_array[:,y_ind,:] = mask_array[:,nb_rows_end,:]

    if axis == 'x_axis':
        for x_ind in range(nb_rows_start):
            mask_array[x_ind,:,:] = mask_array[nb_rows_start,:,:]
        for x_ind in range(nb_rows_end, mask.shape[0]):
            mask_array[x_ind,:,:] = mask_array[nb_rows_end,:,:]

    extended_mask = mask.copy()
    extended_mask.values = mask_array
    return extended_mask





