import numpy as np
import SimpleITK as sitk
import breast_metadata

# def SITKToScan(image, orientationFlag):
#     scanImage = breast_metadata.Scan()
#     scanImage.spacing = image.GetSpacing()
#     scanImage.set_origin(image.GetOrigin())
#     scanImage.num_slices = image.GetSize()[0]
#     values = np.swapaxes(sitk.GetArrayFromImage(image), 2, 0).copy()
#     scanImage.values = values
#     scanImage.orientation = orientationFlag
#     return scanImage
#
# def readNIFTIImage(image_path):
#     reader = sitk.ImageFileReader()
#     reader.SetFileName(image_path)
#     itk_mask = reader.Execute()
#     mask_scan = SITKToScan(itk_mask, 'RAI')
#     return mask_scan
#

def SITKToScan(image, orientationFlag, load_dicom, swap_axes=False):
    scanImage = breast_metadata.Scan(load_dicom=load_dicom)
    scanImage.spacing = image.GetSpacing()
    scanImage.set_origin(image.GetOrigin())
    scanImage.num_slices = image.GetSize()[0]
    if swap_axes:
        values = np.swapaxes(sitk.GetArrayFromImage(image), 2, 0).copy()
    else:
        values = sitk.GetArrayFromImage(image).copy()
    scanImage.values = values
    scanImage.orientation = orientationFlag
    return scanImage


def SITKToScan2(image_path, image, orientationFlag, load_dicom, swap_axes=False):
    scanImage = breast_metadata.Scan(image_path, load_dicom=load_dicom)
    scanImage.spacing = image.GetSpacing()
    scanImage.set_origin(image.GetOrigin())
    scanImage.num_slices = image.GetSize()[0]
    if swap_axes:
        values = np.swapaxes(sitk.GetArrayFromImage(image), 2, 0).copy()
    else:
        values = sitk.GetArrayFromImage(image).copy()
    scanImage.values = values
    scanImage.orientation = orientationFlag
    return scanImage


def readNIFTIImage(image_path, swap_axes=False):
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    itk_mask = reader.Execute()
    mask_scan = SITKToScan(itk_mask, 'RAI', load_dicom=False, swap_axes=swap_axes)
    return mask_scan