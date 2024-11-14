"""Update nifti data and header information.

Shows how to Loads nifti images, transfer header information, and flip data.

Authors: Thiranja Prasad Babarenda Gamage
Organisation: Auckland Bioengineering Institute, University of Auckland
"""

import pathlib
import os
import nibabel as nib
import numpy as np

if __name__ == '__main__':

    mri_path = 'X:\\projects\\volunteer_camri\\data\\images\\00009\\MRI_T2_prone\\study.nii.gz'
    segmentation_path = 'X:\\projects\\volunteer_camri\\processed\\segmentations\\u_net2d_001\\VL01\\body\\T2\\prone\\body_VL00009.nii'
    os.listdir(pathlib.Path('X:\\projects\\volunteer_camri\\processed\\segmentations\\u_net2d_001\\VL01\\body\\T2\\prone'))
    # Load MRI nifti.
    mri_image = nib.load(mri_path)
    print(mri_image.header)

    # Load segmentation nifti.
    segmentation_image = nib.load(segmentation_path)
    print(segmentation_image.header)

    # Rotate the data directly without changing the header/affine matrix.
    values = np.flip(segmentation_image.get_fdata(), axis=1)

    # Replace elements of segmentation header with MRI header.
    updated_segmentation_image = nib.Nifti1Image(
        values,
        mri_image.affine,
        mri_image.header)

    # Save updated segmentation nifti
    nib.save(updated_segmentation_image, 'body_VL00009_updated.nii.gz')

