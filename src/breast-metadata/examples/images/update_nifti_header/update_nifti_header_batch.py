
import pathlib
import os
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    mri_dir = pathlib.Path("/home/clin864/eresearch/projects/volunteer_camri/data/images")
    segmentation_dir = pathlib.Path("/home/clin864/eresearch/projects/volunteer_camri/processed/segmentations/u_net2d_001/VL01/body/T2/prone")
    output_dir = pathlib.Path("/home/clin864/eresearch/projects/volunteer_camri/processed/segmentations/u_net2d_001/VL01/body/T2/prone_corrected_chinchien")

    modality = "MRI_T2_prone"
    mri_filename = "study.nii.gz"
    segmentation_filename_prefix = "body_VL"
    segmentation_extension = ".nii"

    for subject_path in sorted(mri_dir.iterdir()):
        subject = subject_path.name

        # get the image paths
        mri_path = subject_path / modality / mri_filename
        segmentation_path = segmentation_dir / (segmentation_filename_prefix + subject + segmentation_extension)

        if not mri_path.exists() or not segmentation_path.exists():
            continue

        mri_image = nib.load(mri_path)
        segmentation_image = nib.load(segmentation_path)

        # Rotate the data directly without changing the header/affine matrix.
        values = np.flip(segmentation_image.get_fdata(), axis=1)

        # Replace elements of segmentation header with MRI header.
        updated_segmentation_image = nib.Nifti1Image(values, mri_image.affine, mri_image.header)

        # Save updated segmentation nifti
        if not output_dir.exists():
            os.makedirs(output_dir)

        save_path = output_dir / (segmentation_filename_prefix + subject + ".nii.gz")
        nib.save(updated_segmentation_image, save_path)
        print("Saved: " + str(save_path))

