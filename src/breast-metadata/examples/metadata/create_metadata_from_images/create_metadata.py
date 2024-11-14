"""Copies data to a new folder structure and generate metadata.

The script walks through directories, extracting patient ids and paths to
data and stores them in a pandas data frame. This data frame is subsequently
exported as a csv file. The data is then copied to a new location with a
new data structure.

Authors: Thiranja Prasad Babarenda Gamage
Organisation: Auckland Bioengineering Institute, University of Auckland
"""

import os
import pandas as pd
from pathlib import Path, PurePath
import glob
import pydicom
import shutil

if __name__ == "__main__":

    copy_data = True

    old_root_dir = 'X:\\projects\\volunteer_camri\\data\\images'
    new_root_dir = 'X:\\projects\\volunteer_camri\\new_data\\images'

    exclusions = ['implants', 'upsidedown', 'old']

    # Define location to export newly created metatdata.
    old_metadata_filename = 'old_metadata.csv'
    new_metadata_filename = 'new_metadata.csv'
    results_folder = 'results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    subject_idxs = range(115)

    path_mapping = {
        'T1': 'MR_T1',
        'T2': 'MR_T2',
        'prone_arms_raised': 'PR_AR',
        'prone_arms_down': 'PR_AD',
        'supine_arms_raised': 'SP_AR',
        'supine_arms_down': 'SP_AD'
    }

    # Create an empty Pandas data frame with initial fields (column labels).
    df = pd.DataFrame(
        columns=[
            'subject_id',
            'modality',
            'pose',
            'nifti_path',
            'dicom_path',
            'study_instance_uid',
            'series_instance_uid'])

    # Loop through subject data and extract id's and dicom metadata.
    idx = 0
    for subject_idx in subject_idxs:
        subject_id = '{0:05d}'.format(subject_idx)
        dir_path = os.path.join(old_root_dir, subject_id)

        if os.path.exists(dir_path):

            for subdir_path in Path(dir_path).iterdir():
                row_data = []

                # Add subject id to df row.
                row_data.append(subject_id)

                image_description = PurePath(subdir_path).name

                skip = False
                for exclusion in exclusions:
                    if exclusion in image_description:
                        skip = True
                        break

                if not skip:

                    # Add modality and pose to df row.
                    for modality in ['T1', 'T2']:
                        if modality in image_description:
                            row_data.append(path_mapping[modality])
                    if 'prone' in image_description:
                        if 'arms_up' in image_description:
                            row_data.append(path_mapping['prone_arms_raised'])
                        else:
                            row_data.append(path_mapping['prone_arms_down'])
                    elif 'supine' in image_description:
                        if 'arms_up' in image_description:
                            row_data.append(path_mapping['supine_arms_raised'])
                        else:
                            row_data.append(path_mapping['supine_arms_down'])

                    # Add nifti path to df row.
                    row_data.append(os.path.join(subdir_path, 'study.nii'))

                    # Add dicom path to df row.
                    row_data.append(str(subdir_path))

                    # Add study and series instance uids to df row.
                    dcm_files = sorted(glob.glob(str(subdir_path)+'/[!.nii.gz]*'))
                    if dcm_files != []:

                        print(subdir_path)

                        dcm = pydicom.read_file(dcm_files[0])
                        row_data.append(dcm.StudyInstanceUID)
                        row_data.append(dcm.SeriesInstanceUID)

                        # Add row data to df.
                        df.loc[idx] = row_data

                        idx += 1

    # Export df.
    df.to_csv(os.path.join(results_folder, old_metadata_filename), index=False)

    # Move data to new paths.
    for index, row in df.iterrows():
        new_path_structure = os.path.join(
            new_root_dir,
            'sub-' + row['subject_id'],
            row['modality'],
            row['pose'])


        new_dicom_path = os.path.join(new_path_structure, 'dicom')

        old_nifti_path = os.path.join(
            new_path_structure, 'dicom', 'study.nii.gz')
        new_nifti_path = os.path.join(
            new_path_structure, 'nifti', 'study.nii.gz')

        if copy_data:
            os.makedirs(
                os.path.join(new_path_structure, 'nifti'), exist_ok=True)
            shutil.copytree(row['dicom_path'], new_dicom_path)
            shutil.move(old_nifti_path, new_nifti_path)

        homedir = Path(r'X:/projects/volunteer_camri/')
        df.at[index, 'nifti_path'] = Path(new_nifti_path).relative_to(homedir)
        df.at[index, 'dicom_path'] = Path(new_dicom_path).relative_to(homedir)

    # Export df.
    df.to_csv(os.path.join(results_folder, new_metadata_filename), index=False)





