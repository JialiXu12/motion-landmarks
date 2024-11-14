"""Generates metadata files that store a studies participant information.

The script walks through directories, extracting patient ids and paths to
data and stores them in a pandas data frame. This data frame is subsequently
exported as a csv file.

Authors: Thiranja Prasad Babarenda Gamage
Organisation: Auckland Bioengineering Institute, University of Auckland
"""

import os
import numpy as np
import pandas as pd


def extract_ids(mesh_dir):
    """Find existing participant ids by seeing if their mesh directory exists.

    Args:
        mesh_dir: Location of the root mesh directory.
    """
    participants = range(23, 81)
    bad_skin_subject_ids = [26, 32, 38, 42, 43, 44, 53, 54, 56, 61, 64, 65,
                            70, 71, 73, 78, 79, 82, 86, 87, 90, 95, 109,
                            111, 113]
    # bad_lung_ids = [20, 21, 22, 26, 57, 61, 64, 70, 75]
    # if id not in bad_skin_subject_ids and id not in bad_lung_ids:

    id_strs = {}
    for id in participants:
        if id not in bad_skin_subject_ids:
            pi_str = 'VL{0:05d}'.format(id)
            mesh_path = os.path.join(
                mesh_dir, pi_str, 'prone.mesh')
            if os.path.exists(mesh_path):
                id_strs[id] = pi_str
    return id_strs

def path_if_exists(data_dir, process_dir, uid, filename):
    """Returns the path to data if it exists, else returns nan.

    Args:
        data_dir (str): Location of the root data directory where data is
            currently stored.
        process_dir (str): Location of the process directory e.g. (mechanics, or
            'segmentation') that will be inserted into the metadata. Usually,
            this will be defined relative to the metadata folder.
        uid (str): Participant identifier.
        filename (str): Name of the data file to check exists.
    """

    if os.path.exists(os.path.join(data_dir, process_dir, uid, filename)):
        return os.path.join(process_dir, uid, filename)
    else:
        return np.nan

if __name__ == '__main__':


    # Define location to export newly created metatdata.
    metadata_filename = 'metadata_prone_to_supine_t2_2019_05_24.csv'
    results_folder = 'results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Specify location of data and directories to walk through for extracting
    # metadata.
    data_dir = '/home/data/projects/volunteers_camri/metadata'
    image_dir = '..//data/images/vl/'
    segmentation_dir = '../processed/prone_to_supine_t2/2019_05_24/segmented_data'
    mesh_dir = '../processed/prone_to_supine_t2/2019_05_24/volunteer_meshes'
    mechanics_dir = '../processed/prone_to_supine_t2/2019_05_24/mechanics_meshes'
    landmarks_dir = '../processed/landmarks'

    # Define participant list.
    participant_ids = []
    for participant_idx in range(115):
        participant_ids.append("VL{0:05d}".format(participant_idx))
    num_participants = len(participant_ids)

    # Create an empty Pandas data frame with two initial fields
    # (column labels).
    df = pd.DataFrame(columns=['participant_id', 'orientation'])

    # Populate initial columns for each participant by specifying data for an
    # entire column (ie only a call to df.loc with the participant_idx
    # (row number) is required to set data for all fields (columns labels)
    # e.g  df.loc[0] = ['VL00001, 'prone'].
    for participant_idx, uid in enumerate(participant_ids):
        df.loc[participant_idx] = [uid, 'prone']

    # Add new empty fields (columns labels) related to segmentation.
    df['lung_pts_path'] = ''
    df['skin_pts_path'] = ''
    df['nipple_pts_path'] = ''

    # Populate new columns for each participant. Here a call to df.at with the
    # particpant_idx (row number) and the field name to update is required
    # e.g. df.at[0, 'lung_pts_path'] = '/my/path'.
    for participant_idx, uid in enumerate(df['participant_id']):
        df.at[participant_idx, 'lung_pts_path'] = path_if_exists(
            data_dir, segmentation_dir, uid, 'lungs_pts_kmeans.data')
        df.at[participant_idx, 'skin_pts_path'] = path_if_exists(
            data_dir, segmentation_dir, uid, 'skin_pts_kmeans.data')
        df.at[participant_idx, 'nipple_pts_path'] = path_if_exists(
            data_dir, segmentation_dir, uid, 'nipple_pts.data')

    # Add new empty fields (columns labels) related to anatomical modelling.
    df['lung_surface_hermite_mesh_path'] = ''
    df['ribcage_surface_hermite_mesh_path'] = ''
    df['left_skin_surface_hermite_mesh_path'] = ''
    df['right_skin_surface_hermite_mesh_path'] = ''

    # Add values to the empty columns.
    for participant_idx, uid in enumerate(df['participant_id']):
        df.at[participant_idx, 'lung_surface_hermite_mesh_path'] = path_if_exists(
            data_dir, mesh_dir, uid, 'lungs_prone.mesh')
        df.at[participant_idx, 'ribcage_surface_hermite_mesh_path'] = path_if_exists(
            data_dir, mesh_dir, uid, 'ribcage_prone.mesh')
        df.at[participant_idx, 'left_skin_surface_hermite_mesh_path'] = path_if_exists(
            data_dir, mesh_dir, uid, 'skin_left_prone.mesh')
        df.at[participant_idx, 'right_skin_surface_hermite_mesh_path'] = path_if_exists(
            data_dir, mesh_dir, uid, 'skin_right_prone.mesh')

    # Add new empty fields (columns labels) related to mechanics.
    df['volume_lagrange_mesh_path'] = ''

    # Add values to the empty columns.
    for participant_idx, uid in enumerate(df['participant_id']):
        df.at[participant_idx, 'volume_lagrange_mesh_path'] = path_if_exists(
            data_dir, mechanics_dir, uid, 'prone.mesh')

    # Add new empty fields (columns labels) related to images.
    df['mri_t1_prone_arms_down_image_path'] = ''
    df['mri_t1_supine_arms_down_image_path'] = ''
    df['mri_t2_prone_arms_down_image_path'] = ''
    df['mri_t2_supine_arms_down_image_path'] = ''

    # Add values to the empty columns.
    for participant_idx, uid in enumerate(df['participant_id']):
        # Prone.
        # T1.
        df.at[participant_idx, 'mri_t1_prone_arms_down_image_path'] = path_if_exists(
            data_dir, image_dir, os.path.join("mri_t1", uid), 'prone')
        # T2.
        df.at[participant_idx, 'mri_t2_prone_arms_down_image_path'] = path_if_exists(
            data_dir, image_dir, os.path.join("mri_t2", uid), 'prone')

        # Supine.
        # T1.
        df.at[participant_idx, 'mri_t1_supine_arms_down_image_path'] = path_if_exists(
            data_dir, image_dir, os.path.join("mri_t1", uid), 'supine')
        # T2.
        df.at[participant_idx, 'mri_t2_supine_arms_down_image_path'] = path_if_exists(
            data_dir, image_dir, os.path.join("mri_t2", uid), 'supine')

    # Add new empty fields (columns labels) related to landmarks.
    df['landmarks_prone_path'] = ''
    df['landmarks_supine_path'] = ''
    # Add values to the empty columns.
    for participant_idx, uid in enumerate(df['participant_id']):
        # Prone.
        df.at[participant_idx, 'landmarks_prone_path'] = path_if_exists(
            data_dir, os.path.join(landmarks_dir, 'prone'), uid, '')
        # Supine.
        df.at[participant_idx, 'landmarks_supine_path'] = path_if_exists(
            data_dir, os.path.join(landmarks_dir, 'supine'), uid, '')

    # Export to csv.
    df.to_csv(os.path.join(results_folder, metadata_filename))

    # Visualise missing data
    import missingno as msno
    import matplotlib.pyplot as plt
    msno.matrix(df)
    plt.show()
