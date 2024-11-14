"""Copies data by iterating through metadata paths.

The script iterates through metadata paths and copies data to a new folder,
and outputs a corresponding metadata file.

Authors: Thiranja Prasad Babarenda Gamage
Organisation: Auckland Bioengineering Institute, University of Auckland
"""

import os
import shutil
import breast_mech
import pandas as pd

if __name__ == '__main__':

    # Turn off warnings where overwriting
    pd.options.mode.chained_assignment = None  # default='warn'

    results_folder = 'results/'
    metadata_path = '/home/psam012/opt/breast-data/prone_to_supine_t2/2019_05_24/metadata.csv'
    study = breast_mech.Study(metadata_path)
    old_data_path = study.data_path
    new_data_path = os.path.join(results_folder, 'data')
    participant_ids = study.get_participant_ids(with_volume_lagrange_mesh=True)
    participant_id_key = 'participant_id'

    # Get metadata for first participant.
    participant_idx = 0
    metadata_df = study.study_metadata[
        study.study_metadata[participant_id_key] == participant_ids[participant_idx]]
    metadata_dict = metadata_df.to_dict(orient='records')[0]
    old_participant_id = metadata_dict[participant_id_key]
    new_participant_id = 'PCA_00001'
    # Update participant id in metadata dataframe.
    metadata_df[participant_id_key] = metadata_df[participant_id_key].replace(
        old_participant_id, new_participant_id)



    # Copy files in metadata paths to new data path and update metadata
    # pandas dataframe.
    for key, item in metadata_dict.items():
        if 'path' in key:
            old_path = item
            # Check if the participant id is embedded in the path name.
            if old_participant_id in old_path:
                new_path = old_path.replace(
                    old_participant_id, new_participant_id)
                new_path_dirname = os.path.dirname(new_path)
                os.makedirs(
                    os.path.join(new_data_path, new_path_dirname),
                    exist_ok=True)
                shutil.copyfile(
                    os.path.join(old_data_path, old_path),
                    os.path.join(new_data_path, new_path))
                # Update metadata pandas dataframe.
                metadata_df[key] = metadata_df[key].replace(old_path, new_path)

     # Export updated metadata pandas dataframe.
    metadata_df.to_csv(os.path.join(new_data_path, 'metadata.csv'))

    # Turn on warnings where overwriting
    pd.options.mode.chained_assignment = 'warn'