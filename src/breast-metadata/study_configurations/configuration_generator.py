"""Exporting study configurations files.

Generates configuration files containing the location of study metadata. This
provides a method for loading study metadata without needing to hard code
paths to metadata.

Authors: Thiranja Prasad Babarenda Gamage
Auckland Bioengineering Institute.
"""
import os
import breast_metadata

if __name__ == '__main__':

    # Generate testing configuration.
    breast_metadata.generate_testing_configuration()

    # Generate config file for clinical breast volunteer data.

    label = 'volunteers_camri'
    config_folder = './'
    config_filename = label + '.config'
    mount_pt = os.path.join(os.sep + 'breast_shared_drive', 'breast', 'mfre190')
    study_metadata_path = os.path.join(mount_pt, 'projects/volunteer_camri/metadata/metadata_prone_to_supine_t2_2019_05_24.csv')
    description = \
        'Breast Biomechanics Research Group CAMRI volunteer dataset.'
    breast_metadata.config_generator(
        label, config_folder, config_filename, study_metadata_path, description)
