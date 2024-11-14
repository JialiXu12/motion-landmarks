"""Functions and classes related to exporting study configurations files.

Generates configuration files containing the location of study metadata. This
provides a method for loading study metadata without needing to hard code
paths to metadata.

Authors: Thiranja Prasad Babarenda Gamage
Auckland Bioengineering Institute.
"""

import os
import json

def config_generator(
        label, config_folder, config_filename, study_metadata_path,
        description):
    """Generates a configuration file (a json file with a .config extension).

    Args:
        label: Label of the dataset.
        config_folder: Folder to save the configuration to.
        config_filename: Filename to save the configuration to.
        study_metadata_path: Path to the study metadata.
        description: Description of the dataset.
    """

    c = {}
    c["label"] = label
    c["study_metadata_path"] = study_metadata_path
    c["description"] = description

    with open(os.path.join(config_folder, config_filename), 'w') as outfile:
        json.dump(c, outfile, indent=4, sort_keys=True)

def generate_testing_configuration(
        config_folder='./', config_filename='testing.config'):
    """Generates a configuration for testing whose metadata is stored in this
    repository.

    Args:
        config_folder: Folder to save the configuration to.
        config_filename: Filename to save the configuration to.
    """

    # Set the study_metadata_path based on the location of the heart_metadata
    # python module.
    import breast_metadata
    study_metadata_path = os.path.join(
        os.path.dirname(breast_metadata.__file__),
        '../',
        'study_configurations',
        'test_study',
        'metadata/',
        'study_metadata.csv')

    label = 'Testing dataset'
    description = \
        'Testing data.'
    config_generator(
        label, config_folder, config_filename, study_metadata_path,
        description)
