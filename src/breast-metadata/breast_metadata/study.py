"""Functions and classes related to loading and saving study metadata.

Authors: Thiranja Prasad Babarenda Gamage
Auckland Bioengineering Institute.
"""

import os
import numpy as np
import pandas as pd
import pickle
import morphic
import mesh_tools
import ntpath
import posixpath

class Study:
    """A class for accessing study and participant metadata.
    """

    def __init__(self, metadata_path, absolute_paths=True):
        """Loads the study metadata from a csv.

        Args:
            metadata_path: Path to the metadata csv file.
            absolute_paths: Convert relative paths in metadata to csv
        """
        self.metadata_path = metadata_path
        # Data is loaded relative to the metadata folder.
        self.data_path = os.path.dirname(metadata_path)

        self.study_metadata = pd.read_csv(metadata_path, index_col=0)

        # Prepend data_path to all metadata keys ending in _path.

        for key in self.study_metadata.keys():
            for participant_idx in range(len(self.study_metadata)):
                if key.endswith('_path'):
                    value = self.study_metadata.loc[participant_idx, key]
                    if not str(value) == 'nan':
                        if os.name == "posix":  # linux
                            value = value.replace(ntpath.sep, posixpath.sep)
                        elif os.name == "nt":  # windows
                            value = value.replace(posixpath.sep, posixpath.sep)

                        self.study_metadata.loc[participant_idx, key] = os.path.join(
                            self.data_path, value)

    def export_metadata(self, results_folder, metadata_filename):
        """Export study metadata to csv.
        """
        self.study_metadata.to_csv(
            os.path.join(results_folder, metadata_filename))

    def analyse_missing_data(self):
        """Analyses missing data using the missingno python module.
        """
        import missingno as msno
        import matplotlib.pyplot as plt

        msno.matrix(self.study_metadata)
        plt.show()

    def get_metadata_keys(self, ):
        """Get study metadata keys.
        """

        return list(self.study_metadata.keys())


    def get_participant_ids(self, with_volume_lagrange_mesh=False):
        """Get participant ids.

        Args:
            with_volume_lagrange_mesh: If True, only return participants who
                have a populated 'volume_lagrange_mesh_path' field value,
                ignoring with 'nan' field values.
        """
        if with_volume_lagrange_mesh:
            filtered_study_metadata = self.study_metadata[
                self.study_metadata['volume_lagrange_mesh_path'].notnull()]
        else:
            filtered_study_metadata = self.study_metadata

        return sorted(filtered_study_metadata['participant_id'].values)


    def get_participant_metadata(self, id, filter=None):
        """Get metadata for one participant.

        Args:
            id: Participant identifier.
            filter: Key/value pairs of column headings/values to filter by.
        """
        metadata_df = self.study_metadata[
            self.study_metadata['participant_id'] == id]

        # Filter based on key/value pairs of column headings.
        if filter:
            for key, value in filter.items():
                metadata_df = metadata_df[metadata_df[key] == value]

        # Convert pandas dataframe to a dictionary
        metadata = metadata_df.to_dict(orient='records')

        # Append data folder
        for entry in metadata:
            entry['data_root_path'] = self.data_path

        if len(metadata) == 1:
            metadata = metadata[0]

        return metadata

    def get_metadata(self, participants_with_images=False, image_types=None):
        """Gets metadata for all participant.

        Args:
            participants_with_images: Flag to determine if metadata from
                only participants with MRI images is required.
            image_types: Types of images to get metadata for.
        """
        if image_types is None:
            image_types = [
                "prone_t1",
                "prone_t2",
                "supine_t1",
                "supine_t2"
            ]
            image_keys = [
                "mri_t1_prone_arms_down_image_path",
                "mri_t2_prone_arms_down_image_path",
                "mri_t1_supine_arms_down_image_path",
                "mri_t2_supine_arms_down_image_path"
            ]


        all_metadata = {}
        for participant_id in self.get_participant_ids():
            participant_metadata = {}
            metadata = self.get_participant_metadata(participant_id)

            # Only get metadata for participants where image information is
            # available.
            if participants_with_images:
                for image_idx, image_type in enumerate(image_types):
                    # if not str(metadata['skin_pts_path']) == 'nan':
                    #     if not str(metadata['nipple_pts_path']) == 'nan':
                    image_path = image_keys[image_idx]
                    if not str(metadata[image_path]) == 'nan':
                        if os.path.exists(metadata[image_path]):
                            if len(os.listdir(metadata[image_path])):
                                files = os.listdir(metadata[image_path])
                                excluded_formats = [".nii.gz", ".nii"]
                                files = self.filter_files_by_ext(files, excluded_formats)
                                participant_metadata['{0}_dicom_filenames'.format(image_type)] = files

            all_metadata[participant_id] = participant_metadata

        return all_metadata

    def filter_files_by_ext(self, files, extensions):
        """ Given a list of extensions, remove files which their extensions in the list.

        Args:
            files: List. a list of filenames (string)
            extensions: List. a list of extension to be excluded
        """
        for ext in extensions:
            files = list(filter(lambda f: ext not in f, files))
        return files

    def get_mesh(
            self, participant_id, mesh_type='volume_lagrange_mesh',
            mesh_format='json'):
        """Gets a mesh for a participant.

        Args:
            participant_id: ID of participant.
            mesh_type: Type of mesh to get e.g. 'volume_lagrange_mesh'.
            mesh_format: Format to export the mesh e.g. .json.
        """

        metadata = self.get_participant_metadata(
            participant_id)

        if mesh_format == 'json':
            json_data = {'nodes': {}, 'elements': {}}
            if mesh_type == 'volume_lagrange_mesh':
                if str(metadata['volume_lagrange_mesh_path']) != 'nan':
                    if os.path.exists(metadata['volume_lagrange_mesh_path']):
                        morphic_mesh = morphic.Mesh(
                            metadata['volume_lagrange_mesh_path'])
                        json_data = morphic_mesh.export(export_format='json')
                return json_data
            else:
                raise ValueError('Selected mesh_type of {0} is not supported')
        else:
            raise ValueError('Output mesh_format of {0} is not supported')

    def get_data_points(self, participant_id, number_of_samples=None):
        """Gets data points for a participant.

        Args:
            participant_id (str): ID of participant.
            number_of_samples (int): number of random samples of the data
                points to get. If None, all data points are returned (default).
        """

        metadata = self.get_participant_metadata(
            participant_id)
        json_str = {}
        if str(metadata['skin_pts_path']) != 'nan':
            if os.path.exists(metadata['skin_pts_path']):
                data = morphic.Data(metadata['skin_pts_path'])

                if number_of_samples is None:
                    sampled_data = data.values
                else:
                    sample_idxs = np.random.randint(
                        0, data.values.shape[0], number_of_samples)
                    sampled_data = data.values[sample_idxs, :]

                json_str = ''
                json_str += '{\n'
                for c_idx, c_label in enumerate(['x', 'y', 'z']):
                    json_str += '"skin_data_{0}": \n'.format(c_label)
                    json_data = mesh_tools.export_json(sampled_data[:, c_idx])
                    if c_label == 'z':
                        json_str += '{0}\n'.format(json_data)
                    else:
                        json_str += '{0},\n'.format(json_data)
                json_str += '}'
        return json_str

    def load_landmarks(self, image_types=None):
        """Gets landmark metadata for all participant.

        Args:
            image_types: Types of images to get metadata for.
        """
        if image_types is None:
            image_types = [
                "prone",
            ]

        self.all_landmark_data = {}
        valid_participant_ids = []
        for participant_id in self.get_participant_ids():
            landmark_id = []
            landmark_data = []
            landmark_parent = []
            landmark_name = []
            metadata = self.get_participant_metadata(participant_id)
            for image_type in image_types:

                landmark_path = 'landmarks_{0}_path'.format(image_type)
                if not str(metadata[landmark_path]) == 'nan':
                    if os.path.exists(metadata[landmark_path]):
                        files = os.listdir(metadata[landmark_path])
                        for file in files:
                            _, file_extension = os.path.splitext(file)
                            if file_extension == '.point':
                                file_path = os.path.join(metadata[landmark_path], file)
                                with open(file_path, 'rb') as f:
                                    d = pickle.load(f, encoding='bytes')
                                try:
                                    landmark_data.append(d[b'data'][b'default'].tolist())
                                except:
                                    pass
                                else:
                                    landmark_id.append(d[b'id'])
                                    landmark_parent.append(d[b'parent'])
                                    landmark_name.append(d[b'name'])
                                    valid_participant_ids.append(participant_id)
                if participant_id in valid_participant_ids:
                    participant_landmarks = {}
                    participant_landmarks['id'] = landmark_id
                    participant_landmarks['data'] = landmark_data
                    participant_landmarks['parent'] = landmark_parent
                    participant_landmarks['name'] = landmark_name
                    self.all_landmark_data[participant_id] = participant_landmarks

    def get_landmark_data(self):
        return self.all_landmark_data