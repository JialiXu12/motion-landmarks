import os
import h5py

class Params(object):

    def __init__(self, dictionary):
        for key in dictionary.keys():
            if isinstance(dictionary[key], dict):
                self.__dict__[key] = Params(dictionary[key])
            else:
                self.__dict__[key] = dictionary[key]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.__dict__.__str__()

def convert_CMISS_dofs_to_morphic(string, offset=-1):
    """
    Converts a list of CMISS dofs (typically formatted as e.g. "1..2,4,5..6")
    to a python list. An offset of 1 is subtracted from the original dof
    number as morphic dof numbering begins with 0.
    """
    temp = CMISS_int_string_to_array(string)
    renumbered_dofs = [idx+offset for idx in temp]
    print (renumbered_dofs,)
    return renumbered_dofs

def CMISS_int_string_to_array(string):
    string_array = string.split(',')
    data = []
    for value in string_array:
        if value.find('..')>-1:
            temp = value.split('..')
            temp2 = range(int(temp[0]),int(temp[1])+1)
            [data.append(point) for point in temp2]
        else:
            data.append(int(value))
    return data


def extract_zipfile(filepath, directory):
    """
    Extracts a zip file into a directory.
    """
    import zipfile
    # Check zip file exists
    if os.path.exists(filepath):
        zip_obj = zipfile.ZipFile(filepath)
    else:
        return False, 'Zip file not found'
    # Extract zip file to workspace
    zip_obj.extractall(path=directory)
    zip_obj.close()


def load_hdf5(filename, dataset_label):
    hdf5_main_grp = h5py.File(filename, 'r')
    array = hdf5_main_grp[dataset_label][()]
    return array

def save_hdf5(filename, dataset_label, array):
    hdf5_main_grp = h5py.File(filename, 'w')
    hdf5_main_grp.create_dataset(dataset_label, data = array)
    hdf5_main_grp.close()
