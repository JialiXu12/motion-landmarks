import dicom
import numpy as np
import os


class Scan(object):

    def __init__(self, dicom_files=None):
        self.num_slices = 0
        self.origin = np.array([0., 0., 0.])
        self.spacing = np.array([1., 1., 1.])
        self.orientation = 'RAF' # [1,0,0,0,1,0] RAF orientation:
                                                    # The X vector is (1,0,0) meaning it is exactly
                                                    #directed with the image pixel matrix row direction
                                                    #  and the Y vector is (0,1,0) meaning it is exactly
                                                    # directed with the image pixel matrix column direction.
        self.filepaths = []
        self.values = np.zeros((1, 1, 1))
        try:
            self.load_files(dicom_files)
        except:
            pass

    def copy(self, copy_values=True):
        new_scan = Scan()
        new_scan.num_slices = self.num_slices
        new_scan.origin = self.origin
        new_scan.spacing = self.spacing
        new_scan.orientation = self.orientation
        new_scan.filepaths = self.filepaths
        if copy_values:
            new_scan.values = self.values.copy()
        return new_scan

    def reset(self):
        self.values = None
        self.num_slices = 0
        self.values = np.zeros((1, 1, 1))

    @property
    def shape(self):
        return self.values.shape

    def set_origin(self, values):
        self.origin = np.array([float(v) for v in values])

    def set_pixel_spacing(self, values):
        self.spacing[0] = float(values[0])
        self.spacing[1] = float(values[1])
    def set_orientation(self, value):
        self.orientation = value

    def set_slice_thickness(self, value):
        self.spacing[2] = float(value)

    def init_values(self, rows, cols, slices):
        self.num_slices = slices
        self.values = np.zeros((rows, cols, slices))

    def insert_slice(self, index, values):
        self.values[:, :, index] = values

    def load_files(self, dicom_files):
        if isinstance(dicom_files, str):
            dicom_path = dicom_files
            dicom_files = os.listdir(dicom_path)
            for i, filename in enumerate(dicom_files):
                dicom_files[i] = os.path.join(dicom_path, dicom_files[i])

        slice_location = []
        slice_thickness = []
        image_position = []
        image_orientation = []
        remove_files = []

        dicom_file = None

        for i, dicom_file in enumerate(dicom_files):
            try:
                dcm = dicom.read_file(dicom_file)
                valid_dicom = True
            except dicom.filereader.InvalidDicomError:
                remove_files.append(i)
                valid_dicom = False

            if valid_dicom:
                try:
                    slice_location.append(float(dcm.SliceLocation))
                except:
                    print ('No slice location found in ' + dicom_file)
                    return False
                try:
                    slice_thickness.append(float(dcm.SliceThickness))
                except:
                    print ('No slice thickness found in ' + dicom_file)
                    return False
                try:
                    image_position.append([float(v) for v in dcm.ImagePositionPatient])
                except:
                    print ('No image_position found in ' + dicom_file)
                    return False


        # Remove files that are not dicoms
        remove_files.reverse()
        for index in remove_files:
            dicom_files.pop(index)

        slice_location = np.array(slice_location)
        slice_thickness = np.array(slice_thickness)
        image_position = np.array(image_position)

        sorted_index = np.array(slice_location).argsort()
        dt = []
        for i, zi in enumerate(sorted_index[:-1]):
            i0 = sorted_index[i]
            i1 = sorted_index[i+1]
            dt.append(slice_location[i1] - slice_location[i0])
        dt = np.array(dt)

        if slice_thickness.std() > 1e-6 or dt.std() > 1e-6:
            print ('Warning: slices are not regularly spaced')

        self.set_slice_thickness(slice_thickness[0])

        try:
            self.set_pixel_spacing(dcm.PixelSpacing)
        except:
            print ('No pixel spacing vlaues found in' + dicom_file)
            return False

        self.set_origin([image_position.max(0)[0],image_position.min(0)[1],image_position.min(0)[2]])

        try:
            image_orientation.append([int(round(v)) for v in dcm.ImageOrientationPatient])
        except:
            print ('No image_position found in ' + dicom_file)
            return False
        image_orientation = np.reshape(image_orientation,6)
        raf_orientation = [1,0,0,0,1,0]
        if np.array_equal(image_orientation,raf_orientation):
            self.set_orientation('RAF')

        try:
            rows = int(dcm.Rows)
        except:
            print ('Number of rows not found in ' + dicom_file)
            return False
        try:
            cols = int(dcm.Columns)
        except:
            print ('Number of cols not found in ' + dicom_file)
            return False

        self.init_values(rows, cols, slice_location.shape[0])
        for i, index in enumerate(sorted_index):
            self.filepaths.append(dicom_files[index])
            self.insert_slice(i, dicom.read_file(dicom_files[index]).pixel_array)

        self.values = np.swapaxes(self.values, 0, 1)
        self.setAlfOrientation()


        return True
# transform the scan object in the image reference system
    def setRafOrientation(self): # RAF: x= right to left; y = anterior to posterior; z= foot to head

        if self.orientation == 'ALF':
            self.values = np.flip(self.values, 1)
            self.values = np.swapaxes(self.values,0 ,1)
            self.set_origin(np.array([self.origin[1], self.origin[0] , self.origin[2]]))
            self.spacing = np.array([self.spacing[1], self.spacing[0], self.spacing[2]])
            self.orientation = 'RAF'

# translate the scan object to the model reference system

    def setAlfOrientation(self): # ALF: x= anterior to posterior; y=left to right; z= foot to head


        if self.orientation == 'RAF':
            self.values = np.flip(self.values, 0)
            self.values = np.swapaxes(self.values, 0, 1)
            self.set_origin(np.array([self.origin[1], self.origin[0], self.origin[2]]))
            self.spacing = np.array([self.spacing[1], self.spacing[0], self.spacing[2]])
            self.orientation = 'ALF'

# extract pixel coordinates given an array of points in spatial coordinates
    def getPixelCoordinates(self, points):
        spacing  = self.spacing
        origin = self.origin
        pixelCoords = np.divide(points-origin,spacing)
        pixelCoords = [[round(y,2) for y in x] for x in pixelCoords]
        return  pixelCoords

# extract spacial coordinates giving an array of pixel coordinates
    def getWorldCoordinate(self,value):
        spacing  = self.spacing
        origin = self.origin
        pointCoords = np.array(value*spacing+origin)

        return  pointCoords













