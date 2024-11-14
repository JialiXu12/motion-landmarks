import SimpleITK as sitk
import pydicom as dicom
import pyvista as pv
import numpy as np
import sys
import os


class Scan(object):
    """This object is used for loading and minpulating DICOM based medical images.
    In our group we aim inform our biomechanics approaches using medical images.
    Therefore, this object enable transformations that bring the image and biomechanics
    meshes into a common coordinate system.

    Medical images are stored in two parts 1) an image array and 2) the meta data
    which contains the information required to map the image array into a right-hand/real-world
    coordinate system that is with respect to the imaging modality, e.g. the center of the MRI bore.
    The right-hand/real-world coordinate system is crucial for the biomechanics as it allows the
    image and biomechanics mesh to be processed, analysed and visualised in a anatomically and
    physically plausably manner.

    The image array uses a voxel coordinate system I (column) J (row) K (slice).
    Whereas a right-handed/real-world coordinate system X Y Z.
    Characters R (right), L (left), A (anterior), P (posterior), I (inferior), S (superior)
    are used to denote the direction components in the coordinate system, e.g. x = Right to left;
    y = Anterior to posterior; z = Inferior to superior. The DICOM meta data give the patient orientation
    vectors for X and Y (6 direction cosines) and Z as the right-handed cross product. The DICOM
    meta data also contains the voxel spacing and patient origin with respect to the MRI
    coordinate system (typically using the center of the bore as the origin)

    In the BBRG group we have standardised our image coordinate system as RAI. Our rationale being
    that ITK, DICOM and NRRD all use an equivalent coordinate system [1]: ITK RAI [2], DICOM LPS [3] or
    NRRD left-posterior-superior [4].

    Despite standardising RAI, our current automatic meshing tool uses ALS. Therefore, this object can be
    used to convert between RAI and ALS. The DICOM's themselves can be loaded in either RAI and ALS,
    and changed at the users desire.

    References:
    1. Aliza Medical Imaging Orientation Documentation: https://www.aliza-dicom-viewer.com/manual/orientation
    2. ITK Documentation: https://itk.org/Wiki/ITK/MetaIO/Documentation
    3. NEMA DICOM Documentation:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1
    4. Definition of NRRD File Format: https://teem.sourceforge.net/nrrd/format.html

    Attributes (#TO DO)
    ----------
    bar : str
        A string

    Methods
    -------
    __init__(fish):
        Constructor, fish is a str which self.bar will be set to.
    baz():
        A function which does blah

    """

    def __init__(self, dicom_files=None, load_dicom=True):
        self.num_slices = 0
        self.origin = np.array([0., 0., 0.])
        self.spacing = np.array([1., 1., 1.])
        self.orientation = 'RAI'  # [1,0,0,0,1,0] RAI orientation:
        # The BBRG has standardised RAI orientation:
        # The X vector is (1,0,0) meaning it is exactly
        # directed with the image pixel matrix row direction
        #  and the Y vector is (0,1,0) meaning it is exactly
        # directed with the image pixel matrix column direction.
        self.filepaths = []
        self.values = np.zeros((1, 1, 1))
        self.age = None  # subject age, weight and height
        self.weight = None
        self.height = None

        if load_dicom:
            try:
                print('Loading dicom files from: ' + dicom_files)
                self.load_files(dicom_files)
            except FileNotFoundError as e:
                print(f"Can not load DICOM file."
                      f"File {dicom_files} not found: {e}", file=sys.stderr)
                return

    def copy(self, copy_values=True):
        new_scan = Scan(load_dicom=False)
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
                    print('No slice location found in ' + dicom_file)
                    return False
                try:
                    slice_thickness.append(float(dcm.SliceThickness))
                except:
                    print('No slice thickness found in ' + dicom_file)
                    return False
                try:
                    image_position.append([float(v) for v in dcm.ImagePositionPatient])
                except:
                    print('No image_position found in ' + dicom_file)
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
            i1 = sorted_index[i + 1]
            dt.append(slice_location[i1] - slice_location[i0])
        dt = np.array(dt)

        if slice_thickness.std() > 1e-6 or dt.std() > 1e-6:
            print('Warning: slices are not regularly spaced')

        self.set_slice_thickness(slice_thickness[0])

        try:
            self.set_pixel_spacing(dcm.PixelSpacing)
        except:
            print('No pixel spacing vlaues found in' + dicom_file)
            return False

        self.set_origin([image_position.max(0)[0], image_position.min(0)[1], image_position.min(0)[2]])

        try:
            image_orientation.append([int(round(v)) for v in dcm.ImageOrientationPatient])
        except:
            print('No image_position found in ' + dicom_file)
            return False
        image_orientation = np.reshape(image_orientation, 6)
        rai_orientation = [1, 0, 0, 0, 1, 0]
        if np.array_equal(image_orientation, rai_orientation):
            self.set_orientation('RAI')

        try:
            rows = int(dcm.Rows)
        except:
            print('Number of rows not found in ' + dicom_file)
            return False
        try:
            cols = int(dcm.Columns)
        except:
            print('Number of cols not found in ' + dicom_file)
            return False

        self.init_values(rows, cols, slice_location.shape[0])
        for i, index in enumerate(sorted_index):
            self.filepaths.append(dicom_files[index])
            self.insert_slice(i, dicom.read_file(dicom_files[index]).pixel_array)

        self.values = np.swapaxes(self.values, 0, 1)
        self.setRaiOrientation()

        try:
            self.age = int(dcm.PatientAge[0:3])
        except:
            print('Subject {0} age not found in ' + dicom_files)
            pass
        try:
            self.weight = float(dcm.PatientWeight)
        except:
            print('Subject {0} weight not found ' + dicom_files)
            pass
        try:
            self.height = float(dcm.PatientSize)
        except:
            print('Subject {0} height not found ' + dicom_files)
            pass

        return True

    def setRaiOrientation(self):  # RAI: x = Right to left;
        # y = Anterior to posterior;
        # z = Inferior to superior

        if self.orientation == 'ALS':
            self.values = np.flip(self.values, 1)
            self.values = np.swapaxes(self.values, 0, 1)
            self.set_origin(np.array([self.origin[1], self.origin[0], self.origin[2]]))
            self.spacing = np.array([self.spacing[1], self.spacing[0], self.spacing[2]])
            self.orientation = 'RAI'

    def setAlsOrientation(self):  # ALS: x = anterior to posterior;
        # y = left to right;
        # z = Superior to inferior

        if self.orientation == 'RAI':
            self.values = np.flip(self.values, 0)
            self.values = np.swapaxes(self.values, 0, 1)
            self.set_origin(np.array([self.origin[1], self.origin[0], self.origin[2]]))
            self.spacing = np.array([self.spacing[1], self.spacing[0], self.spacing[2]])
            self.orientation = 'ALS'

    def getPixelCoordinates(self, points):
        spacing = self.spacing
        origin = self.origin
        pixelCoords = np.divide(points - origin, spacing)
        # pixelCoords = [[round(y,2) for y in x] for x in pixelCoords]
        return pixelCoords

    def getWorldCoordinate(self, value):
        spacing = self.spacing
        origin = self.origin
        pointCoords = np.array(value * spacing + origin)

        return pointCoords

    def transformPointToImageSpace(self, point):

        spacing = self.spacing
        origin = self.origin
        size = self.values.shape

        if self.orientation == "RAI":
            point = [-point[1] + origin[0] + spacing[0] * size[0],
                     point[0] + origin[1], point[2] + origin[2]]

        else:
            # Work around: If the points are not in RAI they're assumed to be in ALS
            assert self.orientation == "ALS", "Scan assumes either RAI or ALS orientation." \
                                              f"{self.orientation} is not currently supported"
            point = [point[0] + origin[0],
                     point[1] + origin[1], point[2] + origin[2]]
            # assert self.orientation == "RAI", "Please set orientation to " \
            #                                   "RAI using " \
            #                                   "to ensure image and point space" \
            #                                   "align."

        return point

    def exportAsNIFTI(self, path):

        assert path.endswith('.nii') or path.endswith('.nii.gz'), f"File extension not recognised: {path}" \
                                                                  f"Currently support .nii or .nii.gz only."
        image = sitk.GetImageFromArray(self.values)
        image.SetOrigin(self.origin)
        image.SetSpacing(self.spacing)
        sitk.WriteImage(image, path)