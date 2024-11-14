#!/usr/bin/env python

#> \file
#> \author Thiranja Prasad Babarenda Gamage
#> \brief 
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is Breast Modelling in OpenCMISS
#>
#> The Initial Developer of the Original Code is:
#> Thiranja Prasad Babarenda Gamage
#> Auckland, New Zealand, 
#> All Rights Reserved.
#>
#> Contributor(s):
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. If you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. If you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>

#> Main script

# Standard library imports
import sys
import os
import math
import _pickle as cPickle
import scipy
import scipy.ndimage
import scipy.signal
import automesh
from scipy import spatial
import numpy as np

# Related third party imports
import nibabel as nii

class MRImage():

    def __init__(self, label='', filename='', image=None):
        self.label = label
        self.filename = filename
        if filename is not '':
            if os.path.exists(self.filename):
                self.image = nii.load(self.filename)
                self.setup_properties()
            else:
                raise IOError("No such file: " + filename) 
        elif image is not None:
            self.image = image
            self.setup_properties()

    def setup_properties(self,):
        self.image_shape = self.image.shape[0:3];
        self.header = self.image.get_header();
        self.total_number_of_pixels = np.prod(self.image_shape)
        self.data_type = self.image.get_data().dtype
        self.affine = self.image.get_affine();
        self.voxel_size = self.header['pixdim'][1:4];
        self.image_Size = self.image_shape * self.voxel_size

    def set_origin(self, o1, o2, o3):
        qform = self.image.get_qform()
        qform[0,-1] = o1
        qform[1,-1] = o2
        qform[2,-1] = o3
        self.image.set_qform(qform)
        sform = self.image.get_sform()
        sform[0,-1] = o1
        sform[1,-1] = o2
        sform[2,-1] = o3
        self.image.set_sform(sform)

    def return_data(self, ):
        return self.image.get_data()

    def load_data_vector(self, ):
        self.data_vector = self.image.get_data().ravel()

    def create_from_existing(self, pixels, reference_image):
        if len(pixels) == reference_image.total_number_of_pixels:
            pixels = pixels.reshape(reference_image.image_shape)
        self.image = nii.nifti1.Nifti1Image(pixels,
                                            reference_image.affine,
                                            reference_image.header)
        self.setup_properties()

    def calculate_non_zero_and_zero_indices(self, ):
        self.non_zero_indices = scipy.nonzero(self.data_vector)[0]
        self.zero_indices = np.where(self.data_vector == 0)[0]

    def save(self, folder='', filename=''):
        self.save_folder = folder
        self.save_filename = filename
        # todo: simplify
        if filename is '':
            self.image.to_filename(os.path.join(folder,self.label+'.nii.gz'))
        else:
            self.image.to_filename(os.path.join(folder,filename))

def imfill(data):

    struct = scipy.zeros( [3,3,3],scipy.int8 )
    struct[:,:,1] = scipy.ndimage.generate_binary_structure(2,1)
    data = scipy.ndimage.binary_fill_holes(data,struct)

    return data

def dcmstack_dicom_to_nifti(dicom_dir):
    '''
    Load a set of dicom images using dcmstack and convert to nifti.
    '''
    import dcmstack
    from glob import glob
    src_paths = glob(os.path.join(dicom_dir,'*'))
    stacks = dcmstack.parse_and_stack(src_paths)
    stack = stacks[stacks.keys()[0]]
    stack_data = stack.get_data()
    stack_affine = stack.get_affine()
    dicom_to_nifti = stack.to_nifti()
    return dicom_to_nifti

def nifti_set_RAI_orientation(nifti):
    '''
    Set orientation of nifti to RAI (where the first 3 diagonal entries of
    the affine tranform are -1,-1, 1). 
    TODO: Note that current, only the 2nd entry is updated in this function.
    '''
    nifti.affine[1,1]=-nifti.affine[1,1]
    return nifti

def nifti_zero_origin(nifti):
    '''
    Zero the origin of the nifti image
    '''
    # Set origin to 0
    nifti.affine[0:3,-1]=0
    return nifti
    
def generate_image_coordinates(image_shape, spacing,origin=[0, 0, 0]):
    """
    Generates image coordinates for a dicom image given the image shape and 
    slice spacing
    """
    import scipy
    x, y, z = scipy.mgrid[0:image_shape[0], 0:image_shape[1], 0:image_shape[2]]
    x = x*spacing[0]+origin[0]
    y = y*spacing[1]+origin[1]
    z = z*spacing[2]+origin[2]
    image_coor = scipy.vstack((x.ravel(), y.ravel(), z.ravel())).transpose()
    return image_coor, x, y, z
