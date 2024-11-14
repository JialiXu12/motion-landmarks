'''
compute the sternum mask based on landmark positions and ribcage segmentation
'''
import os
import numpy as np
import scipy
from scipy.spatial import cKDTree
import pickle
from tools import landmarks_old as ld
from tools import sitkTools


# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: define the line
    p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction; does not need to be normalized.

    return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane
        return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
        )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
        )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )


def load_landmark(filepath):
    data = []
    if os.path.exists(filepath):
        d = pickle.load(open(filepath, 'r'))
        if 'default' in d['data']:
            if type(d['data']['default']) is list:
                if len(d['data']['default']) == 3:
                    data = d['data']['default']
            else:
                if d['data']['default'].size == 3:
                    data = d['data']['default']
    return data

def generate_image_coordinates(image_shape, spacing):
    #import ipdb; ipdb.set_trace()
    #print scan.values.shape
    x, y, z = scipy.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    #print x.shape, y.shape, z.shape
    x = x*spacing[0]
    y = y*spacing[1]
    z = z*spacing[2]

    #import ipdb; ipdb.set_trace()
    image_coor = scipy.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z

def compute_sternum_mask(image, jugular_landmark,prone_cw_mask_path,width_y=15):


    coor, x, y, z = generate_image_coordinates(image.shape, image.spacing)

    image_mask = sitkTools.readNIFTIImage(prone_cw_mask_path)
    image_mask.setAlfOrientation()

    z_tree = cKDTree(scipy.array([z[0,0,:]]).T)

    _, z_start_idx = z_tree.query(scipy.array([jugular_landmark[0,2]]))
    _, z_end_idx = z_tree.query(scipy.array([jugular_landmark[1,2]]))

    p0 = jugular_landmark[0,:]
    p1 = jugular_landmark[1,:]
    p_no = [0., 0., 1.]
    mask= scipy.zeros(image.shape)

    width_x= 20
    z_lenght = 30

    lenght = abs(z_start_idx-z_lenght- (z_end_idx-10))
    cut_y = 20./lenght

    for step, slice_idx in enumerate(range(z_start_idx-z_lenght, z_end_idx+10)):#range(image.shape[-1]):
        #import ipdb; ipdb.set_trace()

        if slice_idx < z.shape[2]:
            p_co = [0.,0., z[0,0,slice_idx]]
            isect_pt = isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6)


            isect_pt_2D_pixels = scipy.array(isect_pt[0:2])/scipy.array(image.spacing[0:2])
            isect_pt_2D_pixels = isect_pt_2D_pixels.round()


            x_start_index = int(isect_pt_2D_pixels[0]-width_x)

            x_end_index = int(isect_pt_2D_pixels[0]+width_x)
            y_start_index = int(isect_pt_2D_pixels[1]-width_y-cut_y*step)
            y_end_index = int(isect_pt_2D_pixels[1]+width_y+cut_y*step)
            mask[x_start_index:x_end_index,y_start_index:y_end_index,slice_idx] = 1

   # struct1 = ndimage.generate_binary_structure(1, 3)
   # image_mask.values = ndimage.binary_errosion(image_mask.values, structure=struct1, iterations=2).astype(image_mask.values.dtype)
    mask_image = image.copy()
    mask = np.ma.array(data=image_mask.values,mask=np.logical_not(mask),
                                                 fill_value=0).filled()


    mask_image.values = mask

    return mask_image



def get_distance_to_skin(point, surface1, surface2):

    ls_distance, ls_point = ld.get_distance_to_surface(point, surface1)
    rs_distance, rs_point = ld.get_distance_to_surface(point,surface2)
    if ls_distance <= rs_distance:
        return  rs_distance,rs_point
    else:
        return  ls_distance,ls_point
