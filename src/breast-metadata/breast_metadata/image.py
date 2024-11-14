import scipy

def generate_image_coordinates(image_shape, spacing):
    x, y, z = scipy.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    x = x*spacing[0]
    y = y*spacing[1]
    z = z*spacing[2]
    image_coor = scipy.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z
