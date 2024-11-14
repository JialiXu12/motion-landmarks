import pyvista as pv


def SCANToPyvistaImageGrid(scan, orientation_flag):
    if orientation_flag == "ALS":
        scan.setAlsOrientation()
    else:
        assert orientation_flag == "RAI", "RAI or ALS are the only supported orientation flags." \
                                          f"{orientation_flag} is not currently supported"
        scan.setRaiOrientation()

    image_grid = pv.ImageData()
    image_grid.dimensions = scan.values.shape
    image_grid.origin = scan.origin
    image_grid.spacing = scan.spacing
    image_grid.point_data["values"] = scan.values.flatten(order="F")
    return image_grid