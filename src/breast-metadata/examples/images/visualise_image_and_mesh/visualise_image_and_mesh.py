import pyvista as pv
import pandas as pd
import os
import mesh_tools
import morphic
import breast_metadata
import argparse
import pathlib


def visualise_prone_mesh_and_image(meta_data_file,
                                   volunteer,
                                   orientation_flag):

    # get meta data path
    meta_data_path = pathlib.Path(meta_data_file).parents[0]

    # read meta data
    meta_data = pd.read_csv(meta_data_file)

    # locate volunteer image path
    t1w_prone_relative_path = meta_data.loc[(meta_data["participant_id"] == volunteer),
                                            "mri_t1_prone_arms_down_image_path"].values[0]

    t1w_prone_path = os.path.join(meta_data_path,
                                  t1w_prone_relative_path)

    # Load scan
    scan = breast_metadata.Scan(t1w_prone_path)

    # Convert Scan to Pyvista Image Grid in desired orientation
    image_grid = breast_metadata.SCANToPyvistaImageGrid(scan,
                                                        orientation_flag)

    # Load prone mesh in morphic format.
    torso_mesh_path = meta_data.loc[(meta_data["participant_id"] == volunteer),
                                    "volume_lagrange_mesh_path"].values[0]

    torso_mesh = morphic.Mesh(os.path.join(meta_data_path,
                                           torso_mesh_path))

    # Transform morphic mesh node coordinates.
    for node in torso_mesh.nodes:
        node.values[:] = scan.transformPointToImageSpace(node.values)

    meshio_mesh = mesh_tools.morphic_to_meshio(
        torso_mesh, triangulate=True, res=4, exterior_only=True)

    plotter = pv.Plotter()
    plotter.add_mesh(meshio_mesh,
                     show_edges=True,
                     style="wireframe")

    plotter.add_points(meshio_mesh,
                       show_edges=False,
                       scalars=meshio_mesh.points[:,
                               orientation_flag.index('A')],
                       style="points",
                       point_size=8,
                       render_points_as_spheres=True,
                       show_scalar_bar=False)

    plotter.add_volume(image_grid,
                       opacity='sigmoid_6',
                       cmap='coolwarm',
                       show_scalar_bar=False)

    plotter.add_text(f"Image and Mesh in {orientation_flag} Coordinates")
    labels = dict(xlabel=f"{orientation_flag[0]} (mm)",
                  ylabel=f"{orientation_flag[1]} (mm)",
                  zlabel=f"{orientation_flag[2]} (mm)")

    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Visualise Image and Mesh',
        description='An example script that can be used to bring images and meshes from the BBRG workflow'
                    'into a common coordinate system and visualise using PyVista')

    parser.add_argument('-m',
                        '--meta_data_file')

    parser.add_argument('-v',
                        '--volunteer',
                        default='VL00010')

    parser.add_argument('-0',
                        '--orientation_flag',
                        default="RAI")

    args = parser.parse_args()

    visualise_prone_mesh_and_image(meta_data_file=args.meta_data_file,
                                   volunteer=args.volunteer,
                                   orientation_flag=args.orientation_flag)