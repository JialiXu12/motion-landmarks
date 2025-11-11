import pyvista as pv
import breast_metadata
import mesh_tools
import numpy as np

def plot_all(
    scan=None,
    mask=None,
    point_cloud=None,
    mesh=None,
    mesh_points=None,
    anat_landmarks=None,
    soft_tissue_landmarks=None,
):
    """
        Plots various breast-related data (scan, mask, point cloud, mesh, and landmarks)
        using PyVista, skipping any input that is not provided or is None.

        Inputs are made optional by setting their default value to None.

        Args:
            scan (object, optional): Scan data, convertible to PyVista ImageGrid.
            mask (object, optional): Mask data, convertible to PyVista ImageGrid.
            point_cloud (pv.DataSet or array-like, optional): Point cloud data.
            mesh (object, optional): Mesh data, convertible to meshio then PyVista mesh.
            anat_landmarks (pv.DataSet or array-like or list of such, optional): Anatomical landmarks.
            soft_tissue_landmarks (pv.DataSet or array-like or list of such, optional): Soft tissue landmarks.
        """

    orientation_flag = "RAI"
    plotter = pv.Plotter()
    opacity = np.linspace(0, 0.4, 100)

    # 1. Plot Scan Volume
    if scan is not None:
        try:
            image_grid = breast_metadata.SCANToPyvistaImageGrid(scan, orientation_flag)
            plotter.add_volume(
                image_grid,
                scalars='values',
                cmap='gray',
                opacity=opacity
            )
            # plotter.add_volume(image_grid, opacity='sigmoid_6', cmap='coolwarm', show_scalar_bar=False)
            print("INFO: Plotted Scan Volume.")
        except Exception as e:
            print(f"WARNING: Could not plot scan volume. Error: {e}")

    # 2. Plot Segmentation Mask
    if mask is not None:
        try:
            mask_image_grid = breast_metadata.SCANToPyvistaImageGrid(mask, orientation_flag)
            # Threshold the mask to get the surface/shape
            thresholded_mask = mask_image_grid.threshold(value=0.5)
            plotter.add_mesh(
                thresholded_mask,
                color='lightskyblue',
                opacity=0.2,
                show_scalar_bar=False
            )
            print("INFO: Plotted Segmentation Mask.")
        except Exception as e:
            print(f"WARNING: Could not plot segmentation mask. Error: {e}")

    # 3. Plot Point Cloud
    if point_cloud is not None:
        try:
            plotter.add_points(
                point_cloud,
                color="tan",
                label='Point cloud',
                point_size=2,
                render_points_as_spheres=True
            )
            print("INFO: Plotted Point Cloud.")
        except Exception as e:
            print(f"WARNING: Could not plot point cloud. Error: {e}")

    # 4. Plot Mesh
    if mesh is not None:
        try:
            mesh_meshio = mesh_tools.morphic_to_meshio(mesh, triangulate=True, res=4, exterior_only=True)
            plotter.add_mesh(
                mesh_meshio,
                show_edges=False,
                color='#FFCCCC',
                style="surface",
                opacity=0.5,
                label='Surface_mesh'
            )
            print("INFO: Plotted Surface Mesh.")
        except Exception as e:
            print(f"WARNING: Could not plot mesh. Error: {e}")

    if mesh_points is not None:
        try:
            plotter.add_points(
                mesh_points,
                color='gray',
                render_points_as_spheres=True,
                point_size=5,
                label='3D coordinates of a surface mesh'
            )
            print("INFO: Plotted 3D coordinates of a Surface Mesh.")
        except Exception as e:
            print(f"WARNING: Could not plot 3D coordinates of a Surface Mesh. Error: {e}")

    # 5. Plot Anatomical Landmarks (Handling multiple landmarks)
    if anat_landmarks is not None:
        # Check if the input is a list/tuple of landmarks or a single landmark set
        if isinstance(anat_landmarks, (list, tuple)):
            for i, landmark_set in enumerate(anat_landmarks):
                try:
                    plotter.add_points(
                        landmark_set,
                        color='red',
                        render_points_as_spheres=True,
                        point_size=10,
                        label=f'Anatomical landmark Set {i + 1}'
                    )
                except Exception as e:
                    print(f"WARNING: Could not plot anatomical landmark set {i + 1}. Error: {e}")
            print("INFO: Plotted multiple Anatomical Landmark sets.")
        else:
            try:
                plotter.add_points(
                    anat_landmarks,
                    color='red',
                    render_points_as_spheres=True,
                    point_size=10,
                    label='Anatomical landmarks'
                )
                print("INFO: Plotted single Anatomical Landmark set.")
            except Exception as e:
                print(f"WARNING: Could not plot anatomical landmarks. Error: {e}")

    # 6. Plot Soft Tissue Landmarks (Handling multiple landmarks)
    if soft_tissue_landmarks is not None:
        # Check if the input is a list/tuple of landmarks or a single landmark set
        if isinstance(soft_tissue_landmarks, (list, tuple)):
            for i, landmark_set in enumerate(soft_tissue_landmarks):
                try:
                    plotter.add_points(
                        landmark_set,
                        color='green',  # Changed color for visual distinction
                        render_points_as_spheres=True,
                        point_size=12,
                        label=f'Soft Tissue landmark Set {i + 1}'
                    )
                except Exception as e:
                    print(f"WARNING: Could not plot soft tissue landmark set {i + 1}. Error: {e}")
            print("INFO: Plotted multiple Soft Tissue Landmark sets.")
        else:
            try:
                plotter.add_points(
                    soft_tissue_landmarks,
                    color='green',  # Changed color for visual distinction
                    render_points_as_spheres=True,
                    point_size=12,
                    label='Soft Tissue landmarks'
                )
                print("INFO: Plotted single Soft Tissue Landmark set.")
            except Exception as e:
                print(f"WARNING: Could not plot soft tissue landmarks. Error: {e}")

    # --- Final Step ---
    plotter.show()
    return plotter





