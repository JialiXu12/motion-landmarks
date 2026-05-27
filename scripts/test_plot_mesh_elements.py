"""
Test script for plot_mesh_elements function.
Tests the robust mesh plotting with different mesh types.

This is a standalone test that doesn't import the full alignment module
to avoid complex import chain issues.
"""

import sys
from pathlib import Path
import numpy as np

# Setup paths - ORDER MATTERS!
script_dir = Path(__file__).parent
project_root = script_dir.parent

sys.path.insert(0, str(project_root / "src" / "morphic"))
sys.path.insert(0, str(project_root / "src" / "mesh-tools"))
sys.path.insert(0, str(project_root / "external" / "breast_metadata_mdv"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

import morphic
import mesh_tools
import pyvista as pv


def get_surface_mesh_coords(morphic_mesh, res, elems=None):
    """
    Extracts the 3D coordinates of a surface mesh
    """
    if elems is None:
        elems = []

    Xi = morphic_mesh.grid(res, method='center')
    NPPE = Xi.shape[0]

    if elems == []:
        NE = morphic_mesh.elements.size()
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(morphic_mesh.elements):
            eid = element.id
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[eid].evaluate(Xi)
    else:
        NE = len(elems)
        mesh_coords = np.zeros((NE * NPPE, 3))
        for i, element in enumerate(elems):
            mesh_coords[i * NPPE:(i + 1) * NPPE, :] = morphic_mesh.elements[element].evaluate(Xi)

    return mesh_coords


def plot_mesh_elements(mesh, ribcage_point_cloud=None, mesh_resolution=4):
    """
    Extract element centers from morphic mesh and visualize with labels.
    Handles different mesh basis types robustly.
    """
    # Detect mesh dimensionality by checking first element's basis
    first_element = list(mesh.elements)[0]
    mesh_dims = len(first_element.basis)
    print(f"INFO: Mesh has {mesh_dims}D elements with basis {first_element.basis}")

    centers = []
    num_elements = mesh.elements.size()

    # Generate appropriate Xi grid based on mesh dimensionality
    try:
        Xi = mesh.grid(3, method='center')
        for i in range(num_elements):
            elem = list(mesh.elements)[i]
            elem_coords = elem.evaluate(Xi)
            center_idx = elem_coords.shape[0] // 2
            center = elem_coords[center_idx, :]
            centers.append(center)
    except Exception as e:
        print(f"WARNING: Could not compute element centers: {e}")
        # Fallback: use node positions as centers
        for i, elem in enumerate(mesh.elements):
            node_ids = elem.node_ids
            node_coords = np.array([mesh.get_nodes(nid)[0] for nid in node_ids])
            center = node_coords.mean(axis=0)
            centers.append(center)

    centers_array = np.array(centers)
    if centers_array.ndim == 1:
        centers_array = centers_array.reshape(1, 3)

    # Visualize with PyVista
    plt = pv.Plotter()
    mesh_added = False

    # For 3D volume meshes, extract surface
    if mesh_dims == 3:
        print("INFO: Volume mesh detected, extracting surface...")
        try:
            mesh_nodes = mesh.get_nodes()
            mesh_cloud = pv.PolyData(mesh_nodes)
            mesh_surface = mesh_cloud.delaunay_3d().extract_surface()
            plt.add_mesh(mesh_surface, color='#FFCCCC', opacity=0.5,
                         show_edges=True, edge_color='gray', label='Surface_mesh')
            mesh_added = True
            print("INFO: Plotted volume mesh surface using Delaunay 3D")
        except Exception as e:
            print(f"WARNING: Delaunay 3D failed: {e}")
            try:
                mesh_nodes = mesh.get_nodes()
                plt.add_points(mesh_nodes, color='#FFCCCC', point_size=3,
                               render_points_as_spheres=True, label='Mesh_nodes')
                mesh_added = True
            except Exception as e2:
                print(f"WARNING: Could not plot volume mesh: {e2}")
    else:
        # For 2D surface meshes, try morphic_to_meshio first
        try:
            mesh_meshio = mesh_tools.morphic_to_meshio(mesh, triangulate=True,
                                                        res=mesh_resolution, exterior_only=True)
            plt.add_mesh(mesh_meshio, show_edges=False, color='#FFCCCC',
                         style="surface", opacity=0.5, label='Surface_mesh')
            mesh_added = True
            print("INFO: Plotted mesh using morphic_to_meshio triangulation")
        except Exception as e:
            print(f"WARNING: morphic_to_meshio failed: {e}")
            try:
                mesh_coords = get_surface_mesh_coords(mesh, res=mesh_resolution, elems=[])
                mesh_cloud = pv.PolyData(mesh_coords)
                mesh_surface = mesh_cloud.delaunay_3d().extract_surface()
                plt.add_mesh(mesh_surface, color='#FFCCCC', opacity=0.5,
                             show_edges=True, edge_color='gray', label='Surface_mesh')
                mesh_added = True
            except Exception as e2:
                print(f"WARNING: Delaunay failed: {e2}")
                plt.add_points(mesh_coords, color='#FFCCCC', point_size=3,
                               render_points_as_spheres=True, label='Mesh_points')
                mesh_added = True

    if ribcage_point_cloud is not None:
        plt.add_points(ribcage_point_cloud, color='tan', label='Point cloud',
                       point_size=2, render_points_as_spheres=True)

    plt.add_point_labels(
        centers_array,
        labels=[str(i) for i in range(num_elements)],
        font_size=14, text_color='black', point_size=10, point_color='red',
        always_visible=True, shadow=True
    )

    plt.add_axes()
    plt.show()

    return centers_array

# Test parameters
TEST_SUBJECTS = [9, 22]  # VL00009 and VL00022 (which previously failed)
MESH_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")


def test_plot_mesh_for_subject(vl_id: int):
    """Test plotting mesh for a specific subject."""
    vl_id_str = f"VL{vl_id:05d}"
    mesh_path = MESH_ROOT / f"{vl_id_str}_ribcage_prone.mesh"

    print(f"\n{'='*60}")
    print(f"Testing plot_mesh_elements for {vl_id_str}")
    print(f"Mesh path: {mesh_path}")
    print(f"{'='*60}")

    if not mesh_path.exists():
        print(f"ERROR: Mesh file not found: {mesh_path}")
        return False

    try:
        # Load the morphic mesh
        mesh = morphic.Mesh(str(mesh_path))
        print(f"Loaded mesh: {mesh.elements.size()} elements, {mesh.nodes.size()} nodes")

        # Check element basis
        first_elem = list(mesh.elements)[0]
        print(f"First element basis: {first_elem.basis}")

        # Test get_surface_mesh_coords
        print("\nTesting get_surface_mesh_coords...")
        try:
            coords = get_surface_mesh_coords(mesh, res=3, elems=[])
            print(f"Surface mesh coords shape: {coords.shape}")
        except Exception as e:
            print(f"get_surface_mesh_coords failed: {e}")

        # Test plot_mesh_elements
        print("\nTesting plot_mesh_elements...")
        centers = plot_mesh_elements(mesh, ribcage_point_cloud=None, mesh_resolution=4)
        print(f"Element centers shape: {centers.shape}")

        print(f"\nSUCCESS: plot_mesh_elements worked for {vl_id_str}")
        return True

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all test subjects."""
    print("Testing plot_mesh_elements with different mesh types")
    print("="*60)

    results = {}
    for vl_id in TEST_SUBJECTS:
        results[vl_id] = test_plot_mesh_for_subject(vl_id)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for vl_id, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"VL{vl_id:05d}: {status}")


if __name__ == "__main__":
    main()









