"""
Visualise prone ribcage: mesh + point cloud from segmentation.

Two side-by-side subplots are shown:
  Left  – source mesh  (fpan017 workflow folder) + segmentation point cloud
  Right – copied mesh  (jxu759 volunteer_prone_mesh folder) + segmentation point cloud

Usage
-----
Run directly:
    python scripts/visualise_ribcage_prone.py

To change subject, set VL_ID at the bottom of the file.

Mesh path logic
---------------
Source primary : U:\\sandbox\\fpan017\\meshes\\new_workflow\\ribcage\\new_cases\\{VL_ID_STR}\\ribcage.mesh
Source fallback: U:\\sandbox\\fpan017\\meshes\\new_workflow\\ribcage\\updated\\{VL_ID_STR}\\ribcage.mesh
Local          : U:\\sandbox\\jxu759\\volunteer_prone_mesh\\{VL_ID_STR}_ribcage_prone.mesh

Point cloud
-----------
Created from the prone ribcage segmentation:
    U:\\sandbox\\jxu759\\volunteer_seg\\results\\prone\\rib_cage\\rib_cage_{VL_ID_STR}.nii.gz

Mesh copy utility
-----------------
Use scripts/rename_mesh.py to copy and rename meshes into the copied folder.
"""

from pathlib import Path

import breast_metadata
import morphic
import pyvista as pv

from utils import extract_contour_points
from alignment_utils import get_surface_mesh_coords

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

MESH_PRIMARY_ROOT   = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\updated")
# MESH_PRIMARY_ROOT   = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\iter2")
# MESH_PRIMARY_ROOT   = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\new_cases")


MESH_FALLBACK_ROOT  = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\new_cases")
MESH_LOCAL_ROOT    = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh\v0")
# MESH_LOCAL_ROOT    = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\new_cases_backup")
SEG_ROOT            = Path(r"U:\sandbox\jxu759\volunteer_seg\results\prone\rib_cage")

# Number of surface points to sample from the segmentation
NB_POINTS = 20000

ORIENTATION_FLAG = "RAI"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_mesh_path(vl_id_str: str) -> Path:
    """Return the ribcage mesh path, trying primary then fallback location."""
    primary  = MESH_PRIMARY_ROOT  / vl_id_str / "ribcage.mesh"
    fallback = MESH_FALLBACK_ROOT / vl_id_str / "ribcage.mesh"

    if primary.exists():
        print(f"  Mesh found (primary):  {primary}")
        return primary
    if fallback.exists():
        print(f"  Mesh found (fallback): {fallback}")
        return fallback

    raise FileNotFoundError(
        f"Ribcage mesh not found for {vl_id_str}.\n"
        f"  Looked in: {primary}\n"
        f"  Looked in: {fallback}"
    )


def get_copied_mesh_path(vl_id_str: str) -> Path:
    """Return path to the pre-copied, renamed mesh in the volunteer_prone_mesh folder."""
    path = MESH_LOCAL_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
    if not path.exists():
        raise FileNotFoundError(
            f"Copied mesh not found for {vl_id_str}.\n"
            f"  Looked in: {path}\n"
            f"  Run copy_and_rename_meshes() first to populate this folder."
        )
    print(f"  Copied mesh found:     {path}")
    return path


def get_seg_path(vl_id_str: str) -> Path:
    """Return the prone ribcage segmentation (.nii.gz) path."""
    seg_path = SEG_ROOT / f"rib_cage_{vl_id_str}.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Segmentation not found for {vl_id_str}.\n"
            f"  Looked in: {seg_path}"
        )
    print(f"  Segmentation found:    {seg_path}")
    return seg_path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ribcage_mesh(vl_id_str: str) -> morphic.Mesh:
    """Load and return the prone ribcage morphic mesh."""
    mesh_path = get_mesh_path(vl_id_str)
    return morphic.Mesh(str(mesh_path))


def load_ribcage_point_cloud(vl_id_str: str, nb_points: int = NB_POINTS):
    """Load prone ribcage segmentation and return surface point cloud (N, 3)."""
    seg_path = get_seg_path(vl_id_str)
    mask = breast_metadata.readNIFTIImage(
        str(seg_path), orientation_flag=ORIENTATION_FLAG, swap_axes=True
    )
    pc = extract_contour_points(mask, nb_points)
    print(f"  Point cloud extracted: {pc.shape[0]} points")
    return pc


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _add_mesh(plotter: pv.Plotter, mesh: morphic.Mesh, label: str,
              color: str, res: int = 26) -> None:
    """
    Add a morphic mesh to *plotter* as a sampled point cloud.
    Matches alignment.py: get_surface_mesh_coords → add_points.
    """
    mesh_coords = get_surface_mesh_coords(mesh, res=res, elems=[])
    plotter.add_points(mesh_coords, color=color, point_size=2,
                       render_points_as_spheres=True, label=label)
    print(f"    INFO: Plotted 3D coordinates of a Surface Mesh.")


# ---------------------------------------------------------------------------
# Visualisation – two side-by-side subplots
# ---------------------------------------------------------------------------

def plot_ribcage_comparison(
    vl_id_str: str,
    source_mesh: morphic.Mesh,
    copied_mesh: morphic.Mesh,
    point_cloud,
) -> None:
    """
    Side-by-side comparison:
      Left  – source mesh  (fpan017 workflow)  + segmentation point cloud
      Right – copied mesh  (volunteer_prone_mesh folder) + segmentation point cloud
    """
    plotter = pv.Plotter(shape=(1, 2), border=False)
    axes_labels = dict(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")

    # ---- Left subplot: source mesh ----------------------------------------
    plotter.subplot(0, 0)
    plotter.add_text(f"{vl_id_str}  |  source mesh (fpan017)",
                     position="upper_left", font_size=9, color="black")
    print("  [Left] adding source mesh...")
    _add_mesh(plotter, source_mesh, label="Source mesh", color="#FFCCCC")
    plotter.add_points(point_cloud, color="steelblue", point_size=2,
                       render_points_as_spheres=True, label="Seg point cloud")
    plotter.add_axes(**axes_labels)
    plotter.add_legend(bcolor="w", size=(0.25, 0.15))

    # ---- Right subplot: copied mesh ---------------------------------------
    plotter.subplot(0, 1)
    plotter.add_text(f"{vl_id_str}  |  v0 mesh (volunteer_prone_mesh)",
                     position="upper_left", font_size=9, color="black")
    print("  [Right] adding v0 mesh...")
    _add_mesh(plotter, copied_mesh, label="v0 mesh", color="#CCFFCC")
    plotter.add_points(point_cloud, color="steelblue", point_size=2,
                       render_points_as_spheres=True, label="Seg point cloud")
    plotter.add_axes(**axes_labels)
    plotter.add_legend(bcolor="w", size=(0.25, 0.15))

    plotter.link_views()   # sync camera between subplots
    plotter.show()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(vl_id: int) -> None:
    vl_id_str = f"VL{vl_id:05d}"
    print(f"\n=== Visualising prone ribcage for {vl_id_str} ===")

    source_mesh = load_ribcage_mesh(vl_id_str)
    copied_mesh = morphic.Mesh(str(get_copied_mesh_path(vl_id_str)))
    point_cloud = load_ribcage_point_cloud(vl_id_str)

    plot_ribcage_comparison(vl_id_str, source_mesh, copied_mesh, point_cloud)


if __name__ == "__main__":
    VL_ID = [20]
    for id in VL_ID:# change to the subject you want to inspect
        main(id)
