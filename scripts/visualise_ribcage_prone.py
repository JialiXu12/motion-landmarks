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
Copied         : U:\\sandbox\\jxu759\\volunteer_prone_mesh\\{VL_ID_STR}_ribcage_prone.mesh

Point cloud
-----------
Created from the prone ribcage segmentation:
    U:\\sandbox\\jxu759\\volunteer_seg\\results\\prone\\rib_cage\\rib_cage_{VL_ID_STR}.nii.gz

Mesh copy utility
-----------------
Call copy_and_rename_meshes() to batch-copy ribcage.mesh files from the
source tree into a flat destination folder as VL00009_ribcage_prone.mesh etc.
"""

import shutil
from pathlib import Path

import breast_metadata
import morphic
import pyvista as pv

from utils import extract_contour_points
from alignment_utils import get_surface_mesh_coords

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

MESH_PRIMARY_ROOT   = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\new_cases")
MESH_FALLBACK_ROOT  = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\updated")
MESH_COPIED_ROOT    = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SEG_ROOT            = Path(r"U:\sandbox\jxu759\volunteer_seg\results\prone\rib_cage")

# Destination used by the copy utility
MESH_DEST_DIR       = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")

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
    path = MESH_COPIED_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
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
    plotter.add_text(f"{vl_id_str}  |  copied mesh (volunteer_prone_mesh)",
                     position="upper_left", font_size=9, color="black")
    print("  [Right] adding copied mesh...")
    _add_mesh(plotter, copied_mesh, label="Copied mesh", color="#CCFFCC")
    plotter.add_points(point_cloud, color="steelblue", point_size=2,
                       render_points_as_spheres=True, label="Seg point cloud")
    plotter.add_axes(**axes_labels)
    plotter.add_legend(bcolor="w", size=(0.25, 0.15))

    plotter.link_views()   # sync camera between subplots
    plotter.show()


# ---------------------------------------------------------------------------
# Mesh copy utility  (disabled — call manually when needed)
# ---------------------------------------------------------------------------

def copy_and_rename_meshes(
    dest_dir: Path = MESH_DEST_DIR,
    primary_root: Path = MESH_PRIMARY_ROOT,
    fallback_root: Path = MESH_FALLBACK_ROOT,
) -> None:
    """
    Walk both source roots and copy every ribcage.mesh to *dest_dir*,
    renaming each file to  <VL_ID_STR>_ribcage_prone.mesh.

    Primary root is tried first; if a subject folder exists in both roots the
    primary copy wins (already-copied files are not overwritten).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for root in (primary_root, fallback_root):
        if not root.is_dir():
            print(f"  WARNING: Source directory not found, skipping: {root}")
            continue

        for subject_dir in sorted(root.iterdir()):
            if not subject_dir.is_dir():
                continue

            subject_id    = subject_dir.name           # e.g. "VL00009"
            source_file   = subject_dir / "ribcage.mesh"
            dest_file     = dest_dir / f"{subject_id}_ribcage_prone.mesh"

            if not source_file.exists():
                continue

            if dest_file.exists():
                print(f"  [SKIP]   {dest_file.name}  (already exists)")
                skipped += 1
                continue

            try:
                shutil.copy2(source_file, dest_file)
                print(f"  [COPIED] {source_file}  ->  {dest_file.name}")
                copied += 1
            except Exception as exc:
                print(f"  [ERROR]  Could not copy {source_file}: {exc}")

    print(f"\nFinished copying meshes: {copied} copied, {skipped} skipped.")


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
    VL_ID = [32,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50]
    for id in VL_ID:# change to the subject you want to inspect
        main(id)
