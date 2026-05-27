"""Verification test: load a previously SAVED transformed prone ribcage mesh
from disk, sample it, load the matching supine point cloud, and render an
overlay PNG (off-screen).

Proves the saved .mesh file is self-consistent — we never touch the original
prone mesh nor the transformation matrix when rendering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _SCRIPT_DIR,
    _PROJECT_ROOT,
    _PROJECT_ROOT / "src" / "morphic",
    _PROJECT_ROOT / "src" / "mesh-tools",
    _PROJECT_ROOT / "external",
    _PROJECT_ROOT / "external" / "breast_metadata_mdv",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import morphic
import pyvista as pv

from alignment_utils import get_surface_mesh_coords
from save_transformed_prone_mesh import load_supine_pc

VL_ID = 9
OUT_DIR = _PROJECT_ROOT / "output" / "alignment" / "transformed_prone_mesh_v8"
SAVED_MESH = OUT_DIR / f"VL{VL_ID:05d}_ribcage_prone_transformed.mesh"
SUPINE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
SCREENSHOT_OUT = OUT_DIR / "previews" / f"VL{VL_ID:05d}_loaded_overlay.png"

print(f"Loading saved mesh: {SAVED_MESH}")
assert SAVED_MESH.exists(), (
    f"Run: python save_transformed_prone_mesh.py --vl_id {VL_ID} "
    f"--prone_root ../test_data/prone --out_dir {OUT_DIR} --no_plot")

mesh = morphic.Mesh(str(SAVED_MESH))
sample = get_surface_mesh_coords(mesh, res=26)
print(f"  sample shape = {sample.shape}")
print(f"  sample bbox  = X[{sample[:,0].min():.1f}, {sample[:,0].max():.1f}]"
      f"  Y[{sample[:,1].min():.1f}, {sample[:,1].max():.1f}]"
      f"  Z[{sample[:,2].min():.1f}, {sample[:,2].max():.1f}]")

print("Loading supine PC...")
supine_pc = load_supine_pc(VL_ID, SUPINE_ROOT)
if supine_pc is not None:
    print(f"  supine PC shape = {supine_pc.shape}")
    from scipy.spatial import cKDTree
    tree = cKDTree(supine_pc)
    dists, _ = tree.query(sample)
    print(f"  mesh-to-supine nearest distance: "
          f"mean={dists.mean():.2f} mm  median={np.median(dists):.2f} mm  "
          f"max={dists.max():.2f} mm")

SCREENSHOT_OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"Rendering to: {SCREENSHOT_OUT}")
plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
plotter.add_text(f"VL{VL_ID:05d}: loaded-from-disk transformed mesh vs supine PC",
                 font_size=14)
if supine_pc is not None:
    plotter.add_points(supine_pc, color="tan", point_size=2,
                       render_points_as_spheres=True, label="Supine PC")
plotter.add_points(sample, color="cornflowerblue", point_size=2,
                   render_points_as_spheres=True, label="Loaded transformed mesh")
plotter.add_points(np.array([[0.0, 0.0, 0.0]]), color="orange",
                   point_size=12, render_points_as_spheres=True, label="Origin")
plotter.add_legend()
plotter.show(screenshot=str(SCREENSHOT_OUT), auto_close=True)
print(f"Done. Screenshot at: {SCREENSHOT_OUT}")
