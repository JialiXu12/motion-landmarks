"""Apply a saved alignment transformation matrix to the original prone ribcage
mesh, save the transformed mesh back to .mesh format, and (optionally) plot it
together with the supine point cloud.

This is a read-only companion to main_alignment.py / apply_saved_alignment.py:
no existing scripts or modules are modified. The original prone .mesh files
are also never written to — every run loads a fresh copy.

Background
----------
The prone ribcage is a morphic cubic-Hermite (H3 x H3) surface mesh. Each
standard node carries values of shape (3, 4):

    column 0: position (x, y, z)
    column 1: d/dxi_1  (tangent vector)
    column 2: d/dxi_2  (tangent vector)
    column 3: d^2/dxi_1 dxi_2 (cross-derivative vector)

Under an affine transform T = [R | t], **positions** transform as
``p' = R @ p + t``, but **derivative columns are vectors**, so they only
rotate: ``d' = R @ d``. The transformation matrix saved by main_alignment.py
already encodes the sternum-anchoring translation in t.

Usage
-----
    python save_transformed_prone_mesh.py              # all subjects with a saved T_matrix
    python save_transformed_prone_mesh.py --vl_id 9 22
    python save_transformed_prone_mesh.py --no_plot
    python save_transformed_prone_mesh.py --screenshot --no_show
    python save_transformed_prone_mesh.py --t_matrix_dir ../output/alignment/transformation_matrix_v7
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repo-local dependencies are importable regardless of the active env.
# Matches the pattern used in test_alignment_cohort.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _SCRIPT_DIR,
    _PROJECT_ROOT,
    _PROJECT_ROOT / "src" / "morphic",
    _PROJECT_ROOT / "external",
    _PROJECT_ROOT / "external" / "breast_metadata_mdv",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math

import numpy as np
import morphic
import pyvista as pv

from alignment_utils import get_surface_mesh_coords


# ── Local helpers (inlined to avoid pulling in utils.py's heavy import chain) ──
def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to Nx3 points. Mirrors utils.apply_transform."""
    if points.shape[0] == 0:
        return np.empty((0, 3))
    ones = np.ones((len(points), 1))
    return (T @ np.hstack((points, ones)).T)[:-1, :].T


def extract_contour_points(mask, nb_points: int) -> np.ndarray:
    """Sample contour points from a NIFTI mask. Mirrors utils.extract_contour_points."""
    from skimage.segmentation import find_boundaries
    labels = mask.values.copy()
    boundaries = find_boundaries(labels, mode="inner").astype(np.uint8)
    boundary_indices = np.argwhere(boundaries)
    points = boundary_indices.astype(np.float64) * np.array(mask.spacing) + np.array(mask.origin)
    if nb_points < len(points):
        step = math.trunc(len(points) / nb_points)
        return points[range(0, len(points), step), :]
    return points


# ── Defaults ──────────────────────────────────────────────────────────────────
T_MATRIX_DIR = Path(r"../output/alignment/transformation_matrix_v8")
PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
OUTPUT_MESH_DIR = Path(r"../output/alignment/transformed_prone_mesh_v8")

DEFAULT_MESH_RES = 26
DEFAULT_NB_CONTOUR_POINTS = 20000
DEFAULT_ORIENTATION_FLAG = "RAI"

VL_ID_PATTERN = re.compile(r"VL(\d{5})_transform_matrix\.npy$")


# ── Core transform ────────────────────────────────────────────────────────────
def transform_morphic_mesh_inplace(mesh: morphic.Mesh, T_total: np.ndarray) -> dict:
    """Apply ``T_total`` to every standard node of ``mesh`` in place.

    Positions get rotated and translated; derivative columns get rotated only.
    Dependent and PCA nodes are skipped (their values follow from their parent
    standard nodes and would be double-transformed otherwise).

    Parameters
    ----------
    mesh : morphic.Mesh
        Mesh to transform in place. Reload a fresh copy from disk if you do
        not want to mutate ``mesh`` for downstream use.
    T_total : (4, 4) ndarray
        Affine transformation. ``R = T_total[:3, :3]``, ``t = T_total[:3, 3]``.

    Returns
    -------
    info : dict
        Counts of nodes touched, skipped, and any anomalies (unexpected shapes).
    """
    if T_total.shape != (4, 4):
        raise ValueError(f"T_total must be (4, 4), got {T_total.shape}")

    R = T_total[:3, :3]
    t = T_total[:3, 3]

    n_standard = 0
    n_skipped = 0
    skipped_types: dict = {}
    bad_shapes: list = []

    for node in mesh.nodes:
        if node._type != "standard":
            n_skipped += 1
            skipped_types[node._type] = skipped_types.get(node._type, 0) + 1
            continue

        vals = np.array(node.values, copy=True)  # shape (num_fields, num_components)

        if vals.shape[0] != 3:
            bad_shapes.append((node.id, vals.shape))
            continue

        new_vals = vals.copy()
        new_vals[:, 0] = R @ vals[:, 0] + t  # position: rotate + translate
        if vals.shape[1] > 1:
            new_vals[:, 1:] = R @ vals[:, 1:]  # derivative vectors: rotate only

        node.set_values(new_vals)
        n_standard += 1

    mesh.generate(True)

    return {
        "n_standard_transformed": n_standard,
        "n_skipped": n_skipped,
        "skipped_types": skipped_types,
        "bad_shapes": bad_shapes,
    }


# ── Sanity check ─────────────────────────────────────────────────────────────
def _sanity_check(pre_sample: np.ndarray, post_sample: np.ndarray,
                  T_total: np.ndarray, tol: float = 1e-6) -> float:
    """Compute RMSE between the resampled transformed mesh and the analytical
    ``T_total @ pre_sample``. Returns RMSE in mm.
    """
    expected = apply_transform(pre_sample, T_total)
    rmse = float(np.sqrt(np.mean(np.sum((post_sample - expected) ** 2, axis=1))))
    return rmse


# ── Supine PC loader ─────────────────────────────────────────────────────────
def load_supine_pc(
    vl_id: int,
    supine_ribcage_root: Path,
    orientation_flag: str = DEFAULT_ORIENTATION_FLAG,
    nb_contour_points: int = DEFAULT_NB_CONTOUR_POINTS,
) -> Optional[np.ndarray]:
    """Load the supine ribcage segmentation and return a contour point cloud.

    Returns None (with a printed warning) if the file is missing or breast_metadata
    cannot be imported — plotting / saving the mesh still proceeds without it.
    """
    vl_id_str = f"VL{vl_id:05d}"
    seg_file = supine_ribcage_root / f"rib_cage_{vl_id_str}.nii.gz"
    if not seg_file.exists():
        print(f"  [warn] supine segmentation not found: {seg_file}")
        return None

    try:
        import external.breast_metadata_mdv.breast_metadata as breast_metadata
    except ImportError as exc:
        print(f"  [warn] cannot import breast_metadata ({exc}); skipping supine PC")
        return None

    mask = breast_metadata.readNIFTIImage(str(seg_file), orientation_flag, swap_axes=True)
    return extract_contour_points(mask, nb_contour_points)


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_transformed_mesh(
    transformed_mesh_pts: np.ndarray,
    supine_pc: Optional[np.ndarray],
    title: str,
    screenshot_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Overlay the transformed prone mesh sample with the supine point cloud.

    If ``screenshot_path`` is provided, save the figure to that PNG. If ``show``
    is True, open the interactive window.
    """
    off_screen = screenshot_path is not None and not show
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_text(title, font_size=14)

    if supine_pc is not None:
        plotter.add_points(
            supine_pc, color="tan", point_size=2,
            render_points_as_spheres=True, label="Supine PC",
        )

    plotter.add_points(
        transformed_mesh_pts, color="cornflowerblue", point_size=2,
        render_points_as_spheres=True, label="Aligned prone mesh",
    )

    plotter.add_points(
        np.array([[0.0, 0.0, 0.0]]), color="orange",
        point_size=12, render_points_as_spheres=True, label="Origin",
    )
    plotter.add_legend()

    if screenshot_path is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path), auto_close=not show)
        print(f"  saved preview: {screenshot_path}")
    elif show:
        plotter.show()
    else:
        plotter.close()


# ── Per-subject pipeline ─────────────────────────────────────────────────────
def process_one(
    vl_id: int,
    t_matrix_dir: Path,
    prone_ribcage_root: Path,
    supine_ribcage_root: Path,
    output_mesh_dir: Path,
    mesh_res: int = DEFAULT_MESH_RES,
    nb_contour_points: int = DEFAULT_NB_CONTOUR_POINTS,
    orientation_flag: str = DEFAULT_ORIENTATION_FLAG,
    do_plot: bool = True,
    screenshot: bool = False,
    show: bool = True,
    overwrite: bool = True,
    sanity_tol_mm: float = 1e-3,
) -> dict:
    """Transform one subject's prone ribcage mesh and save it.

    Returns a dict with status info for batch reporting.
    """
    vl_id_str = f"VL{vl_id:05d}"
    print(f"\n--- {vl_id_str} ---")

    matrix_path = t_matrix_dir / f"{vl_id_str}_transform_matrix.npy"
    if not matrix_path.exists():
        print(f"  [skip] no transformation matrix: {matrix_path}")
        return {"vl_id": vl_id, "status": "missing_matrix"}

    mesh_path = prone_ribcage_root / f"{vl_id_str}_ribcage_prone.mesh"
    if not mesh_path.exists():
        print(f"  [skip] no prone mesh: {mesh_path}")
        return {"vl_id": vl_id, "status": "missing_mesh"}

    out_path = output_mesh_dir / f"{vl_id_str}_ribcage_prone_transformed.mesh"
    if out_path.exists() and not overwrite:
        print(f"  [skip] output exists and --overwrite not set: {out_path}")
        return {"vl_id": vl_id, "status": "exists"}

    # Load fresh mesh + transformation matrix
    T_total = np.load(str(matrix_path))
    mesh = morphic.Mesh(str(mesh_path))
    pre_sample = get_surface_mesh_coords(mesh, res=mesh_res)

    # Transform
    info = transform_morphic_mesh_inplace(mesh, T_total)
    print(f"  transformed {info['n_standard_transformed']} standard nodes "
          f"(skipped {info['n_skipped']}: {info['skipped_types']})")
    if info["bad_shapes"]:
        print(f"  [warn] nodes with unexpected shape: {info['bad_shapes']}")

    # Sanity check: resampled transformed mesh vs analytical T @ pre_sample
    post_sample = get_surface_mesh_coords(mesh, res=mesh_res)
    rmse = _sanity_check(pre_sample, post_sample, T_total)
    if rmse > sanity_tol_mm:
        print(f"  [warn] sanity RMSE = {rmse:.3e} mm exceeds tol {sanity_tol_mm:.0e} mm")
    else:
        print(f"  sanity RMSE = {rmse:.3e} mm (ok)")

    # Save .mesh
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(out_path), format="pytables")
    print(f"  saved: {out_path}")

    # Round-trip check
    mesh_reloaded = morphic.Mesh(str(out_path))
    reload_sample = get_surface_mesh_coords(mesh_reloaded, res=mesh_res)
    rt_rmse = float(np.sqrt(np.mean(np.sum((reload_sample - post_sample) ** 2, axis=1))))
    if rt_rmse > sanity_tol_mm:
        print(f"  [warn] reload RMSE = {rt_rmse:.3e} mm exceeds tol")
    else:
        print(f"  reload RMSE = {rt_rmse:.3e} mm (ok)")

    # Plot
    if do_plot:
        supine_pc = load_supine_pc(vl_id, supine_ribcage_root,
                                   orientation_flag, nb_contour_points)
        screenshot_path = None
        if screenshot:
            screenshot_path = output_mesh_dir / "previews" / f"{vl_id_str}_overlay.png"
        title = f"{vl_id_str}: aligned prone vs supine ribcage"
        plot_transformed_mesh(post_sample, supine_pc, title,
                              screenshot_path=screenshot_path, show=show)

    return {
        "vl_id": vl_id,
        "status": "ok",
        "n_nodes": info["n_standard_transformed"],
        "sanity_rmse_mm": rmse,
        "reload_rmse_mm": rt_rmse,
        "out_path": str(out_path),
    }


# ── Discovery + CLI ──────────────────────────────────────────────────────────
def discover_vl_ids(t_matrix_dir: Path) -> List[int]:
    """List VL IDs that have a saved transformation matrix in ``t_matrix_dir``."""
    if not t_matrix_dir.is_dir():
        return []
    ids: List[int] = []
    for f in sorted(t_matrix_dir.iterdir()):
        m = VL_ID_PATTERN.search(f.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vl_id", type=int, nargs="+", default=None,
                        help="Subject VL IDs to process. Default: all matrices in --t_matrix_dir.")
    parser.add_argument("--t_matrix_dir", type=Path, default=T_MATRIX_DIR,
                        help=f"Directory of *_transform_matrix.npy (default: {T_MATRIX_DIR})")
    parser.add_argument("--prone_root", type=Path, default=PRONE_RIBCAGE_ROOT,
                        help=f"Root for *_ribcage_prone.mesh (default: {PRONE_RIBCAGE_ROOT})")
    parser.add_argument("--supine_root", type=Path, default=SUPINE_RIBCAGE_ROOT,
                        help=f"Root for rib_cage_VL*.nii.gz (default: {SUPINE_RIBCAGE_ROOT})")
    parser.add_argument("--out_dir", type=Path, default=OUTPUT_MESH_DIR,
                        help=f"Output directory for transformed .mesh files (default: {OUTPUT_MESH_DIR})")
    parser.add_argument("--mesh_res", type=int, default=DEFAULT_MESH_RES,
                        help="Sampling resolution for sanity check / plot")
    parser.add_argument("--nb_contour_points", type=int, default=DEFAULT_NB_CONTOUR_POINTS,
                        help="Supine point-cloud size")
    parser.add_argument("--no_plot", action="store_true",
                        help="Do not produce a plot (also skips supine PC loading)")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save PNG preview to <out_dir>/previews/")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not open an interactive window (use with --screenshot for batch runs)")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="Skip subjects whose output .mesh already exists")
    args = parser.parse_args()

    vl_ids = args.vl_id if args.vl_id is not None else discover_vl_ids(args.t_matrix_dir)
    if not vl_ids:
        print(f"No subjects to process. Looked in: {args.t_matrix_dir}")
        return

    print(f"Processing {len(vl_ids)} subject(s): {vl_ids}")
    print(f"  T matrices : {args.t_matrix_dir}")
    print(f"  prone root : {args.prone_root}")
    print(f"  supine root: {args.supine_root}")
    print(f"  out dir    : {args.out_dir}")

    summary: List[dict] = []
    for vl_id in vl_ids:
        try:
            result = process_one(
                vl_id=vl_id,
                t_matrix_dir=args.t_matrix_dir,
                prone_ribcage_root=args.prone_root,
                supine_ribcage_root=args.supine_root,
                output_mesh_dir=args.out_dir,
                mesh_res=args.mesh_res,
                nb_contour_points=args.nb_contour_points,
                do_plot=not args.no_plot,
                screenshot=args.screenshot,
                show=not args.no_show,
                overwrite=not args.no_overwrite,
            )
        except Exception as exc:
            print(f"  [error] {exc!r}")
            result = {"vl_id": vl_id, "status": "error", "error": repr(exc)}
        summary.append(result)

    # Summary
    n_ok = sum(1 for r in summary if r.get("status") == "ok")
    n_err = sum(1 for r in summary if r.get("status") == "error")
    n_skip = len(summary) - n_ok - n_err
    print(f"\n=== Summary: {n_ok} ok, {n_err} error, {n_skip} skipped ===")
    for r in summary:
        if r.get("status") != "ok":
            print(f"  VL{r['vl_id']:05d}: {r.get('status')} {r.get('error', '')}")


if __name__ == "__main__":
    main()
