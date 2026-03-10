"""
Apply a previously saved alignment transformation matrix to prone data,
compute per-registrar landmark displacements, save results to Excel,
and optionally plot.

Usage:
    python apply_saved_alignment.py
    python apply_saved_alignment.py --vl_id 9
    python apply_saved_alignment.py --vl_id 9 --t_matrix_dir ../output/alignment/transformation_matrix_v7
    python apply_saved_alignment.py --vl_id 9 --no_plot
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import morphic
import pyvista as pv
import external.breast_metadata_mdv.breast_metadata as breast_metadata

from readers import load_subject
from structures import Subject
from utils import (
    apply_transform,
    extract_contour_points,
    get_landmarks_as_array,
    find_corresponding_landmarks,
    add_averaged_landmarks,
    save_alignment_results_to_excel,
    load_alignment_metrics,
)
from alignment import compute_landmark_displacements
from alignment_utils import get_surface_mesh_coords, apply_transform_to_coords


# ── Defaults ──────────────────────────────────────────────────────────────────
SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
T_MATRIX_DIR = Path(r"../output/alignment/transformation_matrix_v7")

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v7_2026_03_10.xlsx"


def load_alignment_data(
    vl_id: int,
    t_matrix_dir: Path = T_MATRIX_DIR,
    soft_tissue_root: Path = SOFT_TISSUE_ROOT,
    anatomical_json_base_root: Path = ANATOMICAL_JSON_BASE_ROOT,
    prone_ribcage_root: Path = PRONE_RIBCAGE_ROOT,
    supine_ribcage_root: Path = SUPINE_RIBCAGE_ROOT,
    orientation_flag: str = 'RAI',
    mesh_res: int = 26,
    nb_contour_points: int = 20000,
    load_ribcage: bool = True,
) -> Dict:
    """
    Load all data needed to visualize an alignment result.

    Parameters
    ----------
    vl_id : int
        Subject ID number (e.g. 9 for VL00009).
    t_matrix_dir : Path
        Directory containing *_transform_matrix.npy files.
    soft_tissue_root, anatomical_json_base_root : Path
        Roots for load_subject().
    prone_ribcage_root : Path
        Directory containing *_ribcage_prone.mesh files.
    supine_ribcage_root : Path
        Directory containing rib_cage_*.nii.gz files.
    orientation_flag : str
        Orientation for reading NIfTI (default 'RAI').
    mesh_res : int
        Resolution for mesh surface coordinate extraction.
    nb_contour_points : int
        Number of contour points to extract from supine segmentation.

    Returns
    -------
    dict with keys:
        T_total, subject, prone_mesh_coords, supine_pc,
        sternum_prone, sternum_supine, nipple_prone, nipple_supine,
        landmarks_prone (dict by registrar), landmarks_supine (dict by registrar)
    """
    vl_id_str = f"VL{vl_id:05d}"

    # Load transformation matrix
    matrix_path = t_matrix_dir / f"{vl_id_str}_transform_matrix.npy"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Transformation matrix not found: {matrix_path}")
    T_total = np.load(str(matrix_path))

    # Load subject
    subject = load_subject(
        vl_id=vl_id,
        positions=["prone", "supine"],
        dicom_root=None,
        anatomical_json_base_root=anatomical_json_base_root,
        soft_tissue_root=soft_tissue_root,
    )

    if "prone" not in subject.scans or "supine" not in subject.scans:
        raise ValueError(f"Subject {vl_id_str} missing prone or supine scan data")

    # Anatomical landmarks
    anat_prone = subject.scans["prone"].anatomical_landmarks
    anat_supine = subject.scans["supine"].anatomical_landmarks

    sternum_prone = np.vstack([anat_prone.sternum_superior, anat_prone.sternum_inferior])
    sternum_supine = np.vstack([anat_supine.sternum_superior, anat_supine.sternum_inferior])
    nipple_prone = np.vstack([anat_prone.nipple_left, anat_prone.nipple_right])
    nipple_supine = np.vstack([anat_supine.nipple_left, anat_supine.nipple_right])

    # Soft-tissue landmarks per registrar
    landmarks_prone = {}
    landmarks_supine = {}
    for reg_name in ["anthony", "holly", "average"]:
        lm_prone = get_landmarks_as_array(subject.scans["prone"], reg_name)
        lm_supine = get_landmarks_as_array(subject.scans["supine"], reg_name)
        if lm_prone.shape[0] > 0:
            landmarks_prone[reg_name] = lm_prone
        if lm_supine.shape[0] > 0:
            landmarks_supine[reg_name] = lm_supine

    # Load prone ribcage mesh and supine point cloud (only needed for plotting)
    prone_mesh_coords = None
    supine_pc = None
    if load_ribcage:
        prone_mesh_file = prone_ribcage_root / f"{vl_id_str}_ribcage_prone.mesh"
        if not prone_mesh_file.exists():
            raise FileNotFoundError(f"Prone mesh not found: {prone_mesh_file}")
        prone_ribcage = morphic.Mesh(str(prone_mesh_file))
        prone_mesh_coords = get_surface_mesh_coords(prone_ribcage, res=mesh_res)

        supine_seg_file = supine_ribcage_root / f"rib_cage_{vl_id_str}.nii.gz"
        if not supine_seg_file.exists():
            raise FileNotFoundError(f"Supine segmentation not found: {supine_seg_file}")
        supine_ribcage_mask = breast_metadata.readNIFTIImage(
            str(supine_seg_file), orientation_flag, swap_axes=True
        )
        supine_pc = extract_contour_points(supine_ribcage_mask, nb_contour_points)

    return {
        'T_total': T_total,
        'subject': subject,
        'prone_mesh_coords': prone_mesh_coords,
        'supine_pc': supine_pc,
        'sternum_prone': sternum_prone,
        'sternum_supine': sternum_supine,
        'nipple_prone': nipple_prone,
        'nipple_supine': nipple_supine,
        'landmarks_prone': landmarks_prone,
        'landmarks_supine': landmarks_supine,
    }


def transform_prone_data(data: Dict) -> Dict:
    """
    Apply T_total to all prone data (mesh, sternum, nipples, landmarks).

    Parameters
    ----------
    data : dict
        Output of load_alignment_data().

    Returns
    -------
    dict with keys:
        prone_mesh_transformed, sternum_prone_transformed,
        nipple_prone_transformed, landmarks_prone_transformed (dict by registrar)
    """
    T = data['T_total']

    prone_mesh_transformed = (apply_transform(data['prone_mesh_coords'], T)
                              if data['prone_mesh_coords'] is not None else None)
    sternum_prone_transformed = apply_transform(data['sternum_prone'], T)
    nipple_prone_transformed = apply_transform(data['nipple_prone'], T)

    landmarks_prone_transformed = {}
    for reg_name, lm in data['landmarks_prone'].items():
        landmarks_prone_transformed[reg_name] = apply_transform(lm, T)

    return {
        'prone_mesh_transformed': prone_mesh_transformed,
        'sternum_prone_transformed': sternum_prone_transformed,
        'nipple_prone_transformed': nipple_prone_transformed,
        'landmarks_prone_transformed': landmarks_prone_transformed,
    }


def build_alignment_results(
    data: Dict,
    transformed: Dict,
    metrics: Dict = None,
) -> Dict:
    """
    Reconstruct the alignment results dict from saved T_matrix and subject data.

    Computes per-registrar landmark displacements using compute_landmark_displacements().
    The returned dict has the same structure as align_prone_to_supine_optimal() output,
    so it can be passed directly to save_alignment_results_to_excel().

    Parameters
    ----------
    data : dict
        Output of load_alignment_data().
    transformed : dict
        Output of transform_prone_data().
    metrics : dict, optional
        Output of load_alignment_metrics(). If None, ribcage error metrics
        are set to None.

    Returns
    -------
    dict matching align_prone_to_supine_optimal() output structure.
    """
    if metrics is None:
        metrics = {}
    T = data['T_total']
    R = T[:3, :3]

    sternum_prone_transformed = transformed['sternum_prone_transformed']
    nipple_prone_transformed = transformed['nipple_prone_transformed']
    sternum_supine = data['sternum_supine']
    nipple_supine = data['nipple_supine']

    # source_anchor = sternum_superior_prone, target_anchor = sternum_superior_supine
    # From T: t = target_anchor - R @ source_anchor
    # So: source_anchor = R^-1 @ (target_anchor - t)
    source_anchor = np.linalg.inv(R) @ (data['sternum_supine'][0] - T[:3, 3])
    target_anchor = data['sternum_supine'][0]

    # Nipple displacements relative to sternum
    ref_sternum_prone = sternum_prone_transformed[0]
    ref_sternum_supine = sternum_supine[0]
    nipple_pos_prone_rel = nipple_prone_transformed - ref_sternum_prone
    nipple_pos_supine_rel = nipple_supine - ref_sternum_supine
    nipple_disp_rel_sternum = nipple_pos_supine_rel - nipple_pos_prone_rel
    nipple_disp_mag_rel_sternum = np.linalg.norm(nipple_disp_rel_sternum, axis=1)

    nipple_disp_left_vec = nipple_disp_rel_sternum[0]
    nipple_disp_right_vec = nipple_disp_rel_sternum[1]

    # Per-registrar displacements
    subject = data['subject']
    prone_scan_data = subject.scans["prone"]
    supine_scan_data = subject.scans["supine"]

    registrar_prefixes = {
        'anthony': 'ld_anthony',
        'holly': 'ld_holly',
        'average': 'ld_ave',
    }

    registrar_displacement_results = {}
    for reg_name, prefix in registrar_prefixes.items():
        lm_prone_raw = get_landmarks_as_array(prone_scan_data, reg_name)
        lm_supine_raw = get_landmarks_as_array(supine_scan_data, reg_name)

        if lm_prone_raw.shape[0] == 0 or lm_supine_raw.shape[0] == 0:
            print(f"  Skipping registrar '{reg_name}': no landmarks found")
            continue

        disp = compute_landmark_displacements(
            landmark_prone_raw=lm_prone_raw,
            landmark_supine_raw=lm_supine_raw,
            R=R,
            source_anchor=source_anchor,
            target_anchor=target_anchor,
            sternum_prone_transformed=sternum_prone_transformed,
            sternum_supine=sternum_supine,
            nipple_prone_transformed=nipple_prone_transformed,
            nipple_supine=nipple_supine,
            nipple_disp_rel_sternum=nipple_disp_rel_sternum,
        )

        for key, value in disp.items():
            registrar_displacement_results[f"{prefix}_{key}"] = value

    results = {
        'T_total': T,
        'R': R,

        # Ribcage error metrics from saved JSON sidecar
        'ribcage_error_rmse': metrics.get('ribcage_error_rmse'),
        'ribcage_error_mean': metrics.get('ribcage_error_mean'),
        'ribcage_error_std': metrics.get('ribcage_error_std'),
        'ribcage_inlier_rmse': metrics.get('ribcage_inlier_rmse'),
        'ribcage_inlier_mean': metrics.get('ribcage_inlier_mean'),
        'ribcage_inlier_std': metrics.get('ribcage_inlier_std'),
        'sternum_error': metrics.get('sternum_error'),

        'sternum_prone_transformed': sternum_prone_transformed,
        'sternum_supine': sternum_supine,
        'nipple_prone_transformed': nipple_prone_transformed,
        'nipple_supine': nipple_supine,

        'nipple_prone_rel_sternum': nipple_pos_prone_rel,
        'nipple_supine_rel_sternum': nipple_pos_supine_rel,

        'nipple_displacement_magnitudes': nipple_disp_mag_rel_sternum,
        'nipple_displacement_vectors': nipple_disp_rel_sternum,
        'nipple_disp_left_vec': nipple_disp_left_vec,
        'nipple_disp_right_vec': nipple_disp_right_vec,

        'method': 'from_saved_t_matrix',
    }

    results.update(registrar_displacement_results)
    return results


def plot_alignment(data: Dict, transformed: Dict, title: str = "") -> pv.Plotter:
    """
    Plot all alignment components in a single 3D view.

    Shows:
        - Supine ribcage point cloud (tan)
        - Aligned prone ribcage mesh (blue)
        - Sternum superior & inferior: prone-transformed (black) and supine (blue)
        - Nipples: prone-transformed (red) and supine (green)
        - Soft-tissue landmarks: prone-transformed (red) and supine (green)
        - Displacement arrows (yellow) from aligned prone to supine landmarks

    Parameters
    ----------
    data : dict
        Output of load_alignment_data().
    transformed : dict
        Output of transform_prone_data().
    title : str
        Plot title.

    Returns
    -------
    pv.Plotter
    """
    plotter = pv.Plotter()
    plot_title = title or f"Alignment: {data['subject'].subject_id}"
    plotter.add_text(plot_title, font_size=16)

    # Supine point cloud
    if data['supine_pc'] is not None:
        plotter.add_points(
            data['supine_pc'], color="tan", point_size=2,
            render_points_as_spheres=True, label='Supine ribcage PC'
        )

    # Aligned prone mesh
    if transformed['prone_mesh_transformed'] is not None:
        plotter.add_points(
            transformed['prone_mesh_transformed'], color="cornflowerblue",
            point_size=2, render_points_as_spheres=True, label='Aligned prone mesh'
        )

    # Sternum — prone transformed
    plotter.add_points(
        transformed['sternum_prone_transformed'], color="black",
        point_size=12, render_points_as_spheres=True,
        label='Aligned prone sternum'
    )

    # Sternum — supine
    plotter.add_points(
        data['sternum_supine'], color="blue",
        point_size=12, render_points_as_spheres=True,
        label='Supine sternum'
    )

    # Nipples — prone transformed
    plotter.add_points(
        transformed['nipple_prone_transformed'], color="red",
        point_size=10, render_points_as_spheres=True,
        label='Aligned prone nipples'
    )

    # Nipples — supine
    plotter.add_points(
        data['nipple_supine'], color="green",
        point_size=10, render_points_as_spheres=True,
        label='Supine nipples'
    )

    # Soft-tissue landmarks + displacement arrows
    for reg_name in transformed['landmarks_prone_transformed']:
        lm_prone_t = transformed['landmarks_prone_transformed'][reg_name]
        lm_supine = data['landmarks_supine'].get(reg_name)

        plotter.add_points(
            lm_prone_t, color="red", point_size=8,
            render_points_as_spheres=True,
            label=f'Aligned prone LM ({reg_name})'
        )

        if lm_supine is not None and lm_supine.shape[0] == lm_prone_t.shape[0]:
            plotter.add_points(
                lm_supine, color="green", point_size=8,
                render_points_as_spheres=True,
                label=f'Supine LM ({reg_name})'
            )

            # Displacement arrows
            disp_vectors = lm_supine - lm_prone_t
            for start_pt, vec in zip(lm_prone_t, disp_vectors):
                plotter.add_arrows(
                    start_pt.reshape(1, 3), vec.reshape(1, 3),
                    mag=1.0, color='yellow'
                )

    # Origin marker
    plotter.add_points(
        np.array([[0, 0, 0]]), color="orange",
        point_size=15, render_points_as_spheres=True,
        label='Origin (0,0,0)'
    )

    plotter.add_legend()
    return plotter


def run_apply_saved_alignment(
    vl_ids: List[int],
    t_matrix_dir: Path = T_MATRIX_DIR,
    excel_path: Path = EXCEL_FILE_PATH,
    show_plot: bool = True,
    **kwargs,
) -> None:
    """
    End-to-end: load saved T_matrix, transform prone data, compute
    per-registrar displacements, save to Excel, and optionally plot.

    Parameters
    ----------
    vl_ids : list of int
        Subject ID numbers to process.
    t_matrix_dir : Path
        Directory containing saved transformation matrices.
    excel_path : Path
        Output Excel file path.
    show_plot : bool
        Whether to show 3D plot for each subject.
    **kwargs
        Additional keyword arguments passed to load_alignment_data().
    """
    # Load all subjects
    all_subjects: Dict[int, Subject] = {}
    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        print(f"--- Loading Subject: {vl_id_str} ---")
        subject = load_subject(
            vl_id=vl_id,
            positions=["prone", "supine"],
            dicom_root=None,
            anatomical_json_base_root=kwargs.get(
                'anatomical_json_base_root', ANATOMICAL_JSON_BASE_ROOT),
            soft_tissue_root=kwargs.get('soft_tissue_root', SOFT_TISSUE_ROOT),
        )
        if subject.scans:
            all_subjects[vl_id] = subject

    # Correspondences + averaged landmarks
    correspondences, all_subjects_filtered = find_corresponding_landmarks(
        all_subjects)
    all_subjects_filtered = add_averaged_landmarks(
        all_subjects_filtered, correspondences)

    # Process each subject
    alignment_results_all = {}

    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"

        if vl_id not in all_subjects_filtered:
            print(f"Skipping {vl_id_str}: not in filtered subjects")
            continue

        print(f"\n--- Applying saved alignment for {vl_id_str} ---")

        # Update the subject reference for load_alignment_data
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ('anatomical_json_base_root',
                                       'soft_tissue_root')}

        data = load_alignment_data(
            vl_id, t_matrix_dir=t_matrix_dir,
            load_ribcage=show_plot, **filtered_kwargs)
        # Replace subject with filtered version (has averaged landmarks)
        data['subject'] = all_subjects_filtered[vl_id]

        transformed = transform_prone_data(data)
        metrics = load_alignment_metrics(t_matrix_dir, vl_id)
        alignment_results = build_alignment_results(data, transformed, metrics)
        alignment_results_all[vl_id] = alignment_results

        print(f"  Displacement computation for {vl_id_str} complete")

        if show_plot:
            plotter = plot_alignment(data, transformed)
            plotter.show()

    # Save alignment results to Excel
    if alignment_results_all:
        save_alignment_results_to_excel(
            excel_path=excel_path,
            correspondences=correspondences,
            all_subjects=all_subjects_filtered,
            alignment_results_all=alignment_results_all,
        )
        print(f"\nAlignment results saved to {excel_path}")

    print("\n=== apply_saved_alignment.py complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply saved alignment, compute displacements, save to Excel."
    )
    parser.add_argument(
        "--vl_id", type=int, nargs='+', default=[11],
        help="Subject VL ID number(s) (default: 9)"
    )
    parser.add_argument(
        "--t_matrix_dir", type=str,
        default=str(T_MATRIX_DIR),
        help="Directory with transformation matrix .npy files"
    )
    parser.add_argument(
        "--excel_path", type=str,
        default=str(EXCEL_FILE_PATH),
        help="Output Excel file path"
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip 3D visualization"
    )
    args = parser.parse_args()

    run_apply_saved_alignment(
        vl_ids=args.vl_id,
        t_matrix_dir=Path(args.t_matrix_dir),
        excel_path=Path(args.excel_path),
        show_plot=not args.no_plot,
    )
