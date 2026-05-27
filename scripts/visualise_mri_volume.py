"""
Interactive MRI volume viewer with orthogonal slicing and optional overlays.

Displays a volumetric MRI dataset with interactive slice planes in three
orthogonal views (axial, sagittal, coronal).  Optionally overlays point
clouds and/or morphic meshes.

Usage (standalone):
    python visualise_mri_volume.py

Usage (from other scripts):
    from visualise_mri_volume import MRIViewer
    viewer = MRIViewer(scan_object, orientation_flag='RAI')
    viewer.add_point_cloud(points, color='tan', point_size=3)
    viewer.add_morphic_mesh(mesh, color='#FFCCCC', opacity=0.5)
    viewer.show()
"""

import sys
import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Optional


class MRIViewer:
    """Interactive MRI volume viewer with orthogonal slice planes."""

    def __init__(
            self,
            image_grid: pv.ImageData,
            title: str = "MRI Volume Viewer",
    ):
        """
        Args:
            image_grid: pv.ImageData with scalar field 'values'
            title: window title
        """
        self.image_grid = image_grid
        self.title = title
        self._overlays = []

    @classmethod
    def from_scan(cls, scan, orientation_flag: str = 'RAI', title: str = "MRI Volume Viewer"):
        """Create viewer from a breast_metadata.Scan object."""
        import external.breast_metadata_mdv.breast_metadata as breast_metadata
        image_grid = breast_metadata.SCANToPyvistaImageGrid(scan, orientation_flag)
        return cls(image_grid, title=title)

    @classmethod
    def from_nifti(cls, nifti_path: str, orientation_flag: str = 'RAI', title: str = "MRI Volume Viewer"):
        """Create viewer from a NIFTI file."""
        import external.breast_metadata_mdv.breast_metadata as breast_metadata
        scan = breast_metadata.readNIFTIImage(nifti_path, orientation_flag, swap_axes=True)
        return cls.from_scan(scan, orientation_flag, title=title)

    def add_point_cloud(
            self,
            points: np.ndarray,
            color: str = 'tan',
            point_size: float = 3.0,
            label: str = 'Point cloud',
            opacity: float = 1.0,
    ):
        """Register a point cloud overlay."""
        self._overlays.append({
            'type': 'points',
            'data': np.asarray(points),
            'color': color,
            'point_size': point_size,
            'label': label,
            'opacity': opacity,
        })

    def add_morphic_mesh(
            self,
            mesh,
            color: str = '#FFCCCC',
            opacity: float = 0.5,
            label: str = 'Mesh',
            res: int = 10,
    ):
        """Register a morphic mesh overlay (converted via mesh_tools)."""
        self._overlays.append({
            'type': 'morphic_mesh',
            'data': mesh,
            'color': color,
            'opacity': opacity,
            'label': label,
            'res': res,
        })

    def add_pyvista_mesh(
            self,
            mesh: pv.DataSet,
            color: str = '#FFCCCC',
            opacity: float = 0.5,
            label: str = 'Mesh',
    ):
        """Register a PyVista mesh overlay."""
        self._overlays.append({
            'type': 'pv_mesh',
            'data': mesh,
            'color': color,
            'opacity': opacity,
            'label': label,
        })

    def add_landmarks(
            self,
            points: np.ndarray,
            color: str = 'red',
            point_size: float = 10.0,
            label: str = 'Landmarks',
    ):
        """Register landmark points (rendered as spheres)."""
        self._overlays.append({
            'type': 'landmarks',
            'data': np.asarray(points),
            'color': color,
            'point_size': point_size,
            'label': label,
        })

    def show(
            self,
            slice_position: Optional[np.ndarray] = None,
            show_volume: bool = True,
    ):
        """
        Launch the interactive viewer.

        Args:
            slice_position: (3,) initial slice position in world coords.
                            Defaults to the volume centre.
            show_volume: render semi-transparent volume behind the slices
        """
        grid = self.image_grid
        bounds = np.array(grid.bounds)  # (xmin,xmax, ymin,ymax, zmin,zmax)
        centre = np.array(grid.center)

        if slice_position is not None:
            centre = np.asarray(slice_position, dtype=float)

        plotter = pv.Plotter()

        # Dark charcoal background — better contrast than pure black,
        # reduces eye strain, and avoids washing out low-intensity MRI
        # voxels that would blend into pure black.
        plotter.set_background('#1A1A2E', top='#16213E')
        plotter.add_text(self.title, font_size=14, color='white')

        # --- Volume rendering (subtle context behind slices) ---
        if show_volume:
            opacity = np.linspace(0, 0.08, 100)
            plotter.add_volume(
                grid,
                scalars='values',
                cmap='gray',
                opacity=opacity,
                show_scalar_bar=False,
            )

        # --- Slice callbacks (one per axis) ---
        # Each slider drives a slice plane through the volume.
        # RAI orientation: X=Right, Y=Anterior, Z=Inferior
        slice_cfg = [
            {
                'axis': 'x', 'normal': [1, 0, 0], 'name': 'slice_sagittal',
                'label': 'Sagittal (X)', 'color': '#E94560',
                'bounds_idx': (0, 1), 'pos': (0.05, 0.25),
            },
            {
                'axis': 'y', 'normal': [0, 1, 0], 'name': 'slice_coronal',
                'label': 'Coronal (Y)', 'color': '#0F3460',
                'bounds_idx': (2, 3), 'pos': (0.375, 0.575),
            },
            {
                'axis': 'z', 'normal': [0, 0, 1], 'name': 'slice_axial',
                'label': 'Axial (Z)', 'color': '#00B4D8',
                'bounds_idx': (4, 5), 'pos': (0.70, 0.90),
            },
        ]

        for cfg in slice_cfg:
            lo = bounds[cfg['bounds_idx'][0]]
            hi = bounds[cfg['bounds_idx'][1]]
            init_val = centre[{'x': 0, 'y': 1, 'z': 2}[cfg['axis']]]

            # Add the initial slice
            origin_pt = centre.copy()
            sliced = grid.slice(normal=cfg['normal'], origin=origin_pt)
            plotter.add_mesh(
                sliced,
                cmap='bone',
                show_scalar_bar=False,
                name=cfg['name'],
            )

            # Build slider callback that updates the slice
            def _make_callback(normal, axis_idx, name):
                def callback(value):
                    origin = centre.copy()
                    origin[axis_idx] = value
                    new_slice = grid.slice(normal=normal, origin=origin)
                    plotter.add_mesh(
                        new_slice,
                        cmap='bone',
                        show_scalar_bar=False,
                        name=name,
                    )
                return callback

            axis_idx = {'x': 0, 'y': 1, 'z': 2}[cfg['axis']]
            plotter.add_slider_widget(
                _make_callback(cfg['normal'], axis_idx, cfg['name']),
                rng=[lo, hi],
                value=init_val,
                title=cfg['label'],
                pointa=(cfg['pos'][0], 0.92),
                pointb=(cfg['pos'][1], 0.92),
                style='modern',
                color=cfg['color'],
                title_color='white',
            )

        # --- Overlays ---
        for overlay in self._overlays:
            _add_overlay(plotter, overlay)

        plotter.add_axes()
        plotter.show()


def _add_overlay(plotter: pv.Plotter, overlay: dict):
    """Add a single overlay to the plotter."""
    otype = overlay['type']

    if otype == 'points':
        plotter.add_points(
            overlay['data'],
            color=overlay['color'],
            point_size=overlay['point_size'],
            render_points_as_spheres=True,
            label=overlay['label'],
            opacity=overlay['opacity'],
        )

    elif otype == 'landmarks':
        plotter.add_points(
            overlay['data'],
            color=overlay['color'],
            point_size=overlay['point_size'],
            render_points_as_spheres=True,
            label=overlay['label'],
        )

    elif otype == 'morphic_mesh':
        try:
            mesh = overlay['data']
            res = overlay['res']
            Xi = mesh.grid(res, method='center')
            nppe = Xi.shape[0]
            ne = mesh.elements.size()
            coords = np.zeros((ne * nppe, 3))
            for i, element in enumerate(mesh.elements):
                coords[i * nppe:(i + 1) * nppe, :] = element.evaluate(Xi)
            plotter.add_points(
                coords,
                color=overlay['color'],
                render_points_as_spheres=True,
                point_size=3,
                opacity=overlay['opacity'],
                label=overlay['label'],
            )
        except Exception as e:
            print(f"WARNING: Could not plot morphic mesh: {e}")

    elif otype == 'pv_mesh':
        plotter.add_mesh(
            overlay['data'],
            color=overlay['color'],
            opacity=overlay['opacity'],
            label=overlay['label'],
        )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'external' / 'automesh'))

    import external.breast_metadata_mdv.breast_metadata as breast_metadata
    from readers import load_subject
    from utils import extract_contour_points

    # ======================================================================
    # Configuration
    # ======================================================================
    vl_ids = [46]
    ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
    SOFT_TISSUE_ROOT = Path(r'U:\projects\dashboard\picker_points')
    ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")
    PRONE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")

    orientation_flag = 'RAI'

    # What to display
    SHOW_POINT_CLOUD = True
    SHOW_MESH = True
    POSITION = "supine"   # "prone" or "supine"

    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        print(f"\n{'='*60}")
        print(f"MRI Volume Viewer: {vl_id_str} ({POSITION})")
        print(f"{'='*60}")

        # Load subject
        subject = load_subject(
            vl_id=vl_id,
            positions=[POSITION],
            dicom_root=ROOT_PATH_MRI,
            anatomical_json_base_root=ANATOMICAL_JSON_BASE_ROOT,
            soft_tissue_root=SOFT_TISSUE_ROOT,
        )

        # Create viewer from scan
        scan = subject.scans[POSITION].scan_object
        viewer = MRIViewer.from_scan(
            scan, orientation_flag,
            title=f"{vl_id_str} {POSITION.capitalize()} MRI",
        )

        # Anatomical landmarks
        anat = subject.scans[POSITION].anatomical_landmarks
        landmarks = []
        for pt in [anat.sternum_superior, anat.sternum_inferior,
                    anat.nipple_left, anat.nipple_right]:
            if pt is not None:
                landmarks.append(pt)
        if landmarks:
            viewer.add_landmarks(
                np.array(landmarks), color='red',
                point_size=12, label='Anatomical landmarks',
            )

        # Point cloud (supine ribcage segmentation)
        if SHOW_POINT_CLOUD and POSITION == "supine":
            seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"
            if seg_file.exists():
                mask = breast_metadata.readNIFTIImage(
                    str(seg_file), orientation_flag, swap_axes=True,
                )
                pc = extract_contour_points(mask, 20000)
                viewer.add_point_cloud(pc, color='tan', point_size=2)
                print(f"  Point cloud: {pc.shape[0]} points")

        # Morphic mesh (prone ribcage)
        if SHOW_MESH and POSITION == "prone":
            import morphic
            mesh_file = PRONE_RIBCAGE_ROOT / f"{vl_id_str}_ribcage_prone.mesh"
            if mesh_file.exists():
                prone_mesh = morphic.Mesh(str(mesh_file))
                viewer.add_morphic_mesh(prone_mesh, color='#FFCCCC', opacity=0.4)
                print(f"  Mesh loaded: {prone_mesh.elements.size()} elements")

        # Slice at sternum superior if available
        slice_pos = None
        if anat.sternum_superior is not None:
            slice_pos = anat.sternum_superior

        viewer.show(slice_position=slice_pos)
