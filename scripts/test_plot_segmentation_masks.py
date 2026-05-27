"""
Test script to load and plot segmentation masks for specific subjects.
This helps visually inspect the quality of the segmentation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import external.breast_metadata_mdv.breast_metadata as breast_metadata
from skimage.segmentation import find_boundaries
import pyvista as pv


def extract_contour_points(mask, nb_points):
    """
    Extract surface/boundary points from a segmentation mask.
    """
    labels = mask.values.copy()
    boundaries = find_boundaries(labels, mode='inner').astype(np.uint8)

    boundary_indices = np.argwhere(boundaries > 0)

    if len(boundary_indices) == 0:
        print("Warning: No boundary points found in mask")
        return np.array([]).reshape(0, 3)

    spacing = np.array(mask.spacing)
    origin = np.array(mask.origin)

    points = boundary_indices.astype(np.float64) * spacing + origin

    if nb_points < len(points):
        step = max(1, len(points) // nb_points)
        indices = np.arange(0, len(points), step)[:nb_points]
        return points[indices, :]
    else:
        return points


def plot_point_cloud_3views(points, title="Point Cloud", figsize=(15, 5)):
    """
    Plot point cloud in 3 views: Axial (top), Coronal (front), Sagittal (side).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Axial view (X-Y plane, looking from above)
    axes[0].scatter(points[:, 0], points[:, 1], s=0.5, alpha=0.5)
    axes[0].set_xlabel('X (Left-Right)')
    axes[0].set_ylabel('Y (Ant-Post)')
    axes[0].set_title(f'{title}\nAxial View (Top)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Coronal view (X-Z plane, looking from front)
    axes[1].scatter(points[:, 0], points[:, 2], s=0.5, alpha=0.5)
    axes[1].set_xlabel('X (Left-Right)')
    axes[1].set_ylabel('Z (Inf-Sup)')
    axes[1].set_title(f'{title}\nCoronal View (Front)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    # Sagittal view (Y-Z plane, looking from side)
    axes[2].scatter(points[:, 1], points[:, 2], s=0.5, alpha=0.5)
    axes[2].set_xlabel('Y (Ant-Post)')
    axes[2].set_ylabel('Z (Inf-Sup)')
    axes[2].set_title(f'{title}\nSagittal View (Side)')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_point_cloud_3d(points, title="Point Cloud 3D", point_size=2.0, color='lightblue',
                        screenshot_path=None, show_interactive=True):
    """
    Plot point cloud in interactive 3D using PyVista.

    Args:
        points: (N, 3) array of point coordinates
        title: Title for the plot
        point_size: Size of rendered points
        color: Color of points
        screenshot_path: Optional path to save screenshot
        show_interactive: If True, show interactive window; if False, only save screenshot
    """
    # Create PyVista point cloud
    cloud = pv.PolyData(points)

    # Create plotter (off_screen if only saving screenshot)
    plotter = pv.Plotter(off_screen=not show_interactive)

    plotter.add_points(
        cloud,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True
    )

    # Add axes and labels
    plotter.add_axes()
    plotter.add_text(title, position='upper_left', font_size=12, color='black')

    # Add axis labels
    plotter.add_text("X: Left-Right", position=(0.02, 0.15), font_size=8)
    plotter.add_text("Y: Ant-Post", position=(0.02, 0.10), font_size=8)
    plotter.add_text("Z: Inf-Sup", position=(0.02, 0.05), font_size=8)

    # Set background
    plotter.set_background('white')

    # Set a good camera view (isometric-ish)
    plotter.camera_position = 'iso'

    # Show and/or save screenshot
    if screenshot_path and show_interactive:
        # Show interactive with screenshot
        plotter.show(screenshot=str(screenshot_path))
        print(f"  Saved 3D screenshot to: {screenshot_path}")
    elif screenshot_path and not show_interactive:
        # Off-screen rendering for screenshot only
        plotter.show(auto_close=False)
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        print(f"  Saved 3D screenshot to: {screenshot_path}")
    elif show_interactive:
        # Just show interactive
        plotter.show()

    return plotter


def plot_multiple_point_clouds_3d(point_clouds_dict, title="Multi-Subject 3D View",
                                   point_size=2.0, screenshot_path=None, show_interactive=True):
    """
    Plot multiple point clouds in a single 3D view with different colors.

    Args:
        point_clouds_dict: Dict of {label: points} where points is (N, 3) array
        title: Title for the plot
        point_size: Size of rendered points
        screenshot_path: Optional path to save screenshot
        show_interactive: If True, show interactive window
    """
    # Color palette for different subjects
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plotter = pv.Plotter(off_screen=not show_interactive)

    for i, (label, points) in enumerate(point_clouds_dict.items()):
        if len(points) == 0:
            continue
        cloud = pv.PolyData(points)
        color = colors[i % len(colors)]
        plotter.add_points(
            cloud,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            label=label
        )

    plotter.add_axes()
    plotter.add_legend()
    plotter.add_text(title, position='upper_left', font_size=12, color='black')
    plotter.set_background('white')
    plotter.camera_position = 'iso'

    # Show and/or save screenshot
    if screenshot_path and show_interactive:
        plotter.show(screenshot=str(screenshot_path))
        print(f"  Saved 3D screenshot to: {screenshot_path}")
    elif screenshot_path and not show_interactive:
        plotter.show(auto_close=False)
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        print(f"  Saved 3D screenshot to: {screenshot_path}")
    elif show_interactive:
        plotter.show()

    return plotter


def plot_segmentation_masks(vl_ids, supine_ribcage_root, output_dir=None, plot_3d=True):
    """
    Load and plot segmentation masks for specified subjects.

    Args:
        vl_ids: List of VL IDs to plot
        supine_ribcage_root: Path to supine ribcage segmentation directory
        output_dir: Optional directory to save figures
        plot_3d: If True, show interactive 3D visualization
    """
    supine_ribcage_root = Path(supine_ribcage_root)

    for vl_id in vl_ids:
        vl_id_str = f"VL{vl_id:05d}"
        seg_file = supine_ribcage_root / f"rib_cage_{vl_id_str}.nii.gz"

        print(f"\n{'='*60}")
        print(f"Loading {vl_id_str}...")
        print(f"{'='*60}")

        if not seg_file.exists():
            print(f"  ERROR: File not found: {seg_file}")
            continue

        try:
            # Load the segmentation mask
            supine_mask = breast_metadata.readNIFTIImage(
                str(seg_file), 'RAI', swap_axes=True
            )

            # Print mask info
            print(f"  Mask shape: {supine_mask.values.shape}")
            print(f"  Mask spacing: {supine_mask.spacing}")
            print(f"  Mask origin: {supine_mask.origin}")
            print(f"  Non-zero voxels: {np.sum(supine_mask.values > 0)}")

            # Extract boundary points
            points = extract_contour_points(supine_mask, 50000)
            print(f"  Boundary points: {len(points)}")

            if len(points) == 0:
                print(f"  WARNING: No boundary points extracted!")
                continue

            # Print point cloud statistics
            print(f"\n  Point cloud statistics:")
            print(f"    X range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] mm")
            print(f"    Y range: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] mm")
            print(f"    Z range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] mm")
            print(f"    X span: {points[:, 0].max() - points[:, 0].min():.1f} mm")
            print(f"    Y span: {points[:, 1].max() - points[:, 1].min():.1f} mm")
            print(f"    Z span: {points[:, 2].max() - points[:, 2].min():.1f} mm")

            # Plot 3D visualization
            if plot_3d:
                screenshot_path = None
                if output_dir:
                    screenshot_path = Path(output_dir) / f"segmentation_3d_{vl_id_str}.png"
                plot_point_cloud_3d(
                    points,
                    title=f"{vl_id_str} Supine Ribcage Segmentation",
                    point_size=3.0,
                    color='lightblue',
                    screenshot_path=screenshot_path
                )

            # Plot 2D views
            fig = plot_point_cloud_3views(points, title=f"{vl_id_str} Supine Ribcage")

            if output_dir:
                output_path = Path(output_dir) / f"segmentation_mask_{vl_id_str}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"\n  Saved 2D figure to: {output_path}")

            plt.show()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


def plot_comparison_all_subjects(vl_ids, supine_ribcage_root, output_dir=None):
    """
    Plot all subjects in a single comparison figure.
    """
    supine_ribcage_root = Path(supine_ribcage_root)

    n_subjects = len(vl_ids)
    fig, axes = plt.subplots(n_subjects, 3, figsize=(15, 4*n_subjects))

    if n_subjects == 1:
        axes = axes.reshape(1, -1)

    for i, vl_id in enumerate(vl_ids):
        vl_id_str = f"VL{vl_id:05d}"
        seg_file = supine_ribcage_root / f"rib_cage_{vl_id_str}.nii.gz"

        print(f"Loading {vl_id_str}...")

        if not seg_file.exists():
            print(f"  File not found: {seg_file}")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'{vl_id_str}\nNot Found',
                               ha='center', va='center', fontsize=12)
                axes[i, j].set_axis_off()
            continue

        try:
            supine_mask = breast_metadata.readNIFTIImage(
                str(seg_file), 'RAI', swap_axes=True
            )
            points = extract_contour_points(supine_mask, 30000)

            if len(points) == 0:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, f'{vl_id_str}\nNo Points',
                                   ha='center', va='center', fontsize=12)
                    axes[i, j].set_axis_off()
                continue

            # Axial view
            axes[i, 0].scatter(points[:, 0], points[:, 1], s=0.3, alpha=0.5)
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            axes[i, 0].set_title(f'{vl_id_str} - Axial')
            axes[i, 0].set_aspect('equal')
            axes[i, 0].grid(True, alpha=0.3)

            # Coronal view
            axes[i, 1].scatter(points[:, 0], points[:, 2], s=0.3, alpha=0.5)
            axes[i, 1].set_xlabel('X')
            axes[i, 1].set_ylabel('Z')
            axes[i, 1].set_title(f'{vl_id_str} - Coronal')
            axes[i, 1].set_aspect('equal')
            axes[i, 1].grid(True, alpha=0.3)

            # Sagittal view
            axes[i, 2].scatter(points[:, 1], points[:, 2], s=0.3, alpha=0.5)
            axes[i, 2].set_xlabel('Y')
            axes[i, 2].set_ylabel('Z')
            axes[i, 2].set_title(f'{vl_id_str} - Sagittal')
            axes[i, 2].set_aspect('equal')
            axes[i, 2].grid(True, alpha=0.3)

            # Add statistics as text
            stats_text = f"N={len(points)}\nX:[{points[:,0].min():.0f},{points[:,0].max():.0f}]\nY:[{points[:,1].min():.0f},{points[:,1].max():.0f}]\nZ:[{points[:,2].min():.0f},{points[:,2].max():.0f}]"
            axes[i, 0].text(0.02, 0.98, stats_text, transform=axes[i, 0].transAxes,
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        except Exception as e:
            print(f"  Error: {e}")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'{vl_id_str}\nError',
                               ha='center', va='center', fontsize=12)
                axes[i, j].set_axis_off()

    plt.suptitle('Supine Ribcage Segmentation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / "segmentation_mask_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison figure to: {output_path}")

    plt.show()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define paths
    SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
    OUTPUT_DIR = Path(r"C:\Users\jxu759\Documents\motion-landmarks\output\figs")

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Subjects with high RMSE to inspect
    VL_IDS_TO_INSPECT = [54, 58, 78, 44]

    print("="*60)
    print("SEGMENTATION MASK INSPECTION (3D)")
    print("="*60)
    print(f"Subjects to inspect: {VL_IDS_TO_INSPECT}")
    print(f"Segmentation root: {SUPINE_RIBCAGE_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all point clouds first
    all_point_clouds = {}
    for vl_id in VL_IDS_TO_INSPECT:
        vl_id_str = f"VL{vl_id:05d}"
        seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

        print(f"\n{'='*60}")
        print(f"Loading {vl_id_str}...")
        print(f"{'='*60}")

        if not seg_file.exists():
            print(f"  ERROR: File not found: {seg_file}")
            continue

        try:
            # Load the segmentation mask
            supine_mask = breast_metadata.readNIFTIImage(
                str(seg_file), 'RAI', swap_axes=True
            )

            # Print mask info
            print(f"  Mask shape: {supine_mask.values.shape}")
            print(f"  Mask spacing: {supine_mask.spacing}")
            print(f"  Mask origin: {supine_mask.origin}")
            print(f"  Non-zero voxels: {np.sum(supine_mask.values > 0)}")

            # Extract boundary points
            points = extract_contour_points(supine_mask, 50000)
            print(f"  Boundary points: {len(points)}")

            if len(points) == 0:
                print(f"  WARNING: No boundary points extracted!")
                continue

            # Print point cloud statistics
            print(f"\n  Point cloud statistics:")
            print(f"    X range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] mm")
            print(f"    Y range: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] mm")
            print(f"    Z range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] mm")
            print(f"    X span: {points[:, 0].max() - points[:, 0].min():.1f} mm")
            print(f"    Y span: {points[:, 1].max() - points[:, 1].min():.1f} mm")
            print(f"    Z span: {points[:, 2].max() - points[:, 2].min():.1f} mm")

            all_point_clouds[vl_id_str] = points

            # Plot individual 3D visualization (interactive)
            screenshot_path = OUTPUT_DIR / f"segmentation_3d_{vl_id_str}.png"
            plot_point_cloud_3d(
                points,
                title=f"{vl_id_str} Supine Ribcage Segmentation",
                point_size=3.0,
                color='lightblue',
                screenshot_path=screenshot_path,
                show_interactive=True  # Show interactive window
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Plot all subjects together in one 3D view for comparison
    print("\n" + "="*60)
    print("GENERATING COMBINED 3D VIEW")
    print("="*60)

    if all_point_clouds:
        # Center each point cloud for comparison
        centered_point_clouds = {}
        for label, points in all_point_clouds.items():
            centroid = points.mean(axis=0)
            centered_point_clouds[label] = points - centroid
            print(f"  {label}: {len(points)} points (centered)")

        plot_multiple_point_clouds_3d(
            centered_point_clouds,
            title="High-RMSE Subjects Comparison (Centered)",
            point_size=2.0,
            screenshot_path=OUTPUT_DIR / "segmentation_3d_comparison.png",
            show_interactive=True
        )

    print("\n" + "="*60)
    print("DONE - Screenshots saved to:")
    print(f"  {OUTPUT_DIR}")
    print("="*60)








