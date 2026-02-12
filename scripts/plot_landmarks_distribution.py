"""
analysis_plot.py - Plotting functions for landmark analysis

This script contains plotting functions for visualizing landmarks in anatomical planes.
Similar to plot_nipple_relative_landmarks but plots only prone landmarks without vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib import cm
from pathlib import Path
import pandas as pd


def plot_nipple_relative_prone_landmarks(
    df_ave,
    vl_id=None,
    title=None,
    save_path=None,
    use_dts_cmap=True
):
    """
    Plots landmarks in prone position relative to nipple in three anatomical planes.
    Similar to plot_nipple_relative_landmarks but without displacement vectors.

    Args:
        df_ave: DataFrame with landmark data including:
                - 'landmark ave prone transformed x/y/z' (prone positions relative to sternum)
                - 'left nipple prone transformed x/y/z' (left nipple prone position)
                - 'right nipple prone transformed x/y/z' (right nipple prone position)
                - 'landmark side (prone)' (LB or RB)
                - 'Distance to skin (prone) [mm]' (optional, for coloring)
                - 'VL_ID' (optional, for filtering by subject)
        vl_id: Optional subject ID to filter data. If None, uses all subjects.
        title: Plot title
        save_path: Path to save the plots
        use_dts_cmap: Whether to use the DTS colormap

    Returns:
        Dictionary with base points for left and right breasts
    """
    print("\n--- Plotting Prone Landmarks Relative to Nipple ---")

    # 1. Filter data if specific subject requested
    if vl_id is not None:
        df_subset = df_ave[df_ave['VL_ID'] == vl_id].copy()
        if df_subset.empty:
            print(f"Warning: No data found for subject VL_{vl_id}")
            return None
    else:
        df_subset = df_ave.copy()

    # 2. Separate into Left (LB) and Right (RB) breasts
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Helper to extract nipple-relative base points (prone positions only)
    def get_nipple_relative_prone_points(sub_df, is_left_breast):
        if sub_df.empty:
            return np.empty((0, 3)), None, None

        # Extract landmark prone positions (relative to sternum)
        prone_x = sub_df['landmark ave prone transformed x'].values
        prone_y = sub_df['landmark ave prone transformed y'].values
        prone_z = sub_df['landmark ave prone transformed z'].values

        # Extract nipple positions from DataFrame columns
        if is_left_breast:
            # Left nipple prone position
            nipple_prone_x = sub_df['left nipple prone transformed x'].values
            nipple_prone_y = sub_df['left nipple prone transformed y'].values
            nipple_prone_z = sub_df['left nipple prone transformed z'].values
        else:
            # Right nipple prone position
            nipple_prone_x = sub_df['right nipple prone transformed x'].values
            nipple_prone_y = sub_df['right nipple prone transformed y'].values
            nipple_prone_z = sub_df['right nipple prone transformed z'].values

        # Get sternum position to make nipple relative
        sternum_prone_x = sub_df['sternum superior prone transformed x'].values
        sternum_prone_y = sub_df['sternum superior prone transformed y'].values
        sternum_prone_z = sub_df['sternum superior prone transformed z'].values

        # Calculate base points: landmark position relative to prone nipple
        base_x = prone_x - (nipple_prone_x - sternum_prone_x)
        base_y = prone_y - (nipple_prone_y - sternum_prone_y)
        base_z = prone_z - (nipple_prone_z - sternum_prone_z)
        base_points = np.column_stack((base_x, base_y, base_z))

        # Extract DTS values if available
        dts_col = 'Distance to skin (prone) [mm]'
        dts_values = sub_df[dts_col].values if dts_col in sub_df.columns else None
        vl_ids = sub_df['VL_ID'].values if 'VL_ID' in sub_df.columns else None

        return base_points, dts_values, vl_ids

    # Extract data for both breasts
    base_left, dts_left, vl_ids_left = get_nipple_relative_prone_points(left_df, is_left_breast=True)
    base_right, dts_right, vl_ids_right = get_nipple_relative_prone_points(right_df, is_left_breast=False)

    # 4. Call the specialized plotting function
    _plot_prone_landmarks_three_planes(
        base_point_left=base_left,
        base_point_right=base_right,
        dts_left=dts_left,
        dts_right=dts_right,
        subject_ids_left=vl_ids_left,
        subject_ids_right=vl_ids_right,
        title=title,
        save_path=save_path,
        use_dts_cmap=use_dts_cmap
    )

    return {
        'base_left': base_left,
        'base_right': base_right,
        'dts_left': dts_left,
        'dts_right': dts_right,
        'vl_ids_left': vl_ids_left,
        'vl_ids_right': vl_ids_right
    }


def _plot_prone_landmarks_three_planes(
        base_point_left: np.ndarray,
        base_point_right: np.ndarray,
        dts_left: np.ndarray = None,
        dts_right: np.ndarray = None,
        subject_ids_left: np.ndarray = None,
        subject_ids_right: np.ndarray = None,
        title: str = None,
        save_path: str = None,
        use_dts_cmap: bool = True
):
    """
    Plots prone landmarks relative to nipple in three anatomical planes: Coronal, Sagittal, and Axial.
    Similar to plot_nipple_relative_vectors but without vectors.

    Args:
        base_point_left: (N, 3) array of left breast landmark positions relative to nipple (prone)
        base_point_right: (M, 3) array of right breast landmark positions relative to nipple (prone)
        dts_left: (N,) array of DTS values for left breast landmarks (optional, for coloring)
        dts_right: (M,) array of DTS values for right breast landmarks (optional, for coloring)
        subject_ids_left: (N,) array of subject IDs for left breast landmarks (optional, for coloring by subject)
        subject_ids_right: (M,) array of subject IDs for right breast landmarks (optional, for coloring by subject)
        title: Overall title for the plots
        save_path: Path to save the figure (optional)
        use_dts_cmap: If True, color by DTS (viridis colormap with colorbar).
                      If False and subject_ids provided, color by subject (viridis colormap, no legend).
                      If False and no subject_ids, color by breast side (blue=right, green=left).

    Coordinate system:
        X (0): Right/Left (positive = right, negative = left)
        Y (1): Anterior/Posterior (positive = anterior, negative = posterior)
        Z (2): Inferior/Superior (positive = superior, negative = inferior)
    """

    # Plane configurations
    PLANE_CONFIG = {
        'Coronal': {
            'axes': (0, 2),  # X (R/L) vs Z (I/S)
            'xlabel': 'Right-Left (mm)',
            'ylabel': 'Inf-Sup (mm)',
            'shape': 'Circle',
            'quadrants_right': ('UI', 'UO', 'LI', 'LO'),
            'quadrants_left': ('UO', 'UI', 'LO', 'LI')
        },
        'Sagittal': {
            'axes': (1, 2),  # Y (A/P) vs Z (I/S)
            'xlabel': 'Ant-Post (mm)',
            'ylabel': 'Inf-Sup (mm)',
            'shape': 'SemiCircle',
            'quadrants_right': ('upper', '', 'lower', ''),
            'quadrants_left': ('upper', '', 'lower', '')
        },
        'Axial': {
            'axes': (0, 1),  # X (R/L) vs Y (A/P)
            'xlabel': 'Right-Left (mm)',
            'ylabel': 'Ant-Post (mm)',
            'shape': 'SemiCircle',
            'quadrants_right': ('inner', 'outer', '', ''),
            'quadrants_left': ('outer', 'inner', '', '')
        }
    }

    # Plot each plane
    for plane_name, config in PLANE_CONFIG.items():
        _plot_prone_landmarks_single_plane(
            plane_name, config,
            base_point_left,
            base_point_right,
            dts_left, dts_right,
            subject_ids_left, subject_ids_right,
            title, save_path,
            use_dts_cmap
        )


def _plot_prone_landmarks_single_plane(
        plane_name: str,
        config: dict,
        base_point_left: np.ndarray,
        base_point_right: np.ndarray,
        dts_left: np.ndarray,
        dts_right: np.ndarray,
        subject_ids_left: np.ndarray,
        subject_ids_right: np.ndarray,
        title: str,
        save_path: str,
        use_dts_cmap: bool
):
    """Helper function to plot a single anatomical plane with prone landmarks only."""

    AXIS_X, AXIS_Y = config['axes']

    # Create figure with two subplots (Right and Left breast)
    fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(18/2.54, 8/2.54), constrained_layout=True)
    fig.suptitle(f"{title} {plane_name} view", fontsize=10,x=0.11, ha='left', fontweight='bold')

    # Set axis limits based on plane
    if plane_name == 'Sagittal':
        lims_x = (0, 200)  # Anterior (0) to Posterior (300)
        lims_y = (-100, 100)  # Inf-Sup
        radius = 100
    elif plane_name == 'Axial':
        lims_x = (-100, 100)  # Right-Left
        lims_y = (0, 200)  # Anterior (0) to Posterior (300)
        radius = 100
    else:
        # For Coronal: use symmetric limits
        lims_x = (-100, 100)
        lims_y = (-100, 100)
        radius = 100

    # Plot Right Breast
    scatter_right = _plot_prone_breast_side(
        ax_right, plane_name, config,
        base_point_right, dts_right, subject_ids_right,
        'Right breast', 'right', lims_x, lims_y, radius,
        use_dts_cmap
    )

    # Plot Left Breast
    _plot_prone_breast_side(
        ax_left, plane_name, config,
        base_point_left, dts_left, subject_ids_left,
        'Left breast', 'left', lims_x, lims_y, radius,
        use_dts_cmap
    )

    if use_dts_cmap and scatter_right is not None:
        cbar = fig.colorbar(scatter_right, ax=[ax_right, ax_left], location='right', aspect=30, pad=0.02)
        cbar.set_label('DTS (mm)', rotation=270, labelpad=15)

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if use_dts_cmap:
            filename = f"{save_path}_{plane_name.lower()}_prone_DTS.png"
        elif subject_ids_left is not None or subject_ids_right is not None:
            filename = f"{save_path}_{plane_name.lower()}_prone_by_subject.png"
        else:
            filename = f"{save_path}_{plane_name.lower()}_prone_by_breast.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Only show if not using Agg backend
    if plt.get_backend().lower() != 'agg':
        plt.show()

    plt.close(fig)


def _plot_prone_breast_side(
        ax, plane_name: str, config: dict,
        base_points: np.ndarray,
        dts_values: np.ndarray,
        subject_ids: np.ndarray,
        side_title: str,
        side: str,
        lims_x: tuple,
        lims_y: tuple,
        radius: float,
        use_dts_cmap: bool
):
    """Helper function to plot prone landmarks for one breast side on a given axis."""

    AXIS_X, AXIS_Y = config['axes']

    # Set title and labels
    ax.set_title(side_title, fontsize=10,loc='left',pad=10)
    ax.set_xlabel(config['xlabel'], fontsize=9)
    ax.set_ylabel(config['ylabel'], fontsize=9)

    # Set limits
    ax.set_xlim(lims_x)
    ax.set_ylim(lims_y)

    # Set ticks with 50mm intervals
    if plane_name == 'Sagittal':
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_yticks(np.arange(-100, 101, 50))
    elif plane_name == 'Axial':
        ax.set_xticks(np.arange(-100, 101, 50))
        ax.set_yticks(np.arange(0, 201, 50))
    else:
        ax.set_xticks(np.arange(-100, 101, 50))
        ax.set_yticks(np.arange(-100, 101, 50))

    ax.set_aspect('equal', adjustable='box')
    # ax.grid(True, linestyle='--', alpha=0.5)

    # Draw anatomical shape
    _draw_anatomical_shape(ax, plane_name, config, radius, side)

    # Draw reference lines through nipple origin
    _draw_reference_lines(ax, plane_name)

    # Plot nipple at origin
    ax.plot(0, 0, 'ro', markersize=5, zorder=10, label='Nipple (Origin)')

    scatter = None
    # Plot landmarks (prone positions only, no vectors)
    if len(base_points) > 0:
        if use_dts_cmap and dts_values is not None and len(dts_values) == len(base_points):
            # Color by DTS (viridis colormap)
            scatter = ax.scatter(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                c=dts_values,
                cmap='viridis',
                s=10,
                zorder=5,
                vmin=0,
                vmax=40,
                # edgecolors='black',
                # linewidths=0.5,
                alpha=0.8
            )

            # # Add colorbar
            # cbar = plt.colorbar(scatter, ax=ax)
            # cbar.set_label('DTS (mm)', rotation=270, labelpad=15)

        elif not use_dts_cmap and subject_ids is not None and len(subject_ids) == len(base_points):
            # Color by subject ID (viridis colormap)
            unique_subjects = np.unique(subject_ids)
            n_subjects = len(unique_subjects)
            cmap_subj = cm.get_cmap('viridis', max(n_subjects, 1))

            subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}

            # Plot each subject with a different color
            for subj in unique_subjects:
                mask = subject_ids == subj
                color = cmap_subj(subject_to_idx[subj])

                ax.scatter(
                    base_points[mask, AXIS_X],
                    base_points[mask, AXIS_Y],
                    c=[color],
                    s=30,
                    zorder=5,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5
                )

        else:
            # Default: color by breast side
            color = 'blue' if side == 'right' else 'green'
            ax.scatter(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                c=color,
                s=30,
                zorder=5,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f'{side.capitalize()} Breast Landmarks'
            )

    # Add legend
    # ax.legend(loc='upper right', fontsize=10)
    # ax.legend(bbox_to_anchor=(0.7, 0.9), loc='upper left', fontsize=10)
    return scatter



def _draw_anatomical_shape(ax, plane_name: str, config: dict, radius: float, side: str):
    """
    Helper function to draw anatomical reference shape (circle or semicircle).

    Args:
        ax: Matplotlib axis
        plane_name: Name of the anatomical plane
        config: Configuration dictionary for the plane
        radius: Radius of the shape
        side: 'left' or 'right' breast
    """
    shape_type = config.get('shape', 'Circle')

    if shape_type == 'Circle':
        # Draw full circle for coronal view
        circle = Circle((0, 0), radius, fill=False, linestyle='-',
                        color='black', alpha=0.5, linewidth=1)
        ax.add_patch(circle)

        # Horizontal line along the X-axis (y=0)
        ax.axhline(0, color='red', linestyle='-', linewidth=0.5, alpha=0.7)

        # Vertical line along the Y-axis (x=0)
        ax.axvline(0, color='red', linestyle='-', linewidth=0.5, alpha=0.7)

        # Add quadrant labels
        quadrant_labels = config.get('quadrants_right' if side == 'right' else 'quadrants_left', ())
        if quadrant_labels:
            offset = radius * 0.9  # Slightly outside the circle
            positions = [
                (offset, offset),      # UI
                (-offset, offset),      # UO
                (offset, -offset),     # LI
                (-offset, -offset)      # LO
            ]
            for i, (label, pos) in enumerate(zip(quadrant_labels, positions)):
                if label:
                    ax.text(pos[0], pos[1], label, ha='center', va='center',
                           fontsize=9, color='black', alpha=0.8)

    elif shape_type == 'SemiCircle':
        # Draw semicircle for sagittal and axial views
        if plane_name == 'Sagittal':
            # Semicircle opening to the right (posterior)
            arc = Arc((radius, 0), radius*2, radius*2, theta1=90, theta2=270,
                     linestyle='-', color='black', alpha=0.5, linewidth=1)
            ax.axhline(0, color='red', linestyle='-', linewidth=0.5, alpha=0.7)


        elif plane_name == 'Axial':
            # Semicircle opening downward (posterior)
            arc = Arc((0, radius), radius*2, radius*2, angle=0, theta1=180, theta2=360,
                     linestyle='-', color='black', alpha=0.5, linewidth=1)
            ax.axvline(0, color='red', linestyle='-', linewidth=0.5, alpha=0.7)

        else:
            arc = None

        if arc:
            ax.add_patch(arc)

        # Add quadrant labels for sagittal/axial
        quadrant_labels = config.get('quadrants_right' if side == 'right' else 'quadrants_left', ())
        if plane_name == 'Sagittal' and quadrant_labels:
            offset = radius * 0.7
            positions = [
                (offset * 0.3, offset * 1.3),    # upper
                (offset, 0),               # (not used)
                (offset * 0.3, -offset * 1.3),   # lower
                (0, 0)                     # (not used)
            ]
            for i, (label, pos) in enumerate(zip(quadrant_labels, positions)):
                if label:
                    ax.text(pos[0], pos[1], label, ha='center', va='center',
                           fontsize=9, color='black', alpha=0.8)

        elif plane_name == 'Axial' and quadrant_labels:
            offset = radius * 0.7
            positions = [
                (offset * 1.1, offset * 0.1),  # outer
                (-offset * 1.1, offset * 0.1),         # inner
                (0, 0),                    # (not used)
                (0, 0)                     # (not used)
            ]
            for i, (label, pos) in enumerate(zip(quadrant_labels, positions)):
                if label:
                    ax.text(pos[0], pos[1], label, ha='center', va='center',
                           fontsize=9, color='black', alpha=0.8)


def _draw_reference_lines(ax, plane_name: str):
    """
    Helper function to draw reference lines through the nipple origin.

    Args:
        ax: Matplotlib axis
        plane_name: Name of the anatomical plane
    """
    if plane_name == 'Coronal':
        # Vertical and horizontal lines through origin
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')
        ax.axvline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')
    elif plane_name == 'Sagittal':
        # Horizontal line through origin (inf-sup axis)
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')
    elif plane_name == 'Axial':
        # Vertical line through origin (right-left axis)
        ax.axvline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Test with sample data
    print("Running analysis_plot.py test...")

    # Try to load data from Excel file
    excel_path = Path("../output/landmark_results_v6_2026_02_10.xlsx")

    if excel_path.exists():
        print(f"Loading data from: {excel_path}")
        df_ave = pd.read_excel(excel_path, sheet_name='processed_ave_data')

        # Filter for valid alignment data
        df_valid = df_ave.dropna(subset=['landmark ave prone transformed x'])
        print(f"Valid landmarks with alignment data: {len(df_valid)}")

        if len(df_valid) > 0:
            # Test the plotting function
            result = plot_nipple_relative_prone_landmarks(
                df_valid,
                title="",
                save_path="../output/figs/v6/prone_landmarks_rel_nipple",
                use_dts_cmap=True
            )

            if result:
                print(f"\nPlotted {len(result['base_left'])} left breast landmarks")
                print(f"Plotted {len(result['base_right'])} right breast landmarks")
        else:
            print("No valid alignment data found")
    else:
        print(f"Excel file not found: {excel_path}")
        print("Run main.py first to generate the data file")

