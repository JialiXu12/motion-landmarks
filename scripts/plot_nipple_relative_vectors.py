"""
Function for plotting landmark vectors relative to nipple from prone to supine
in three anatomical planes (Axial, Coronal, Sagittal).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib import cm
from pathlib import Path


def plot_nipple_relative_vectors(
        base_point_left: np.ndarray,
        vector_left: np.ndarray,
        base_point_right: np.ndarray,
        vector_right: np.ndarray,
        dts_left: np.ndarray = None,
        dts_right: np.ndarray = None,
        subject_ids_left: np.ndarray = None,
        subject_ids_right: np.ndarray = None,
        title: str = None,
        save_path: str = None,
        use_dts_cmap: bool = True
):
    """
    Plots landmark motion vectors relative to nipple from prone to supine
    in three anatomical planes: Coronal, Sagittal, and Axial.
    
    Args:
        base_point_left: (N, 3) array of left breast landmark positions relative to nipple (prone)
        vector_left: (N, 3) array of left breast displacement vectors
        base_point_right: (M, 3) array of right breast landmark positions relative to nipple (prone)
        vector_right: (M, 3) array of right breast displacement vectors
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
        _plot_single_plane(
            plane_name, config,
            base_point_left, vector_left,
            base_point_right, vector_right,
            dts_left, dts_right,
            subject_ids_left, subject_ids_right,
            title, save_path,
            use_dts_cmap
        )


def _plot_single_plane(
        plane_name: str,
        config: dict,
        base_point_left: np.ndarray,
        vector_left: np.ndarray,
        base_point_right: np.ndarray,
        vector_right: np.ndarray,
        dts_left: np.ndarray,
        dts_right: np.ndarray,
        subject_ids_left: np.ndarray,
        subject_ids_right: np.ndarray,
        title: str,
        save_path: str,
        use_dts_cmap: bool
):
    """Helper function to plot a single anatomical plane."""

    AXIS_X, AXIS_Y = config['axes']
    
    # Create figure with two subplots (Right and Left breast)
    fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(f"{title} ({plane_name.lower()} view)", fontsize=14) #, fontweight='bold')

    # Set axis limits based on plane
    if plane_name == 'Sagittal':
        # For sagittal: x-axis starts at 0 (anterior/nipple), 500 units wide
        lims_x = (0, 300)  # Anterior (0) to Posterior (500)
        lims_y = (-150, 150)  # Inf-Sup (500 units tall)
        radius = 150
    elif plane_name == 'Axial':
        # For axial: y-axis starts at 0 (anterior/nipple), 500 units wide
        lims_x = (-150, 150)  # Right-Left (500 units wide)
        lims_y = (0, 300)  # Anterior (0) to Posterior (500)
        radius = 150
    else:
        # For Coronal: use symmetric limits
        lims_x = (-150, 150)
        lims_y = (-150, 150)
        radius = 150


    # Plot Right Breast
    _plot_breast_side(
        ax_right, plane_name, config,
        base_point_right, vector_right, dts_right, subject_ids_right,
        'Right breast', 'right', lims_x, lims_y, radius,
        use_dts_cmap
    )

    # Plot Left Breast
    _plot_breast_side(
        ax_left, plane_name, config,
        base_point_left, vector_left, dts_left, subject_ids_left,
        'Left breast', 'left', lims_x, lims_y, radius,
        use_dts_cmap
    )

    # plt.tight_layout()

    # Save if path provided
    if save_path:
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if use_dts_cmap:
            filename = f"{save_path}_{plane_name.lower()}_DTS.png"
        elif subject_ids_left is not None or subject_ids_right is not None:
            filename = f"{save_path}_{plane_name.lower()}_by_subject.png"
        else:
            filename = f"{save_path}_{plane_name.lower()}_by_breast.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Only show if not using Agg backend
    if plt.get_backend().lower() != 'agg':
        plt.show()

    plt.close(fig)


def _plot_breast_side(
        ax, plane_name: str, config: dict,
        base_points: np.ndarray,
        vectors: np.ndarray,
        dts_values: np.ndarray,
        subject_ids: np.ndarray,
        side_title: str,
        side: str,
        lims_x: tuple,
        lims_y: tuple,
        radius: float,
        use_dts_cmap: bool
):
    """Helper function to plot one breast side on a given axis."""

    AXIS_X, AXIS_Y = config['axes']

    # Set title and labels
    ax.set_title(side_title, fontsize=12)
    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)

    # Set limits
    ax.set_xlim(lims_x)
    ax.set_ylim(lims_y)

    # Set ticks with 50mm intervals
    if plane_name == 'Sagittal':
        # For sagittal, x-axis starts at 0 and extends to 500
        ax.set_xticks(np.arange(0, 301, 50))
        ax.set_yticks(np.arange(-150, 151, 50))
    elif plane_name == 'Axial':
        # For axial, y-axis starts at 0 and extends to 500
        ax.set_xticks(np.arange(-150, 151, 50))
        ax.set_yticks(np.arange(0, 301, 50))
    else:
        # For Coronal, both axes symmetric
        ax.set_xticks(np.arange(-150, 151, 50))
        ax.set_yticks(np.arange(-150, 151, 50))

    ax.set_aspect('equal', adjustable='box')
    # Grid style (dashed, alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Draw anatomical shape
    _draw_anatomical_shape(ax, plane_name, config, radius, side)

    # Draw reference lines through nipple origin
    _draw_reference_lines(ax, plane_name)

    # Plot nipple at origin (matching plot_vectors_rel_sternum origin marker)
    ax.plot(0, 0, 'ro', markersize=5, zorder=10, label='Nipple (Origin)')

    # Plot vectors
    if len(base_points) > 0:
        if use_dts_cmap and dts_values is not None and len(dts_values) == len(base_points):
            # Color by DTS (viridis colormap)
            colors = dts_values

            # Create scatter plot for color mapping
            scatter = ax.scatter(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                c=colors,
                cmap='viridis',
                s=20,
                zorder=5,
                vmin=0,
                vmax=40
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('DTS (mm)', rotation=270, labelpad=15)

            # Plot arrows with same colors
            for i in range(len(base_points)):
                ax.arrow(
                    base_points[i, AXIS_X],
                    base_points[i, AXIS_Y],
                    vectors[i, AXIS_X],
                    vectors[i, AXIS_Y],
                    head_width=2,
                    head_length=2,
                    fc=plt.cm.viridis(colors[i] / 40),
                    ec=plt.cm.viridis(colors[i] / 40),
                    alpha=0.7,
                    zorder=4
                )

        elif not use_dts_cmap and subject_ids is not None and len(subject_ids) == len(base_points):
            # Color by subject ID (viridis colormap, same as plot_vectors_rel_sternum)
            unique_subjects = np.unique(subject_ids)
            n_subjects = len(unique_subjects)
            cmap_subj = cm.get_cmap('viridis', max(n_subjects, 1))

            subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}

            # Plot each subject with a different color (no legend, same as sternum plot)
            for i, subj in enumerate(unique_subjects):
                mask = subject_ids == subj
                color = cmap_subj(subject_to_idx[subj])

                # Plot scatter points for this subject
                ax.scatter(
                    base_points[mask, AXIS_X],
                    base_points[mask, AXIS_Y],
                    c=[color],
                    s=20,
                    zorder=5,
                    alpha=0.7
                )

                # Plot arrows for this subject
                subj_base = base_points[mask]
                subj_vec = vectors[mask]
                for j in range(len(subj_base)):
                    ax.arrow(
                        subj_base[j, AXIS_X],
                        subj_base[j, AXIS_Y],
                        subj_vec[j, AXIS_X],
                        subj_vec[j, AXIS_Y],
                        head_width=2,
                        head_length=2,
                        fc=color,
                        ec=color,
                        alpha=0.7,
                        zorder=4
                    )

        else:
            # Default: color by breast side (same as plot_vectors_rel_sternum)
            # Blue for right breast, green for left breast
            if side == 'right':
                breast_color = 'blue'
            else:
                breast_color = 'green'

            # Plot scatter points
            ax.scatter(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                c=breast_color,
                s=20,
                zorder=5,
                alpha=0.7
            )

            # Plot quiver
            ax.quiver(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                vectors[:, AXIS_X],
                vectors[:, AXIS_Y],
                angles='xy',
                scale_units='xy',
                scale=1,
                color=breast_color,
                width=0.003,
                headwidth=3,
                headlength=4,
                alpha=0.7,
                zorder=4
            )

    # Add quadrant labels
    _add_quadrant_labels(ax, config, side, radius, plane_name)


def _draw_anatomical_shape(ax, plane_name: str, config: dict, radius: float, side: str):
    """Draw the anatomical background shape (circle or semicircle)."""

    if config['shape'] == 'Circle':
        # Full circle for coronal plane
        circle = Circle((0, 0), radius, fill=False, color='gray', lw=1.5, linestyle='-')
        ax.add_artist(circle)

    elif plane_name == 'Sagittal':
        # Semicircle for sagittal plane (anterior side)
        # The breast extends from posterior (x=0) to anterior (x=radius)
        # Arc centered at (radius, 0) from 90° to 270° (left half of circle)
        arc = Arc(
            (radius, 0), radius * 2, radius * 2,
            theta1=90, theta2=270,
            color='gray', lw=1.5, linestyle='-'
        )
        ax.add_artist(arc)
        # Straight line at posterior edge (x=0)
        ax.plot([0, 0], [-radius, radius], color='gray', lw=1.5, linestyle='-')

    elif plane_name == 'Axial':
        # Semicircle for axial plane (anterior side)
        # The breast extends from posterior (y=0) to anterior (y=radius)
        # Arc centered at (0, radius) from 180° to 360° (lower half of circle)
        arc = Arc(
            (0, radius), radius * 2, radius * 2,
            theta1=180, theta2=360,
            color='gray', lw=1.5, linestyle='-'
        )
        ax.add_artist(arc)
        # Straight line at posterior edge (y=0)
        ax.plot([-radius, radius], [0, 0], color='gray', lw=1.5, linestyle='-')


def _draw_reference_lines(ax, plane_name: str):
    """Draw reference lines through nipple."""

    if plane_name == 'Coronal':
        # Both horizontal and vertical lines
        ax.axhline(0, color='red', lw=1, alpha=0.5, zorder=1)
        ax.axvline(0, color='red', lw=1, alpha=0.5, zorder=1)
    elif plane_name == 'Sagittal':
        # Only horizontal line (inf-sup)
        ax.axhline(0, color='red', lw=1, alpha=0.5, zorder=1)
    elif plane_name == 'Axial':
        # Only vertical line (right-left)
        ax.axvline(0, color='red', lw=1, alpha=0.5, zorder=1)


def _add_quadrant_labels(ax, config: dict, side: str, radius: float, plane_name: str):
    """Add quadrant labels to the plot."""

    quadrants = config['quadrants_right'] if side == 'right' else config['quadrants_left']

    if plane_name == 'Coronal':
        # Position labels on diagonal at ~45 degrees, outside the circle
        diagonal_offset = radius * 0.85

        # Top-right quadrant
        if quadrants[0]:
            ax.text(diagonal_offset, diagonal_offset, quadrants[0],
                    ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)

        # Top-left quadrant
        if quadrants[1]:
            ax.text(-diagonal_offset, diagonal_offset, quadrants[1],
                    ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)

        # Bottom-right quadrant
        if quadrants[2]:
            ax.text(diagonal_offset, -diagonal_offset, quadrants[2],
                    ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)

        # Bottom-left quadrant
        if quadrants[3]:
            ax.text(-diagonal_offset, -diagonal_offset, quadrants[3],
                    ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)

    elif plane_name == 'Sagittal':
        # Sagittal: semicircle, place labels to the LEFT of the arc
        # quadrants[0] = 'upper' (top), quadrants[2] = 'lower' (bottom)
        # X-axis starts at 0 (anterior/nipple), extends to posterior
        # Place labels at x=0 (left edge of the arc)
        x_pos = 40  # Slightly to the right of the posterior edge
        y_upper = radius * 0.9  # Upper position
        y_lower = -radius * 0.9  # Lower position

        # Upper label
        if quadrants[0]:
            ax.text(x_pos, y_upper, quadrants[0],
                    ha='right', va='center', fontsize=10, fontweight='bold', alpha=0.6)

        # Lower label
        if quadrants[2]:
            ax.text(x_pos, y_lower, quadrants[2],
                    ha='right', va='center', fontsize=10, fontweight='bold', alpha=0.6)

    elif plane_name == 'Axial':
        # Axial: semicircle, place labels to the BOTTOM of the arc
        # quadrants[0] = 'inner', quadrants[1] = 'outer' (for right breast)
        # Y-axis starts at 0 (anterior/nipple), extends to posterior
        # Place labels at y=0 (bottom edge of the arc)
        y_pos = 30  # Slightly above the posterior edge
        x_inner = radius * 0.9  # Inner position (closer to midline)
        x_outer = -radius * 0.9 if side == 'right' else radius * 0.9  # Outer position (away from midline)

        # For right breast: inner is positive x, outer is negative x
        # For left breast: inner is negative x, outer is positive x
        if side == 'right':
            x_inner_pos = radius * 0.9
            x_outer_pos = -radius * 0.9
        else:
            x_inner_pos = -radius * 0.9
            x_outer_pos = radius * 0.9

        # Inner label (quadrants[0])
        if quadrants[0]:
            ax.text(x_inner_pos, y_pos, quadrants[0],
                    ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.6)

        # Outer label (quadrants[1])
        if quadrants[1]:
            ax.text(x_outer_pos, y_pos, quadrants[1],
                    ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.6)
