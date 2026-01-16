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
        title: str = "Landmark Motion Relative to Nipple",
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
        title: Overall title for the plots
        save_path: Path to save the figure (optional)
    
    Coordinate system:
        X (0): Right/Left (positive = right, negative = left)
        Y (1): Anterior/Posterior (positive = anterior, negative = posterior)
        Z (2): Inferior/Superior (positive = superior, negative = inferior)
    """
    
    # Plane configurations
    PLANE_CONFIG = {
        'Coronal': {
            'axes': (0, 2),  # X (R/L) vs Z (I/S)
            'xlabel': 'right-left (mm)',
            'ylabel': 'inf-sup (mm)',
            'shape': 'Circle',
            'quadrants_right': ('UI', 'UO', 'LI', 'LO'),
            'quadrants_left': ('UO', 'UI', 'LO', 'LI')
        },
        'Sagittal': {
            'axes': (1, 2),  # Y (A/P) vs Z (I/S)
            'xlabel': 'post-ant (mm)',
            'ylabel': 'inf-sup (mm)',
            'shape': 'SemiCircle',
            'quadrants_right': ('upper', '', 'lower', ''),
            'quadrants_left': ('upper', '', 'lower', '')
        },
        'Axial': {
            'axes': (0, 1),  # X (R/L) vs Y (A/P)
            'xlabel': 'right-left (mm)',
            'ylabel': 'post-ant (mm)',
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
        title: str,
        save_path: str,
        use_dts_cmap: bool
):
    """Helper function to plot a single anatomical plane."""
    
    AXIS_X, AXIS_Y = config['axes']
    
    # Create figure with two subplots (Right and Left breast)
    fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{plane_name} plane\n{title}", fontsize=14, fontweight='bold')
    
    # Set axis limits based on plane to match reference images
    if plane_name == 'Coronal':
        # Reference: -60 to 60 for both axes
        lims_x = (-60, 60)
        lims_y = (-60, 60)
        radius = 60
    elif plane_name == 'Sagittal':
        # Reference: post-ant from 0 to 140, inf-sup from -60 to 60
        lims_x = (0, 140)  # Posterior (0) to Anterior (140)
        lims_y = (-60, 60)  # Inferior to Superior
        radius = 60
    else:  # Axial
        # Reference: right-left from -60 to 60, post-ant from 0 to 140
        lims_x = (-60, 60)  # Right to Left
        lims_y = (0, 140)  # Posterior (0) to Anterior (140)
        radius = 60
    
    
    # Plot Right Breast
    _plot_breast_side(
        ax_right, plane_name, config,
        base_point_right, vector_right, dts_right,
        'Right breast', 'right', lims_x, lims_y, radius,
        use_dts_cmap
    )
    
    # Plot Left Breast
    _plot_breast_side(
        ax_left, plane_name, config,
        base_point_left, vector_left, dts_left,
        'Left breast', 'left', lims_x, lims_y, radius,
        use_dts_cmap
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        filename = f"{save_path}_{plane_name.lower()}.png"
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
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    
    # Set limits
    ax.set_xlim(lims_x)
    ax.set_ylim(lims_y)
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Draw anatomical background shape
    _draw_anatomical_shape(ax, plane_name, config, radius, side)
    
    # Draw reference lines
    _draw_reference_lines(ax, plane_name)
    
    # Plot nipple at origin
    ax.plot(0, 0, 'ro', markersize=8, zorder=10, label='Nipple')
    
    # Plot vectors
    if len(base_points) > 0:
        # Determine colors
        if use_dts_cmap and dts_values is not None and len(dts_values) == len(base_points):
            # Color by DTS
            colors = dts_values
            cmap = 'viridis'
        else:
            # Default color
            colors = 'darkblue'
            cmap = None
        
        # Plot quiver
        if cmap:
            # Create scatter plot for color mapping
            scatter = ax.scatter(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                c=colors,
                cmap=cmap,
                s=30,
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
                    fc=plt.cm.viridis(colors[i] / 40) if cmap else colors,
                    ec=plt.cm.viridis(colors[i] / 40) if cmap else colors,
                    alpha=0.7,
                    zorder=4
                )
        else:
            # Plot scatter points
            ax.scatter(
                base_points[:, AXIS_X],
                base_points[:, AXIS_Y],
                c=colors,
                s=30,
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
                color=colors,
                width=0.003,
                headwidth=3,
                headlength=4,
                alpha=0.7,
                zorder=4
            )
    
    # Add quadrant labels
    _add_quadrant_labels(ax, config, side, radius)


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


def _add_quadrant_labels(ax, config: dict, side: str, radius: float):
    """Add quadrant labels to the plot."""
    
    quadrants = config['quadrants_right'] if side == 'right' else config['quadrants_left']
    text_offset = radius * 0.7
    
    # Top-right quadrant
    if quadrants[0]:
        ax.text(text_offset, text_offset, quadrants[0],
                ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)
    
    # Top-left quadrant
    if quadrants[1]:
        ax.text(-text_offset, text_offset, quadrants[1],
                ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)
    
    # Bottom-right quadrant
    if quadrants[2]:
        ax.text(text_offset, -text_offset, quadrants[2],
                ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)
    
    # Bottom-left quadrant
    if quadrants[3]:
        ax.text(-text_offset, -text_offset, quadrants[3],
                ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.6)
