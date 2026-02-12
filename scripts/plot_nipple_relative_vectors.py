"""
Function for plotting landmark vectors relative to nipple from prone to supine
in three anatomical planes (Axial, Coronal, Sagittal).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from pathlib import Path
import pandas as pd


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
    fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(18/2.54, 8/2.54), constrained_layout=True)
    fig.suptitle(f"{title} ({plane_name.lower()} view)", fontsize=10) #, fontweight='bold')

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
        lims_x = (-100, 100)
        lims_y = (-100, 100)
        radius = 100


    # Plot Right Breast
    subplot_right = _plot_breast_side(
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

    if use_dts_cmap and subplot_right is not None:
        cbar = fig.colorbar(subplot_right, ax=[ax_right, ax_left], location='right', aspect=30, pad=0.02)
        cbar.set_label('DTS (mm)', rotation=270, labelpad=15)

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
    scatter = None  # Initialize scatter for colorbar

    # Set title and labels
    ax.set_title(side_title, fontsize=10,loc='left',pad=10)
    ax.set_xlabel(config['xlabel'], fontsize=9)
    ax.set_ylabel(config['ylabel'], fontsize=9)

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
        ax.set_xticks(np.arange(-100, 101, 50))
        ax.set_yticks(np.arange(-100, 101, 50))

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
                s=0,
                zorder=5,
                vmin=0,
                vmax=40
            )

            # Add colorbar
            # cbar = plt.colorbar(scatter, ax=ax)
            # cbar.set_label('DTS (mm)', rotation=270, labelpad=15)

            # Plot arrows with same colors
            for i in range(len(base_points)):
                ax.arrow(
                    base_points[i, AXIS_X],
                    base_points[i, AXIS_Y],
                    vectors[i, AXIS_X],
                    vectors[i, AXIS_Y],
                    width=0.8,
                    head_width=6,
                    head_length=8,
                    fc=plt.cm.viridis(colors[i] / 40),
                    ec=plt.cm.viridis(colors[i] / 40),
                    alpha=0.7,
                    zorder=4,
                    length_includes_head=True
                )

        elif not use_dts_cmap and subject_ids is not None and len(subject_ids) == len(base_points):
            # Color by subject ID (viridis colormap, same as plot_vectors_rel_sternum)
            unique_subjects = np.unique(subject_ids)
            n_subjects = len(unique_subjects)
            cmap_subj = plt.colormaps.get_cmap('viridis').resampled(max(n_subjects, 1))

            subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}

            # Plot each subject with a different color (no legend, same as sternum plot)
            for i, subj in enumerate(unique_subjects):
                mask = subject_ids == subj
                color = cmap_subj(subject_to_idx[subj])

                # # Plot scatter points for this subject
                # ax.scatter(
                #     base_points[mask, AXIS_X],
                #     base_points[mask, AXIS_Y],
                #     c=[color],
                #     s=20,
                #     zorder=5,
                #     alpha=0.7
                # )

                # Plot arrows for this subject
                subj_base = base_points[mask]
                subj_vec = vectors[mask]
                for j in range(len(subj_base)):
                    ax.arrow(
                        subj_base[j, AXIS_X],
                        subj_base[j, AXIS_Y],
                        subj_vec[j, AXIS_X],
                        subj_vec[j, AXIS_Y],
                        head_width=4,
                        head_length=6,
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
                s=0,
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
                width=0.007,
                headwidth=6,
                headlength=8,
                alpha=0.7,
                zorder=4
            )

    # Add quadrant labels
    _add_quadrant_labels(ax, config, side, radius, plane_name)

    # Return scatter object for colorbar (only defined in DTS mode)
    if use_dts_cmap and dts_values is not None and len(dts_values) == len(base_points):
        return scatter
    return None


def _draw_anatomical_shape(ax, plane_name: str, config: dict, radius: float, side: str):
    """Draw the anatomical background shape (circle or semicircle)."""

    if config['shape'] == 'Circle':
        # Full circle for coronal plane
        circle = Circle((0, 0), radius, fill=False, color='black', lw=1, linestyle='-', alpha=0.5,)
        ax.add_artist(circle)

    elif plane_name == 'Sagittal':
        # Semicircle for sagittal plane (anterior side)
        # The breast extends from posterior (x=0) to anterior (x=radius)
        # Arc centered at (radius, 0) from 90° to 270° (left half of circle)
        arc = Arc(
            (radius, 0), radius * 2, radius * 2,
            theta1=90, theta2=270,
            color='black', lw=1, linestyle='-',alpha=0.5
        )
        ax.add_artist(arc)
        # Straight line at posterior edge (x=0)
        ax.plot([0, 0], [-radius, radius], color='black', lw=1, linestyle='-')

    elif plane_name == 'Axial':
        # Semicircle for axial plane (anterior side)
        # The breast extends from posterior (y=0) to anterior (y=radius)
        # Arc centered at (0, radius) from 180° to 360° (lower half of circle)
        arc = Arc(
            (0, radius), radius * 2, radius * 2,
            theta1=180, theta2=360,
            color='black', lw=1, linestyle='-',alpha=0.5
        )
        ax.add_artist(arc)
        # Straight line at posterior edge (y=0)
        ax.plot([-radius, radius], [0, 0], color='black', lw=1, linestyle='-')


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
                    ha='center', va='center', fontsize=9, alpha=0.8)

        # Top-left quadrant
        if quadrants[1]:
            ax.text(-diagonal_offset, diagonal_offset, quadrants[1],
                    ha='center', va='center', fontsize=9,  alpha=0.8)

        # Bottom-right quadrant
        if quadrants[2]:
            ax.text(diagonal_offset, -diagonal_offset, quadrants[2],
                    ha='center', va='center', fontsize=9, alpha=0.6)

        # Bottom-left quadrant
        if quadrants[3]:
            ax.text(-diagonal_offset, -diagonal_offset, quadrants[3],
                    ha='center', va='center', fontsize=9, alpha=0.6)

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
                    ha='right', va='center', fontsize=9,  alpha=0.6)

        # Lower label
        if quadrants[2]:
            ax.text(x_pos, y_lower, quadrants[2],
                    ha='right', va='center', fontsize=9, alpha=0.6)

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
                    ha='center', va='top', fontsize=9, alpha=0.6)

        # Outer label (quadrants[1])
        if quadrants[1]:
            ax.text(x_outer_pos, y_pos, quadrants[1],
                    ha='center', va='top', fontsize=9, alpha=0.6)


def load_data_from_excel(excel_path):
    """Load landmark data from Excel file."""
    try:
        all_sheets = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl', header=0)
        df_ave = all_sheets['processed_ave_data']
        print(f"Successfully loaded data from {excel_path}")
        return df_ave
    except FileNotFoundError:
        print(f"Error: The file {excel_path} was not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def get_nipple_relative_points_and_vectors(sub_df, is_left_breast):
    """
    Extract nipple-relative base points and vectors from a DataFrame.

    Args:
        sub_df: DataFrame containing landmark and nipple positions
        is_left_breast: True if processing left breast landmarks

    Returns:
        base_points: (N, 3) array of landmark positions relative to nipple (prone)
        vectors: (N, 3) array of displacement vectors relative to nipple
        dts_values: (N,) array of DTS values
        vl_ids: (N,) array of subject IDs
        lm_displacement: (N, 3) array of landmark displacement vectors
        nipple_displacement: (N, 3) array of nipple displacement vectors
    """
    if sub_df.empty:
        return np.empty((0, 3)), np.empty((0, 3)), None, None, np.empty((0, 3)), np.empty((0, 3))

    # Extract landmark prone positions (relative to sternum)
    prone_x = sub_df['landmark ave prone transformed x'].values
    prone_y = sub_df['landmark ave prone transformed y'].values
    prone_z = sub_df['landmark ave prone transformed z'].values

    # Extract landmark supine positions (relative to sternum)
    supine_x = sub_df['landmark ave supine x'].values
    supine_y = sub_df['landmark ave supine y'].values
    supine_z = sub_df['landmark ave supine z'].values

    # Extract nipple positions from DataFrame columns
    if is_left_breast:
        # Left nipple prone position
        nipple_prone_x = sub_df['left nipple prone transformed x'].values
        nipple_prone_y = sub_df['left nipple prone transformed y'].values
        nipple_prone_z = sub_df['left nipple prone transformed z'].values

        # Left nipple supine position
        nipple_supine_x = sub_df['left nipple supine x'].values
        nipple_supine_y = sub_df['left nipple supine y'].values
        nipple_supine_z = sub_df['left nipple supine z'].values
    else:
        # Right nipple prone position
        nipple_prone_x = sub_df['right nipple prone transformed x'].values
        nipple_prone_y = sub_df['right nipple prone transformed y'].values
        nipple_prone_z = sub_df['right nipple prone transformed z'].values

        # Right nipple supine position
        nipple_supine_x = sub_df['right nipple supine x'].values
        nipple_supine_y = sub_df['right nipple supine y'].values
        nipple_supine_z = sub_df['right nipple supine z'].values

    sternum_prone_x = sub_df['sternum superior prone transformed x'].values
    sternum_prone_y = sub_df['sternum superior prone transformed y'].values
    sternum_prone_z = sub_df['sternum superior prone transformed z'].values

    sternum_supine_x = sub_df['sternum superior supine x'].values
    sternum_supine_y = sub_df['sternum superior supine y'].values
    sternum_supine_z = sub_df['sternum superior supine z'].values

    # Calculate nipple displacement vector (supine - prone)
    nipple_disp_x = (nipple_supine_x - sternum_supine_x) - (nipple_prone_x - sternum_prone_x)
    nipple_disp_y = (nipple_supine_y - sternum_supine_y) - (nipple_prone_y - sternum_prone_y)
    nipple_disp_z = (nipple_supine_z - sternum_supine_z) - (nipple_prone_z - sternum_prone_z)
    nipple_displacement = np.column_stack((nipple_disp_x, nipple_disp_y, nipple_disp_z))

    # Calculate base points: landmark position relative to prone nipple
    base_x = prone_x - (nipple_prone_x - sternum_prone_x)
    base_y = prone_y - (nipple_prone_y - sternum_prone_y)
    base_z = prone_z - (nipple_prone_z - sternum_prone_z)
    base_points = np.column_stack((base_x, base_y, base_z))

    # Calculate landmark displacement relative to sternum
    lm_disp_x = supine_x - prone_x
    lm_disp_y = supine_y - prone_y
    lm_disp_z = supine_z - prone_z
    lm_displacement = np.column_stack((lm_disp_x, lm_disp_y, lm_disp_z))

    # Calculate vectors: landmark motion relative to nipple
    # Subtract nipple motion to show intrinsic deformation
    ld_vectors_rel_nipple = lm_displacement - nipple_displacement

    # Extract DTS values if available
    dts_col = 'Distance to skin (prone) [mm]'
    dts_values = sub_df[dts_col].values if dts_col in sub_df.columns else None
    vl_ids = sub_df['VL_ID'].values if 'VL_ID' in sub_df.columns else None

    return base_points, ld_vectors_rel_nipple, dts_values, vl_ids, lm_displacement, nipple_displacement


def run_plot(df_ave, vl_id=None, title="Landmark displacement relative to nipple",
             save_path=None, use_dts_cmap=True):
    """
    Main function to extract data and generate nipple-relative plots.

    Args:
        df_ave: DataFrame with landmark data
        vl_id: Optional subject ID to filter data. If None, uses all subjects.
        title: Plot title
        save_path: Path to save the plots
        use_dts_cmap: Whether to use the DTS colormap
    """
    print("\n--- Plotting Nipple-Relative Motion ---")

    # 1. Filter data if specific subject requested
    if vl_id is not None:
        df_subset = df_ave[df_ave['VL_ID'] == vl_id].copy()
        if df_subset.empty:
            print(f"Warning: No data found for subject VL_{vl_id}")
            return
    else:
        df_subset = df_ave.copy()

    # Remove rows with missing essential data
    required_cols = ['landmark ave prone transformed x', 'landmark ave supine x',
                     'landmark side (prone)']
    df_subset = df_subset.dropna(subset=required_cols)

    # 2. Separate into Left (LB) and Right (RB) breasts
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Extract data for both breasts
    base_left, vec_left, dts_left, vl_ids_left, lm_disp_left, nipple_disp_left = \
        get_nipple_relative_points_and_vectors(left_df, is_left_breast=True)
    base_right, vec_right, dts_right, vl_ids_right, lm_disp_right, nipple_disp_right = \
        get_nipple_relative_points_and_vectors(right_df, is_left_breast=False)

    print(f"Left breast landmarks: {len(base_left)}")
    print(f"Right breast landmarks: {len(base_right)}")

    # 4. Call the specialized plotting function
    plot_nipple_relative_vectors(
        base_point_left=base_left,
        vector_left=vec_left,
        base_point_right=base_right,
        vector_right=vec_right,
        dts_left=dts_left,
        dts_right=dts_right,
        subject_ids_left=vl_ids_left,
        subject_ids_right=vl_ids_right,
        title=title,
        save_path=save_path,
        use_dts_cmap=use_dts_cmap
    )

    return base_left, base_right, vec_left, vec_right


if __name__ == "__main__":
    # Default Excel file path
    OUTPUT_DIR = Path(__file__).parent.parent / "output"
    EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v6_2026_02_10.xlsx"

    # Load data from Excel
    df_ave = load_data_from_excel(EXCEL_FILE_PATH)

    if df_ave is not None:
        # Set up save path
        save_dir = OUTPUT_DIR / "figs" / "v6" / "landmark vectors"
        save_path = save_dir / "Vectors_rel_nipple"

        # Run plots with DTS coloring
        print("\n=== Generating plots colored by DTS ===")
        run_plot(
            df_ave,
            vl_id=None,  # Use all subjects
            title="Landmark displacement relative to nipple",
            save_path=str(save_path),
            use_dts_cmap=True
        )

        # Run plots colored by subject
        print("\n=== Generating plots colored by subject ===")
        run_plot(
            df_ave,
            vl_id=None,
            title="Landmark displacement relative to nipple",
            save_path=str(save_path),
            use_dts_cmap=False
        )

        print("\n=== Plotting complete ===")

