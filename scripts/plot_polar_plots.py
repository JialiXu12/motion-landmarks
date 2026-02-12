"""
Polar Plot Functions for Clock Position Rotation Analysis
=========================================================
This script contains functions to generate polar plots showing landmark rotation
from prone to supine position relative to the nipple.

The main function `analyse_clock_position_rotation` analyzes and visualizes
clock position changes using polar plots with proper arrow directions.

Usage:
    python plot_polar_plots.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def circular_mean_angle(angles_rad):
    """
    Calculate the circular mean of angles (in radians).
    This correctly handles the circular nature of angles (e.g., 350° and 10° average to 0°, not 180°).

    Args:
        angles_rad: Array of angles in radians

    Returns:
        mean_angle_rad: Circular mean angle in radians
    """
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    return mean_angle


def polar_to_cartesian(theta, r):
    """Convert polar coordinates to Cartesian for arrow drawing."""
    x = r * np.sin(theta)  # theta=0 is at top, increases clockwise
    y = r * np.cos(theta)
    return x, y


def draw_polar_arrow(ax, theta_start, r_start, theta_end, r_end,
                     color='black', alpha=0.6, linewidth=0.8,
                     head_width=0.03, head_length=0.02, is_mean=False):
    """
    Draw an arrow on a polar plot with correct direction.

    The issue with matplotlib's polar scatter markers like '>' is that they don't
    rotate based on the direction from start to end point. This function draws
    a proper arrow that points in the correct direction.

    Args:
        ax: Matplotlib polar axes
        theta_start: Starting angle in radians
        r_start: Starting radius
        theta_end: Ending angle in radians
        r_end: Ending radius
        color: Arrow color
        alpha: Transparency
        linewidth: Line width
        head_width: Arrow head width (relative)
        head_length: Arrow head length (relative)
        is_mean: If True, use thicker style for mean vectors
    """
    # Draw the line
    line, = ax.plot([theta_start, theta_end], [r_start, r_end],
                    color=color, alpha=alpha,
                    linewidth=3 if is_mean else linewidth,
                    zorder=10 if is_mean else 3)

    # Calculate arrow head position and direction
    # For polar plots, we need to calculate the direction in Cartesian space
    # then add a small marker at the end pointing in the right direction

    # Convert to Cartesian
    x_start = r_start * np.sin(theta_start)
    y_start = r_start * np.cos(theta_start)
    x_end = r_end * np.sin(theta_end)
    y_end = r_end * np.cos(theta_end)

    # Calculate direction angle
    dx = x_end - x_start
    dy = y_end - y_start

    if np.sqrt(dx**2 + dy**2) > 1e-6:  # Only draw if there's movement
        # Direction angle in radians (from positive x-axis)
        direction = np.arctan2(dx, dy)  # Note: arctan2(dy, dx) for standard, but we use (dx, dy) because of polar orientation

        # Draw arrowhead using a triangular marker at the endpoint
        # The marker angle needs to match the direction of movement
        marker_size = 80 if is_mean else 20

        # Use a rotated triangle marker
        # Create custom marker that points in the direction of movement
        # For polar plots with theta=0 at top, we need to adjust the rotation
        rotation_angle = np.degrees(direction)

        # Draw a small arrowhead using a scatter with rotated marker
        ax.scatter(theta_end, r_end,
                  marker=(3, 0, rotation_angle),  # Triangle marker with rotation
                  s=marker_size,
                  color=color,
                  alpha=alpha,
                  zorder=11 if is_mean else 4)


def draw_polar_arrow_annotate(ax, theta_start, r_start, theta_end, r_end,
                              color='black', alpha=0.6, linewidth=0.8, is_mean=False):
    """
    Draw an arrow on a polar plot using FancyArrowPatch for proper arrow direction.

    This approach uses matplotlib's annotation system which handles arrow directions
    correctly in polar coordinates.

    Args:
        ax: Matplotlib polar axes
        theta_start: Starting angle in radians
        r_start: Starting radius
        theta_end: Ending angle in radians
        r_end: Ending radius
        color: Arrow color
        alpha: Transparency
        linewidth: Line width
        is_mean: If True, use thicker style for mean vectors
    """
    # Determine arrow style based on is_mean
    if is_mean:
        arrowstyle = '->,head_width=8,head_length=10'
        lw = 3
        zorder = 10
    else:
        arrowstyle = '->,head_width=3,head_length=4'
        lw = linewidth
        zorder = 3

    # Use annotate with empty string to draw arrow
    ax.annotate('',
                xy=(theta_end, r_end),
                xytext=(theta_start, r_start),
                arrowprops=dict(arrowstyle=arrowstyle,
                               color=color,
                               alpha=alpha,
                               lw=lw,
                               shrinkA=0,
                               shrinkB=0),
                zorder=zorder)


# ------------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ------------------------------------------------------------------------------

def analyse_clock_position_rotation(df_ave, base_left=None, base_right=None,
                                    vec_left=None, vec_right=None, save_dir=None):
    """
    Analyze clock position rotation from prone to supine, focusing on gravity-induced
    lateral displacement. Creates polar plots showing rotation patterns with proper arrows.

    Tests the hypothesis: "We observed a mean clockwise rotation of X hours (Y degrees)
    for left-sided tumors, consistent with gravity-induced lateral displacement in the
    supine position."

    Args:
        df_ave: DataFrame with landmark data including nipple-relative positions
        base_left: Nx3 array of prone landmark positions relative to left nipple
        base_right: Nx3 array of prone landmark positions relative to right nipple
        vec_left: Nx3 array of displacement vectors for left breast
        vec_right: Nx3 array of displacement vectors for right breast
        save_dir: Directory to save plots (default: ../output/figs/clock_analysis/)

    Returns:
        summary: Dictionary with rotation statistics for left and right breasts
    """
    print("\n" + "="*80)
    print("CLOCK POSITION ROTATION ANALYSIS")
    print("="*80)

    if save_dir is None:
        save_dir = Path("..") / "output" / "figs" / "v6" / "clock_analysis"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = df_ave.copy()

    # Separate left and right breasts
    df_left = df[df['landmark side (prone)'] == 'LB'].copy()
    df_right = df[df['landmark side (prone)'] == 'RB'].copy()

    def compute_clock_positions(base_points, vectors):
        """
        Compute clock positions (angle and radius) from nipple-relative coordinates.
        Uses coronal plane projection (X, Z) for 2D clock face.
        """
        if len(base_points) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        prone_x = base_points[:, 0]  # X: right-left
        prone_z = base_points[:, 2]  # Z: inf-sup

        supine_x = prone_x + vectors[:, 0]
        supine_z = prone_z + vectors[:, 2]

        # Clock face uses coronal plane (X, Z)
        # Angle: 0° = superior (12 o'clock), positive = clockwise
        angle_prone = np.degrees(np.arctan2(prone_x, prone_z))
        angle_supine = np.degrees(np.arctan2(supine_x, supine_z))

        # Normalize to [0, 360)
        angle_prone = (angle_prone + 360) % 360
        angle_supine = (angle_supine + 360) % 360

        # Distance (radius on coronal plane)
        distance_prone = np.sqrt(prone_x**2 + prone_z**2)
        distance_supine = np.sqrt(supine_x**2 + supine_z**2)

        # Calculate rotation (handle wraparound)
        angle_diff = angle_supine - angle_prone
        angle_diff = np.where(angle_diff > 180, angle_diff - 360, angle_diff)
        angle_diff = np.where(angle_diff < -180, angle_diff + 360, angle_diff)

        clock_rotation = angle_diff / 30.0  # Convert to hours

        return angle_prone, angle_supine, distance_prone, distance_supine, angle_diff, clock_rotation

    # Calculate clock positions for each breast
    processed_dfs = {}

    for breast_side, df_orig, base_points, vectors in [
        ('Left', df_left, base_left, vec_left),
        ('Right', df_right, base_right, vec_right)
    ]:
        if len(df_orig) == 0 or base_points is None or len(base_points) == 0:
            continue

        df_subset = df_orig.copy()

        if len(df_subset) != len(base_points):
            print(f"Warning: DataFrame length ({len(df_subset)}) doesn't match base_points length ({len(base_points)}) for {breast_side} breast")
            continue

        angle_prone, angle_supine, distance_prone, distance_supine, angle_rotation, clock_rotation = \
            compute_clock_positions(base_points, vectors)

        df_subset['angle_prone'] = angle_prone
        df_subset['angle_supine'] = angle_supine
        df_subset['distance_prone'] = distance_prone
        df_subset['distance_supine'] = distance_supine
        df_subset['angle_rotation'] = angle_rotation
        df_subset['clock_rotation'] = clock_rotation

        df_subset['clock_prone'] = ((angle_prone / 30.0) % 12)
        df_subset['clock_supine'] = ((angle_supine / 30.0) % 12)
        df_subset['clock_prone'] = df_subset['clock_prone'].replace(0, 12)
        df_subset['clock_supine'] = df_subset['clock_supine'].replace(0, 12)

        df_subset['distance_change'] = distance_supine - distance_prone

        df_clean = df_subset.dropna(subset=['angle_prone', 'angle_supine']).copy()
        processed_dfs[breast_side] = df_clean

    df_left = processed_dfs.get('Left', pd.DataFrame())
    df_right = processed_dfs.get('Right', pd.DataFrame())

    # --- PART 1: CLOCK FREQUENCY ANALYSIS ---
    print("\n" + "-"*80)
    print("CLOCK POSITION FREQUENCY ANALYSIS")
    print("-"*80)

    def round_to_half_hour(h):
        if pd.isna(h):
            return h
        r = round(h * 2) / 2.0
        if r == 0:
            r = 12.0
        elif r > 12:
            r = r - 12
        return r

    def format_clock_time(h):
        if pd.isna(h):
            return "N/A"
        whole = int(h)
        frac = h - whole
        if abs(frac - 0.5) < 0.01:
            return f"{whole}:30"
        else:
            return f"{whole}:00"

    for df_subset, breast_side in [(df_left, 'Left'), (df_right, 'Right')]:
        if len(df_subset) == 0:
            continue

        print(f"\n{breast_side} Breast Frequency Distribution (Half-Hour Precision):")

        hours_prone = df_subset['clock_prone'].apply(round_to_half_hour)
        hours_supine = df_subset['clock_supine'].apply(round_to_half_hour)

        counts_prone = hours_prone.value_counts().sort_index()
        counts_supine = hours_supine.value_counts().sort_index()

        half_hour_bins = []
        for h in range(1, 13):
            half_hour_bins.append(float(h))
            half_hour_bins.append(h + 0.5)

        for hh in half_hour_bins:
            if hh not in counts_prone.index:
                counts_prone.loc[hh] = 0
            if hh not in counts_supine.index:
                counts_supine.loc[hh] = 0

        counts_prone = counts_prone.sort_index()
        counts_supine = counts_supine.sort_index()

        print(f"{'Time':<8} | {'Prone N':<10} | {'Supine N':<10}")
        print("-" * 35)
        for hh in half_hour_bins:
            time_str = format_clock_time(hh)
            print(f"{time_str:<8} | {int(counts_prone.loc[hh]):<10} | {int(counts_supine.loc[hh]):<10}")

        # --- ROSE PLOT (Histogram) ---
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        theta_centers = []
        widths = []
        radii_prone_plot = []
        radii_supine_plot = []

        for hh in half_hour_bins:
            if hh >= 12:
                deg = (hh - 12) * 30
            else:
                deg = hh * 30
            theta_centers.append(np.radians(deg))
            widths.append(np.radians(15))
            radii_prone_plot.append(counts_prone.loc[hh])
            radii_supine_plot.append(counts_supine.loc[hh])

        ax.bar(theta_centers, radii_prone_plot, width=np.radians(15), bottom=0.0,
               color='blue', alpha=0.3, edgecolor='blue', linewidth=1, label='Prone')
        ax.bar(theta_centers, radii_supine_plot, width=np.radians(12), bottom=0.0,
               color='red', alpha=0.5, edgecolor='red', linewidth=1, label='Supine')

        ax.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax.set_title(f'{breast_side} Breast: Clock Position Frequency (Half-Hour Bins)',
                    pad=20, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=11)

        filename = f"clock_frequency_{breast_side.lower()}_breast_half_hour.png"
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved frequency plot: {save_path}")
        plt.show()
        plt.close()

    # --- PART 2: STATISTICAL SUMMARY ---
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY (Rotation & Shift)")
    print("-"*80)

    for df_subset, breast_side in [(df_left, 'Left'), (df_right, 'Right')]:
        if len(df_subset) == 0:
            print(f"\n{breast_side} Breast: No data available")
            continue

        print(f"\n{breast_side} Breast (n={len(df_subset)}):")
        print("-"*40)

        mean_rotation_deg = df_subset['angle_rotation'].mean()
        std_rotation_deg = df_subset['angle_rotation'].std()
        mean_rotation_hours = df_subset['clock_rotation'].mean()

        print(f"Mean rotation: {mean_rotation_hours:.2f} hours ({mean_rotation_deg:.1f}°)")
        print(f"Std rotation: {std_rotation_deg:.1f}°")
        print(f"Median rotation: {df_subset['angle_rotation'].median():.1f}°")
        print(f"Range: [{df_subset['angle_rotation'].min():.1f}°, {df_subset['angle_rotation'].max():.1f}°]")

        t_stat, p_val = stats.ttest_1samp(df_subset['angle_rotation'], 0)
        print(f"\nTest if rotation ≠ 0: t={t_stat:.3f}, p={p_val:.4e}")

        if p_val < 0.05:
            direction = "clockwise" if mean_rotation_deg > 0 else "counterclockwise"
            print(f"✓ Significant {direction} rotation detected!")
        else:
            print("✗ No significant rotation detected")

        mean_dist_change = df_subset['distance_change'].mean()
        print(f"\nMean distance change: {mean_dist_change:.2f} mm")
        print(f"Mean prone distance: {df_subset['distance_prone'].mean():.2f} mm")
        print(f"Mean supine distance: {df_subset['distance_supine'].mean():.2f} mm")

        t_stat_dist, p_val_dist = stats.ttest_rel(df_subset['distance_prone'],
                                                   df_subset['distance_supine'])
        print(f"Test if distance changed: t={t_stat_dist:.3f}, p={p_val_dist:.4e}")

    # --- PART 3: POLAR PLOTS WITH PROPER ARROWS ---
    print("\n" + "-"*80)
    print("GENERATING POLAR PLOTS WITH PROPER ARROW DIRECTION")
    print("-"*80)

    for df_subset, breast_side in [(df_left, 'Left'), (df_right, 'Right')]:
        if len(df_subset) == 0:
            continue

        fig = plt.figure(figsize=(14, 6))

        # Plot 1: Individual trajectories
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)

        theta_prone = np.radians(df_subset['angle_prone'].values)
        theta_supine = np.radians(df_subset['angle_supine'].values)
        r_prone = df_subset['distance_prone'].values
        r_supine = df_subset['distance_supine'].values

        # Plot individual vectors using annotate arrows
        for i in range(len(df_subset)):
            draw_polar_arrow_annotate(ax1, theta_prone[i], r_prone[i],
                                      theta_supine[i], r_supine[i],
                                      color='black', alpha=0.6, linewidth=0.8)

        ax1.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax1.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax1.set_ylabel('Distance from Nipple (mm)', labelpad=30)
        ax1.set_title(f'{breast_side} Breast: Prone→Supine Shift\n(Individual Landmarks)',
                     fontsize=12, pad=20)

        legend_elements_ax1 = [
            Line2D([0], [0], color='black', lw=1, alpha=0.6,
                   marker='>', markersize=5, label='Individual vectors (Prone→Supine)')
        ]
        ax1.legend(handles=legend_elements_ax1, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax1.grid(True, alpha=0.3)

        # Plot 2: Mean shift with confidence
        ax2 = fig.add_subplot(122, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)

        # Calculate mean positions using circular mean for angles
        theta_prone_rad = np.radians(df_subset['angle_prone'].values)
        theta_supine_rad = np.radians(df_subset['angle_supine'].values)
        mean_theta_prone = circular_mean_angle(theta_prone_rad)
        mean_theta_supine = circular_mean_angle(theta_supine_rad)
        mean_r_prone = df_subset['distance_prone'].mean()
        mean_r_supine = df_subset['distance_supine'].mean()

        # Plot individual vectors (lighter, in background)
        for i in range(len(df_subset)):
            draw_polar_arrow_annotate(ax2, theta_prone[i], r_prone[i],
                                      theta_supine[i], r_supine[i],
                                      color='black', alpha=0.3, linewidth=0.6)

        # Plot mean vector (thicker, more prominent)
        draw_polar_arrow_annotate(ax2, mean_theta_prone, mean_r_prone,
                                  mean_theta_supine, mean_r_supine,
                                  color='blue', alpha=1.0, is_mean=True)

        ax2.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax2.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax2.set_ylabel('Distance from Nipple (mm)', labelpad=30)

        mean_rotation = df_subset['angle_rotation'].mean()
        mean_rotation_hours = df_subset['clock_rotation'].mean()
        direction = "Clockwise" if mean_rotation > 0 else "Counterclockwise"

        title_text = f'{breast_side} Breast: Mean Rotation\n'
        title_text += f'{direction}: {abs(mean_rotation_hours):.2f} hours ({abs(mean_rotation):.1f}°)'
        ax2.set_title(title_text, fontsize=12, pad=20)

        legend_elements_ax2 = [
            Line2D([0], [0], color='black', lw=1, alpha=0.3, marker='>', markersize=5, label='Individual vectors'),
            Line2D([0], [0], color='blue', lw=3, marker='>', markersize=8, label='Mean vector')
        ]
        ax2.legend(handles=legend_elements_ax2, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"clock_rotation_{breast_side.lower()}_breast.png"
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
        plt.close()

    # Create combined comparison plot
    if len(df_left) > 0 and len(df_right) > 0:
        print("\nGenerating combined polar plot with individual points...")
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6),
                                                subplot_kw=dict(projection='polar'))

        # LEFT SUBPLOT: RIGHT BREAST
        ax_left.set_theta_zero_location('N')
        ax_left.set_theta_direction(-1)

        theta_prone_r = np.radians(df_right['angle_prone'].values)
        theta_supine_r = np.radians(df_right['angle_supine'].values)
        r_prone_r = df_right['distance_prone'].values
        r_supine_r = df_right['distance_supine'].values

        # Plot individual vectors using annotate arrows
        for i in range(len(df_right)):
            draw_polar_arrow_annotate(ax_left, theta_prone_r[i], r_prone_r[i],
                                      theta_supine_r[i], r_supine_r[i],
                                      color='black', alpha=0.5, linewidth=0.8)

        # Mean trajectory
        mean_theta_prone_r = circular_mean_angle(theta_prone_r)
        mean_theta_supine_r = circular_mean_angle(theta_supine_r)
        mean_r_prone_r = df_right['distance_prone'].mean()
        mean_r_supine_r = df_right['distance_supine'].mean()

        draw_polar_arrow_annotate(ax_left, mean_theta_prone_r, mean_r_prone_r,
                                  mean_theta_supine_r, mean_r_supine_r,
                                  color='blue', alpha=1.0, is_mean=True)

        ax_left.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_left.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_left.set_ylabel('Distance from Nipple (mm)', labelpad=30, fontsize=11)

        mean_rotation_r = df_right['angle_rotation'].mean()
        mean_rotation_hours_r = df_right['clock_rotation'].mean()
        direction_r = "Clockwise" if mean_rotation_r > 0 else "Counterclockwise"

        ax_left.set_title(
            f'Landmarks in right breast\n{direction_r}: {abs(mean_rotation_hours_r):.2f}h ({abs(mean_rotation_r):.1f}°)',
            fontsize=12, color='black', pad=20)

        legend_elements = [
            Line2D([0], [0], color='black', lw=1, alpha=0.5, marker='>', markersize=5, label='Individual vectors'),
            Line2D([0], [0], color='blue', lw=3, marker='>', markersize=8, label='Mean vector')
        ]
        ax_left.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, ncol=1)
        ax_left.grid(True, alpha=0.3)

        # RIGHT SUBPLOT: LEFT BREAST
        ax_right.set_theta_zero_location('N')
        ax_right.set_theta_direction(-1)

        theta_prone_l = np.radians(df_left['angle_prone'].values)
        theta_supine_l = np.radians(df_left['angle_supine'].values)
        r_prone_l = df_left['distance_prone'].values
        r_supine_l = df_left['distance_supine'].values

        for i in range(len(df_left)):
            draw_polar_arrow_annotate(ax_right, theta_prone_l[i], r_prone_l[i],
                                      theta_supine_l[i], r_supine_l[i],
                                      color='black', alpha=0.5, linewidth=0.8)

        mean_theta_prone_l = circular_mean_angle(theta_prone_l)
        mean_theta_supine_l = circular_mean_angle(theta_supine_l)
        mean_r_prone_l = df_left['distance_prone'].mean()
        mean_r_supine_l = df_left['distance_supine'].mean()

        draw_polar_arrow_annotate(ax_right, mean_theta_prone_l, mean_r_prone_l,
                                  mean_theta_supine_l, mean_r_supine_l,
                                  color='blue', alpha=1.0, is_mean=True)

        ax_right.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_right.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_right.set_ylabel('Distance from Nipple (mm)', labelpad=30, fontsize=11)

        mean_rotation_l = df_left['angle_rotation'].mean()
        mean_rotation_hours_l = df_left['clock_rotation'].mean()
        direction_l = "Clockwise" if mean_rotation_l > 0 else "Counterclockwise"

        ax_right.set_title(
            f'Landmarks in left breast: \n{direction_l}: {abs(mean_rotation_hours_l):.2f}h ({abs(mean_rotation_l):.1f}°)',
            fontsize=12, pad=20)

        ax_right.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, ncol=1)
        ax_right.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_dir / "clock_rotation_comparison_individual.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot (individual): {save_path}")
        plt.show()
        plt.close()

        # Individual landmarks only plot (no mean overlay)
        print("Generating combined polar plot with individual landmarks only (no mean)...")
        fig_indiv, (ax_left_indiv, ax_right_indiv) = plt.subplots(1, 2, figsize=(14, 6),
                                                                  subplot_kw=dict(projection='polar'))

        # LEFT SUBPLOT: RIGHT BREAST
        ax_left_indiv.set_theta_zero_location('N')
        ax_left_indiv.set_theta_direction(-1)

        for i in range(len(df_right)):
            draw_polar_arrow_annotate(ax_left_indiv, theta_prone_r[i], r_prone_r[i],
                                      theta_supine_r[i], r_supine_r[i],
                                      color='black', alpha=0.6, linewidth=0.8)

        ax_left_indiv.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_left_indiv.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_left_indiv.set_ylabel('Distance from Nipple (mm)', labelpad=30)
        ax_left_indiv.set_title(f'Right Breast: Prone->Supine Shift\n(n={len(df_right)} landmarks)',
                                fontsize=12, pad=20)

        legend_elements_indiv = [
            Line2D([0], [0], color='black', lw=1, alpha=0.6, marker='>', markersize=5,
                   label='Individual vectors (Prone→Supine)')
        ]
        ax_left_indiv.legend(handles=legend_elements_indiv, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_left_indiv.grid(True, alpha=0.3)

        # RIGHT SUBPLOT: LEFT BREAST
        ax_right_indiv.set_theta_zero_location('N')
        ax_right_indiv.set_theta_direction(-1)

        for i in range(len(df_left)):
            draw_polar_arrow_annotate(ax_right_indiv, theta_prone_l[i], r_prone_l[i],
                                      theta_supine_l[i], r_supine_l[i],
                                      color='black', alpha=0.6, linewidth=0.8)

        ax_right_indiv.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_right_indiv.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_right_indiv.set_ylabel('Distance from Nipple (mm)', labelpad=30)
        ax_right_indiv.set_title(f'Left Breast: Prone->Supine Shift\n(n={len(df_left)} landmarks)',
                                 fontsize=12, pad=20)
        ax_right_indiv.legend(handles=legend_elements_indiv, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_right_indiv.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path_indiv = save_dir / "clock_rotation_individual_landmarks_only.png"
        plt.savefig(save_path_indiv, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot (individual landmarks only): {save_path_indiv}")
        plt.show()
        plt.close()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Return summary statistics
    summary = {}
    if len(df_left) > 0:
        summary['left'] = {
            'n': len(df_left),
            'mean_rotation_deg': df_left['angle_rotation'].mean(),
            'mean_rotation_hours': df_left['clock_rotation'].mean(),
            'std_rotation_deg': df_left['angle_rotation'].std(),
            'p_value': stats.ttest_1samp(df_left['angle_rotation'], 0)[1],
            'mean_distance_change': df_left['distance_change'].mean()
        }

    if len(df_right) > 0:
        summary['right'] = {
            'n': len(df_right),
            'mean_rotation_deg': df_right['angle_rotation'].mean(),
            'mean_rotation_hours': df_right['clock_rotation'].mean(),
            'std_rotation_deg': df_right['angle_rotation'].std(),
            'p_value': stats.ttest_1samp(df_right['angle_rotation'], 0)[1],
            'mean_distance_change': df_right['distance_change'].mean()
        }

    return summary


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

def read_data(excel_path):
    """Read data from Excel file."""
    try:
        all_sheets = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl', header=0)
        df_raw = all_sheets['raw_data']
        df_ave = all_sheets['processed_ave_data']
        df_demo = all_sheets['demographic']
        print(f"Successfully loaded {len(all_sheets)} sheets.")
        return df_raw, df_ave, df_demo
    except FileNotFoundError:
        print(f"Error: The file {excel_path} was not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None


def get_nipple_relative_points_and_vectors(df, is_left_breast=True):
    """
    Get landmark positions relative to nipple and displacement vectors.

    Args:
        df: DataFrame with landmark data (filtered for one breast side)
        is_left_breast: True for left breast, False for right breast

    Returns:
        base_points: Nx3 array of prone landmark positions relative to nipple
        vectors: Nx3 array of displacement vectors
        dts_values: Array of distance to skin values
    """
    df = df.dropna(subset=['landmark ave prone transformed x',
                           'landmark ave supine x']).copy()

    if len(df) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([])

    # Get nipple columns based on breast side
    if is_left_breast:
        nipple_prone_x = 'left nipple prone transformed x'
        nipple_prone_y = 'left nipple prone transformed y'
        nipple_prone_z = 'left nipple prone transformed z'
        nipple_supine_x = 'left nipple supine x'
        nipple_supine_y = 'left nipple supine y'
        nipple_supine_z = 'left nipple supine z'
    else:
        nipple_prone_x = 'right nipple prone transformed x'
        nipple_prone_y = 'right nipple prone transformed y'
        nipple_prone_z = 'right nipple prone transformed z'
        nipple_supine_x = 'right nipple supine x'
        nipple_supine_y = 'right nipple supine y'
        nipple_supine_z = 'right nipple supine z'

    # Calculate landmark positions relative to nipple
    # Prone position relative to nipple
    prone_x = df['landmark ave prone transformed x'].values - df[nipple_prone_x].values
    prone_y = df['landmark ave prone transformed y'].values - df[nipple_prone_y].values
    prone_z = df['landmark ave prone transformed z'].values - df[nipple_prone_z].values

    # Supine position relative to nipple
    supine_x = df['landmark ave supine x'].values - df[nipple_supine_x].values
    supine_y = df['landmark ave supine y'].values - df[nipple_supine_y].values
    supine_z = df['landmark ave supine z'].values - df[nipple_supine_z].values

    # Base points (prone position relative to nipple)
    base_points = np.column_stack([prone_x, prone_y, prone_z])

    # Vectors (displacement from prone to supine, relative to nipple)
    vectors = np.column_stack([supine_x - prone_x, supine_y - prone_y, supine_z - prone_z])

    # DTS values
    dts_values = df['Distance to skin (prone) [mm]'].values

    return base_points, vectors, dts_values


if __name__ == "__main__":
    print("="*80)
    print("POLAR PLOT ANALYSIS - CLOCK POSITION ROTATION")
    print("="*80)

    # Define paths
    OUTPUT_DIR = Path("../output")
    EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v6_2026_02_10.xlsx"

    # Read data
    print(f"\nReading data from: {EXCEL_FILE_PATH}")
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)

    if df_ave is None:
        print("Error: Could not load data. Exiting.")
        exit(1)

    # Filter for landmarks with alignment data
    df_ave_filtered = df_ave.dropna(subset=['landmark ave prone transformed x',
                                             'landmark ave supine x']).copy()
    print(f"Landmarks with alignment data: {len(df_ave_filtered)}")

    # Separate left and right breasts
    df_left = df_ave_filtered[df_ave_filtered['landmark side (prone)'] == 'LB'].copy()
    df_right = df_ave_filtered[df_ave_filtered['landmark side (prone)'] == 'RB'].copy()

    print(f"Left breast landmarks: {len(df_left)}")
    print(f"Right breast landmarks: {len(df_right)}")

    # Get nipple-relative positions and vectors
    base_left, vec_left, dts_left = get_nipple_relative_points_and_vectors(df_left, is_left_breast=True)
    base_right, vec_right, dts_right = get_nipple_relative_points_and_vectors(df_right, is_left_breast=False)

    print(f"\nBase points shape - Left: {base_left.shape}, Right: {base_right.shape}")
    print(f"Vectors shape - Left: {vec_left.shape}, Right: {vec_right.shape}")

    # Run the analysis
    summary = analyse_clock_position_rotation(
        df_ave_filtered,
        base_left=base_left,
        base_right=base_right,
        vec_left=vec_left,
        vec_right=vec_right
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if 'left' in summary:
        s = summary['left']
        print(f"\nLeft Breast (n={s['n']}):")
        print(f"  Mean rotation: {s['mean_rotation_hours']:.2f} hours ({s['mean_rotation_deg']:.1f}°)")
        print(f"  Std rotation: {s['std_rotation_deg']:.1f}°")
        print(f"  P-value: {s['p_value']:.4e}")
        print(f"  Mean distance change: {s['mean_distance_change']:.2f} mm")

    if 'right' in summary:
        s = summary['right']
        print(f"\nRight Breast (n={s['n']}):")
        print(f"  Mean rotation: {s['mean_rotation_hours']:.2f} hours ({s['mean_rotation_deg']:.1f}°)")
        print(f"  Std rotation: {s['std_rotation_deg']:.1f}°")
        print(f"  P-value: {s['p_value']:.4e}")
        print(f"  Mean distance change: {s['mean_distance_change']:.2f} mm")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)

