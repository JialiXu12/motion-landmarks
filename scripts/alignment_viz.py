"""
Alignment Visualization

All visualization functions for alignment results, iteration debugging,
convergence plots, and correspondence diagnostics.

Collected from alignment.py and surface_to_point_alignment.py in Stage 3
of the refactoring plan.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Dict


# ---------------------------------------------------------------------------
# 1. Alignment error visualization (from alignment.py)
# ---------------------------------------------------------------------------
def visualize_alignment_errors(
        source_mesh_coords: np.ndarray,
        target_pc: np.ndarray,
        source_aligned: np.ndarray = None,
        error_magnitudes: np.ndarray = None,
        error_indices: np.ndarray = None,
        source_sternum: np.ndarray = None,
        target_sternum: np.ndarray = None,
        selected_elements_coords: np.ndarray = None,
        iteration: int = None,
        title: str = "Alignment Visualization",
        cmap: str = "coolwarm",
        point_size: int = 4,
        show_error_arrows: bool = True,
        worst_n_arrows: int = 50,
        show_sternum: bool = True,
        show_legend: bool = True,
        screenshot_path: str = None
) -> None:
    """
    Visualize mesh and point cloud with alignment errors during alignment process.

    Colors the source (prone) mesh by per-source-point distance to the nearest
    target (supine) point (prone to supine query direction).

    Args:
        source_mesh_coords: (N, 3) source mesh coordinates (prone ribcage)
        target_pc: (M, 3) target point cloud (supine ribcage)
        source_aligned: (N, 3) aligned source coordinates (if None, uses source_mesh_coords)
        error_magnitudes: (N,) per-source-point distance to nearest target point
        error_indices: (N,) index into target_pc of nearest point for each source point
        source_sternum: (3,) source sternum superior position
        target_sternum: (3,) target sternum superior position
        selected_elements_coords: (K, 3) coordinates of selected elements only (subset for alignment)
        iteration: iteration number (for title display)
        title: plot title
        cmap: colormap for error visualization
        point_size: size of point cloud points
        show_error_arrows: whether to show arrows for worst errors
        worst_n_arrows: number of worst error arrows to show
        show_sternum: whether to show sternum positions
        show_legend: whether to show legend
        screenshot_path: if provided, save screenshot to this path
    """
    import pyvista as pv

    plotter = pv.Plotter()
    plotter.set_background('white')

    # Update title with iteration if provided
    if iteration is not None:
        title = f"{title} (Iteration {iteration})"
    plotter.add_text(title, font_size=14, position='upper_left')

    # Use aligned source if provided, otherwise use original
    display_source = source_aligned if source_aligned is not None else source_mesh_coords

    # --- Target Point Cloud (plain overlay) ---
    plotter.add_points(
        target_pc,
        color='blue',
        point_size=max(1, point_size - 2),
        render_points_as_spheres=True,
        opacity=0.3,
        label='Target (Supine)'
    )

    # --- Source Mesh colored by error (prone to supine) ---
    if error_magnitudes is not None:
        source_cloud = pv.PolyData(display_source)
        source_cloud['Alignment Error (mm)'] = error_magnitudes
        plotter.add_points(
            source_cloud,
            scalars='Alignment Error (mm)',
            cmap=cmap,
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Error (mm)',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1,
                'width': 0.1,
                'height': 0.6
            }
        )
    elif selected_elements_coords is not None:
        # Full mesh in gray (not used for alignment)
        plotter.add_points(
            display_source,
            color='lightgray',
            point_size=max(1, point_size - 2),
            render_points_as_spheres=True,
            opacity=0.3,
            label='Full Mesh (not used)'
        )
        # Selected elements in red (used for alignment)
        plotter.add_points(
            selected_elements_coords,
            color='red',
            point_size=point_size,
            render_points_as_spheres=True,
            label='Selected Elements (used)'
        )
    else:
        # All mesh in red
        plotter.add_points(
            display_source,
            color='red',
            point_size=max(2, point_size - 1),
            render_points_as_spheres=True,
            label='Source Mesh (Prone)'
        )

    # --- Error Arrows (worst errors) ---
    if show_error_arrows and error_magnitudes is not None and error_indices is not None:
        # Find worst N source points by error magnitude
        worst_n = min(worst_n_arrows, len(error_magnitudes))
        worst_src_indices = np.argsort(error_magnitudes)[-worst_n:]

        # Draw arrows from worst source points to their nearest target points
        for src_idx in worst_src_indices:
            start = display_source[src_idx]
            end = target_pc[error_indices[src_idx]]
            direction = end - start

            error = error_magnitudes[src_idx]
            plotter.add_arrows(
                cent=start.reshape(1, 3),
                direction=direction.reshape(1, 3),
                mag=1.0,
                color='orange' if error > 10 else 'yellow',
            )

    # --- Sternum Markers ---
    if show_sternum:
        if source_sternum is not None:
            plotter.add_points(
                source_sternum.reshape(1, 3),
                color='green',
                point_size=15,
                render_points_as_spheres=True,
                label='Source Sternum'
            )
        if target_sternum is not None:
            plotter.add_points(
                target_sternum.reshape(1, 3),
                color='purple',
                point_size=15,
                render_points_as_spheres=True,
                label='Target Sternum'
            )

    # --- Add Legend ---
    if show_legend:
        legend_entries = []
        if selected_elements_coords is not None:
            legend_entries.append(['Selected Elements', 'red'])
            legend_entries.append(['Full Mesh', 'lightgray'])
        else:
            legend_entries.append(['Source Mesh', 'red'])

        if error_magnitudes is None:
            legend_entries.append(['Target PC', 'blue'])

        if show_sternum:
            if source_sternum is not None:
                legend_entries.append(['Source Sternum', 'green'])
            if target_sternum is not None:
                legend_entries.append(['Target Sternum', 'purple'])

        if legend_entries:
            plotter.add_legend(legend_entries, bcolor='white')

    # --- Statistics Text Box ---
    if error_magnitudes is not None:
        stats_text = (
            f"Error Statistics:\n"
            f"  Mean: {np.mean(error_magnitudes):.2f} mm\n"
            f"  Std: {np.std(error_magnitudes):.2f} mm\n"
            f"  RMSE: {np.sqrt(np.mean(error_magnitudes**2)):.2f} mm\n"
            f"  Min: {np.min(error_magnitudes):.2f} mm\n"
            f"  Max: {np.max(error_magnitudes):.2f} mm"
        )
        plotter.add_text(stats_text, font_size=10, position='lower_left')

    plotter.add_axes()

    # Save screenshot if path provided
    if screenshot_path:
        plotter.show(screenshot=screenshot_path, auto_close=True)
    else:
        plotter.show()


# ---------------------------------------------------------------------------
# 2. ICP iteration visualization (from alignment.py)
# ---------------------------------------------------------------------------
def visualize_alignment_during_iteration(
        source_centered: np.ndarray,
        target_centered: np.ndarray,
        iteration: int,
        correspondences: np.ndarray = None,
        correspondence_distances: np.ndarray = None,
        valid_mask: np.ndarray = None,
        max_correspondence_distance: float = 15.0,
        show_correspondences: bool = True,
        n_correspondence_lines: int = 100,
        show_unused_points: bool = True
) -> None:
    """
    Visualize alignment state during an ICP iteration (sternum-centered coordinates).

    Shows the ENTIRE mesh/point cloud with UNUSED points in muted colors
    and USED points (after trimming and max correspondence distance filtering)
    highlighted in bright colors.

    Args:
        source_centered: (N, 3) source points centered on sternum (origin)
        target_centered: (M, 3) target points centered on sternum (origin)
        iteration: current iteration number
        correspondences: (N,) indices of corresponding target points for each source
        correspondence_distances: (N,) distances to corresponding points
        valid_mask: (N,) boolean mask of valid correspondences (after trimming)
        max_correspondence_distance: maximum distance for valid correspondences
        show_correspondences: whether to draw lines between correspondences
        n_correspondence_lines: number of correspondence lines to draw
        show_unused_points: if True, show whole point cloud with unused in muted colors (default: True)
    """
    import pyvista as pv

    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_text(f"ICP Iteration {iteration}", font_size=14, position='upper_left')

    # Get valid indices
    if valid_mask is not None:
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]
    else:
        valid_indices = np.arange(len(source_centered))
        invalid_indices = np.array([], dtype=int)

    # --- SOURCE POINTS (Prone mesh) ---
    # FIRST: Show ALL source points in muted color (whole mesh/point cloud)
    if show_unused_points and len(invalid_indices) > 0:
        invalid_source = source_centered[invalid_indices]
        plotter.add_points(
            invalid_source,
            color='lightgray',
            point_size=3,
            render_points_as_spheres=True,
            opacity=0.4,
            label='Source (not used)'
        )

    # SECOND: Highlight USED source points in bright color
    if len(valid_indices) > 0:
        valid_source = source_centered[valid_indices]

        if correspondence_distances is not None:
            # Color valid source points by correspondence distance
            valid_distances = correspondence_distances[valid_indices]
            source_cloud = pv.PolyData(valid_source)
            source_cloud['Distance (mm)'] = valid_distances

            plotter.add_points(
                source_cloud,
                scalars='Distance (mm)',
                cmap='plasma',  # More visible colormap
                point_size=6,
                render_points_as_spheres=True,
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Correspondence\nDistance (mm)'}
            )
        else:
            plotter.add_points(
                valid_source,
                color='red',
                point_size=6,
                render_points_as_spheres=True,
                label='Source (used for alignment)'
            )

    # --- TARGET POINTS (Supine point cloud) ---
    if correspondences is not None and valid_mask is not None and len(valid_indices) > 0:
        # Get target points that have valid correspondences
        valid_correspondences = correspondences[valid_indices]
        unique_target_indices = np.unique(valid_correspondences)

        # FIRST: Show ALL target points in muted color (whole point cloud)
        if show_unused_points:
            all_target_indices = np.arange(len(target_centered))
            unused_target_indices = np.setdiff1d(all_target_indices, unique_target_indices)
            if len(unused_target_indices) > 0:
                unused_target = target_centered[unused_target_indices]
                plotter.add_points(
                    unused_target,
                    color='lightblue',
                    point_size=3,
                    render_points_as_spheres=True,
                    opacity=0.3,
                    label='Target (not used)'
                )

        # SECOND: Highlight USED target points in bright color
        used_target = target_centered[unique_target_indices]
        plotter.add_points(
            used_target,
            color='blue',
            point_size=5,
            render_points_as_spheres=True,
            label='Target (used for alignment)'
        )
    else:
        # No valid mask - show all target points
        plotter.add_points(
            target_centered,
            color='blue',
            point_size=4,
            render_points_as_spheres=True,
            label='Target (Supine)'
        )

    # --- Origin Marker (Sternum) ---
    plotter.add_points(
        np.array([[0, 0, 0]]),
        color='green',
        point_size=20,
        render_points_as_spheres=True,
        label='Origin (Sternum)'
    )

    # --- Correspondence Lines (only for used points) ---
    if show_correspondences and correspondences is not None and len(valid_indices) > 0:
        # Sample from valid correspondences only
        if len(valid_indices) > n_correspondence_lines:
            # Sample evenly from valid indices
            sample_idx = valid_indices[::len(valid_indices)//n_correspondence_lines][:n_correspondence_lines]
        else:
            sample_idx = valid_indices

        # Draw lines for sampled correspondences
        lines = []
        for src_idx in sample_idx:
            tgt_idx = correspondences[src_idx]
            start = source_centered[src_idx]
            end = target_centered[tgt_idx]
            lines.append([start, end])

        if lines:
            for line in lines:
                plotter.add_lines(
                    np.array(line),
                    color='yellow',
                    width=1
                )

    # --- Statistics (for used points only) ---
    n_used_source = len(valid_indices)
    n_total_source = len(source_centered)
    n_used_target = len(np.unique(correspondences[valid_indices])) if correspondences is not None and len(valid_indices) > 0 else 0
    n_total_target = len(target_centered)

    if correspondence_distances is not None and len(valid_indices) > 0:
        valid_dists = correspondence_distances[valid_indices]
        stats_text = (
            f"Iteration {iteration} Statistics:\n"
            f"  Source (bright): {n_used_source}/{n_total_source} ({100*n_used_source/n_total_source:.1f}%)\n"
            f"  Target (blue): {n_used_target}/{n_total_target} ({100*n_used_target/n_total_target:.1f}%)\n"
            f"  Mean distance: {np.mean(valid_dists):.2f} mm\n"
            f"  RMSE: {np.sqrt(np.mean(valid_dists**2)):.2f} mm\n"
            f"  Max distance: {np.max(valid_dists):.2f} mm\n"
            f"  Gray/Light = not used for alignment"
        )
        plotter.add_text(stats_text, font_size=10, position='lower_left')

    plotter.add_legend(bcolor='white')
    plotter.add_axes()
    plotter.show()


# ---------------------------------------------------------------------------
# 3. Convergence diagram (from surface_to_point_alignment.py)
# ---------------------------------------------------------------------------
def plot_convergence_diagram(info: Dict, save_path: str = None):
    """
    Plot optimization convergence diagram showing RMSE and rotation angles.

    Args:
        info: dict returned by surface_to_point_align containing iteration_history
        save_path: optional path to save the figure (can be relative or absolute)
    """
    import matplotlib
    import matplotlib.pyplot as plt


    print(f"  Plotting convergence diagram (backend: {matplotlib.get_backend()})")

    history = info.get('iteration_history', [])
    if not history:
        print("  No iteration history to plot")
        return

    print(f"  Found {len(history)} iterations to plot")

    iterations = [h['iteration'] for h in history]
    rmse_values = [h['rmse'] for h in history]
    angle_x_values = [h.get('angle_x_deg', 0) for h in history]
    angle_y_values = [h.get('angle_y_deg', 0) for h in history]
    angle_z_values = [h.get('angle_z_deg', 0) for h in history]
    total_angle_values = [h.get('total_angle_deg', 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Surface-to-Point ICP Convergence', fontsize=14, fontweight='bold')

    # Plot 1: RMSE convergence
    ax1 = axes[0, 0]
    ax1.plot(iterations, rmse_values, 'b-o', linewidth=2, markersize=4, label='RMSE')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('RMSE (mm)', fontsize=11)
    ax1.set_title('RMSE Convergence', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(iterations) + 1)

    # Add convergence annotation
    if len(rmse_values) > 1:
        initial_rmse = rmse_values[0]
        final_rmse = rmse_values[-1]
        reduction = (initial_rmse - final_rmse) / initial_rmse * 100 if initial_rmse > 0 else 0
        ax1.annotate(f'Initial: {initial_rmse:.4f} mm\nFinal: {final_rmse:.4f} mm\nReduction: {reduction:.1f}%',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Total rotation angle
    ax2 = axes[0, 1]
    ax2.plot(iterations, total_angle_values, 'r-s', linewidth=2, markersize=4, label='Total Angle')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Total Rotation (deg)', fontsize=11)
    ax2.set_title('Total Rotation Angle Convergence', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(iterations) + 1)

    # Add final angle annotation
    if total_angle_values:
        ax2.annotate(f'Final: {total_angle_values[-1]:.3f} deg',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Plot 3: Individual Euler angles (X, Y, Z)
    ax3 = axes[1, 0]
    ax3.plot(iterations, angle_x_values, 'g-^', linewidth=1.5, markersize=3, label='Angle X')
    ax3.plot(iterations, angle_y_values, 'b-v', linewidth=1.5, markersize=3, label='Angle Y')
    ax3.plot(iterations, angle_z_values, 'r-o', linewidth=1.5, markersize=3, label='Angle Z')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Angle (deg)', fontsize=11)
    ax3.set_title('Euler Angles (X, Y, Z)', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(iterations) + 1)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add final angles annotation
    if angle_x_values and angle_y_values and angle_z_values:
        ax3.annotate(f'Final:\n  X: {angle_x_values[-1]:+.3f} deg\n  Y: {angle_y_values[-1]:+.3f} deg\n  Z: {angle_z_values[-1]:+.3f} deg',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 4: Rotation change per iteration (convergence rate)
    ax4 = axes[1, 1]
    rotation_changes = [h.get('rotation_change', 0) for h in history]
    ax4.semilogy(iterations, rotation_changes, 'm-d', linewidth=2, markersize=4, label='Rotation Change')
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('||R_delta - I|| (log scale)', fontsize=11)
    ax4.set_title('Rotation Update Magnitude (Convergence Rate)', fontsize=12)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0, max(iterations) + 1)

    # Add convergence threshold line
    ax4.axhline(y=1e-4, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Convergence threshold')
    ax4.legend(loc='best', fontsize=9)

    plt.tight_layout()

    # Save the figure
    if save_path:
        try:
            save_path = Path(save_path)
            if not save_path.is_absolute():
                # Assume relative to script directory
                script_dir = Path(__file__).parent
                save_path = script_dir / save_path

            # Ensure the directory exists
            save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving to: {save_path}")

            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')

            # Verify the file was actually saved
            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f"  Convergence diagram saved: {save_path} ({file_size:,} bytes)")
            else:
                print(f"  WARNING: File was not created at {save_path}")
        except Exception as e:
            print(f"  ERROR saving convergence diagram: {e}")
            import traceback
            traceback.print_exc()

    # Display the figure
    try:
        plt.show(block=False)  # Non-blocking show
        plt.pause(1.0)  # Display for 1 second
        print("  Convergence diagram displayed")
    except Exception as e:
        print(f"  Warning: Could not display figure interactively: {e}")
    finally:
        plt.close(fig)  # Close the figure to free memory


# ---------------------------------------------------------------------------
# 4. Many-to-one correspondence diagnostic (from surface_to_point_alignment.py)
# ---------------------------------------------------------------------------
def _plot_many_to_one(
        src_centered: np.ndarray,
        tgt_centered: np.ndarray,
        src_idx: np.ndarray,
        tgt_idx: np.ndarray,
        counts: np.ndarray,
        unique_targets: np.ndarray,
) -> None:
    """
    Visualise many-to-one correspondences using PyVista.

    Shows all source and target points involved in correspondences, with
    many-to-one pairs highlighted in red and connected by lines.

    Args:
        src_centered: (N, 3) full source points (sternum-centred)
        tgt_centered: (M, 3) full target points (sternum-centred)
        src_idx: (K,) indices of matched source points
        tgt_idx: (K,) indices of matched target points
        counts: per-unique-target match counts from np.unique
        unique_targets: unique target indices from np.unique
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  WARNING: pyvista not available, skipping many-to-one plot")
        return

    # Identify many-to-one correspondences (target matched by >1 source)
    shared_mask = counts > 1
    shared_tgt_set = set(unique_targets[shared_mask])

    # Separate 1-to-1 and many-to-one correspondence indices
    is_many = np.array([t in shared_tgt_set for t in tgt_idx])
    m2o_src_idx = src_idx[is_many]
    m2o_tgt_idx = tgt_idx[is_many]
    one2one_src_idx = src_idx[~is_many]
    one2one_tgt_idx = tgt_idx[~is_many]

    plotter = pv.Plotter()
    plotter.set_background('white')

    # All source points (small, grey)
    plotter.add_points(
        pv.PolyData(src_centered[src_idx]),
        color='grey', point_size=3,
        render_points_as_spheres=True, label='Source (1-to-1)',
    )

    # All target points (small, blue)
    plotter.add_points(
        pv.PolyData(tgt_centered[one2one_tgt_idx]),
        color='steelblue', point_size=3,
        render_points_as_spheres=True, label='Target (1-to-1)',
    )

    # Many-to-one source points (red)
    if len(m2o_src_idx) > 0:
        plotter.add_points(
            pv.PolyData(src_centered[m2o_src_idx]),
            color='red', point_size=8,
            render_points_as_spheres=True, label='Source (many-to-1)',
        )

        # Shared target points (orange, larger)
        shared_tgt_unique = np.array(list(shared_tgt_set))
        plotter.add_points(
            pv.PolyData(tgt_centered[shared_tgt_unique]),
            color='orange', point_size=10,
            render_points_as_spheres=True, label='Target (shared)',
        )

        # Draw lines from each many-to-one source to its target
        lines_pts = []
        lines_conn = []
        for i in range(len(m2o_src_idx)):
            base = len(lines_pts)
            lines_pts.append(src_centered[m2o_src_idx[i]])
            lines_pts.append(tgt_centered[m2o_tgt_idx[i]])
            lines_conn.append([2, base, base + 1])

        if lines_pts:
            lines_mesh = pv.PolyData(
                np.array(lines_pts),
                lines=np.hstack(lines_conn),
            )
            plotter.add_mesh(
                lines_mesh, color='red', line_width=1.5,
                opacity=0.6, label='Many-to-1 links',
            )

    # Origin marker (sternum)
    plotter.add_points(
        pv.PolyData(np.zeros((1, 3))),
        color='green', point_size=15,
        render_points_as_spheres=True, label='Sternum (origin)',
    )

    plotter.add_legend(face=None)
    plotter.add_axes()
    plotter.add_title('Many-to-One Correspondence Diagnostic (Iter 1)')
    plotter.show()


# ---------------------------------------------------------------------------
# 5. Alignment result visualization (from surface_to_point_alignment.py)
# ---------------------------------------------------------------------------
def plot_alignment_result(
        aligned_mesh_pts: np.ndarray,
        target_pts: np.ndarray,
        sternum_pos: np.ndarray = None,
        title: str = "Plane-to-Point ICP Alignment Result",
        save_path: str = None,
        show_correspondences: bool = False,
        max_distance: float = 15.0,
) -> None:
    """
    Visualise the aligned mesh overlaid on the target point cloud.

    Shows only the regions used for alignment (filtered mesh elements
    and filtered target point cloud).

    Args:
        aligned_mesh_pts: (N, 3) aligned source mesh points (in target frame)
        target_pts: (M, 3) target point cloud (filtered to alignment region)
        sternum_pos: (3,) sternum position to mark (optional)
        title: plot title
        save_path: if provided, save screenshot to this path
        show_correspondences: if True, draw lines between closest pairs
        max_distance: max distance for correspondence lines (mm)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  WARNING: pyvista not available, skipping alignment plot")
        return

    plotter = pv.Plotter()
    plotter.set_background('white')

    # Target point cloud (supine) - blue
    plotter.add_points(
        pv.PolyData(target_pts),
        color='steelblue',
        point_size=4,
        render_points_as_spheres=True,
        opacity=0.6,
        label=f'Target PC ({len(target_pts)} pts)',
    )

    # Aligned source mesh (prone) - red
    plotter.add_points(
        pv.PolyData(aligned_mesh_pts),
        color='red',
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.9,
        label=f'Aligned Mesh ({len(aligned_mesh_pts)} pts)',
    )

    # Draw correspondence lines if requested
    if show_correspondences:
        tree = cKDTree(target_pts)
        dists, indices = tree.query(aligned_mesh_pts, k=1)

        # Only show correspondences within max_distance
        valid_mask = dists <= max_distance
        valid_src = aligned_mesh_pts[valid_mask]
        valid_tgt = target_pts[indices[valid_mask]]

        if len(valid_src) > 0:
            # Subsample for cleaner visualization (max 500 lines)
            n_lines = min(500, len(valid_src))
            step = max(1, len(valid_src) // n_lines)

            lines_pts = []
            lines_conn = []
            for i in range(0, len(valid_src), step):
                base = len(lines_pts)
                lines_pts.append(valid_src[i])
                lines_pts.append(valid_tgt[i])
                lines_conn.append([2, base, base + 1])

            if lines_pts:
                lines_mesh = pv.PolyData(
                    np.array(lines_pts),
                    lines=np.hstack(lines_conn),
                )
                plotter.add_mesh(
                    lines_mesh, color='yellow', line_width=1.0,
                    opacity=0.4, label='Correspondences',
                )

    # Sternum marker (if provided)
    if sternum_pos is not None:
        plotter.add_points(
            pv.PolyData(sternum_pos.reshape(1, 3)),
            color='green',
            point_size=15,
            render_points_as_spheres=True,
            label='Sternum Superior',
        )

    # Calculate and display alignment statistics
    tree = cKDTree(target_pts)
    dists, _ = tree.query(aligned_mesh_pts, k=1)
    rmse = np.sqrt(np.mean(dists ** 2))
    mean_dist = np.mean(dists)

    stats_text = (
        f"RMSE: {rmse:.2f} mm\n"
        f"Mean: {mean_dist:.2f} mm\n"
        f"Mesh pts: {len(aligned_mesh_pts)}\n"
        f"Target pts: {len(target_pts)}"
    )
    plotter.add_text(stats_text, position='upper_right', font_size=10, color='black')

    plotter.add_legend(face=None, bcolor='white')
    plotter.add_axes()
    plotter.add_title(title)

    if save_path:
        plotter.show(screenshot=save_path)
        print(f"  Alignment plot saved to: {save_path}")
    else:
        plotter.show()
