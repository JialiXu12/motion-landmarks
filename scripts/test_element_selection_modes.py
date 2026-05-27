"""
Test script to visualize element selection with AND vs OR combine modes.

This script demonstrates why using 'or' (union) mode is usually better
for ribcage alignment than 'and' (intersection) mode.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add morphic path only
scripts_dir = Path(__file__).parent
project_dir = scripts_dir.parent
sys.path.insert(0, str(project_dir / 'src' / 'morphic'))

import morphic


def get_mesh_elements_centers(mesh):
    """Extract element centers from morphic mesh."""
    centers = []
    num_elements = mesh.elements.size()
    Xi = mesh.grid(3, method='center')

    for i in range(num_elements):
        elem = list(mesh.elements)[i]
        elem_coords = elem.evaluate(Xi)
        center_idx = elem_coords.shape[0] // 2
        center = elem_coords[center_idx, :]
        centers.append(center)

    return np.array(centers), num_elements


def debug_element_selection(mesh_path, y_percentile=50, z_percentile=50):
    """
    Debug why certain elements might be missing from the selection.
    Shows detailed per-element analysis.
    """
    mesh = morphic.Mesh(str(mesh_path))
    centers_array, num_elements = get_mesh_elements_centers(mesh)

    print(f"\n{'='*70}")
    print(f"DETAILED ELEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Mesh: {mesh_path.name}")
    print(f"Total elements: {num_elements}")

    # Calculate thresholds
    y_thresh = np.percentile(centers_array[:, 1], y_percentile)
    z_thresh = np.percentile(centers_array[:, 2], z_percentile)

    print(f"\nThresholds (percentile-based):")
    print(f"  Y threshold (p{y_percentile}): {y_thresh:.1f} mm")
    print(f"  Z threshold (p{z_percentile}): {z_thresh:.1f} mm")
    print(f"  Anterior = Y < {y_thresh:.1f}")
    print(f"  Superior = Z > {z_thresh:.1f}")

    # Analyze each element
    print(f"\n{'Elem':<5} {'X':>8} {'Y':>8} {'Z':>8} {'Ant?':>6} {'Sup?':>6} {'AND':>5}")
    print("-" * 55)

    selected_and = []
    for i in range(num_elements):
        x, y, z = centers_array[i]
        is_ant = y < y_thresh
        is_sup = z > z_thresh
        is_selected = is_ant and is_sup

        if is_selected:
            selected_and.append(i)

        ant_str = "Yes" if is_ant else "No"
        sup_str = "Yes" if is_sup else "No"
        sel_str = "✓" if is_selected else ""

        print(f"{i:<5} {x:>8.1f} {y:>8.1f} {z:>8.1f} {ant_str:>6} {sup_str:>6} {sel_str:>5}")

    print("-" * 55)
    print(f"Selected (AND): {len(selected_and)} elements: {selected_and}")

    # Show Y and Z distributions
    print(f"\nY coordinate distribution:")
    print(f"  Min: {centers_array[:, 1].min():.1f}")
    print(f"  Max: {centers_array[:, 1].max():.1f}")
    print(f"  Median (p50): {np.percentile(centers_array[:, 1], 50):.1f}")

    print(f"\nZ coordinate distribution:")
    print(f"  Min: {centers_array[:, 2].min():.1f}")
    print(f"  Max: {centers_array[:, 2].max():.1f}")
    print(f"  Median (p50): {np.percentile(centers_array[:, 2], 50):.1f}")

    return centers_array, selected_and, y_thresh, z_thresh


def visualize_element_grid(mesh_path, y_percentile=50, z_percentile=50):
    """
    Visualize elements in a grid layout showing their spatial positions
    and which quadrant they fall into.
    """
    mesh = morphic.Mesh(str(mesh_path))
    centers_array, num_elements = get_mesh_elements_centers(mesh)

    y_thresh = np.percentile(centers_array[:, 1], y_percentile)
    z_thresh = np.percentile(centers_array[:, 2], z_percentile)

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Sagittal view (Y vs Z) - most informative for anterior/superior
    ax1 = fig.add_subplot(2, 2, 1)
    y = centers_array[:, 1]
    z = centers_array[:, 2]

    # Color by quadrant
    colors = []
    for i in range(num_elements):
        is_ant = y[i] < y_thresh
        is_sup = z[i] > z_thresh
        if is_ant and is_sup:
            colors.append('green')  # Selected (AND)
        elif is_ant:
            colors.append('blue')   # Anterior only
        elif is_sup:
            colors.append('red')    # Superior only
        else:
            colors.append('gray')   # Neither (posterior-inferior)

    ax1.scatter(y, z, c=colors, s=300, alpha=0.7, edgecolors='black', linewidths=2)
    for i in range(num_elements):
        ax1.annotate(str(i), (y[i], z[i]), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    ax1.axvline(y_thresh, color='purple', linestyle='--', linewidth=2,
                label=f'Y threshold ({y_thresh:.0f}mm)')
    ax1.axhline(z_thresh, color='orange', linestyle='--', linewidth=2,
                label=f'Z threshold ({z_thresh:.0f}mm)')

    ax1.set_xlabel('Y (Posterior → Anterior)', fontsize=12)
    ax1.set_ylabel('Z (Inferior → Superior)', fontsize=12)
    ax1.set_title(f'Sagittal View (Y-Z)\nGreen = AND selection (Anterior AND Superior)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.invert_xaxis()  # Anterior on the right
    ax1.grid(True, alpha=0.3)

    # Add quadrant labels
    y_mid = (y.min() + y_thresh) / 2
    y_ant_mid = (y_thresh + y.max()) / 2
    z_mid = (z.min() + z_thresh) / 2
    z_sup_mid = (z_thresh + z.max()) / 2

    # Count elements in each quadrant
    n_post_sup = sum(1 for i in range(num_elements) if y[i] >= y_thresh and z[i] > z_thresh)
    n_ant_sup = sum(1 for i in range(num_elements) if y[i] < y_thresh and z[i] > z_thresh)
    n_post_inf = sum(1 for i in range(num_elements) if y[i] >= y_thresh and z[i] <= z_thresh)
    n_ant_inf = sum(1 for i in range(num_elements) if y[i] < y_thresh and z[i] <= z_thresh)

    ax1.text(y_mid, z_sup_mid, f'Post-Sup\n({n_post_sup})', fontsize=11,
             ha='center', va='center', color='red', fontweight='bold')
    ax1.text(y_ant_mid, z_sup_mid, f'Ant-Sup\n({n_ant_sup}) ✓', fontsize=11,
             ha='center', va='center', color='green', fontweight='bold')
    ax1.text(y_mid, z_mid, f'Post-Inf\n({n_post_inf})', fontsize=11,
             ha='center', va='center', color='gray', fontweight='bold')
    ax1.text(y_ant_mid, z_mid, f'Ant-Inf\n({n_ant_inf})', fontsize=11,
             ha='center', va='center', color='blue', fontweight='bold')

    # Plot 2: Coronal view (X vs Z)
    ax2 = fig.add_subplot(2, 2, 2)
    x = centers_array[:, 0]
    ax2.scatter(x, z, c=colors, s=300, alpha=0.7, edgecolors='black', linewidths=2)
    for i in range(num_elements):
        ax2.annotate(str(i), (x[i], z[i]), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    ax2.axhline(z_thresh, color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('X (Right → Left)', fontsize=12)
    ax2.set_ylabel('Z (Inferior → Superior)', fontsize=12)
    ax2.set_title('Coronal View (X-Z)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Axial view (X vs Y)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(x, y, c=colors, s=300, alpha=0.7, edgecolors='black', linewidths=2)
    for i in range(num_elements):
        ax3.annotate(str(i), (x[i], y[i]), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    ax3.axhline(y_thresh, color='purple', linestyle='--', linewidth=2)
    ax3.set_xlabel('X (Right → Left)', fontsize=12)
    ax3.set_ylabel('Y (Posterior → Anterior)', fontsize=12)
    ax3.set_title('Axial View (X-Y)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()  # Anterior at top
    ax3.grid(True, alpha=0.3)

    # Plot 4: Legend and summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = f"""
ELEMENT SELECTION SUMMARY
{'='*40}

Mesh: {mesh_path.name}
Total elements: {num_elements}

Thresholds:
  Y (Ant/Post): {y_thresh:.1f} mm (p{y_percentile})
  Z (Sup/Inf): {z_thresh:.1f} mm (p{z_percentile})

Quadrant counts:
  Anterior-Superior (AND): {n_ant_sup} elements ✓ SELECTED
  Anterior-Inferior:       {n_ant_inf} elements
  Posterior-Superior:      {n_post_sup} elements  
  Posterior-Inferior:      {n_post_inf} elements

Color legend:
  🟢 Green = Selected (Anterior AND Superior)
  🔵 Blue  = Anterior only (not superior)
  🔴 Red   = Superior only (not anterior)
  ⚫ Gray  = Neither (posterior-inferior)

NOTE: If you want MORE elements in the AND selection,
try adjusting the percentiles:
  - y_percentile=60 → keeps more posterior elements
  - z_percentile=40 → keeps more inferior elements
"""
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / 'output' / 'figs' / 'element_selection_debug.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved debug figure to: {output_path}")

    plt.show()

    return centers_array, n_ant_sup


def visualize_selection_modes(mesh_path):
    """
    Visualize the difference between AND and OR combine modes.

    Creates a 2x2 grid showing:
    - All elements
    - Anterior only
    - AND mode (anterior AND superior) - only corner
    - OR mode (anterior OR superior) - excludes posterior-inferior
    """
    # Load mesh
    mesh = morphic.Mesh(str(mesh_path))
    centers_array, num_elements = get_mesh_elements_centers(mesh)

    print(f"Mesh has {num_elements} elements")
    print(f"Element centers shape: {centers_array.shape}")

    # Get thresholds (50th percentile = median)
    y_thresh = np.percentile(centers_array[:, 1], 50)
    z_thresh = np.percentile(centers_array[:, 2], 50)

    print(f"\nThresholds:")
    print(f"  Y (Ant-Post): {y_thresh:.1f} mm (anterior < threshold)")
    print(f"  Z (Inf-Sup): {z_thresh:.1f} mm (superior > threshold)")

    # Classify each element
    is_anterior = centers_array[:, 1] < y_thresh
    is_superior = centers_array[:, 2] > z_thresh

    # Different selection modes
    and_mask = is_anterior & is_superior  # Both conditions
    or_mask = is_anterior | is_superior   # Either condition

    # Count elements in each region
    print(f"\nElement counts:")
    print(f"  Total: {num_elements}")
    print(f"  Anterior: {np.sum(is_anterior)}")
    print(f"  Superior: {np.sum(is_superior)}")
    print(f"  AND (anterior AND superior): {np.sum(and_mask)}")
    print(f"  OR (anterior OR superior): {np.sum(or_mask)}")
    print(f"  Excluded by OR (posterior-inferior): {num_elements - np.sum(or_mask)}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot Y vs Z (sagittal view - most informative)
    y = centers_array[:, 1]  # Anterior-Posterior
    z = centers_array[:, 2]  # Inferior-Superior

    # 1. All elements
    ax = axes[0, 0]
    ax.scatter(y, z, c='gray', s=200, alpha=0.7, edgecolors='black')
    for i in range(num_elements):
        ax.annotate(str(i), (y[i], z[i]), ha='center', va='center', fontsize=8)
    ax.axvline(y_thresh, color='blue', linestyle='--', label=f'Y threshold ({y_thresh:.0f})')
    ax.axhline(z_thresh, color='red', linestyle='--', label=f'Z threshold ({z_thresh:.0f})')
    ax.set_xlabel('Y (Posterior → Anterior)')
    ax.set_ylabel('Z (Inferior → Superior)')
    ax.set_title(f'All Elements ({num_elements})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_xaxis()  # Anterior on the right

    # Add quadrant labels
    ax.text(y.min() + 10, z.max() - 10, 'Posterior-Superior', fontsize=10, ha='left')
    ax.text(y.max() - 10, z.max() - 10, 'Anterior-Superior', fontsize=10, ha='right')
    ax.text(y.min() + 10, z.min() + 10, 'Posterior-Inferior', fontsize=10, ha='left')
    ax.text(y.max() - 10, z.min() + 10, 'Anterior-Inferior', fontsize=10, ha='right')

    # 2. Anterior only
    ax = axes[0, 1]
    colors = ['green' if a else 'lightgray' for a in is_anterior]
    ax.scatter(y, z, c=colors, s=200, alpha=0.7, edgecolors='black')
    for i in range(num_elements):
        ax.annotate(str(i), (y[i], z[i]), ha='center', va='center', fontsize=8)
    ax.axvline(y_thresh, color='blue', linestyle='--')
    ax.set_xlabel('Y (Posterior → Anterior)')
    ax.set_ylabel('Z (Inferior → Superior)')
    ax.set_title(f'Anterior Only ({np.sum(is_anterior)} elements)', fontsize=12, fontweight='bold')
    ax.invert_xaxis()

    # 3. AND mode (intersection)
    ax = axes[1, 0]
    colors = ['red' if m else 'lightgray' for m in and_mask]
    ax.scatter(y, z, c=colors, s=200, alpha=0.7, edgecolors='black')
    for i in range(num_elements):
        ax.annotate(str(i), (y[i], z[i]), ha='center', va='center', fontsize=8)
    ax.axvline(y_thresh, color='blue', linestyle='--')
    ax.axhline(z_thresh, color='red', linestyle='--')
    ax.set_xlabel('Y (Posterior → Anterior)')
    ax.set_ylabel('Z (Inferior → Superior)')
    ax.set_title(f'AND Mode: Anterior AND Superior ({np.sum(and_mask)} elements)\n'
                 f'⚠️ MISSING MIDDLE REGION!', fontsize=12, fontweight='bold', color='red')
    ax.invert_xaxis()

    # Highlight missing regions
    ax.fill_betweenx([z.min(), z_thresh], y_thresh, y.max(), alpha=0.2, color='orange',
                     label='Missing: Anterior-Inferior')
    ax.fill_betweenx([z_thresh, z.max()], y.min(), y_thresh, alpha=0.2, color='purple',
                     label='Missing: Posterior-Superior')
    ax.legend(loc='lower right', fontsize=8)

    # 4. OR mode (union)
    ax = axes[1, 1]
    colors = ['green' if m else 'lightgray' for m in or_mask]
    ax.scatter(y, z, c=colors, s=200, alpha=0.7, edgecolors='black')
    for i in range(num_elements):
        ax.annotate(str(i), (y[i], z[i]), ha='center', va='center', fontsize=8)
    ax.axvline(y_thresh, color='blue', linestyle='--')
    ax.axhline(z_thresh, color='red', linestyle='--')
    ax.set_xlabel('Y (Posterior → Anterior)')
    ax.set_ylabel('Z (Inferior → Superior)')
    ax.set_title(f'OR Mode: Anterior OR Superior ({np.sum(or_mask)} elements)\n'
                 f'✓ Excludes only posterior-inferior corner', fontsize=12, fontweight='bold', color='green')
    ax.invert_xaxis()

    # Highlight excluded region
    ax.fill_betweenx([z.min(), z_thresh], y.min(), y_thresh, alpha=0.3, color='red',
                     label='Excluded: Posterior-Inferior')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.suptitle('Element Selection: AND vs OR Mode Comparison', fontsize=14, fontweight='bold', y=1.02)

    # Save figure
    output_path = Path(__file__).parent.parent / 'output' / 'figs' / 'element_selection_modes.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

    plt.show()

    return and_mask, or_mask


if __name__ == "__main__":
    # Find a mesh file to test with
    MESH_ROOT = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")

    # Try to find a mesh file
    mesh_files = list(MESH_ROOT.glob("*_ribcage_prone.mesh"))

    if mesh_files:
        mesh_path = mesh_files[0]  # Use first available mesh
        print(f"Using mesh: {mesh_path}")

        # Run detailed debugging
        debug_element_selection(mesh_path, y_percentile=50, z_percentile=50)

        # Visualize with grid layout
        visualize_element_grid(mesh_path, y_percentile=50, z_percentile=50)

        # Also test with adjusted percentiles to see effect
        print("\n" + "="*70)
        print("TESTING DIFFERENT PERCENTILE SETTINGS")
        print("="*70)

        for y_pct, z_pct in [(50, 50), (60, 40), (70, 30)]:
            mesh = morphic.Mesh(str(mesh_path))
            centers, num_elem = get_mesh_elements_centers(mesh)
            y_thresh = np.percentile(centers[:, 1], y_pct)
            z_thresh = np.percentile(centers[:, 2], z_pct)

            selected = [i for i in range(num_elem)
                       if centers[i, 1] < y_thresh and centers[i, 2] > z_thresh]

            print(f"\n  y_percentile={y_pct}, z_percentile={z_pct}:")
            print(f"    Y threshold: {y_thresh:.1f} mm")
            print(f"    Z threshold: {z_thresh:.1f} mm")
            print(f"    Selected: {len(selected)}/{num_elem} elements")
            print(f"    Indices: {selected}")
    else:
        print(f"No mesh files found in {MESH_ROOT}")
        print("Please update the MESH_ROOT path to point to your mesh directory")











