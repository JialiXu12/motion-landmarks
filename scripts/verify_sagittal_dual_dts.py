"""
Visual verification: Count vectors in plot_sagittal_dual_axes DTS mode
"""

from pathlib import Path
from analysis import read_data
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("VISUAL VERIFICATION FOR plot_sagittal_dual_axes")
print("=" * 80)

# Load data
print("\nLoading data...")
df_raw, df_ave, df_demo = read_data(Path('../output/landmark_results_v4_2026_01_12.xlsx'))

# Separate by breast
left_df = df_ave[df_ave['landmark side (prone)'] == 'LB']
right_df = df_ave[df_ave['landmark side (prone)'] == 'RB']

print(f"\nDataset contains:")
print(f"  - Right breast: {len(right_df)} landmarks")
print(f"  - Left breast: {len(left_df)} landmarks")
print(f"  - Total: {len(df_ave)} landmarks")

# Extract data
def get_points_and_vectors(sub_df):
    if sub_df.empty:
        return np.empty((0, 3)), np.empty((0, 3)), None

    prone_x = sub_df['landmark ave prone transformed x'].values
    prone_y = sub_df['landmark ave prone transformed y'].values
    prone_z = sub_df['landmark ave prone transformed z'].values
    base_points = np.column_stack((prone_x, prone_y, prone_z))

    supine_x = sub_df['landmark ave supine x'].values
    supine_y = sub_df['landmark ave supine y'].values
    supine_z = sub_df['landmark ave supine z'].values
    end_points = np.column_stack((supine_x, supine_y, supine_z))

    vectors = end_points - base_points

    dts_col = 'Distance to skin (prone) [mm]'
    dts_values = sub_df[dts_col].values if dts_col in sub_df.columns else None

    return base_points, vectors, dts_values

base_left, vec_left, dts_left = get_points_and_vectors(left_df)
base_right, vec_right, dts_right = get_points_and_vectors(right_df)

print(f"\nExtracted vectors:")
print(f"  - Right breast: {len(base_right)} base points, {len(vec_right)} vectors, {len(dts_right) if dts_right is not None else 0} DTS values")
print(f"  - Left breast: {len(base_left)} base points, {len(vec_left)} vectors, {len(dts_left) if dts_left is not None else 0} DTS values")

# Create test plot with DTS coloring
print("\nCreating test plot (Sagittal Dual Axes with DTS coloring)...")
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
fig.suptitle("TEST: DTS Coloring - Sagittal Dual View", fontsize=14)
plt.subplots_adjust(wspace=0.0)

ylim_val = 250
ax_left.set_ylim(-ylim_val, ylim_val)
ax_right.set_ylim(-ylim_val, ylim_val)
yticks = np.arange(-250, 251, 50)
ax_left.set_yticks(yticks)
ax_right.set_yticks(yticks)

# LEFT PLOT (RIGHT BREAST)
ax_left.set_xlim(150, -250)
ax_left.set_xticks([150, 100, 50, 0, -50, -100, -150, -200, -250])
ax_left.set_xlabel("Post-Ant (mm)", fontsize=12, fontweight='bold')
ax_left.set_ylabel("Inf-Sup (mm)", fontsize=12, fontweight='bold')
ax_left.spines['right'].set_position(('data', 0))
ax_left.spines['left'].set_visible(False)
ax_left.spines['top'].set_visible(False)
ax_left.spines['bottom'].set_visible(True)
ax_left.plot(0, 0, 'ko', markersize=6, zorder=10)
ax_left.grid(True, linestyle='--', alpha=0.5)
ax_left.set_aspect('equal', adjustable='box')

vector_count_right = 0
scatter_left_for_colorbar = None

if len(base_right) > 0 and dts_right is not None:
    norm = plt.Normalize(vmin=0, vmax=40)
    cmap = plt.cm.viridis
    colors_right = cmap(norm(dts_right))

    for i in range(len(base_right)):
        ax_left.quiver(
            base_right[i, 1], base_right[i, 2],
            vec_right[i, 1], vec_right[i, 2],
            angles='xy', scale_units='xy', scale=1,
            color=colors_right[i],
            width=0.003, headwidth=3, alpha=0.7
        )
        vector_count_right += 1

    scatter_left_for_colorbar = ax_left.scatter(
        base_right[:, 1], base_right[:, 2],
        c=dts_right, cmap='viridis', s=20, vmin=0, vmax=40, zorder=5
    )
    print(f"  ✓ Plotted {vector_count_right} RIGHT breast vectors on left subplot")

ax_left.text(0, ylim_val*0.9, "RIGHT BREAST", ha='center', va='center',
            fontweight='bold', fontsize=11)

# RIGHT PLOT (LEFT BREAST)
ax_right.set_xlim(-250, 150)
ax_right.set_xticks(np.arange(-250, 151, 50))
ax_right.set_xlabel("Ant-Post (mm)", fontsize=12, fontweight='bold')
ax_right.spines['left'].set_position(('data', 0))
ax_right.spines['right'].set_visible(False)
ax_right.spines['top'].set_visible(False)
ax_right.spines['bottom'].set_visible(True)
ax_right.plot(0, 0, 'ko', markersize=6, zorder=10)
ax_right.grid(True, linestyle='--', alpha=0.5)
ax_right.set_aspect('equal', adjustable='box')

vector_count_left = 0
scatter_right_for_colorbar = None

if len(base_left) > 0 and dts_left is not None:
    norm = plt.Normalize(vmin=0, vmax=40)
    cmap = plt.cm.viridis
    colors_left = cmap(norm(dts_left))

    for i in range(len(base_left)):
        ax_right.quiver(
            base_left[i, 1], base_left[i, 2],
            vec_left[i, 1], vec_left[i, 2],
            angles='xy', scale_units='xy', scale=1,
            color=colors_left[i],
            width=0.003, headwidth=3, alpha=0.7
        )
        vector_count_left += 1

    scatter_right_for_colorbar = ax_right.scatter(
        base_left[:, 1], base_left[:, 2],
        c=dts_left, cmap='viridis', s=20, vmin=0, vmax=40, zorder=5
    )
    print(f"  ✓ Plotted {vector_count_left} LEFT breast vectors on right subplot")

ax_right.text(0, ylim_val*0.9, "LEFT BREAST", ha='center', va='center',
             fontweight='bold', fontsize=11)

# Add colorbar
scatter_for_cbar = scatter_right_for_colorbar if scatter_right_for_colorbar is not None else scatter_left_for_colorbar
if scatter_for_cbar is not None:
    cbar = plt.colorbar(scatter_for_cbar, ax=ax_right, pad=0.02)
    cbar.set_label('DTS (mm)', rotation=270, labelpad=15, fontsize=12)

# Save
save_path = Path("..") / "output" / "figs" / "landmark vectors" / "TEST_sagittal_dual_verification.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"\n✓ Test plot saved: {save_path}")

total_vectors = vector_count_right + vector_count_left
expected_total = len(df_ave)

print(f"\n" + "=" * 80)
print(f"VERIFICATION COMPLETE")
print(f"=" * 80)
print(f"Total vectors plotted: {total_vectors}")
print(f"  - Right breast (left subplot): {vector_count_right}")
print(f"  - Left breast (right subplot): {vector_count_left}")
print(f"Expected: {expected_total} ({len(right_df)} right + {len(left_df)} left)")

if total_vectors == expected_total:
    print("✅ SUCCESS: All vectors are being plotted!")
else:
    print(f"⚠️  WARNING: Only {total_vectors}/{expected_total} vectors plotted")
print(f"=" * 80)

plt.close()
