"""
Visual verification: Count how many vectors are actually plotted in DTS mode
"""

from pathlib import Path
from analysis import read_data
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading data...")
df_raw, df_ave, df_demo = read_data(Path('../output/landmark_results_v4_2026_01_12.xlsx'))

# Separate by breast
left_df = df_ave[df_ave['landmark side (prone)'] == 'LB']
right_df = df_ave[df_ave['landmark side (prone)'] == 'RB']

print(f"\nDataset contains:")
print(f"  - Right breast: {len(right_df)} landmarks")
print(f"  - Left breast: {len(left_df)} landmarks")
print(f"  - Total: {len(df_ave)} landmarks")

# Extract data for one plane (Coronal: X vs Z)
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

# Create a test plot with DTS coloring for Coronal plane
print("\nCreating test plot (Coronal plane with DTS coloring)...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle("Test: DTS Coloring - Coronal View", fontsize=14)

axis_x_idx, axis_y_idx = 0, 2  # Coronal: X vs Z

ax.set_xlabel("Right-Left (mm)", fontsize=12)
ax.set_ylabel("Inf-Sup (mm)", fontsize=12)
ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.5)
ax.plot(0, 0, 'ko', markersize=5, label='Sternum (Origin)', zorder=5)

scatter_for_colorbar = None
vector_count = 0

# Plot Right Breast
if len(base_right) > 0 and dts_right is not None:
    norm = plt.Normalize(vmin=0, vmax=40)
    cmap = plt.cm.viridis
    colors_right = cmap(norm(dts_right))

    for i in range(len(base_right)):
        ax.quiver(
            base_right[i, axis_x_idx],
            base_right[i, axis_y_idx],
            vec_right[i, axis_x_idx],
            vec_right[i, axis_y_idx],
            angles='xy', scale_units='xy', scale=1,
            color=colors_right[i],
            width=0.003,
            headwidth=3,
            headlength=4,
            alpha=0.7,
            zorder=4
        )
        vector_count += 1

    scatter_right = ax.scatter(
        base_right[:, axis_x_idx],
        base_right[:, axis_y_idx],
        c=dts_right,
        cmap='viridis',
        s=20,
        vmin=0,
        vmax=40,
        zorder=5
    )
    scatter_for_colorbar = scatter_right
    print(f"  ✓ Plotted {len(base_right)} RIGHT breast vectors")

# Plot Left Breast
if len(base_left) > 0 and dts_left is not None:
    norm = plt.Normalize(vmin=0, vmax=40)
    cmap = plt.cm.viridis
    colors_left = cmap(norm(dts_left))

    for i in range(len(base_left)):
        ax.quiver(
            base_left[i, axis_x_idx],
            base_left[i, axis_y_idx],
            vec_left[i, axis_x_idx],
            vec_left[i, axis_y_idx],
            angles='xy', scale_units='xy', scale=1,
            color=colors_left[i],
            width=0.003,
            headwidth=3,
            headlength=4,
            alpha=0.7,
            zorder=4
        )
        vector_count += 1

    scatter_left = ax.scatter(
        base_left[:, axis_x_idx],
        base_left[:, axis_y_idx],
        c=dts_left,
        cmap='viridis',
        s=20,
        vmin=0,
        vmax=40,
        zorder=5
    )
    if scatter_for_colorbar is None:
        scatter_for_colorbar = scatter_left
    print(f"  ✓ Plotted {len(base_left)} LEFT breast vectors")

# Add colorbar
if scatter_for_colorbar is not None:
    cbar = plt.colorbar(scatter_for_colorbar, ax=ax)
    cbar.set_label('DTS (mm)', rotation=270, labelpad=15, fontsize=12)

# Save
save_path = Path("..") / "output" / "figs" / "landmark vectors" / "TEST_DTS_verification.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"\n✓ Test plot saved: {save_path}")

print(f"\n" + "=" * 80)
print(f"VERIFICATION COMPLETE")
print(f"=" * 80)
print(f"Total vectors plotted: {vector_count}")
print(f"Expected: {len(df_ave)} (77 right + 79 left)")
if vector_count == len(df_ave):
    print("✅ SUCCESS: All vectors are being plotted!")
else:
    print(f"⚠️  WARNING: Only {vector_count}/{len(df_ave)} vectors plotted")
print(f"=" * 80)

plt.close()
