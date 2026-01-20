"""
Visual quality check - Opens all generated plots for manual inspection
This helps verify the text visualization improvements are correct and presentable
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

def display_plot_comparison():
    """Display all three plot variants for visual inspection"""

    output_path = Path("..") / "output" / "figs" / "landmark vectors"

    plot_files = [
        ('Vectors_rel_sternum_sagittal_dual.png', 'Breast Coloring (Blue/Green)'),
        ('Vectors_rel_sternum_sagittal_dual_DTS.png', 'DTS Coloring (Viridis)'),
        ('Vectors_rel_sternum_sagittal_dual_by_subject.png', 'Subject Coloring')
    ]

    print("="*80)
    print("VISUAL QUALITY CHECK: Text Labels in plot_sagittal_dual_axes")
    print("="*80)
    print("\nLoading plots for visual inspection...\n")

    # Create a figure to display all plots
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Text Visualization Quality Check - plot_sagittal_dual_axes',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, (filename, title) in enumerate(plot_files, 1):
        plot_path = output_path / filename

        if not plot_path.exists():
            print(f"⚠️ Plot not found: {filename}")
            continue

        print(f"✓ Loading: {filename}")

        ax = fig.add_subplot(3, 1, idx)
        img = mpimg.imread(plot_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout()

    # Save comparison
    comparison_path = output_path / "text_visualization_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {comparison_path}")

    print("\n" + "="*80)
    print("CHECKLIST FOR MANUAL INSPECTION")
    print("="*80)
    print("\nPlease verify the following in each plot:")
    print("\n1. TEXT CONTENT:")
    print("   □ Right breast labeled as 'Right Breast' (not 'RIGHT BREAST')")
    print("   □ Left breast labeled as 'Left Breast' (not 'LEFT BREAST')")
    print("\n2. TEXT POSITIONING:")
    print("   □ Labels positioned in upper portion of each subplot")
    print("   □ Labels centered at x=0 (sternum position)")
    print("   □ Labels do not overlap with data points/vectors")
    print("   □ Position at ~82% of y-axis looks appropriate")
    print("\n3. TEXT STYLING:")
    print("   □ Font size (14) is clearly readable")
    print("   □ Font weight is bold")
    print("   □ Text color matches subplot (blue for right, green for left)")
    print("\n4. BACKGROUND BOX:")
    print("   □ White background with rounded corners visible")
    print("   □ Colored border matches text color")
    print("   □ Border thickness (2) is appropriate")
    print("   □ Padding (0.6) provides good spacing")
    print("   □ Alpha (0.9) provides good contrast")
    print("\n5. OVERALL APPEARANCE:")
    print("   □ Labels appear on top of all other elements (zorder=100)")
    print("   □ Presentation looks professional and scientific")
    print("   □ Style is consistent across all three plot variants")
    print("   □ No layout warnings or visual artifacts")
    print("\n" + "="*80)

    plt.show()

if __name__ == "__main__":
    display_plot_comparison()
