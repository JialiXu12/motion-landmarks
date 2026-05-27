"""
Visual comparison showing why the two plotting functions produce different results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def create_comparison_diagram():
    """
    Create side-by-side comparison of the two reference frames.
    """

    fig = plt.figure(figsize=(16, 6))

    # ========== LEFT PLOT: STERNUM REFERENCE (alignment.py) ==========
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-150, 150)
    ax1.set_ylim(-150, 50)
    ax1.set_aspect('equal')

    # Sternum (origin)
    ax1.plot(0, 0, 'ks', markersize=15, label='Sternum (origin)', zorder=5)
    ax1.text(0, -15, 'Sternum\n(0,0,0)', ha='center', fontsize=10, fontweight='bold')

    # Left breast nipple
    left_nipple_prone = np.array([-80, -100])
    left_nipple_supine = np.array([-120, -50])
    ax1.plot(left_nipple_prone[0], left_nipple_prone[1], 'bo', markersize=10, label='Nipple (prone)')
    ax1.plot(left_nipple_supine[0], left_nipple_supine[1], 'ro', markersize=10, label='Nipple (supine)')
    ax1.arrow(left_nipple_prone[0], left_nipple_prone[1],
              left_nipple_supine[0]-left_nipple_prone[0],
              left_nipple_supine[1]-left_nipple_prone[1],
              head_width=5, head_length=5, fc='purple', ec='purple', lw=2, alpha=0.7)
    ax1.text(left_nipple_prone[0]-10, left_nipple_prone[1]-15, 'Nipple\ndisplacement',
             fontsize=9, color='purple', fontweight='bold')

    # Left breast landmark
    lm_prone = np.array([-70, -80])
    lm_supine = np.array([-100, -40])
    ax1.plot(lm_prone[0], lm_prone[1], 'b^', markersize=12, label='Landmark (prone)')
    ax1.plot(lm_supine[0], lm_supine[1], 'r^', markersize=12, label='Landmark (supine)')
    ax1.arrow(lm_prone[0], lm_prone[1],
              lm_supine[0]-lm_prone[0],
              lm_supine[1]-lm_prone[1],
              head_width=5, head_length=5, fc='green', ec='green', lw=3, alpha=0.8)
    ax1.text(lm_prone[0]+15, lm_prone[1], 'Landmark\ndisplacement',
             fontsize=9, color='green', fontweight='bold')

    # Right breast (mirror)
    right_nipple_prone = np.array([80, -100])
    right_nipple_supine = np.array([120, -50])
    ax1.plot(right_nipple_prone[0], right_nipple_prone[1], 'bo', markersize=10)
    ax1.plot(right_nipple_supine[0], right_nipple_supine[1], 'ro', markersize=10)
    ax1.arrow(right_nipple_prone[0], right_nipple_prone[1],
              right_nipple_supine[0]-right_nipple_prone[0],
              right_nipple_supine[1]-right_nipple_prone[1],
              head_width=5, head_length=5, fc='purple', ec='purple', lw=2, alpha=0.7)

    rm_prone = np.array([70, -80])
    rm_supine = np.array([100, -40])
    ax1.plot(rm_prone[0], rm_prone[1], 'b^', markersize=12)
    ax1.plot(rm_supine[0], rm_supine[1], 'r^', markersize=12)
    ax1.arrow(rm_prone[0], rm_prone[1],
              rm_supine[0]-rm_prone[0],
              rm_supine[1]-rm_prone[1],
              head_width=5, head_length=5, fc='green', ec='green', lw=3, alpha=0.8)

    # Axes
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Medial ← → Lateral (mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Anterior ← → Posterior (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('STERNUM REFERENCE\n(alignment.py, line 775)\n\nShows: Absolute displacement from sternum',
                  fontsize=13, fontweight='bold', color='darkblue')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add reference frame indicator
    ax1.text(0, 40, 'Origin = Sternum Superior', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
             ha='center', fontweight='bold')

    # ========== RIGHT PLOT: NIPPLE REFERENCE (main.py) ==========
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-80, 80)
    ax2.set_ylim(-50, 80)
    ax2.set_aspect('equal')

    # Left nipple (now origin)
    ax2.plot(0, 0, 'ko', markersize=15, label='Left Nipple (origin)', zorder=5)
    ax2.text(0, -10, 'Left Nipple\n(0,0,0)', ha='center', fontsize=10, fontweight='bold')

    # Left landmark relative to left nipple
    lm_rel_nipple_prone = lm_prone - left_nipple_prone
    lm_disp_abs = lm_supine - lm_prone
    nipple_disp = left_nipple_supine - left_nipple_prone
    lm_disp_diff = lm_disp_abs - nipple_disp  # Differential

    ax2.plot(lm_rel_nipple_prone[0], lm_rel_nipple_prone[1], 'b^', markersize=12,
             label='Landmark (prone, rel to nipple)')
    lm_end = lm_rel_nipple_prone + lm_disp_diff
    ax2.plot(lm_end[0], lm_end[1], 'r^', markersize=12,
             label='Landmark (after differential motion)')
    ax2.arrow(lm_rel_nipple_prone[0], lm_rel_nipple_prone[1],
              lm_disp_diff[0], lm_disp_diff[1],
              head_width=3, head_length=3, fc='orange', ec='orange', lw=3, alpha=0.8)
    ax2.text(lm_rel_nipple_prone[0]+5, lm_rel_nipple_prone[1]+20,
             'Differential\ndisplacement\n(Landmark - Nipple)',
             fontsize=9, color='orange', fontweight='bold')

    # Show what the differential means
    ax2.annotate('', xy=(lm_rel_nipple_prone[0], lm_rel_nipple_prone[1]),
                 xytext=(lm_rel_nipple_prone[0] + lm_disp_abs[0], lm_rel_nipple_prone[1] + lm_disp_abs[1]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.4, linestyle='--'))
    ax2.text(lm_rel_nipple_prone[0] + lm_disp_abs[0]/2 - 15,
             lm_rel_nipple_prone[1] + lm_disp_abs[1]/2,
             'Absolute\nmotion', fontsize=8, color='green', alpha=0.6)

    ax2.annotate('', xy=(0, 0), xytext=(nipple_disp[0], nipple_disp[1]),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, alpha=0.4, linestyle='--'))
    ax2.text(nipple_disp[0]/2 + 10, nipple_disp[1]/2,
             'Nipple\nmotion', fontsize=8, color='purple', alpha=0.6)

    # Axes
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Medial ← → Lateral (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Anterior ← → Posterior (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('NIPPLE REFERENCE\n(main.py, line 180)\n\nShows: Differential displacement\n(Landmark motion - Nipple motion)',
                  fontsize=13, fontweight='bold', color='darkred')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add reference frame indicator
    ax2.text(0, 70, 'Origin = Nipple Position', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../output/figs/reference_frame_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison diagram saved to: ../output/figs/reference_frame_comparison.png")
    plt.show()

    # ========== PRINT SUMMARY ==========
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\nLEFT PLOT (Sternum Reference):")
    print("  Origin: Sternum superior")
    print("  Base points: Landmark positions relative to sternum")
    print("  Vectors: Absolute displacement (supine - prone)")
    print("  Code: alignment.py, line 775")
    print("  Title: 'Landmark Displacement Relative to Sternal Superior (Jugular Notch)'")

    print("\nRIGHT PLOT (Nipple Reference):")
    print("  Origin: Nipple (left or right)")
    print("  Base points: Landmark positions relative to nipple")
    print("  Vectors: Differential displacement (landmark - nipple motion)")
    print("  Code: main.py, line 180")
    print("  Title: 'Displacement of landmarks from prone to supine'")

    print("\n" + "="*80)
    print("KEY DIFFERENCE:")
    print("="*80)
    print("\nSternum Reference:")
    print("  Vector = Landmark_supine - Landmark_prone")
    print("  Shows how much landmark moved in absolute terms")

    print("\nNipple Reference:")
    print("  Vector = (Landmark_supine - Landmark_prone) - (Nipple_supine - Nipple_prone)")
    print("  Shows how much MORE/LESS landmark moved compared to nipple")
    print("  Negative = Landmark moved LESS than nipple (or opposite direction)")

    print("\n" + "="*80)
    print("CLINICAL INTERPRETATION:")
    print("="*80)
    print("\nSternum Reference:")
    print("  ✓ Shows gravity effect on breast")
    print("  ✓ Shows whole-breast deformation")
    print("  ✓ Reference: Skeletal (fixed)")

    print("\nNipple Reference:")
    print("  ✓ Shows tissue heterogeneity (deep vs superficial)")
    print("  ✓ Shows if tumor moves with skin or stays fixed to chest")
    print("  ✓ Reference: Anatomical (moves with breast)")
    print("  ✓ Important for surgical planning")

    print("\n" + "="*80)


if __name__ == "__main__":
    create_comparison_diagram()
