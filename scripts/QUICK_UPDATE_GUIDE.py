"""
Quick Guide: Update main.py to use the new alignment module

Follow these steps to switch to the optimal sternum-fixed alignment:
"""

# ==============================================================================
# STEP 1: Update the import at the top of main.py
# ==============================================================================

# OLD (line ~17):
# from align_fixed_sternum import align_prone_to_supine_fixed_sternum

# NEW:
from alignment import align_prone_to_supine_optimal

# ==============================================================================
# STEP 2: Update the function call (around line ~131)
# ==============================================================================

# OLD:
"""
alignment_results = align_prone_to_supine_fixed_sternum(
    subject=filtered_subject,
    prone_ribcage_mesh_path=prone_mesh_file,
    supine_ribcage_seg_path=supine_seg_file,
    orientation_flag='RAI',
    plot_for_debug=True
)
"""

# NEW:
"""
alignment_results = align_prone_to_supine_optimal(
    subject=filtered_subject,
    prone_ribcage_mesh_path=prone_mesh_file,
    supine_ribcage_seg_path=supine_seg_file,
    orientation_flag='RAI',
    plot_for_debug=True
)
"""

# ==============================================================================
# OPTIONAL: Add both methods and make it switchable
# ==============================================================================

"""
# At top of main.py:
from alignment import align_prone_to_supine_optimal
from align_fixed_sternum import align_prone_to_supine_fixed_sternum

# Add a configuration flag:
USE_OPTIMAL_ALIGNMENT = True  # Switch between methods

# In the alignment loop:
if USE_OPTIMAL_ALIGNMENT:
    alignment_results = align_prone_to_supine_optimal(
        subject=filtered_subject,
        prone_ribcage_mesh_path=prone_mesh_file,
        supine_ribcage_seg_path=supine_seg_file,
        orientation_flag='RAI',
        plot_for_debug=True
    )
else:
    alignment_results = align_prone_to_supine_fixed_sternum(
        subject=filtered_subject,
        prone_ribcage_mesh_path=prone_mesh_file,
        supine_ribcage_seg_path=supine_seg_file,
        orientation_flag='RAI',
        plot_for_debug=True
    )
"""

# ==============================================================================
# That's it! No other changes needed.
# ==============================================================================

print("The new alignment module has the same interface as the old one.")
print("It returns the same dictionary structure, so all downstream code")
print("(saving to Excel, plotting, etc.) will work without changes.")
