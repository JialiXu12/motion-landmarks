"""
ISSUE RESOLUTION: Breast Labels Missing When color_by='subject'
================================================================

PROBLEM:
--------
When calling plot_sagittal_dual_axes(df_ave, color_by='subject'), the text
labels "Right Breast" and "Left Breast" were not displayed on the plots.

ROOT CAUSE:
-----------
The breast label text was wrapped in a condition:
    if color_by != 'subject':
        # display labels

This prevented labels from appearing when coloring by subject, making it
difficult to identify which subplot represented which breast.

SOLUTION:
---------
Removed the conditional check and made breast labels always visible,
regardless of the coloring scheme. The labels are essential for identifying
the left and right breast subplots.

CODE CHANGES:
-------------

Location 1: Right Breast Label (analysis.py, ~line 1112)
BEFORE:
    # Add breast label (only for non-subject coloring)
    if color_by != 'subject':
        label_color = 'blue' if color_by == 'breast' else 'black'
        ax_left.text(0, ylim_val*0.82, "Right Breast", ...)

AFTER:
    # Add breast label (always shown for clarity)
    label_color = 'blue' if color_by == 'breast' else 'black'
    ax_left.text(0, ylim_val*0.82, "Right Breast", ...)

Location 2: Left Breast Label (analysis.py, ~line 1193)
BEFORE:
    # Add breast label (only for non-subject coloring)
    if color_by != 'subject':
        label_color = 'green' if color_by == 'breast' else 'black'
        ax_right.text(0, ylim_val*0.82, "Left Breast", ...)

AFTER:
    # Add breast label (always shown for clarity)
    label_color = 'green' if color_by == 'breast' else 'black'
    ax_right.text(0, ylim_val*0.82, "Left Breast", ...)

LABEL STYLING BY COLOR SCHEME:
-------------------------------
1. color_by='breast' (default):
   - Right Breast: Blue text with blue border
   - Left Breast: Green text with green border

2. color_by='dts' (distance to skin):
   - Right Breast: Black text with black border
   - Left Breast: Black text with black border

3. color_by='subject' (per-subject):
   - Right Breast: Black text with black border
   - Left Breast: Black text with black border

All labels maintain:
- Font size: 14 (bold)
- White background box with rounded corners
- Border thickness: 2
- Padding: 0.6
- Transparency: alpha=0.9
- Z-order: 100 (always on top)
- Position: 82% of y-axis

TESTING:
--------
✅ Created test_subject_labels.py to verify the fix
✅ Test passed - labels now visible with color_by='subject'
✅ Labels styled correctly (black text, white background)
✅ No errors or warnings
✅ All three coloring schemes tested and working

VERIFICATION CHECKLIST:
-----------------------
[✅] Breast labels visible when color_by='breast'
[✅] Breast labels visible when color_by='dts'
[✅] Breast labels visible when color_by='subject' (FIXED!)
[✅] Label colors appropriate for each scheme
[✅] Labels clearly readable and well-positioned
[✅] No overlap with data points
[✅] Professional appearance maintained

RESULT:
-------
✅ ISSUE RESOLVED

The breast labels now appear in all three coloring schemes, making it easy
to identify which subplot represents the right breast and which represents
the left breast, regardless of how the data points are colored.

Date: January 19, 2026
Status: COMPLETE AND VALIDATED
"""

if __name__ == "__main__":
    print(__doc__)
