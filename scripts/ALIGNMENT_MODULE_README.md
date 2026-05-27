# Alignment Module Documentation

## Overview

The `alignment.py` module provides an optimal sternum-fixed alignment method for prone-to-supine breast MRI registration. This method guarantees that the sternum superior landmark remains fixed during alignment, providing a stable anatomical reference point.

## Key Features

1. **Sternum Superior Fixed at Origin**: Mathematically guaranteed zero drift
2. **Optimal Rotation**: Uses SVD (Singular Value Decomposition) for closed-form optimal rotation
3. **Robust Correspondence**: Trimmed ICP with outlier rejection
4. **Compatible Interface**: Drop-in replacement for existing alignment functions

## Usage in main.py

### Method 1: Replace existing alignment function

Replace this import:
```python
from align_fixed_sternum import align_prone_to_supine_fixed_sternum
```

With:
```python
from alignment import align_prone_to_supine_optimal
```

Replace this function call:
```python
alignment_results = align_prone_to_supine_fixed_sternum(
    subject=filtered_subject,
    prone_ribcage_mesh_path=prone_mesh_file,
    supine_ribcage_seg_path=supine_seg_file,
    orientation_flag='RAI',
    plot_for_debug=True
)
```

With:
```python
alignment_results = align_prone_to_supine_optimal(
    subject=filtered_subject,
    prone_ribcage_mesh_path=prone_mesh_file,
    supine_ribcage_seg_path=supine_seg_file,
    orientation_flag='RAI',
    plot_for_debug=True
)
```

### Method 2: Use as alternative alignment method

You can also keep both methods and switch between them:

```python
from alignment import align_prone_to_supine_optimal
from align_fixed_sternum import align_prone_to_supine_fixed_sternum

# Choose which method to use
USE_OPTIMAL_METHOD = True  # Set to False to use the original method

if USE_OPTIMAL_METHOD:
    alignment_results = align_prone_to_supine_optimal(...)
else:
    alignment_results = align_prone_to_supine_fixed_sternum(...)
```

## Function Parameters

### Required Parameters

- `subject`: Subject object containing scan data and anatomical landmarks
- `prone_ribcage_mesh_path`: Path to prone ribcage mesh file (.mesh)
- `supine_ribcage_seg_path`: Path to supine ribcage segmentation (.nii.gz)

### Optional Parameters

- `orientation_flag`: Image orientation (default: 'RAI')
- `plot_for_debug`: Show debug visualization plots (default: False)
- `max_correspondence_distance`: Maximum distance for ICP correspondences in mm (default: 15.0)
- `max_iterations`: Maximum ICP iterations (default: 100)
- `verbose`: Print progress information (default: True)

## Return Value

The function returns a dictionary containing:

### Transformation
- `T_total`: (4, 4) homogeneous transformation matrix
- `R`: (3, 3) rotation matrix

### Error Metrics
- `ribcage_error_mean`: Mean ribcage alignment error (mm)
- `ribcage_error_std`: Standard deviation of ribcage alignment error (mm)
- `ribcage_inlier_rmse`: RMSE of inlier points (mm)
- `sternum_error`: Sternum alignment error (should be ~0 mm)

### Transformed Anatomical Landmarks
- `sternum_prone_transformed`: Transformed prone sternum positions
- `sternum_supine`: Supine sternum positions
- `nipple_prone_transformed`: Transformed prone nipple positions
- `nipple_supine`: Supine nipple positions

### Landmark Data (Average Registrar)
- `ld_ave_prone_transformed`: Transformed prone landmark positions
- `ld_ave_supine`: Supine landmark positions
- `ld_ave_displacement_magnitudes`: Landmark displacement magnitudes
- `ld_ave_displacement_vectors`: Landmark displacement vectors

### Positions Relative to Sternum
- `ld_ave_prone_rel_sternum`: Prone landmarks relative to sternum
- `ld_ave_supine_rel_sternum`: Supine landmarks relative to sternum
- `nipple_prone_rel_sternum`: Prone nipples relative to sternum
- `nipple_supine_rel_sternum`: Supine nipples relative to sternum

### Nipple Displacements
- `nipple_displacement_magnitudes`: Nipple displacement magnitudes
- `nipple_displacement_vectors`: Nipple displacement vectors
- `nipple_disp_left_vec`: Left nipple displacement vector
- `nipple_disp_right_vec`: Right nipple displacement vector

### Relative to Nipple
- `ld_ave_displacement_rel_nipple`: Landmark displacement relative to nipple
- `ld_ave_displacement_mag_rel_nipple`: Magnitude of displacement relative to nipple
- `is_left_breast`: Boolean array indicating which breast each landmark belongs to

### Algorithm Information
- `alignment_info`: Detailed alignment algorithm information
- `method`: Alignment method name ('optimal_sternum_fixed_svd')

## Algorithm Details

### Step 1: Data Loading
- Loads prone ribcage mesh and supine ribcage point cloud
- Extracts anatomical landmarks (sternum, nipples)
- Extracts registrar landmarks

### Step 2: Centering
- Centers both prone and supine data on their respective sternum superior positions
- This moves sternum superior to the origin (0, 0, 0)

### Step 3: Iterative Alignment
1. Find nearest neighbor correspondences between prone and supine ribcages
2. Filter correspondences by distance threshold
3. Apply trimmed ICP (reject worst correspondences)
4. Compute optimal rotation using SVD (Kabsch algorithm)
5. Apply rotation around origin (sternum stays at 0,0,0)
6. Check convergence
7. Repeat until convergence or max iterations

### Step 4: Transformation
- Apply final rotation to all prone data
- Translate back to supine coordinate frame

### Step 5: Displacement Calculation
- Calculate displacements relative to sternum
- Calculate displacements relative to nipple
- Separate data by breast side (left/right)

## Mathematical Guarantee

Because all coordinates are centered on sternum superior before alignment:

1. Sternum superior in prone: (0, 0, 0)
2. Sternum superior in supine: (0, 0, 0)
3. Any rotation R around origin: R × [0, 0, 0]ᵀ = [0, 0, 0]ᵀ

Therefore, sternum superior **cannot drift** - it's mathematically fixed at the origin.

## Comparison with Other Methods

| Feature | align_prone_to_supine | align_prone_to_supine_fixed_sternum | align_prone_to_supine_optimal |
|---------|----------------------|-------------------------------------|-------------------------------|
| Method | Open3D ICP | scipy minimize | SVD (Kabsch) |
| Sternum Fixed | No | Soft constraint (weight) | Hard constraint (mathematical) |
| Optimization | Point-to-plane | Combined objective | Point-to-point |
| Global Optimum | No | No | Yes (for rotation) |
| Sternum Drift | Possible | Minimal | **Zero** |
| Robustness | Good | Good | Excellent |
| Speed | Fast | Medium | Fast |

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the correct directory with the virtual environment activated:
```bash
cd C:\Users\jxu759\Documents\motion-landmarks\scripts
..\venv1\Scripts\activate
python main.py
```

### Alignment Fails
If alignment fails for a subject:
- Check that sternum landmarks exist in both prone and supine
- Verify ribcage mesh and segmentation files exist
- Try adjusting `max_correspondence_distance` (increase for larger subjects)
- Try increasing `max_iterations`

### High Sternum Error
If `sternum_error` is not near zero (> 0.001 mm), this indicates a bug. Please report this issue.

## Testing

To test the alignment module:
```bash
python test_alignment_module.py
```

This will verify that the module can be imported and has the correct interface.

## Questions or Issues

If you encounter any problems or have questions about the alignment module, please check:
1. This documentation
2. The inline code comments in `alignment.py`
3. The test script `test_alignment_module.py`
