# Project Structure Improvement Recommendations

## Executive Summary

After analyzing the current project structure, I've identified several areas for improvement to make the codebase more maintainable, testable, and scalable.

---

## Current Structure Analysis

### Current File Organization
```
motion-landmarks/
├── scripts/              # 50+ files, mixing different concerns
│   ├── utils.py         # 1873 lines, 18 functions
│   ├── utils_plot.py    # Plotting utilities
│   ├── analysis.py      # Main analysis script
│   ├── readers.py       # Data loading
│   ├── structures.py    # Data structures
│   ├── align_fixed_sternum.py  # Alignment method
│   ├── main.py         # Entry point
│   └── 40+ other files (tests, plots, backups, etc.)
├── src/                 # morphic library
├── external/            # External dependencies
├── output/              # Results
└── docs/                # Documentation
```

### Key Issues Identified

1. **`utils.py` is a "God Module"** (1873 lines)
   - Contains 18 unrelated functions
   - Mixes alignment, data processing, distance calculations, I/O
   - Hard to test individual components
   - Difficult to navigate

2. **scripts/ folder is cluttered** (50+ files)
   - No clear organization
   - Mix of production code, tests, backups, plots
   - Hard to find specific functionality

3. **Tight coupling**
   - Alignment function contains displacement calculations
   - Hard to reuse components independently

4. **Missing module boundaries**
   - No clear separation of concerns
   - Functions not grouped by purpose

---

## 📋 Recommended Improvements

### 1. **Refactor `utils.py` into Focused Modules**

#### Current: Monolithic `utils.py` (1873 lines, 18 functions)

**Proposed Split**:

```
scripts/
├── alignment/
│   ├── __init__.py
│   ├── transformation.py        # apply_transform, rotation matrices
│   ├── alignment_original.py    # Original align_prone_to_supine
│   ├── alignment_fixed_sternum.py  # Fixed sternum method
│   ├── icp.py                   # run_point_to_plane_icp, compute_icp_metrics
│   └── point_cloud.py           # extract_contour_points, filter_point_cloud_asymmetric
│
├── displacement/
│   ├── __init__.py
│   ├── calculator.py            # calculate_displacements (NEW)
│   ├── relative_sternum.py      # Displacement relative to sternum
│   └── relative_nipple.py       # Displacement relative to nipple
│
├── data/
│   ├── __init__.py
│   ├── readers.py               # (existing) load_all_subjects
│   ├── structures.py            # (existing) Subject, ScanData
│   ├── correspondence.py        # find_corresponding_landmarks, add_averaged_landmarks
│   ├── extractors.py            # get_landmarks_as_array
│   └── io.py                    # save_results_to_excel, save_raw_data_to_excel
│
├── distance/
│   ├── __init__.py
│   ├── calculator.py            # calculate_landmark_distances, calculate_distance
│   ├── analyzer.py              # analyse_landmark_distances
│   └── clockface.py             # calculate_clockface_coordinates
│
├── visualization/
│   ├── __init__.py
│   ├── utils_plot.py            # (existing) plot_all, plot_vector_three_views
│   ├── alignment_viz.py         # plot_evaluate_alignment
│   └── analysis_plots.py        # Plots from analysis.py
│
├── analysis/
│   ├── __init__.py
│   ├── analysis.py              # (existing) main analysis script
│   ├── partial_correlation.py   # (existing)
│   └── statistical_tests.py     # Statistical analysis functions
│
├── utils/
│   ├── __init__.py
│   ├── geometry.py              # calculate_distance, vector operations
│   └── subject_utils.py         # copy_subject
│
└── main.py                       # Entry point (unchanged)
```

---

### 2. **Separate Alignment from Displacement Calculation**

#### Current Problem:
```python
def align_prone_to_supine_fixed_sternum(...):
    # ... 700 lines of code ...
    
    # PHASE 6: CALCULATE DISPLACEMENTS (should be separate!)
    lm_pos_prone_rel_sternum = prone_landmarks_aligned - ref_sternum_prone
    lm_pos_supine_rel_sternum = landmark_supine_ave_raw - ref_sternum_supine
    # ... 50 more lines of displacement calculations ...
    
    return results  # Contains both alignment AND displacement data
```

#### Proposed Solution:

**File: `scripts/alignment/alignment_fixed_sternum.py`**
```python
def align_prone_to_supine_fixed_sternum(
    subject: Subject,
    prone_ribcage_mesh_path: Path,
    supine_ribcage_seg_path: Path,
    ...
) -> AlignmentResult:
    """
    Align prone to supine with sternum superior locked.
    
    Returns ONLY alignment information:
    - Transformation matrix
    - Rotation matrix
    - Aligned coordinates (in centered coordinate system)
    - Alignment quality metrics
    """
    # ... alignment code (phases 1-5) ...
    
    return AlignmentResult(
        vl_id=subject.subject_id,
        T_total=T_total,
        R_total=R_total,
        anchor_prone=anchor_prone,
        anchor_supine=anchor_supine,
        
        # Aligned coordinates (CENTERED on sternum superior = origin)
        prone_sternum_centered_aligned=prone_sternum_final,
        prone_nipple_centered_aligned=prone_nipple_final,
        prone_landmarks_centered_aligned=prone_landmarks_final,
        
        # Supine coordinates (CENTERED on sternum superior = origin)
        supine_sternum_centered=supine_sternum_centered,
        supine_nipple_centered=supine_nipple_centered,
        supine_landmarks_centered=supine_landmarks_centered,
        
        # Quality metrics
        sternum_error=sternum_error_final,
        ribcage_error_mean=np.mean(distances_final),
        ribcage_error_std=np.std(distances_final),
        ribcage_inlier_RMSE=icp_info['inlier_rmse'],
        
        method="fixed_sternum_rotation_only"
    )
```

**File: `scripts/displacement/calculator.py`**
```python
def calculate_displacements(
    alignment_result: AlignmentResult,
    reference_frame: str = 'sternum'  # 'sternum' or 'nipple'
) -> DisplacementResult:
    """
    Calculate landmark displacements from alignment result.
    
    All coordinates are in CENTERED coordinate system (sternum superior = origin).
    This is the TRUE displacement without needing to transform back.
    
    Args:
        alignment_result: Result from alignment function
        reference_frame: 'sternum' (default) or 'nipple'
        
    Returns:
        DisplacementResult with displacement vectors and magnitudes
    """
    # Extract centered coordinates (sternum superior already at origin)
    prone_landmarks = alignment_result.prone_landmarks_centered_aligned
    supine_landmarks = alignment_result.supine_landmarks_centered
    prone_nipples = alignment_result.prone_nipple_centered_aligned
    supine_nipples = alignment_result.supine_nipple_centered
    
    # Calculate displacements (already relative to sternum = origin)
    landmark_displacement_vectors = supine_landmarks - prone_landmarks
    landmark_displacement_magnitudes = np.linalg.norm(landmark_displacement_vectors, axis=1)
    
    nipple_displacement_vectors = supine_nipples - prone_nipples
    nipple_displacement_magnitudes = np.linalg.norm(nipple_displacement_vectors, axis=1)
    
    # Calculate nipple-relative displacements
    if reference_frame == 'nipple':
        # Assign each landmark to nearest nipple
        dist_to_left = np.linalg.norm(supine_landmarks - supine_nipples[0], axis=1)
        dist_to_right = np.linalg.norm(supine_landmarks - supine_nipples[1], axis=1)
        is_left_breast = dist_to_left < dist_to_right
        
        # Get corresponding nipple displacement for each landmark
        closest_nipple_disp = np.where(
            is_left_breast[:, np.newaxis],
            nipple_displacement_vectors[0],
            nipple_displacement_vectors[1]
        )
        
        # Displacement relative to nipple = landmark displacement - nipple displacement
        landmark_rel_nipple_vectors = landmark_displacement_vectors - closest_nipple_disp
        landmark_rel_nipple_magnitudes = np.linalg.norm(landmark_rel_nipple_vectors, axis=1)
    else:
        landmark_rel_nipple_vectors = None
        landmark_rel_nipple_magnitudes = None
        is_left_breast = None
    
    return DisplacementResult(
        vl_id=alignment_result.vl_id,
        
        # Absolute displacements (relative to sternum at origin)
        landmark_displacement_vectors=landmark_displacement_vectors,
        landmark_displacement_magnitudes=landmark_displacement_magnitudes,
        nipple_displacement_vectors=nipple_displacement_vectors,
        nipple_displacement_magnitudes=nipple_displacement_magnitudes,
        
        # Nipple-relative (if requested)
        landmark_rel_nipple_vectors=landmark_rel_nipple_vectors,
        landmark_rel_nipple_magnitudes=landmark_rel_nipple_magnitudes,
        is_left_breast=is_left_breast,
        
        reference_frame=reference_frame
    )
```

**Usage Example**:
```python
# Step 1: Align
alignment_result = align_prone_to_supine_fixed_sternum(
    subject, prone_mesh_path, supine_seg_path
)

# Step 2: Calculate displacements (sternum = origin)
displacements_sternum = calculate_displacements(
    alignment_result, 
    reference_frame='sternum'
)

# Step 3: Calculate nipple-relative displacements (optional)
displacements_nipple = calculate_displacements(
    alignment_result,
    reference_frame='nipple'
)

# Coordinates are already centered - no need to transform back!
print(f"Landmark displacement: {displacements_sternum.landmark_displacement_magnitudes}")
```

---

### 3. **Organize `scripts/` Folder**

#### Current State:
- 50+ files in flat structure
- Mix of production, test, backup, output files
- Hard to navigate

#### Proposed Structure:

```
scripts/
├── alignment/              # Alignment algorithms
│   ├── __init__.py
│   ├── transformation.py
│   ├── alignment_original.py
│   ├── alignment_fixed_sternum.py
│   ├── icp.py
│   └── point_cloud.py
│
├── displacement/           # Displacement calculations
│   ├── __init__.py
│   ├── calculator.py
│   ├── relative_sternum.py
│   └── relative_nipple.py
│
├── data/                   # Data loading and structures
│   ├── __init__.py
│   ├── readers.py
│   ├── structures.py
│   ├── correspondence.py
│   ├── extractors.py
│   └── io.py
│
├── distance/               # Distance calculations
│   ├── __init__.py
│   ├── calculator.py
│   ├── analyzer.py
│   └── clockface.py
│
├── visualization/          # All plotting functions
│   ├── __init__.py
│   ├── utils_plot.py
│   ├── alignment_viz.py
│   └── analysis_plots.py
│
├── analysis/               # Analysis scripts
│   ├── __init__.py
│   ├── analysis.py
│   ├── partial_correlation.py
│   └── statistical_tests.py
│
├── utils/                  # Generic utilities
│   ├── __init__.py
│   ├── geometry.py
│   └── subject_utils.py
│
├── tests/                  # ALL test files
│   ├── __init__.py
│   ├── test_alignment_validation.py
│   ├── test_fixed_sternum_standalone.py
│   ├── test_icp_standalone.py
│   └── ...
│
├── deprecated/             # OLD/backup files
│   ├── analysis_backup_20260123_232525.py
│   ├── constrained_icp_fixed_sternum.py
│   └── ...
│
└── main.py                 # Entry point
```

---

### 4. **Create Data Structure Classes**

#### Current: Dictionary-based returns
```python
results = {
    "vl_id": subject.subject_id,
    "T_total": T_total,
    "sternum_error": sternum_error,
    # ... 30 more key-value pairs ...
}
```

#### Proposed: Typed dataclasses

**File: `scripts/data/result_types.py`**
```python
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class AlignmentResult:
    """Result from prone-to-supine alignment (centered coordinates)"""
    vl_id: int
    T_total: np.ndarray  # 4x4 transformation matrix
    R_total: np.ndarray  # 3x3 rotation matrix
    anchor_prone: np.ndarray  # Original prone sternum superior
    anchor_supine: np.ndarray  # Original supine sternum superior
    
    # Aligned coordinates (CENTERED: sternum superior = origin)
    prone_sternum_centered_aligned: np.ndarray  # (2, 3)
    prone_nipple_centered_aligned: np.ndarray  # (2, 3)
    prone_landmarks_centered_aligned: np.ndarray  # (N, 3)
    
    # Target coordinates (CENTERED: sternum superior = origin)
    supine_sternum_centered: np.ndarray  # (2, 3)
    supine_nipple_centered: np.ndarray  # (2, 3)
    supine_landmarks_centered: np.ndarray  # (N, 3)
    
    # Quality metrics
    sternum_error: np.ndarray  # (2,) - superior and inferior
    ribcage_error_mean: float
    ribcage_error_std: float
    ribcage_inlier_RMSE: float
    
    method: str  # 'fixed_sternum_rotation_only' or 'original'
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility"""
        return {...}


@dataclass
class DisplacementResult:
    """Result from displacement calculation"""
    vl_id: int
    
    # Displacements (relative to sternum at origin)
    landmark_displacement_vectors: np.ndarray  # (N, 3)
    landmark_displacement_magnitudes: np.ndarray  # (N,)
    nipple_displacement_vectors: np.ndarray  # (2, 3)
    nipple_displacement_magnitudes: np.ndarray  # (2,)
    
    # Nipple-relative (optional)
    landmark_rel_nipple_vectors: Optional[np.ndarray]  # (N, 3)
    landmark_rel_nipple_magnitudes: Optional[np.ndarray]  # (N,)
    is_left_breast: Optional[np.ndarray]  # (N,) boolean
    
    reference_frame: str  # 'sternum' or 'nipple'
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Excel export"""
        return {...}
```

**Benefits**:
- Type hints for better IDE support
- Clear interface contracts
- Easy to extend
- Validation in constructors
- Easy conversion to dict for Excel export

---

### 5. **Module Import Structure**

**File: `scripts/alignment/__init__.py`**
```python
"""
Alignment module for prone-to-supine registration.
"""
from .alignment_original import align_prone_to_supine
from .alignment_fixed_sternum import align_prone_to_supine_fixed_sternum
from .transformation import apply_transform, rotation_matrix_from_euler
from .icp import run_point_to_plane_icp

__all__ = [
    'align_prone_to_supine',
    'align_prone_to_supine_fixed_sternum',
    'apply_transform',
    'rotation_matrix_from_euler',
    'run_point_to_plane_icp'
]
```

**File: `scripts/displacement/__init__.py`**
```python
"""
Displacement calculation module.
"""
from .calculator import calculate_displacements

__all__ = ['calculate_displacements']
```

**Usage**:
```python
# Clean imports
from alignment import align_prone_to_supine_fixed_sternum
from displacement import calculate_displacements
from data import load_all_subjects
from visualization import plot_all
```

---

### 6. **Configuration Management**

**File: `scripts/config.py`**
```python
"""
Central configuration for alignment and analysis parameters.
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AlignmentConfig:
    """Configuration for alignment algorithms"""
    # ICP parameters
    max_correspondence_distance: float = 10.0
    max_iterations: int = 200
    huber_delta: float = 2.0
    convergence_threshold: float = 1e-6
    
    # Weights
    w_rib: float = 1.0
    w_sternum: float = 100.0
    
    # Paths
    segmentation_root: Path = Path(r"U:\sandbox\jxu759\volunteer_seg")
    mesh_root: Path = Path("../output/transformation_matrix_v5")
    
    # Debugging
    plot_for_debug: bool = False
    verbose: bool = True


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    alpha: float = 0.05  # Significance level
    min_landmarks: int = 3
    output_dir: Path = Path("../output")
```

---

## 📊 Benefits of Proposed Structure

### Maintainability
- ✅ **Single Responsibility**: Each module does one thing
- ✅ **Clear Organization**: Easy to find functionality
- ✅ **Smaller Files**: Easier to understand and modify
- ✅ **Typed Interfaces**: Clear contracts between modules

### Testability
- ✅ **Isolated Components**: Test alignment without displacement
- ✅ **Mock-Friendly**: Easy to mock dependencies
- ✅ **Organized Tests**: All tests in one folder

### Reusability
- ✅ **Modular Design**: Use alignment without displacement
- ✅ **Clear APIs**: Well-defined inputs and outputs
- ✅ **Composable**: Mix and match components

### Scalability
- ✅ **Easy to Extend**: Add new alignment methods easily
- ✅ **Parallel Development**: Multiple people can work on different modules
- ✅ **Version Control**: Smaller diffs, easier merges

---

## 🔄 Migration Strategy

### Phase 1: Create New Structure (No Breaking Changes)
1. Create new folder structure
2. Copy functions to new modules
3. Keep old `utils.py` as compatibility layer
4. Update imports gradually

### Phase 2: Refactor Alignment
1. Separate alignment from displacement
2. Create `AlignmentResult` dataclass
3. Update alignment functions
4. Add tests

### Phase 3: Refactor Displacement
1. Create `calculate_displacements` function
2. Update to use centered coordinates
3. Create `DisplacementResult` dataclass
4. Add tests

### Phase 4: Update Main Scripts
1. Update `main.py`
2. Update `analysis.py`
3. Update tests
4. Remove old `utils.py`

### Phase 5: Clean Up
1. Move old files to `deprecated/`
2. Update documentation
3. Final testing

---

## 📝 Immediate Action Items (For Your Request)

### Request 1: Keep Centered Coordinates
✅ **Already identified** - Change Phase 5 to NOT uncenter
```python
# Current (WRONG - transforms back to absolute):
prone_sternum_aligned = uncenter_from_anchor(prone_sternum_final, anchor_supine)

# Proposed (CORRECT - keep centered):
prone_sternum_aligned = prone_sternum_final  # Already centered at origin
supine_sternum = supine_sternum_centered  # Already centered at origin

# Displacement is now truly relative to sternum (at origin):
displacement = supine_landmarks_centered - prone_landmarks_centered_aligned
```

### Request 2: Separate Alignment and Displacement
✅ **Already designed** - See section 2 above

### Request 3: Project Structure Improvements
✅ **Already documented** - See entire document above

---

## 📈 Impact Assessment

### File Count Reduction
- **Before**: 50+ files in flat `scripts/` folder
- **After**: ~25 organized files in 7 modules + tests folder

### Code Organization
- **Before**: 1873-line `utils.py` with 18 functions
- **After**: 7 focused modules, ~200-300 lines each

### Function Clarity
- **Before**: Alignment returns 30+ dictionary keys
- **After**: Typed `AlignmentResult` with clear fields

### Testability
- **Before**: Hard to test alignment without displacement
- **After**: Completely independent, easily testable

---

## 🎯 Recommendation Priority

### High Priority (Do First)
1. ✅ **Separate alignment from displacement** (Request #2)
2. ✅ **Keep centered coordinates** (Request #1)
3. Create `AlignmentResult` and `DisplacementResult` dataclasses

### Medium Priority (Do Next)
4. Split `utils.py` into alignment/, displacement/, data/
5. Create module-level `__init__.py` files
6. Move tests to `tests/` folder

### Low Priority (Nice to Have)
7. Create configuration management
8. Move deprecated files
9. Update documentation

---

## 🚀 Next Steps

1. **Review this document** - Provide feedback on proposed structure
2. **Prioritize changes** - Which improvements are most important?
3. **Start refactoring** - Begin with high-priority items
4. **Test thoroughly** - Ensure no regressions
5. **Update documentation** - Keep README.md current

---

**Author**: Analysis Team  
**Date**: February 3, 2026  
**Status**: 📋 RECOMMENDATIONS READY FOR REVIEW
