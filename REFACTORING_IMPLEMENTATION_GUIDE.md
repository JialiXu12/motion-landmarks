"""
Step-by-Step Implementation Guide for Refactoring analysis.py
==============================================================

PHASE 1: SETUP (30 minutes)
============================

Step 1: Create Directory Structure
-----------------------------------
Run these commands in terminal (from scripts/ directory):

mkdir -p analysis
mkdir -p analysis/data
mkdir -p analysis/statistics  
mkdir -p analysis/plotting
mkdir -p analysis/utils

# Create __init__.py files
touch analysis/__init__.py
touch analysis/data/__init__.py
touch analysis/statistics/__init__.py
touch analysis/plotting/__init__.py
touch analysis/utils/__init__.py

Step 2: Create Config File
---------------------------
File: scripts/analysis/config.py
- Copy lines 50-55 from analysis.py (paths and constants)
- Add any other configuration variables

Step 3: Backup Original
-----------------------
cp analysis.py analysis_original_backup.py


PHASE 2: MOVE UTILITIES (1 hour)
=================================

Step 4: Move Logging Class
---------------------------
File: scripts/analysis/utils/logging.py

1. Copy lines 34-48 from analysis.py (Tee class)
2. Add imports: import sys
3. Test:
   from analysis.utils.logging import Tee
   tee = Tee("test.log")
   print("Hello")
   tee.close()

Step 5: Move Data Loader
-------------------------
File: scripts/analysis/data/loader.py

1. Copy lines 58-75 from analysis.py (read_data function)
2. Add imports: import pandas as pd
3. Test:
   from analysis.data.loader import read_data
   df_raw, df_ave, df_demo = read_data("../output/landmark_results_v6_2026_02_10.xlsx")
   print(df_ave.shape)


PHASE 3: MOVE STATISTICS (2 hours)
===================================

Step 6: Move Group Analysis
----------------------------
File: scripts/analysis/statistics/group_analysis.py

1. Copy perform_group_analysis() [lines 77-176]
2. Copy perform_two_group_analysis() [lines 178-261]
3. Add all necessary imports
4. Test each function independently

Step 7: Move Repeated Measures
-------------------------------
File: scripts/analysis/statistics/repeated_measures.py

1. Copy perform_repeated_measures_analysis() [lines 263-419]
2. Add all necessary imports
3. Test with sample data

Step 8: Move Correlation Analysis
----------------------------------
File: scripts/analysis/statistics/correlations.py

1. Copy plot_bmi_correlations() [lines 421-474]
2. Copy investigate_proximity_effect() [lines 476-503]
3. Copy plot_anatomical_correlation_matrix() [lines 3011+]
4. Import test_partial_correlation from partial_correlation.py

Step 9: Move Clock Analysis
----------------------------
File: scripts/analysis/statistics/clock_position.py

1. Copy parse_clock_time() [line 2061]
2. Copy circular_mean_angle() [line 2093]
3. Copy calculate_clock_position() [line 2117]
4. Copy analyse_clock_position_rotation() [line 2153]


PHASE 4: MOVE PLOTTING (2 hours)
=================================

Step 10: Move Vector Plots
---------------------------
File: scripts/analysis/plotting/vector_plots.py

1. Copy plot_vectors_for_vl81() [line 505]
2. Copy plot_vectors_rel_sternum() [line 634]
3. Copy _plot_dual_sagittal_view_sternum() [line 1073]
4. Copy plot_nipple_relative_landmarks() [line 1893]
5. Test each plot function

Step 11: Move Anatomical Views
-------------------------------
File: scripts/analysis/plotting/anatomical_views.py

1. Copy plot_3panel_displacement_mechanism() [line 1256]
2. Copy plot_3panel_anatomical_views() [line 2772]


PHASE 5: CREATE MAIN (1 hour)
==============================

Step 12: Create Main Execution File
------------------------------------
File: scripts/analysis/main.py

1. Import all functions from submodules
2. Copy main execution logic from bottom of analysis.py [lines 3200+]
3. Wrap in main() function
4. Add if __name__ == '__main__': main()

Step 13: Update __init__.py Files
----------------------------------
Make key functions easily importable:

# analysis/__init__.py
from .data.loader import read_data
from .statistics.group_analysis import perform_group_analysis
from .plotting.vector_plots import plot_vectors_rel_sternum
# ... etc

Step 14: Test Full Pipeline
----------------------------
python analysis/main.py

Compare output with original analysis.py


PHASE 6: CLEANUP (1 hour)
==========================

Step 15: Update External References
------------------------------------
Check if any other scripts import from analysis.py:
- Update imports to use new module structure
- Test each dependent script

Step 16: Archive Original
--------------------------
mv analysis.py analysis_legacy.py
# Keep as reference but don't use

Step 17: Update Documentation
------------------------------
- Update README.md with new structure
- Add docstrings to each module
- Create usage examples


TESTING CHECKLIST:
==================

After each phase, verify:
□ Imports work correctly
□ Functions execute without errors
□ Output matches original
□ No circular dependencies
□ All paths resolve correctly

Final Integration Test:
□ Run full pipeline on all subjects
□ Compare all outputs with original
□ Check all plots are generated
□ Verify all statistics match
□ Ensure logging works


ROLLBACK PLAN:
==============

If something goes wrong:
1. Revert to analysis_original_backup.py
2. Delete analysis/ directory
3. Debug issues offline
4. Try again with fixes


COMMON ISSUES & SOLUTIONS:
==========================

Issue: ImportError: cannot import name 'X'
Solution: Check __init__.py files, verify function names

Issue: FileNotFoundError for output paths
Solution: Update paths in config.py, use Path().resolve()

Issue: Circular imports
Solution: Reorganize dependencies, use lazy imports

Issue: Tests pass individually but fail together
Solution: Check for global state, initialize properly

Issue: Plots look different
Solution: Check matplotlib backend, figure size settings


TIME ESTIMATES:
===============

- Phase 1 (Setup): 30 minutes
- Phase 2 (Utilities): 1 hour
- Phase 3 (Statistics): 2 hours
- Phase 4 (Plotting): 2 hours
- Phase 5 (Main): 1 hour
- Phase 6 (Cleanup): 1 hour

Total: 7.5 hours

With testing and debugging: 8-10 hours


RECOMMENDATION:
===============

Do this refactoring in stages over 2-3 days:

Day 1: Phases 1-2 (Setup + Utilities)
Day 2: Phases 3-4 (Statistics + Plotting)
Day 3: Phases 5-6 (Main + Cleanup)

This allows time for testing between phases and reduces risk.
"""

