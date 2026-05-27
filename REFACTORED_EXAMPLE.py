"""
Example: How the Refactored Code Would Look
=============================================

BEFORE (Current - Everything in analysis.py):
---------------------------------------------
# File: analysis.py (3,917 lines)
from pathlib import Path
import pandas as pd
import numpy as np
# ... 20+ more imports

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v6_2026_02_10.xlsx"

class Tee:
    # logging code...

def read_data(excel_path):
    # 30 lines of data loading...

def perform_group_analysis(data_df, dv_col, group_col):
    # 100 lines of statistics...

def plot_vectors_rel_sternum(df_ave, color_by='breast', ...):
    # 440 lines of plotting...

# ... 15 more functions ...

# Main execution code at bottom (700 lines)
if __name__ == '__main__':
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)
    # ... 600+ more lines ...


AFTER (Refactored - Modular Structure):
----------------------------------------

# File: analysis/config.py (20 lines)
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v6_2026_02_10.xlsx"
LOG_FILE_PATH = OUTPUT_DIR / f"analysis_output_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.txt"
ALPHA = 0.05


# File: analysis/utils/logging.py (30 lines)
import sys

class Tee:
    '''Write output to both console and file simultaneously.'''
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    # ... rest of class


# File: analysis/data/loader.py (40 lines)
import pandas as pd

def read_data(excel_path):
    '''Load all sheets from the Excel file.'''
    all_sheets = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl', header=0)
    return all_sheets['raw_data'], all_sheets['processed_ave_data'], all_sheets['demographic']


# File: analysis/statistics/group_analysis.py (150 lines)
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_group_analysis(data_df, dv_col, group_col):
    '''Performs assumption checks (Normality, Levene's) and runs ANOVA/Kruskal-Wallis.'''
    # ... 100 lines of clean, focused statistics code ...


def perform_two_group_analysis(data_df, dv_col, group_col):
    '''Performs T-test or Mann-Whitney U test for two groups.'''
    # ... 50 lines of clean, focused statistics code ...


# File: analysis/plotting/vector_plots.py (500 lines)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_vectors_rel_sternum(df_ave, color_by='breast', vl_id=None, data_type='landmarks',
                             include_dual_sagittal=False):
    '''Main function for plotting displacement vectors relative to sternum.'''
    # ... 400 lines of clean plotting code ...

def _plot_dual_sagittal_view_sternum(...):
    '''Helper function for dual sagittal view - not exposed to users.'''
    # ... 100 lines of helper code ...


# File: analysis/main.py (100 lines - clean and readable!)
from analysis.config import EXCEL_FILE_PATH, LOG_FILE_PATH
from analysis.utils.logging import Tee
from analysis.data.loader import read_data
from analysis.statistics.group_analysis import perform_group_analysis
from analysis.plotting.vector_plots import plot_vectors_rel_sternum
from analysis.statistics.clock_position import analyse_clock_position_rotation
from analysis.plotting.anatomical_views import plot_3panel_anatomical_views

def main():
    '''Main analysis pipeline.'''
    # Setup logging
    tee = Tee(LOG_FILE_PATH)

    # Load data
    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)

    # Statistical analyses
    perform_group_analysis(df_ave, 'Landmark displacement [mm]', 'Landmark type')

    # Visualizations
    plot_vectors_rel_sternum(df_ave, color_by='breast', data_type='landmarks',
                            include_dual_sagittal=True)

    # Clock position analysis
    analyse_clock_position_rotation(df_ave)

    # Close logging
    tee.close()

if __name__ == '__main__':
    main()


USAGE EXAMPLES:
===============

# Example 1: Just load data
from analysis.data.loader import read_data
df_raw, df_ave, df_demo = read_data("../output/landmark_results_v6_2026_02_10.xlsx")

# Example 2: Run specific statistical test
from analysis.statistics.group_analysis import perform_group_analysis
perform_group_analysis(df_ave, 'Landmark displacement [mm]', 'Landmark type')

# Example 3: Create specific plot
from analysis.plotting.vector_plots import plot_vectors_rel_sternum
plot_vectors_rel_sternum(df_ave, color_by='dts', data_type='landmarks')

# Example 4: Run full pipeline
from analysis.main import main
main()

# Example 5: Use in Jupyter notebook
import sys
sys.path.append('../scripts')
from analysis.data.loader import read_data
from analysis.plotting.vector_plots import plot_vectors_rel_sternum

df_raw, df_ave, df_demo = read_data("../output/landmark_results_v6_2026_02_10.xlsx")
plot_vectors_rel_sternum(df_ave, color_by='subject')


BENEFITS DEMONSTRATED:
======================

✅ Clear imports - know exactly where each function comes from
✅ Reusable - can import just what you need
✅ Testable - each module can be tested independently
✅ Maintainable - easy to find and update functions
✅ Documented - clear module structure
✅ Collaborative - multiple people can work on different modules
✅ IDE-friendly - better autocomplete and navigation

"""

