
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Import from analysis
try:
    import analysis
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    import analysis

# Patch plt.show to save instead
original_show = plt.show
def save_and_show():
    plt.savefig("sagittal_dual_plot.png", dpi=300)
    print("Plot saved to sagittal_dual_plot.png")
    # original_show() # Don't show to avoid blocking execution

plt.show = save_and_show

if __name__ == "__main__":
    print("Loading data...")
    try:
        df_raw, df_ave, df_demo = analysis.read_data(analysis.EXCEL_FILE_PATH)
        print("Running plot_sagittal_dual_axes from analysis.py...")
        analysis.plot_vectors_rel_sternum(df_ave)
    except Exception as e:
        print(f"Error: {e}")
