# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys
from datetime import datetime

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.patches import Circle, Arc
import matplotlib.patches as patches

# Statistical analysis
import pingouin as pg
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from pingouin import welch_anova, pairwise_gameshowell, rm_anova, pairwise_ttests

from plot_nipple_relative_vectors import plot_nipple_relative_vectors

from partial_correlation import test_partial_correlation


# ------------------------------------------------------------------------------
# LOGGING SETUP - Automatically save terminal output to file
# ------------------------------------------------------------------------------
class Tee:
    """Write output to both console and file simultaneously."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v5_2026_01_21.xlsx"
LOG_FILE_PATH = OUTPUT_DIR / f"analysis_output_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.txt"


def read_data(excel_path):
    try:
        # Reads ALL sheets into an OrderedDict where keys are sheet names and values are DataFrames
        all_sheets = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl',header=0)

        # Example: Accessing the 'processed_data' sheet
        df_raw = all_sheets['raw_data']
        df_ave = all_sheets['processed_ave_data']
        df_demo = all_sheets['demographic']
        print(f"Successfully loaded {len(all_sheets)} sheets.")

    except FileNotFoundError:
        print(f"Error: The file {excel_path} was not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

    return df_raw, df_ave, df_demo


def perform_group_analysis(data_df, dv_col, group_col):
    """Performs assumption checks (Normality, Levene's) and runs ANOVA/Kruskal-Wallis."""

    print(f"\n{'=' * 70}\nANALYSIS: {dv_col} grouped by {group_col}\n{'=' * 70}")

    # 1. Prepare data (list of arrays, one for each group)
    grouped_data_series = data_df.groupby(group_col)[dv_col]
    grouped_data_arrays = [group.values for name, group in grouped_data_series]

    num_groups = data_df[group_col].nunique()
    if num_groups < 3:
        print(f"⚠️ Warning: Only {num_groups} groups found. ANOVA/Kruskal-Wallis requires 3+ groups.")
        return

    # --- ASSUMPTION CHECKS ---
    print("\n## 1. Assumption Check: Normality (Shapiro-Wilk)")
    non_normal_groups = 0
    for name, data in grouped_data_series:
        if len(data) >= 3:
            stat, p_norm = stats.shapiro(data)
            norm_status = "NON-NORMAL" if p_norm < 0.05 else "Normal"
            print(f"  - Group '{name}' (N={len(data)}): p={p_norm:.4f} ({norm_status})")
            if p_norm < 0.05:
                non_normal_groups += 1

    print("\n## 2. Assumption Check: Homogeneity of Variances (Levene's Test)")
    stat_levene, p_levene = stats.levene(*grouped_data_arrays, center='median')
    is_equal_var = p_levene >= 0.05
    levene_status = "ASSUMED EQUAL (p >= 0.05)" if is_equal_var else "UNEQUAL (p < 0.05)"
    print(f"  - Levene's Test (Brown-Forsythe): F={stat_levene:.4f}, p={p_levene:.4f} ({levene_status})")

    # --- STATISTICAL TEST SELECTION ---
    ALPHA = 0.05
    print("\n## 3. Inferential Testing")

    # A. Parametric Path (ANOVA or Welch's)
    if is_equal_var:
        # One-way ANOVA
        print("\n--- Running One-way ANOVA (Equal Variances) ---")
        formula = f'{dv_col} ~ C({group_col})'
        model = ols(formula, data=data_df).fit()
        anova_results = anova_lm(model)
        p_anova = anova_results.loc[f'C({group_col})', 'PR(>F)']

        print(f"  - Result: F={anova_results.loc[f'C({group_col})', 'F']:.4f}, p={p_anova:.4f}")

        if p_anova < ALPHA:
            print("  - POST-HOC: Tukey's HSD (for equal variances)")
            mc = MultiComparison(data_df[dv_col], data_df[group_col])
            tukey_results = mc.tukeyhsd(alpha=ALPHA)
            print(tukey_results)
        else:
            print("  - No significant difference found.")

    else:
        # Welch's ANOVA (Robust to Unequal Variances)
        print("\n--- Running Welch's ANOVA (Unequal Variances) ---")
        welch_results = welch_anova(data=data_df, dv=dv_col, between=group_col)
        p_welch = welch_results.loc[0, 'p-unc']

        print(f"  - Result: F={welch_results.loc[0, 'F']:.4f}, p={p_welch:.4f}")

        if p_welch < ALPHA:
            print("  - POST-HOC: Games-Howell (for unequal variances)")
            gh_results = pairwise_gameshowell(data=data_df, dv=dv_col, between=group_col)
            print(gh_results[['A', 'B', 'mean(A)', 'mean(B)', 'diff', 'pval', 'hedges']])
        else:
            print("  - No significant difference found.")

    # B. Non-Parametric Alternative
    print("\n--- Running Kruskal-Wallis H Test (Non-parametric Alternative) ---")
    h_stat, p_kw = stats.kruskal(*grouped_data_arrays)
    print(f"  - Result: H={h_stat:.4f}, p={p_kw:.4f}")

    if p_kw < ALPHA:
        print("  - Significant difference found. Use Dunn's Test with Bonferroni correction for post-hoc.")
        # NOTE: Dunn's test requires 'scikit-posthocs' library, not included here for simplicity.
    else:
        print("  - No significant difference found.")


    diff_current = dv_col.split('_')[1]
    group_col_current = group_col.split('_')[0]+ " " +group_col.split('_')[1]

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y=dv_col, data=data_df, palette="Pastel2")
    plt.title(f'Magnitude of Distance Difference ({diff_current}) by {group_col_current}', fontsize=14)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel(f'|Prone - Supine| distance difference (mm)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    #  # The boxplot visualizes the group difference

    print(f"\n{'=' * 70}")
    print(f"STATISTICAL SUMMARY for {group_col}")
    print(data_df.groupby(group_col)[dv_col].agg(['count', 'mean', 'std', 'median']))
    print(f"{'=' * 70}")


def perform_two_group_analysis(data_df, dv_col, group_col):
    """Performs t-test or Mann-Whitney U test for exactly two independent groups."""

    print(f"\n{'=' * 70}\nTWO-GROUP ANALYSIS: {dv_col} grouped by {group_col}\n{'=' * 70}")

    # Check for exactly two groups
    group_names = data_df[group_col].unique()
    if len(group_names) != 2:
        print(f"ERROR: This function requires exactly 2 groups. Found {len(group_names)}.")
        return

    group_A, group_B = group_names[0], group_names[1]

    # 1. Prepare Data
    data_A = data_df[data_df[group_col] == group_A][dv_col].dropna()
    data_B = data_df[data_df[group_col] == group_B][dv_col].dropna()

    if len(data_A) < 5 or len(data_B) < 5:
        print(f"⚠️ Warning: Small sample sizes (N_A={len(data_A)}, N_B={len(data_B)}) may affect test reliability.")

    # --- ASSUMPTION CHECKS ---
    print("\n## 1. Assumption Check: Normality (Shapiro-Wilk)")
    stat_A, p_norm_A = stats.shapiro(data_A)
    stat_B, p_norm_B = stats.shapiro(data_B)
    is_normal = (p_norm_A >= 0.05) and (p_norm_B >= 0.05)

    print(
        f"  - Group '{group_A}' (N={len(data_A)}): p={p_norm_A:.4f} ({'Normal' if p_norm_A >= 0.05 else 'Non-Normal'})")
    print(
        f"  - Group '{group_B}' (N={len(data_B)}): p={p_norm_B:.4f} ({'Normal' if p_norm_B >= 0.05 else 'Non-Normal'})")

    print("\n## 2. Assumption Check: Homogeneity of Variances (Levene's Test)")
    stat_levene, p_levene = stats.levene(data_A, data_B, center='median')
    is_equal_var = p_levene >= 0.05
    levene_status = "ASSUMED EQUAL (p >= 0.05)" if is_equal_var else "UNEQUAL (p < 0.05)"
    print(f"  - Levene's Test: F={stat_levene:.4f}, p={p_levene:.4f} ({levene_status})")

    # --- INFERENTIAL TESTING ---
    ALPHA = 0.05
    print("\n## 3. Inferential Testing")

    if is_normal:
        # --- Parametric Tests (t-test) ---
        print("\n--- Running T-test (Parametric) ---")

        if is_equal_var:
            # Independent Samples t-test (assumes equal variances)
            test_type = "Student's t-test (Equal Var)"
            t_stat, p_ttest = stats.ttest_ind(data_A, data_B, equal_var=True)
        else:
            # Welch's t-test (does not assume equal variances)
            test_type = "Welch's t-test (Unequal Var)"
            t_stat, p_ttest = stats.ttest_ind(data_A, data_B, equal_var=False)

        print(f"  - Test: {test_type}")
        print(
            f"  - Result: t={t_stat:.4f}, p={p_ttest:.4f} ({'Significant' if p_ttest < ALPHA else 'Not Significant'})")

    else:
        # --- Non-Parametric Test (Mann-Whitney U) ---
        print("\n--- Running Mann-Whitney U test (Non-parametric) ---")

        # Mann-Whitney U test
        u_stat, p_mw = stats.mannwhitneyu(data_A, data_B, alternative='two-sided')

        print("  - Test: Mann-Whitney U")
        print(f"  - Result: U={u_stat:.4f}, p={p_mw:.4f} ({'Significant' if p_mw < ALPHA else 'Not Significant'})")

    diff_current = dv_col.split('_')[1]
    group_col_current = group_col.split('_')[0]+ " " +group_col.split('_')[1]
    # --- VISUALIZATION ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=group_col, y=dv_col, data=data_df, palette="coolwarm")
    plt.title(f'Comparison of {diff_current} difference by {group_col_current}', fontsize=14)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel(f'{dv_col} (mm)', fontsize=12)
    plt.tight_layout()
    plt.show()  #

    print(f"\n{'=' * 70}")
    print(f"STATISTICAL SUMMARY for {group_col}")
    print(data_df.groupby(group_col)[dv_col].agg(['count', 'mean', 'std', 'median']))
    print(f"{'=' * 70}")


def perform_repeated_measures_analysis(df_input, subject_id_col, dv_cols):
    """
    Performs Repeated Measures ANOVA or Friedman Test to compare three or more
    dependent difference magnitudes across different measurement types.

    Args:
        df_input (pd.DataFrame): The DataFrame containing the difference columns.
        subject_id_col (str): The column name identifying the unique landmark/subject (e.g., 'VL_ID').
        dv_cols (list): List of column names representing the dependent measurements
                        (e.g., ['diff_DTS_Skin', 'diff_DTN_Nipple', 'diff_DTR_Rib_Cage']).
    """

    print("\n" + "=" * 80)
    print("REPEATED MEASURES ANALYSIS: Comparing Multiple Dependent Shifts")
    print("=" * 80)

    # 1. Clean Data: Keep only rows where all DVs and the Subject ID exist
    data_wide = df_input.dropna(subset=[subject_id_col] + dv_cols).copy()

    if len(data_wide) == 0:
        print("ERROR: No complete cases found for the selected columns.")
        return

    # 2. Reshape Data to Long Format
    df_long = pd.melt(
        data_wide,
        id_vars=[subject_id_col],
        value_vars=dv_cols,
        var_name='Measurement_Type',
        value_name='Difference_Magnitude'
    )

    N_landmarks = df_long[subject_id_col].nunique()
    print(f"Total N for Analysis: {N_landmarks} landmarks.")

    # --- Try Parametric RM-ANOVA ---
    try:

        # Run Repeated Measures ANOVA (RM-ANOVA)
        aov_rm = rm_anova(
            data=df_long,
            dv='Difference_Magnitude',
            within='Measurement_Type',
            subject=subject_id_col,
            detailed=True
        )

        print("\n" + "=" * 80)
        print("REPEATED MEASURES ANOVA RESULTS")
        print("=" * 80)

        # Create formatted table for RM-ANOVA results
        rm_anova_summary = pd.DataFrame({
            'Source': ['Measurement Type'],
            'F-statistic': [f"{aov_rm['F'].iloc[0]:.3f}"],
            'P-value': [f"{aov_rm['p-unc'].iloc[0]:.4e}"],
            'Sig.': ['***' if aov_rm['p-unc'].iloc[0] < 0.001 else
                     '**' if aov_rm['p-unc'].iloc[0] < 0.01 else
                     '*' if aov_rm['p-unc'].iloc[0] < 0.05 else 'ns'],
        })
        rm_anova_summary.set_index('Source', inplace=True)
        print(rm_anova_summary)
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("η² = partial eta squared (effect size)")

        # Post-hoc Analysis if RM-ANOVA is significant
        ALPHA = 0.05
        if aov_rm['p-unc'].iloc[0] < ALPHA:
            print("\nRM-ANOVA is significant. Running Post-hoc Paired T-tests (Bonferroni corrected).")

            post_hoc = pairwise_ttests(
                data=df_long,
                dv='Difference_Magnitude',
                within='Measurement_Type',
                subject=subject_id_col,
                padjust='bonf'
            )

            group_means = df_long.groupby('Measurement_Type')['Difference_Magnitude'].mean()

            # Map the means to the post_hoc dataframe
            post_hoc['mean(A)'] = post_hoc['A'].map(group_means)
            post_hoc['mean(B)'] = post_hoc['B'].map(group_means)

            # Format post-hoc results - include mean difference
            post_hoc_formatted = post_hoc[['A', 'B', 'mean(A)', 'mean(B)', 'T', 'dof', 'p-unc', 'p-corr', 'hedges']].copy()

            # Calculate mean difference (A - B)
            post_hoc_formatted['Mean Diff'] = post_hoc_formatted['mean(A)'] - post_hoc_formatted['mean(B)']

            # Reorder and rename columns
            post_hoc_formatted = post_hoc_formatted[['A', 'B', 'mean(A)', 'mean(B)', 'Mean Diff', 'T', 'dof', 'p-unc', 'p-corr', 'hedges']]
            post_hoc_formatted.columns = ['Comparison A', 'Comparison B', 'Mean A [mm]', 'Mean B [mm]',
                                          'Mean Diff [mm]', 't-statistic', 'df',
                                          'P-value (uncorrected)', 'P-value (Bonferroni)', "Hedges' g"]

            # Add significance markers
            post_hoc_formatted['Sig.'] = post_hoc_formatted['P-value (Bonferroni)'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            )

            # Clean up metric names
            post_hoc_formatted['Comparison A'] = post_hoc_formatted['Comparison A'].str.replace('diff_', '').str.replace('_', ' ')
            post_hoc_formatted['Comparison B'] = post_hoc_formatted['Comparison B'].str.replace('diff_', '').str.replace('_', ' ')

            # Format numeric columns
            post_hoc_formatted['Mean A [mm]'] = post_hoc_formatted['Mean A [mm]'].apply(lambda x: f"{x:.2f}")
            post_hoc_formatted['Mean B [mm]'] = post_hoc_formatted['Mean B [mm]'].apply(lambda x: f"{x:.2f}")
            post_hoc_formatted['Mean Diff [mm]'] = post_hoc_formatted['Mean Diff [mm]'].apply(lambda x: f"{x:.2f}")
            post_hoc_formatted['t-statistic'] = post_hoc_formatted['t-statistic'].apply(lambda x: f"{x:.3f}")
            post_hoc_formatted['P-value (uncorrected)'] = post_hoc_formatted['P-value (uncorrected)'].apply(lambda x: f"{x:.4e}")
            post_hoc_formatted['P-value (Bonferroni)'] = post_hoc_formatted['P-value (Bonferroni)'].apply(lambda x: f"{x:.4e}")
            post_hoc_formatted["Hedges' g"] = post_hoc_formatted["Hedges' g"].apply(lambda x: f"{x:.3f}")

            print(post_hoc_formatted.to_string(index=False))
            print("\nMean Diff = Mean A - Mean B")
            print("Significance based on Bonferroni-corrected p-values")
        else:
            print("\nRM-ANOVA is not significant. No post-hoc tests needed.")

    except NotImplementedError as e:
        print(f"\n ! RM-ANOVA Skipped: {e}")

    except Exception as e:
        print(f"\n ! RM-ANOVA failed (Switching to Non-Parametric Friedman Test): {e}")

        # --- Non-Parametric Fallback (Friedman Test) ---
        print("\n" + "-" * 40)
        print("NON-PARAMETRIC FALLBACK: Friedman Test")
        print("-" * 40)

        # Friedman test requires wide format, passing the arrays directly
        friedman_stat, p_friedman = stats.friedmanchisquare(
            *[data_wide[col] for col in dv_cols]
        )

        print(f"  - Result: Chi-squared={friedman_stat:.4f}, p={p_friedman:.4f}")

        if p_friedman < 0.05:
            print("  - Significant difference found. The shifts are different across the 3 metrics.")
            print("  - Post-hoc requires Wilcoxon Signed-Rank Test with correction.")
        else:
            print("  - No significant difference found.")

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Measurement_Type', y='Difference_Magnitude', data=df_long, palette="magma")
    plt.title('Shift Magnitude by Reference Point', fontsize=14)
    plt.xlabel('Distance Metric', fontsize=12)
    plt.ylabel('Difference Magnitude (mm)', fontsize=12)
    # Clean up x-axis labels for display
    display_labels = [col.replace('diff_', '').replace('_', ' ') for col in dv_cols]
    plt.xticks(ticks=range(len(dv_cols)), labels=display_labels, rotation=15)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)

def plot_bmi_correlations(df, output_filename='BMI_Shift_Correlations.png'):
    """
    Generates regression plots to visualize the relationship between BMI
    and landmark shifts relative to the Nipple and Rib Cage.
    """
    # Check if necessary columns exist
    required_cols = ['BMI', 'diff_DTN_Nipple', 'diff_DTR_Rib_Cage']
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        print(f"Error: Missing columns in DataFrame: {missing}")
        return

    # Set the visual style
    sns.set_theme(style="whitegrid")

    # Create a figure with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: BMI vs. Nipple Shift ---
    sns.regplot(
        data=df,
        x='BMI',
        y='diff_DTN_Nipple',
        ax=axes[0],
        scatter_kws={'alpha': 0.4, 'color': 'teal', 's': 40},
        line_kws={'color': 'firebrick', 'linewidth': 2}
    )
    axes[0].set_title('Impact of BMI on Difference in DTN\n(β = -4.01, p < 0.001)', fontsize=14)
    axes[0].set_xlabel('Body Mass Index (BMI) [kg/m²]', fontsize=12)
    axes[0].set_ylabel('Difference in DTN [mm]', fontsize=12)

    # --- Plot 2: BMI vs. Rib Cage Shift ---
    sns.regplot(
        data=df,
        x='BMI',
        y='diff_DTR_Rib_Cage',
        ax=axes[1],
        scatter_kws={'alpha': 0.4, 'color': 'darkblue', 's': 40},
        line_kws={'color': 'firebrick', 'linewidth': 2}
    )
    axes[1].set_title('Impact of BMI on Difference in DTR\n(β = -1.44, p < 0.001)', fontsize=14)
    axes[1].set_xlabel('Body Mass Index (BMI) [kg/m²]', fontsize=12)
    axes[1].set_ylabel('Difference in DTR [mm]', fontsize=12)

    # Clean up the layout
    plt.tight_layout()

    # Save and show
    save_path = Path("..") / "output" / "figs" / "v5" / "BMI" / output_filename
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Visualization complete. Plot saved as '{save_path}'")


def investigate_proximity_effect(df):
    # use the Prone DTS as the "Baseline" position
    baseline_dts = 'Distance to skin (prone) [mm]'
    shift_metrics = ['diff_DTS_Skin', 'diff_DTN_Nipple', 'diff_DTR_Rib_Cage']

    print("\n" + "=" * 60)
    print("EFFECT OF INITIAL PROXIMITY TO SKIN ON SHIFT MAGNITUDE")
    print("=" * 60)

    for metric in shift_metrics:
        # 1. Correlation Analysis
        temp_df = df[[baseline_dts, metric]].dropna()
        rho, pval = stats.spearmanr(temp_df[baseline_dts], temp_df[metric])

        # 2. Categorical Comparison (Threshold = 10mm)
        superficial = temp_df[temp_df[baseline_dts] <= 20][metric]
        deep = temp_df[temp_df[baseline_dts] > 20][metric]

        t_stat, p_comp = stats.mannwhitneyu(superficial, deep)

        print(f"\nMetric: {metric}")
        print(f"  - Correlation with Prone DTS: rho={rho:.3f}, p={pval:.4e}")
        print(f"  - Superficial Mean (N={len(superficial)}): {superficial.mean():.2f} mm")
        print(f"  - Deep Mean (N={len(deep)}): {deep.mean():.2f} mm")
        print(f"  - Difference Significance: p={p_comp:.4e}")




def plot_vectors_for_vl81(df_ave):
    """
    Plots displacement vectors (Prone -> Supine) for VL_ID 81 using
    the style defined in utils_plot.plot_vector_three_views.
    """
    print("\n--- Plotting Vectors for VL 81 ---")

    # 1. Filter for VL 81
    df_subset = df_ave[df_ave['VL_ID'] == 81].copy()

    if df_subset.empty:
        print("No data found for VL_ID 81.")
        return

    # 2. Separate into Left (LB) and Right (RB) breasts
    # Note: Adjust column names if they differ slightly in your Excel
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Helper to extract Base Points (Prone) and Vectors (Supine - Prone)
    def get_points_and_vectors(sub_df):
        if sub_df.empty:
            return np.empty((0, 3)), np.empty((0, 3))

        # Extract Prone (Start/Base points)
        # Using 'landmark ave prone transformed' columns based on your file snippet
        prone_x = sub_df['landmark ave prone transformed x'].values
        prone_y = sub_df['landmark ave prone transformed y'].values
        prone_z = sub_df['landmark ave prone transformed z'].values
        base_points = np.column_stack((prone_x, prone_y, prone_z))

        # Extract Supine (End points)
        supine_x = sub_df['landmark ave supine x'].values
        supine_y = sub_df['landmark ave supine y'].values
        supine_z = sub_df['landmark ave supine z'].values
        end_points = np.column_stack((supine_x, supine_y, supine_z))

        # Calculate Vector = End - Start
        vectors = end_points - base_points
        return base_points, vectors

    base_left, vec_left = get_points_and_vectors(left_df)
    base_right, vec_right = get_points_and_vectors(right_df)

    # 4. Define Plane Configuration (Copied from utils_plot.py)
    # 0: X (Right/Left), 1: Y (Ant/Post), 2: Z (Inf/Sup)
    PLANE_CONFIG = {
        'Coronal': {
            'axes': (0, 2),  # X vs Z
            'xlabel': "Right-Left (mm)", 'ylabel': "Inf-Sup (mm)",
            'shape': 'Circle'
        },
        'Sagittal': {
            'axes': (1, 2),  # Y vs Z
            'xlabel': "Ant-Post (mm)", 'ylabel': "Inf-Sup (mm)",
            'shape': 'SemiCircle'
        },
        'Axial': {
            'axes': (0, 1),  # X vs Y
            'xlabel': "Right-Left (mm)", 'ylabel': "Ant-Post (mm)",
            'shape': 'SemiCircle'
        }
    }

    lims = (-200, 200)  # Adjust limits as needed, utils_plot used (-400, 400)
    radius = 150  # Radius for the breast outline representation

    # 5. Plotting Loop
    for plane_name, config in PLANE_CONFIG.items():
        axis_x_idx, axis_y_idx = config['axes']

        fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        fig.suptitle(f"VL 81 - {plane_name} Plane: Prone to Supine Displacement", fontsize=14)

        # --- Subplot Helper ---
        def plot_breast_side(ax, base, vec, title, side_color):
            ax.set_title(title)
            ax.set_xlabel(config['xlabel'])
            ax.set_ylabel(config['ylabel'])
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Draw Quadrant Lines (Nipple Centered at 0,0)
            ax.plot(0, 0, 'ro', markersize=8, label='Nipple (Prone)', zorder=5)

            if plane_name == 'Coronal':
                ax.axhline(0, color='red', lw=1)
                ax.axvline(0, color='red', lw=1)
                # Full Circle
                circle = Circle((0, 0), radius, fill=False, color='black', lw=1, linestyle='--')
                ax.add_artist(circle)

            elif plane_name == 'Sagittal':
                ax.axhline(0, color='red', lw=1)
                # SemiCircle (Anterior is usually +Y)
                arc = Arc((0, 0), radius * 2, radius * 2, theta1=0, theta2=180, color='black', linestyle='--')
                ax.add_artist(arc)
                ax.plot([-radius, radius], [0, 0], color='black', lw=1)

            elif plane_name == 'Axial':
                ax.axvline(0, color='red', lw=1)
                # SemiCircle (Anterior is usually +Y)
                arc = Arc((0, 0), radius * 2, radius * 2, theta1=0, theta2=180, color='black', linestyle='--')
                ax.add_artist(arc)
                ax.plot([-radius, radius], [0, 0], color='black', lw=1)

            # Plot Vectors
            if len(base) > 0:
                ax.quiver(
                    base[:, axis_x_idx], base[:, axis_y_idx],  # X, Y start
                    vec[:, axis_x_idx], vec[:, axis_y_idx],  # U, V components
                    angles='xy', scale_units='xy', scale=1,
                    color=side_color, width=0.003, headwidth=3
                )
                # Plot start points
                ax.scatter(base[:, axis_x_idx], base[:, axis_y_idx], c=side_color, s=20)

        # Plot Right Breast (on the left subplot usually, or labeled explicitly)
        plot_breast_side(ax_right, base_right, vec_right, "Right Breast", 'blue')

        # Plot Left Breast
        plot_breast_side(ax_left, base_left, vec_left, "Left Breast", 'green')

        plt.tight_layout()
        plt.show()


def plot_vectors_rel_sternum(df_ave, color_by='breast', vl_id=None, data_type='landmarks', include_dual_sagittal=False):
    """
    Plots displacement vectors (Prone -> Supine) for both breasts relative to sternum.
    Supports plotting either landmark positions or nipple positions.

    Args:
        df_ave: DataFrame with landmark data
        color_by: Coloring scheme - 'breast' (default, blue/green),
                  'subject' (color by VL_ID), or 'dts' (color by distance to skin)
        vl_id: Optional subject ID to filter data. If None, uses all subjects.
        data_type: Type of data to plot - 'landmarks' (default) or 'nipples'
        include_dual_sagittal: Whether to include the dual sagittal view plot
    """
    data_type_name = "Landmarks" if data_type == 'landmarks' else "Nipples"
    print(f"\n--- Plotting {data_type_name} Relative to Sternum ---")
    print(f"Color scheme: {color_by}")

    # 1. Filter data if specific subject requested
    if vl_id is not None:
        df_subset = df_ave[df_ave['VL_ID'] == vl_id].copy()
        if df_subset.empty:
            print(f"Warning: No data found for subject VL_{vl_id}")
            return
        print(f"Filtering for subject VL_{vl_id}")
    else:
        df_subset = df_ave.copy()

    if df_subset.empty:
        print("No data found.")
        return

    # 2. Separate into Left (LB) and Right (RB) breasts
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Helper to extract Base Points (Prone) and Vectors (Supine - Prone)
    def get_points_and_vectors(sub_df, is_left_breast=True):
        if sub_df.empty:
            return np.empty((0, 3)), np.empty((0, 3)), None, None

        if data_type == 'landmarks':
            # Extract Prone (Start/Base points) - landmarks are already relative to sternum
            prone_x = sub_df['landmark ave prone transformed x'].values
            prone_y = sub_df['landmark ave prone transformed y'].values
            prone_z = sub_df['landmark ave prone transformed z'].values

            # Extract Supine (End points)
            supine_x = sub_df['landmark ave supine x'].values
            supine_y = sub_df['landmark ave supine y'].values
            supine_z = sub_df['landmark ave supine z'].values

        elif data_type == 'nipples':
            # Get sternum positions for calculating nipple relative positions
            sternum_prone_x = sub_df['sternum superior prone transformed x'].values
            sternum_prone_y = sub_df['sternum superior prone transformed y'].values
            sternum_prone_z = sub_df['sternum superior prone transformed z'].values
            sternum_supine_x = sub_df['sternum superior supine x'].values
            sternum_supine_y = sub_df['sternum superior supine y'].values
            sternum_supine_z = sub_df['sternum superior supine z'].values

            # Get nipple columns based on breast side
            if is_left_breast:
                nipple_prone_x_raw = sub_df['left nipple prone transformed x'].values
                nipple_prone_y_raw = sub_df['left nipple prone transformed y'].values
                nipple_prone_z_raw = sub_df['left nipple prone transformed z'].values
                nipple_supine_x_raw = sub_df['left nipple supine x'].values
                nipple_supine_y_raw = sub_df['left nipple supine y'].values
                nipple_supine_z_raw = sub_df['left nipple supine z'].values
            else:
                nipple_prone_x_raw = sub_df['right nipple prone transformed x'].values
                nipple_prone_y_raw = sub_df['right nipple prone transformed y'].values
                nipple_prone_z_raw = sub_df['right nipple prone transformed z'].values
                nipple_supine_x_raw = sub_df['right nipple supine x'].values
                nipple_supine_y_raw = sub_df['right nipple supine y'].values
                nipple_supine_z_raw = sub_df['right nipple supine z'].values

            # Calculate nipple positions relative to sternum
            prone_x = nipple_prone_x_raw - sternum_prone_x
            prone_y = nipple_prone_y_raw - sternum_prone_y
            prone_z = nipple_prone_z_raw - sternum_prone_z
            supine_x = nipple_supine_x_raw - sternum_supine_x
            supine_y = nipple_supine_y_raw - sternum_supine_y
            supine_z = nipple_supine_z_raw - sternum_supine_z
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'landmarks' or 'nipples'.")

        base_points = np.column_stack((prone_x, prone_y, prone_z))

        # Extract Supine (End points)
        end_points = np.column_stack((supine_x, supine_y, supine_z))

        # Calculate Vector = End - Base
        vectors = end_points - base_points

        # Extract DTS values if available (only relevant for landmarks)
        dts_col = 'Distance to skin (prone) [mm]'
        dts_values = sub_df[dts_col].values if dts_col in sub_df.columns else None

        # Extract VL_ID for subject coloring
        vl_ids = sub_df['VL_ID'].values if 'VL_ID' in sub_df.columns else None

        return base_points, vectors, dts_values, vl_ids

    base_left, vec_left, dts_left, vl_ids_left = get_points_and_vectors(left_df, is_left_breast=True)
    base_right, vec_right, dts_right, vl_ids_right = get_points_and_vectors(right_df, is_left_breast=False)

    # 4. Define Plane Configuration
    # 0: X (Right/Left), 1: Y (Ant/Post), 2: Z (Inf/Sup)
    PLANE_CONFIG = {
        'Coronal': {
            'axes': (0, 2),  # X vs Z
            'xlabel': "Right-Left (mm)", 'ylabel': "Inf-Sup (mm)",
            'shape': 'Circle'
        },
        'Sagittal': {
            'axes': (1, 2),  # Y vs Z
            'xlabel': "Ant-Post (mm)", 'ylabel': "Inf-Sup (mm)",
            'shape': 'SemiCircle'
        },
        'Axial': {
            'axes': (0, 1),  # X vs Y
            'xlabel': "Right-Left (mm)", 'ylabel': "Ant-Post (mm)",
            'shape': 'SemiCircle'
        }
    }

    lims = (-250, 250)
    radius = 150

    # 5. Prepare colormaps for 'subject' coloring
    subject_color_map = {}  # Initialize empty dict
    if color_by == 'subject':
        # Get unique subject IDs from combined left and right data
        all_vl_ids = []
        if vl_ids_left is not None:
            all_vl_ids.extend(vl_ids_left)
        if vl_ids_right is not None:
            all_vl_ids.extend(vl_ids_right)
        unique_subjects = sorted(list(set(all_vl_ids)))
        n_subjects = len(unique_subjects)

        # Create colormap for subjects
        subject_cmap = cm.get_cmap('viridis', n_subjects)
        subject_color_map = {subj: subject_cmap(i) for i, subj in enumerate(unique_subjects)}
        print(f"Number of unique subjects: {n_subjects}")

    # 6. Plotting Loop - Single Plot for Both Breasts
    for plane_name, config in PLANE_CONFIG.items():
        axis_x_idx, axis_y_idx = config['axes']

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create title based on coloring scheme and data type
        data_label = "landmarks" if data_type == 'landmarks' else "nipples"
        if color_by == 'dts':
            title_suffix = " (colored by DTS)"
        elif color_by == 'subject':
            title_suffix = " (colored by subject)"
        else:
            title_suffix = ""

        fig.suptitle(f"Displacement of {data_label} relative to the sternum ({plane_name.lower()} view){title_suffix}", fontsize=14)

        # Setup axes
        ax.set_xlabel(config['xlabel'], fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_xticks(np.arange(-250, 251, 50))
        ax.set_yticks(np.arange(-250, 251, 50))

        # Set limits based on plane
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Mark Sternum at origin (0,0) with a dot
        ax.plot(0, 0, 'ko', markersize=5, label='Sternum (Origin)', zorder=5)

        # Add reference lines through sternum
        if plane_name == 'Axial':
            ax.axvline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')
        elif plane_name == 'Coronal':
            ax.axhline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')
            ax.axvline(0, color='gray', lw=0.8, alpha=0.5, linestyle=':')

        # Initialize colorbar placeholder
        scatter_for_colorbar = None

        # Plot Right Breast Vectors
        if len(base_right) > 0:
            if color_by == 'dts' and dts_right is not None:
                # Color by DTS - normalize colors for colormap
                norm = plt.Normalize(vmin=0, vmax=40)
                cmap = plt.cm.viridis

                # Create color array
                colors_right = cmap(norm(dts_right))

                # Plot quiver with individual colors
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

                # Plot scatter for colorbar
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

            elif color_by == 'subject' and vl_ids_right is not None:
                # Color by subject
                for i in range(len(base_right)):
                    color = subject_color_map.get(vl_ids_right[i], 'blue')
                    ax.scatter(
                        base_right[i, axis_x_idx],
                        base_right[i, axis_y_idx],
                        c=[color],
                        s=20,
                        zorder=5
                    )
                    ax.quiver(
                        base_right[i, axis_x_idx],
                        base_right[i, axis_y_idx],
                        vec_right[i, axis_x_idx],
                        vec_right[i, axis_y_idx],
                        angles='xy', scale_units='xy', scale=1,
                        color=color, width=0.003, headwidth=3,
                        alpha=0.7, zorder=4
                    )
            else:
                # Default: color by breast (blue for right)
                ax.quiver(
                    base_right[:, axis_x_idx], base_right[:, axis_y_idx],
                    vec_right[:, axis_x_idx], vec_right[:, axis_y_idx],
                    angles='xy', scale_units='xy', scale=1,
                    color='blue', width=0.003, headwidth=3,
                    label='Right Breast', alpha=0.7, zorder=4
                )
                ax.scatter(base_right[:, axis_x_idx], base_right[:, axis_y_idx],
                          c='blue', s=20, zorder=5)

            # Add "Right Breast" region label
            if color_by == 'breast':
                # For breast coloring: use blue color
                right_x_pos = np.mean(base_right[:, axis_x_idx])
                # For sagittal view, move text further to the right
                if plane_name == 'Sagittal':
                    right_x_pos = lims[0] + (lims[1] - lims[0]) * 0.25  # Position at 25% from left edge
                right_y_pos = lims[1] * 0.85
                ax.text(right_x_pos, right_y_pos, 'Right Breast',
                       ha='center', va='center', fontsize=11,
                       color='blue', alpha=0.9)
            elif color_by == 'dts':
                # For DTS coloring: black color for axial and coronal, no text for sagittal
                if plane_name != 'Sagittal':
                    right_x_pos = np.mean(base_right[:, axis_x_idx])
                    right_y_pos = lims[1] * 0.85
                    ax.text(right_x_pos, right_y_pos, 'Right Breast',
                           ha='center', va='center', fontsize=11,
                           color='black', alpha=0.9)
            elif color_by == 'subject':
                # For subject coloring: black color for axial and coronal only
                if plane_name != 'Sagittal':
                    right_x_pos = np.mean(base_right[:, axis_x_idx])
                    right_y_pos = lims[1] * 0.85
                    ax.text(right_x_pos, right_y_pos, 'Right Breast',
                           ha='center', va='center', fontsize=11,
                           color='black', alpha=0.9)

        # Plot Left Breast Vectors
        if len(base_left) > 0:
            if color_by == 'dts' and dts_left is not None:
                # Color by DTS - normalize colors for colormap
                norm = plt.Normalize(vmin=0, vmax=40)
                cmap = plt.cm.viridis

                # Create color array
                colors_left = cmap(norm(dts_left))

                # Plot quiver with individual colors
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

                # Plot scatter for colorbar
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

            elif color_by == 'subject' and vl_ids_left is not None:
                # Color by subject
                for i in range(len(base_left)):
                    color = subject_color_map.get(vl_ids_left[i], 'black')
                    ax.scatter(
                        base_left[i, axis_x_idx],
                        base_left[i, axis_y_idx],
                        c=[color],
                        s=20,
                        zorder=5
                    )
                    ax.quiver(
                        base_left[i, axis_x_idx],
                        base_left[i, axis_y_idx],
                        vec_left[i, axis_x_idx],
                        vec_left[i, axis_y_idx],
                        angles='xy', scale_units='xy', scale=1,
                        color=color, width=0.003, headwidth=3,
                        alpha=0.7, zorder=4
                    )
            else:
                # Default: color by breast (green for left)
                ax.quiver(
                    base_left[:, axis_x_idx], base_left[:, axis_y_idx],
                    vec_left[:, axis_x_idx], vec_left[:, axis_y_idx],
                    angles='xy', scale_units='xy', scale=1,
                    color='green', width=0.003, headwidth=3,
                    label='Left Breast', alpha=0.7, zorder=4
                )
                ax.scatter(base_left[:, axis_x_idx], base_left[:, axis_y_idx],
                          c='green', s=20, zorder=5)

            # Add "Left Breast" region label
            if color_by == 'breast':
                # For breast coloring: use green color
                left_x_pos = np.mean(base_left[:, axis_x_idx])
                # For sagittal view, move text further to the left
                if plane_name == 'Sagittal':
                    left_x_pos = lims[0] + (lims[1] - lims[0]) * 0.75  # Position at 75% from left edge
                left_y_pos = lims[1] * 0.85
                ax.text(left_x_pos, left_y_pos, 'Left Breast',
                       ha='center', va='center', fontsize=12,
                       color='green', alpha=0.9)
            elif color_by == 'dts':
                # For DTS coloring: black color for axial and coronal, no text for sagittal
                if plane_name != 'Sagittal':
                    left_x_pos = lims[0] + (lims[1] - lims[0]) * 0.75
                    left_y_pos = lims[1] * 0.85
                    ax.text(left_x_pos, left_y_pos, 'Left Breast',
                           ha='center', va='center', fontsize=12,
                           color='black', alpha=0.9)
            elif color_by == 'subject':
                # For subject coloring: black color for axial and coronal only
                if plane_name != 'Sagittal':
                    left_x_pos = lims[0] + (lims[1] - lims[0]) * 0.75
                    left_y_pos = lims[1] * 0.85
                    ax.text(left_x_pos, left_y_pos, 'Left Breast',
                           ha='center', va='center', fontsize=12,
                           color='black', alpha=0.9)

        # Add colorbar for DTS coloring
        if color_by == 'dts' and scatter_for_colorbar is not None:
            cbar = plt.colorbar(scatter_for_colorbar, ax=ax)
            cbar.set_label('DTS (mm)', rotation=270, labelpad=15, fontsize=12)

        # Add legend (only for breast coloring or with sternum)
        if color_by == 'breast':
            ax.legend(loc='lower right', fontsize=10)
        else:
            # Just show sternum origin
            ax.legend(['Sternum (Origin)'], loc='lower right', fontsize=10)

        plt.tight_layout()

        # Create filename based on coloring scheme and data type
        data_prefix = "Landmarks" if data_type == 'landmarks' else "Nipples"
        if color_by == 'dts':
            filename = f"{data_prefix}_rel_sternum_{plane_name}_DTS.png"
        elif color_by == 'subject':
            filename = f"{data_prefix}_rel_sternum_{plane_name}_by_subject.png"
        else:
            filename = f"{data_prefix}_rel_sternum_{plane_name}.png"

        save_path = Path("..") / "output" / "figs" / "v5" / "landmark vectors" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.show()
        plt.close(fig)

    # 7. Dual Sagittal View (optional)
    if include_dual_sagittal:
        _plot_dual_sagittal_view_sternum(
            base_left, vec_left, dts_left, vl_ids_left,
            base_right, vec_right, dts_right, vl_ids_right,
            color_by=color_by, data_type=data_type
        )

    print(f"[OK] Completed plotting {data_type} relative to sternum with '{color_by}' coloring")

    return {
        'base_left': base_left,
        'vec_left': vec_left,
        'base_right': base_right,
        'vec_right': vec_right,
        'dts_left': dts_left,
        'dts_right': dts_right,
        'vl_ids_left': vl_ids_left,
        'vl_ids_right': vl_ids_right
    }


def _plot_dual_sagittal_view_sternum(
    base_left, vec_left, dts_left, vl_ids_left,
    base_right, vec_right, dts_right, vl_ids_right,
    color_by='breast', data_type='landmarks'
):
    """
    Internal helper function for dual sagittal view with two x-axis origins.
    Used by plot_vectors_rel_sternum.

    Args:
        base_left, vec_left: Base points and vectors for left breast
        dts_left: Distance to skin values for left breast
        vl_ids_left: VL_ID values for left breast
        base_right, vec_right: Base points and vectors for right breast
        dts_right: Distance to skin values for right breast
        vl_ids_right: VL_ID values for right breast
        color_by: Coloring scheme - 'breast', 'subject', or 'dts'
        data_type: 'landmarks' or 'nipples'
    """
    data_label = "landmarks" if data_type == 'landmarks' else "nipples"
    print(f"\n--- Plotting Dual Sagittal View ({data_label}) ---")

    # Title
    title_suffix = ""
    if color_by == 'dts':
        title_suffix = " (colored by DTS)"
    elif color_by == 'subject':
        title_suffix = " (colored by subject)"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8), sharey=True, constrained_layout=True)
    fig.suptitle(f"Displacement of {data_label} relative to sternum (sagittal dual view){title_suffix}", fontsize=14)

    # Common Settings - consistent limits
    ylim_val = 250
    ax_left.set_ylim(-ylim_val, ylim_val)
    ax_right.set_ylim(-ylim_val, ylim_val)

    yticks = np.arange(-ylim_val, ylim_val+1, 50)
    ax_left.set_yticks(yticks)
    ax_right.set_yticks(yticks)

    # X limits
    ax_left.set_xlim(150, -250)
    ax_right.set_xlim(-250, 150)
    ax_left.set_xticks([150, 100, 50, 0, -50, -100, -150, -200, -250])
    ax_right.set_xticks([-250, -200, -150, -100, -50, 0, 50, 100, 150])

    # Labels
    xlabel_color_left = 'blue' if color_by == 'breast' else 'black'
    xlabel_color_right = 'green' if color_by == 'breast' else 'black'
    ax_left.set_xlabel("Post-Ant (mm)", color=xlabel_color_left, fontsize=12)
    ax_right.set_xlabel("Ant-Post (mm)", color=xlabel_color_right, fontsize=12)
    ax_left.set_ylabel("Inf-Sup (mm)", fontsize=12)

    # Spine configuration for shared center
    ax_left.spines['right'].set_position(('data', 0))
    ax_left.spines['left'].set_visible(False)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_color('black')

    ax_right.spines['left'].set_position(('data', 0))
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_color('black')

    # Origins and grid
    ax_left.plot(0, 0, 'o', color='black', markersize=6, zorder=10)
    ax_right.plot(0, 0, 'o', color='black', markersize=6, zorder=10)
    ax_left.grid(True, linestyle='--', alpha=0.5)
    ax_right.grid(True, linestyle='--', alpha=0.5)
    ax_left.set_aspect('equal', adjustable='box')
    ax_right.set_aspect('equal', adjustable='box')

    # Prepare subject colormap if needed
    subject_color_map = {}
    if color_by == 'subject':
        all_vl_ids = []
        if vl_ids_left is not None:
            all_vl_ids.extend(vl_ids_left)
        if vl_ids_right is not None:
            all_vl_ids.extend(vl_ids_right)
        unique_subjects = sorted(list(set(all_vl_ids)))
        n_subjects = len(unique_subjects)
        subject_cmap = cm.get_cmap('viridis', n_subjects)
        subject_color_map = {subj: subject_cmap(i) for i, subj in enumerate(unique_subjects)}

    scatter_left_cbar = None
    scatter_right_cbar = None

    # --- LEFT PLOT (RIGHT BREAST) ---
    if len(base_right) > 0:
        if color_by == 'dts' and dts_right is not None:
            norm = plt.Normalize(vmin=0, vmax=40)
            cmap_dts = plt.cm.viridis
            colors = cmap_dts(norm(dts_right))
            for i in range(len(base_right)):
                ax_left.quiver(base_right[i, 1], base_right[i, 2],
                              vec_right[i, 1], vec_right[i, 2],
                              angles='xy', scale_units='xy', scale=1,
                              color=colors[i], width=0.003, headwidth=3, alpha=0.7)
            scatter_left_cbar = ax_left.scatter(base_right[:, 1], base_right[:, 2],
                                                c=dts_right, cmap='viridis', s=20, vmin=0, vmax=40, zorder=5)
        elif color_by == 'subject' and vl_ids_right is not None and subject_color_map:
            for i in range(len(base_right)):
                color = subject_color_map.get(vl_ids_right[i], 'blue')
                ax_left.quiver(base_right[i, 1], base_right[i, 2],
                              vec_right[i, 1], vec_right[i, 2],
                              angles='xy', scale_units='xy', scale=1,
                              color=color, width=0.003, headwidth=3, alpha=0.7)
                ax_left.scatter(base_right[i, 1], base_right[i, 2], c=[color], s=20, zorder=5)
        else:
            ax_left.quiver(base_right[:, 1], base_right[:, 2],
                          vec_right[:, 1], vec_right[:, 2],
                          angles='xy', scale_units='xy', scale=1,
                          color='blue', width=0.003, headwidth=3, alpha=0.6)
            ax_left.scatter(base_right[:, 1], base_right[:, 2], c='blue', s=10, alpha=0.6)

    # Breast label
    label_color = 'blue' if color_by == 'breast' else 'black'
    ax_left.text(0, ylim_val*0.82, "Right Breast", ha='center', va='center',
                color=label_color, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                         edgecolor=label_color, alpha=0.9, linewidth=0), zorder=100)

    # --- RIGHT PLOT (LEFT BREAST) ---
    if len(base_left) > 0:
        if color_by == 'dts' and dts_left is not None:
            norm = plt.Normalize(vmin=0, vmax=40)
            cmap_dts = plt.cm.viridis
            colors = cmap_dts(norm(dts_left))
            for i in range(len(base_left)):
                ax_right.quiver(base_left[i, 1], base_left[i, 2],
                               vec_left[i, 1], vec_left[i, 2],
                               angles='xy', scale_units='xy', scale=1,
                               color=colors[i], width=0.003, headwidth=3, alpha=0.7)
            scatter_right_cbar = ax_right.scatter(base_left[:, 1], base_left[:, 2],
                                                  c=dts_left, cmap='viridis', s=20, vmin=0, vmax=40, zorder=5)
        elif color_by == 'subject' and vl_ids_left is not None and subject_color_map:
            for i in range(len(base_left)):
                color = subject_color_map.get(vl_ids_left[i], 'green')
                ax_right.quiver(base_left[i, 1], base_left[i, 2],
                               vec_left[i, 1], vec_left[i, 2],
                               angles='xy', scale_units='xy', scale=1,
                               color=color, width=0.003, headwidth=3, alpha=0.7)
                ax_right.scatter(base_left[i, 1], base_left[i, 2], c=[color], s=20, zorder=5)
        else:
            ax_right.quiver(base_left[:, 1], base_left[:, 2],
                           vec_left[:, 1], vec_left[:, 2],
                           angles='xy', scale_units='xy', scale=1,
                           color='green', width=0.003, headwidth=3, alpha=0.6)
            ax_right.scatter(base_left[:, 1], base_left[:, 2], c='green', s=10, alpha=0.6)

    # Breast label
    label_color = 'green' if color_by == 'breast' else 'black'
    ax_right.text(0, ylim_val*0.82, "Left Breast", ha='center', va='center',
                 color=label_color, fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                          edgecolor=label_color, alpha=0.9, linewidth=0), zorder=100)

    # Colorbar for DTS
    if color_by == 'dts':
        scatter_for_cbar = scatter_right_cbar if scatter_right_cbar is not None else scatter_left_cbar
        if scatter_for_cbar is not None:
            cbar = plt.colorbar(scatter_for_cbar, ax=ax_right, pad=0.02)
            cbar.set_label('DTS (mm)', rotation=270, labelpad=15, fontsize=12)

    # Save
    data_prefix = "Landmarks" if data_type == 'landmarks' else "Nipples"
    if color_by == 'dts':
        filename = f"{data_prefix}_rel_sternum_sagittal_dual_DTS.png"
    elif color_by == 'subject':
        filename = f"{data_prefix}_rel_sternum_sagittal_dual_by_subject.png"
    else:
        filename = f"{data_prefix}_rel_sternum_sagittal_dual.png"

    save_path = Path("..") / "output" / "figs" / "v5" / "landmark vectors" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_3panel_displacement_mechanism(df_ave, save_path=None):
    """
    Create a comprehensive 3-panel figure explaining the displacement mechanism:

    Panel A: Absolute displacement (sternum reference) - coronal view with mean vectors
    Panel B: Vector subtraction explanation - mathematical decomposition
    Panel C: Surgical view (nipple reference) - polar plot showing medial shift

    Args:
        df_ave: DataFrame with landmark and nipple position data
        save_path: Path to save the figure (default: ../output/figs/displacement_mechanism_3panel.png)
    """
    print("\n" + "="*80)
    print("GENERATING 3-PANEL DISPLACEMENT MECHANISM FIGURE")
    print("="*80)

    # Separate left and right breasts
    df_left = df_ave[df_ave['landmark side (prone)'] == 'LB'].copy()
    df_right = df_ave[df_ave['landmark side (prone)'] == 'RB'].copy()

    if len(df_left) == 0 and len(df_right) == 0:
        print("No data available")
        return

    # ========================================================================
    # CALCULATE MEAN DISPLACEMENTS (Coronal Plane: X, Z)
    # ========================================================================

    # Combine both breasts for overall statistics
    df_all = pd.concat([df_left, df_right])

    # Calculate mean landmark displacement (absolute, sternum reference)
    # Landmark coordinates are already relative to sternum (transformed)
    landmark_prone_x = df_all['landmark ave prone transformed x'].mean()
    landmark_prone_z = df_all['landmark ave prone transformed z'].mean()
    landmark_supine_x = df_all['landmark ave supine x'].mean()
    landmark_supine_z = df_all['landmark ave supine z'].mean()

    # Landmark displacement vectors
    landmark_dx_abs = landmark_supine_x - landmark_prone_x
    landmark_dz_abs = landmark_supine_z - landmark_prone_z
    landmark_disp_mag = np.sqrt(landmark_dx_abs**2 + landmark_dz_abs**2)

    # Calculate mean nipple displacement (absolute, sternum reference)
    # Nipple coordinates are NOT relative to sternum, so we need to subtract sternum position
    if len(df_left) > 0:
        # Get nipple positions from left breast (raw, not relative to sternum)
        nipple_prone_x_raw = df_left['left nipple prone transformed x'].mean()
        nipple_prone_z_raw = df_left['left nipple prone transformed z'].mean()
        nipple_supine_x_raw = df_left['left nipple supine x'].mean()
        nipple_supine_z_raw = df_left['left nipple supine z'].mean()

        # Get sternum positions to make nipple coordinates relative to sternum
        sternum_prone_x = df_left['sternum superior prone transformed x'].mean()
        sternum_prone_z = df_left['sternum superior prone transformed z'].mean()
        sternum_supine_x = df_left['sternum superior supine x'].mean()
        sternum_supine_z = df_left['sternum superior supine z'].mean()

        breast_label = "Left Breast"
    else:
        # Use right breast
        nipple_prone_x_raw = df_right['right nipple prone transformed x'].mean()
        nipple_prone_z_raw = df_right['right nipple prone transformed z'].mean()
        nipple_supine_x_raw = df_right['right nipple supine x'].mean()
        nipple_supine_z_raw = df_right['right nipple supine z'].mean()

        # Get sternum positions to make nipple coordinates relative to sternum
        sternum_prone_x = df_right['sternum superior prone transformed x'].mean()
        sternum_prone_z = df_right['sternum superior prone transformed z'].mean()
        sternum_supine_x = df_right['sternum superior supine x'].mean()
        sternum_supine_z = df_right['sternum superior supine z'].mean()

        breast_label = "Right Breast"

    # Make nipple coordinates relative to sternum
    nipple_prone_x = nipple_prone_x_raw - sternum_prone_x
    nipple_prone_z = nipple_prone_z_raw - sternum_prone_z
    nipple_supine_x = nipple_supine_x_raw - sternum_supine_x
    nipple_supine_z = nipple_supine_z_raw - sternum_supine_z

    # Nipple displacement vectors (now relative to sternum)
    nipple_dx_abs = nipple_supine_x - nipple_prone_x
    nipple_dz_abs = nipple_supine_z - nipple_prone_z
    nipple_disp_mag = np.sqrt(nipple_dx_abs**2 + nipple_dz_abs**2)

    # Relative displacement (landmark relative to nipple)
    rel_dx = landmark_dx_abs - nipple_dx_abs
    rel_dz = landmark_dz_abs - nipple_dz_abs
    rel_disp_mag = np.sqrt(rel_dx**2 + rel_dz**2)

    print(f"\nCalculated Mean Displacements ({breast_label}, coronal plane):")
    print(f"  Nipple displacement: X={nipple_dx_abs:.1f}, Z={nipple_dz_abs:.1f}, Mag={nipple_disp_mag:.1f} mm")
    print(f"  Landmark displacement: X={landmark_dx_abs:.1f}, Z={landmark_dz_abs:.1f}, Mag={landmark_disp_mag:.1f} mm")
    print(f"  Relative displacement: X={rel_dx:.1f}, Z={rel_dz:.1f}, Mag={rel_disp_mag:.1f} mm")

    # Create 3-panel figure
    fig = plt.figure(figsize=(20, 6))

    # ========================================================================
    # PANEL A: Absolute Displacement (Sternum Reference) - Coronal View
    # ========================================================================
    ax1 = fig.add_subplot(131)
    ax1.set_xlim(-250, 250)
    ax1.set_ylim(-250, 250)
    ax1.set_aspect('equal')
    ax1.set_xlabel("Right-Left (mm)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Inf-Sup (mm)", fontsize=12, fontweight='bold')
    ax1.set_xticks(np.arange(-250, 251, 50))
    ax1.set_yticks(np.arange(-250, 251, 50))
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Draw sternum (reference line at x=0)
    ax1.axvline(x=0, color='black', linewidth=2, label='Sternum', zorder=1)
    ax1.axhline(y=0, color='gray', linewidth=1, alpha=0.5, linestyle=':', zorder=1)
    ax1.plot(0, 0, 'ko', markersize=8, zorder=5)
    ax1.text(5, -240, 'Sternum\n(Fixed)', ha='left', va='bottom', fontsize=10, fontweight='bold')

    # Plot mean landmark positions and vector
    ax1.plot(landmark_prone_x, landmark_prone_z, 'bo', markersize=12, label='Landmark (Prone)', zorder=5)
    ax1.plot(landmark_supine_x, landmark_supine_z, 'b^', markersize=12, label='Landmark (Supine)', zorder=5)

    # Landmark vector (BLUE)
    ax1.arrow(landmark_prone_x, landmark_prone_z,
             landmark_dx_abs, landmark_dz_abs,
             head_width=10, head_length=8, fc='blue', ec='blue', linewidth=3,
             length_includes_head=True, zorder=10)
    ax1.text(landmark_prone_x + landmark_dx_abs/2, landmark_prone_z + landmark_dz_abs/2 - 20,
            f'Landmark: {landmark_disp_mag:.0f} mm',
            fontsize=11, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Plot mean nipple positions and vector
    ax1.plot(nipple_prone_x, nipple_prone_z, 'ro', markersize=12, label='Nipple (Prone)', zorder=5)
    ax1.plot(nipple_supine_x, nipple_supine_z, 'r^', markersize=12, label='Nipple (Supine)', zorder=5)

    # Nipple vector (RED - should be longer)
    ax1.arrow(nipple_prone_x, nipple_prone_z,
             nipple_dx_abs, nipple_dz_abs,
             head_width=10, head_length=8, fc='red', ec='red', linewidth=3,
             length_includes_head=True, zorder=10)
    ax1.text(nipple_prone_x + nipple_dx_abs/2, nipple_prone_z + nipple_dz_abs/2 + 20,
            f'Nipple: {nipple_disp_mag:.0f} mm',
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.set_title('Panel A: Absolute Displacement\n(Sternum Reference - Coronal View)',
                 fontsize=13, fontweight='bold', pad=15)

    # Add annotation
    ax1.text(-240, 220, 'Displacement relative to sternum.\nNote: Nipple translates further\nthan the deep tissue.',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=10, ha='left', fontweight='bold')

    # ========================================================================
    # PANEL B: Vector Subtraction (Differential Mechanism)
    # ========================================================================
    ax2 = fig.add_subplot(132)
    ax2.set_xlim(-20, 250)
    ax2.set_ylim(-80, 180)
    ax2.axis('off')

    # Vector diagram origin
    origin_x, origin_y = 30, 100
    scale = 1.0

    # Title
    ax2.text(125, 165, 'Vector Subtraction Mathematics',
            fontsize=13, ha='center', fontweight='bold')

    # Draw vectors stacked vertically for subtraction
    # 1. Nipple vector (long red) - top
    ax2.arrow(origin_x, origin_y, nipple_dx_abs*scale, 0,
             head_width=6, head_length=10, fc='red', ec='red', linewidth=3,
             length_includes_head=True, zorder=5)
    ax2.text(origin_x + nipple_dx_abs*scale/2, origin_y + 12,
            r'$\vec{V}_{Nipple}$' + f' = {nipple_disp_mag:.0f} mm',
            fontsize=12, color='red', fontweight='bold', ha='center')

    # 2. Landmark vector (short blue) - middle
    offset_y = -30
    ax2.arrow(origin_x, origin_y + offset_y, landmark_dx_abs*scale, 0,
             head_width=6, head_length=10, fc='blue', ec='blue', linewidth=3,
             length_includes_head=True, zorder=5)
    ax2.text(origin_x + landmark_dx_abs*scale/2, origin_y + offset_y + 12,
            r'$\vec{V}_{Landmark}$' + f' = {landmark_disp_mag:.0f} mm',
            fontsize=12, color='blue', fontweight='bold', ha='center')

    # 3. Subtraction symbol
    ax2.text(origin_x + max(nipple_dx_abs, landmark_dx_abs)*scale + 15, origin_y - 15,
            '−', fontsize=28, fontweight='bold')

    # 4. Dividing line
    ax2.plot([origin_x - 10, origin_x + max(nipple_dx_abs, landmark_dx_abs)*scale + 10],
            [origin_y + offset_y - 15, origin_y + offset_y - 15],
            'k-', linewidth=2)

    # 5. Relative vector (difference) - bottom, pointing MEDIALLY (backwards)
    offset_y2 = -65
    # Start from end of nipple vector to show the gap
    start_x = origin_x + nipple_dx_abs*scale
    ax2.arrow(start_x, origin_y + offset_y2,
             rel_dx*scale, 0,  # This will be NEGATIVE (medial)
             head_width=6, head_length=10, fc='green', ec='green', linewidth=3,
             length_includes_head=True, zorder=5)
    ax2.text(start_x + rel_dx*scale/2, origin_y + offset_y2 - 12,
            r'$\vec{V}_{Rel}$' + f' = {rel_disp_mag:.0f} mm\n(MEDIAL)',
            fontsize=12, color='green', fontweight='bold', ha='center')

    # Add equation
    ax2.text(125, 140,
            r'$\vec{V}_{Rel} = \vec{V}_{Landmark} - \vec{V}_{Nipple}$',
            fontsize=13, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))

    # Explanation box
    explanation = (
        "Differential Displacement Effect:\n\n"
        "The superficial tissue 'outruns'\n"
        "the deep tissue.\n\n"
        "Result: Relative vector points\n"
        "MEDIALLY (backwards)."
    )
    ax2.text(125, 20, explanation,
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9,
                     edgecolor='black', linewidth=2))

    ax2.set_title('Panel B: Differential Mechanism\n(Vector Subtraction)',
                 fontsize=13, fontweight='bold', pad=15)

    # ========================================================================
    # PANEL C: Surgical View (Nipple Reference) - Polar Plot
    # ========================================================================
    ax3 = fig.add_subplot(133, projection='polar')
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)

    # Get nipple-relative coordinates for plotting
    # Use left breast if available, otherwise right
    if len(df_left) > 0:
        df_plot = df_left
        nipple_prone_x_col = 'left nipple prone transformed x'
        nipple_prone_z_col = 'left nipple prone transformed z'
        nipple_supine_x_col = 'left nipple supine x'
        nipple_supine_z_col = 'left nipple supine z'
    else:
        df_plot = df_right
        nipple_prone_x_col = 'right nipple prone transformed x'
        nipple_prone_z_col = 'right nipple prone transformed z'
        nipple_supine_x_col = 'right nipple supine x'
        nipple_supine_z_col = 'right nipple supine z'

    # Calculate positions relative to nipple
    prone_x_rel = df_plot['landmark ave prone transformed x'].values - df_plot[nipple_prone_x_col].values
    prone_z_rel = df_plot['landmark ave prone transformed z'].values - df_plot[nipple_prone_z_col].values
    supine_x_rel = df_plot['landmark ave supine x'].values - df_plot[nipple_supine_x_col].values
    supine_z_rel = df_plot['landmark ave supine z'].values - df_plot[nipple_supine_z_col].values

    # Convert to polar coordinates (coronal plane: X, Z)
    theta_prone = np.arctan2(prone_x_rel, prone_z_rel)
    r_prone = np.sqrt(prone_x_rel**2 + prone_z_rel**2)

    theta_supine = np.arctan2(supine_x_rel, supine_z_rel)
    r_supine = np.sqrt(supine_x_rel**2 + supine_z_rel**2)

    # Plot sample trajectories
    sample_indices = np.linspace(0, len(df_plot)-1, min(25, len(df_plot)), dtype=int)
    for i in sample_indices:
        ax3.plot([theta_prone[i], theta_supine[i]],
                [r_prone[i], r_supine[i]],
                'gray', alpha=0.2, linewidth=1)

    # Plot all points
    ax3.scatter(theta_prone, r_prone, c='lightblue', s=25, alpha=0.4,
               label='Prone', zorder=3, edgecolors='blue', linewidths=0.5)
    ax3.scatter(theta_supine, r_supine, c='lightcoral', s=25, alpha=0.4,
               label='Supine', zorder=3, edgecolors='red', linewidths=0.5)

    # Calculate and plot mean positions using circular mean
    mean_theta_prone = circular_mean_angle(theta_prone)
    mean_r_prone = np.mean(r_prone)
    mean_theta_supine = circular_mean_angle(theta_supine)
    mean_r_supine = np.mean(r_supine)

    # Plot mean trajectory
    ax3.plot([mean_theta_prone, mean_theta_supine],
            [mean_r_prone, mean_r_supine],
            'green', linewidth=4, zorder=10, label='Mean Shift')
    ax3.scatter([mean_theta_prone], [mean_r_prone], c='blue', s=250,
               marker='*', edgecolors='darkblue', linewidths=2,
               label='Mean Prone', zorder=11)
    ax3.scatter([mean_theta_supine], [mean_r_supine], c='red', s=250,
               marker='*', edgecolors='darkred', linewidths=2,
               label='Mean Supine', zorder=11)

    # Add annotation
    ax3.text(np.pi, np.max(r_prone) * 1.15,
            'Displacement relative to nipple.\nDeep landmarks appear\nto shift medially.',
            ha='center', va='center', fontsize=10, color='purple', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='purple', linewidth=2))

    ax3.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax3.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    ax3.set_ylabel('Distance from Nipple (mm)', labelpad=35, fontsize=11)
    ax3.set_title('Panel C: Surgical View\n(Nipple Reference - Polar Plot)',
                 fontsize=13, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Main title
    fig.suptitle('Displacement Mechanism: From Absolute to Relative Reference Frames\n' +
                f'Mean Displacements - Nipple: {nipple_disp_mag:.0f}mm, Landmark: {landmark_disp_mag:.0f}mm, Relative: {rel_disp_mag:.0f}mm (MEDIAL)',
                fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    if save_path is None:
        save_path = Path("..") / "output" / "figs" / "displacement_mechanism_3panel.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 3-panel figure: {save_path}")
    plt.show()
    plt.close()

    print("\n" + "="*80)
    print("3-PANEL FIGURE GENERATION COMPLETE")
    print("="*80)

    return {
        'nipple_displacement': nipple_disp_mag,
        'landmark_displacement': landmark_disp_mag,
        'relative_displacement': rel_disp_mag,
        'nipple_dx': nipple_dx_abs,
        'landmark_dx': landmark_dx_abs,
        'relative_dx': rel_dx
    }


# def plot_sagittal_dual_axes(df_ave, color_by='breast', vl_id=None):
#     """
#     Creates a detailed dual-plot for the Sagittal plane (Left and Right breasts side-by-side)
#     with specific axis configurations:
#     - Shared vertical axis at x=0 (Inf-Sup)
#     - Two separate x-axis origins
#     - Right Breast (Right side): Ant-Post (mm), Blue
#     - Left Breast (Left side): Post-Ant (mm), Green
#
#     Args:
#         df_ave: DataFrame with landmark data
#         color_by: Coloring scheme - 'breast' (default, blue/green),
#                   'subject' (color by VL_ID), or 'dts' (color by distance to skin)
#         vl_id: Optional subject ID to filter data. If None, uses all subjects.
#     """
#     print("\n--- Plotting Dual Sagittal Axes ---")
#     print(f"Color scheme: {color_by}")
#
#     # 1. Filter data if specific subject requested
#     if vl_id is not None:
#         df_subset = df_ave[df_ave['VL_ID'] == vl_id].copy()
#         if df_subset.empty:
#             print(f"Warning: No data found for subject VL_{vl_id}")
#             return
#         print(f"Filtering for subject VL_{vl_id}")
#     else:
#         df_subset = df_ave.copy()
#
#     if df_subset.empty:
#         print("No data found.")
#         return
#
#     # 2. Separate into Left (LB) and Right (RB) breasts
#     left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
#     right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']
#
#     # 3. Helper to extract Base Points and Vectors
#     def get_points_and_vectors(sub_df):
#         if sub_df.empty:
#             return np.empty((0, 3)), np.empty((0, 3)), None, None
#
#         prone_x = sub_df['landmark ave prone transformed x'].values
#         prone_y = sub_df['landmark ave prone transformed y'].values
#         prone_z = sub_df['landmark ave prone transformed z'].values
#         base_points = np.column_stack((prone_x, prone_y, prone_z))
#
#         supine_x = sub_df['landmark ave supine x'].values
#         supine_y = sub_df['landmark ave supine y'].values
#         supine_z = sub_df['landmark ave supine z'].values
#         end_points = np.column_stack((supine_x, supine_y, supine_z))
#
#         vectors = end_points - base_points
#
#         # Extract DTS values if available
#         dts_col = 'Distance to skin (prone) [mm]'
#         dts_values = sub_df[dts_col].values if dts_col in sub_df.columns else None
#
#         # Extract VL_ID for subject coloring
#         vl_ids = sub_df['VL_ID'].values if 'VL_ID' in sub_df.columns else None
#
#         return base_points, vectors, dts_values, vl_ids
#
#     base_left, vec_left, dts_left, vl_ids_left = get_points_and_vectors(left_df)
#     base_right, vec_right, dts_right, vl_ids_right = get_points_and_vectors(right_df)
#
#     # 3a. Prepare colormaps for 'subject' coloring
#     if color_by == 'subject':
#         # Get unique subject IDs from combined left and right data
#         all_vl_ids = []
#         if vl_ids_left is not None:
#             all_vl_ids.extend(vl_ids_left)
#         if vl_ids_right is not None:
#             all_vl_ids.extend(vl_ids_right)
#         unique_subjects = sorted(list(set(all_vl_ids)))
#         n_subjects = len(unique_subjects)
#
#         # Create colormap for subjects
#         subject_cmap = cm.get_cmap('viridis', n_subjects)
#         subject_color_map = {subj: subject_cmap(i) for i, subj in enumerate(unique_subjects)}
#         print(f"Number of unique subjects: {n_subjects}")
#
#     # 4. Setup Plot
#     # Sagittal Plane: Y (Ant-Post) vs Z (Inf-Sup)
#
#     # Create title based on coloring scheme
#     if color_by == 'dts':
#         title_suffix = " (colored by DTS)"
#     elif color_by == 'subject':
#         title_suffix = " (colored by subject)"
#     else:
#         title_suffix = ""
#
#     # Create subplots with sharey=True and constrained_layout for proper spacing
#     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8), sharey=True, constrained_layout=True)
#     fig.suptitle(f"Displacement of landmarks relative to the sternum (sagittal dual view){title_suffix}", fontsize=14)
#
#     # Common Settings
#     ylim_val = 250
#     ax_left.set_ylim(-ylim_val, ylim_val)
#     ax_right.set_ylim(-ylim_val, ylim_val)
#
#     # Set Y-axis Ticks explicitly to include endpoints
#     yticks = np.arange(-250, 251, 50)
#     ax_left.set_yticks(yticks)
#     ax_right.set_yticks(yticks)
#
#     # --- LEFT PLOT (RIGHT BREAST) ---
#     # Metrics: Post-Ant (mm)
#     # Ticks: 150, 100, 50, 0, -50, -100, -150, -200, -250 (Anterior -> Posterior)
#     ax_left.set_xlim(150, -250)
#     ax_left.set_xticks([150, 100, 50, 0, -50, -100, -150, -200, -250])
#
#     # Color xlabel based on mode
#     xlabel_color = 'blue' if color_by == 'breast' else 'black'
#     ax_left.set_xlabel("Post-Ant (mm)", color=xlabel_color, fontsize=12)
#     ax_left.set_ylabel("Inf-Sup (mm)", fontsize=12)
#
#     # Spine Config: Shared Central Axis Effect
#     ax_left.spines['right'].set_position(('data', 0))
#     ax_left.spines['left'].set_visible(False)
#     ax_left.spines['top'].set_visible(False)
#     ax_left.spines['bottom'].set_visible(True)
#     ax_left.spines['right'].set_color('black')
#     ax_left.spines['right'].set_linewidth(1)
#
#     # Origin and Grid
#     ax_left.plot(0, 0, 'ko', markersize=6, zorder=10)
#     ax_left.grid(True, linestyle='--', alpha=0.5)
#     ax_left.set_aspect('equal', adjustable='box')
#
#     # Initialize colorbar placeholder for left plot
#     scatter_left_for_colorbar = None
#
#     # Plot Logic for Right Breast on Left Plot
#     if len(base_right) > 0:
#         if color_by == 'dts' and dts_right is not None:
#             # Color by DTS
#             norm = plt.Normalize(vmin=0, vmax=40)
#             cmap_dts = plt.cm.viridis
#             colors_right = cmap_dts(norm(dts_right))
#
#             for i in range(len(base_right)):
#                 ax_left.quiver(
#                     base_right[i, 1], base_right[i, 2],
#                     vec_right[i, 1], vec_right[i, 2],
#                     angles='xy', scale_units='xy', scale=1,
#                     color=colors_right[i],
#                     width=0.003, headwidth=3, alpha=0.7
#                 )
#
#             scatter_left_for_colorbar = ax_left.scatter(
#                 base_right[:, 1], base_right[:, 2],
#                 c=dts_right, cmap='viridis', s=20, vmin=0, vmax=40, zorder=5
#             )
#
#         elif color_by == 'subject' and vl_ids_right is not None:
#             # Color by subject
#             for i in range(len(base_right)):
#                 color = subject_color_map.get(vl_ids_right[i], 'blue')
#                 ax_left.quiver(
#                     base_right[i, 1], base_right[i, 2],
#                     vec_right[i, 1], vec_right[i, 2],
#                     angles='xy', scale_units='xy', scale=1,
#                     color=color, width=0.003, headwidth=3, alpha=0.7
#                 )
#                 ax_left.scatter(
#                     base_right[i, 1], base_right[i, 2],
#                     c=[color], s=20, zorder=5
#                 )
#         else:
#             # Default: Blue
#             ax_left.quiver(
#                 base_right[:, 1], base_right[:, 2],
#                 vec_right[:, 1], vec_right[:, 2],
#                 angles='xy', scale_units='xy', scale=1,
#                 color='blue', width=0.003, headwidth=3, alpha=0.6
#             )
#             ax_left.scatter(base_right[:, 1], base_right[:, 2], c='blue', s=10, alpha=0.6)
#
#     # Add breast label (always shown for clarity)
#     label_color = 'blue' if color_by == 'breast' else 'black'
#     # Place label in upper portion with optimal positioning and styling for scientific presentation
#     ax_left.text(0, ylim_val*0.82, "Right Breast", ha='center', va='center',
#                 color=label_color, fontsize=14,
#                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
#                          edgecolor=label_color, alpha=0.9, linewidth=0,
#                 zorder=100)) # High zorder ensures text appears on top of data
#
#     # --- RIGHT PLOT (LEFT BREAST) ---
#     # Metrics: Ant-Post (mm)
#     # Ticks: -250, -200, ..., 150 (Posterior -> Anterior)
#     ax_right.set_xlim(-250, 150)
#     ax_right.set_xticks(np.arange(-250, 151, 50))
#
#     # Color xlabel based on mode
#     xlabel_color = 'green' if color_by == 'breast' else 'black'
#     ax_right.set_xlabel("Ant-Post (mm)", color=xlabel_color, fontsize=12)
#
#     # Move left spine to x=0
#     ax_right.spines['left'].set_position(('data', 0))
#     ax_right.spines['right'].set_visible(False)
#     ax_right.spines['top'].set_visible(False)
#     ax_right.spines['bottom'].set_visible(True)
#     ax_right.spines['left'].set_color('black')
#     ax_right.spines['left'].set_linewidth(1)
#
#     # Origin and Grid
#     ax_right.plot(0, 0, 'ko', markersize=6, zorder=10)
#     ax_right.grid(True, linestyle='--', alpha=0.5)
#     ax_right.set_aspect('equal', adjustable='box')
#
#     # Initialize colorbar placeholder for right plot
#     scatter_right_for_colorbar = None
#
#     # Plot Logic for Left Breast on Right Plot
#     if len(base_left) > 0:
#         if color_by == 'dts' and dts_left is not None:
#             # Color by DTS
#             norm = plt.Normalize(vmin=0, vmax=40)
#             cmap_dts = plt.cm.viridis
#             colors_left = cmap_dts(norm(dts_left))
#
#             for i in range(len(base_left)):
#                 ax_right.quiver(
#                     base_left[i, 1], base_left[i, 2],
#                     vec_left[i, 1], vec_left[i, 2],
#                     angles='xy', scale_units='xy', scale=1,
#                     color=colors_left[i],
#                     width=0.003, headwidth=3, alpha=0.7
#                 )
#
#             scatter_right_for_colorbar = ax_right.scatter(
#                 base_left[:, 1], base_left[:, 2],
#                 c=dts_left, cmap='viridis', s=20, vmin=0, vmax=40, zorder=5
#             )
#
#         elif color_by == 'subject' and vl_ids_left is not None:
#             # Color by subject
#             for i in range(len(base_left)):
#                 color = subject_color_map.get(vl_ids_left[i], 'green')
#                 ax_right.quiver(
#                     base_left[i, 1], base_left[i, 2],
#                     vec_left[i, 1], vec_left[i, 2],
#                     angles='xy', scale_units='xy', scale=1,
#                     color=color, width=0.003, headwidth=3, alpha=0.7
#                 )
#                 ax_right.scatter(
#                     base_left[i, 1], base_left[i, 2],
#                     c=[color], s=20, zorder=5
#                 )
#         else:
#             # Default: Green
#             ax_right.quiver(
#                 base_left[:, 1], base_left[:, 2],
#                 vec_left[:, 1], vec_left[:, 2],
#                 angles='xy', scale_units='xy', scale=1,
#                 color='green', width=0.003, headwidth=3, alpha=0.6
#             )
#             ax_right.scatter(base_left[:, 1], base_left[:, 2], c='green', s=10, alpha=0.6)
#
#     # Add breast label (always shown for clarity)
#     label_color = 'green' if color_by == 'breast' else 'black'
#     # Place label in upper portion with optimal positioning and styling for scientific presentation
#     ax_right.text(0, ylim_val*0.82, "Left Breast", ha='center', va='center',
#                  color=label_color, fontsize=14,
#                  bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
#                           edgecolor=label_color, alpha=0.9, linewidth=0),
#                  zorder=100)  # High zorder ensures text appears on top of data
#
#     # Add colorbar for DTS coloring
#     if color_by == 'dts':
#         # Use the scatter from either plot that has data
#         scatter_for_cbar = scatter_right_for_colorbar if scatter_right_for_colorbar is not None else scatter_left_for_colorbar
#         if scatter_for_cbar is not None:
#             # Add colorbar on the right side
#             cbar = plt.colorbar(scatter_for_cbar, ax=ax_right, pad=0.02)
#             cbar.set_label('DTS (mm)', rotation=270, labelpad=15, fontsize=12)
#
#     # Create filename based on coloring scheme
#     if color_by == 'dts':
#         filename = "Vectors_rel_sternum_sagittal_dual_DTS.png"
#     elif color_by == 'subject':
#         filename = "Vectors_rel_sternum_sagittal_dual_by_subject.png"
#     else:
#         filename = "Vectors_rel_sternum_sagittal_dual.png"
#
#     save_path = Path("..") / "output" / "figs" / "v5" / "landmark vectors" / filename
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(save_path, dpi=300)
#     print(f"Saved: {save_path}")
#     plt.show()
#     plt.close(fig)
#     print(f"✓ Completed dual sagittal plot with '{color_by}' coloring")


def plot_nipple_relative_landmarks(
    df_ave,
    vl_id=None,
    title="Nipple Relative Landmarks",
    save_path=None,
    use_dts_cmap=True
):
    """
    Derives Nipple-relative plotting data from DataFrame.

    Args:
        df_ave: DataFrame with landmark data including:
                - 'landmark ave prone transformed x/y/z' (prone positions relative to sternum)
                - 'landmark ave supine x/y/z' (supine positions relative to sternum)
                - 'left nipple prone transformed x/y/z' (left nipple prone position)
                - 'left nipple supine x/y/z' (left nipple supine position)
                - 'right nipple prone transformed x/y/z' (right nipple prone position)
                - 'right nipple supine x/y/z' (right nipple supine position)
                - 'landmark side (prone)' (LB or RB)
                - 'Distance to skin (prone) [mm]' (optional, for coloring)
                - 'VL_ID' (optional, for filtering by subject)
        vl_id: Optional subject ID to filter data. If None, uses all subjects.
        title: Plot title
        save_path: Path to save the plots
        use_dts_cmap: Whether to use the DTS colormap
    """
    print("\n--- Plotting Nipple-Relative Motion ---")

    # 1. Filter data if specific subject requested
    if vl_id is not None:
        df_subset = df_ave[df_ave['VL_ID'] == vl_id].copy()
        if df_subset.empty:
            print(f"Warning: No data found for subject VL_{vl_id}")
            return
    else:
        df_subset = df_ave.copy()

    # 2. Separate into Left (LB) and Right (RB) breasts
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Helper to extract nipple-relative base points and vectors
    def get_nipple_relative_points_and_vectors(sub_df, is_left_breast):
        if sub_df.empty:
            return np.empty((0, 3)), np.empty((0, 3)), None, np.empty((0, 3)), np.empty((0, 3))

        # Extract landmark prone positions (relative to sternum)
        prone_x = sub_df['landmark ave prone transformed x'].values
        prone_y = sub_df['landmark ave prone transformed y'].values
        prone_z = sub_df['landmark ave prone transformed z'].values

        # Extract landmark supine positions (relative to sternum)
        supine_x = sub_df['landmark ave supine x'].values
        supine_y = sub_df['landmark ave supine y'].values
        supine_z = sub_df['landmark ave supine z'].values

        # Extract nipple positions from DataFrame columns
        if is_left_breast:
            # Left nipple prone position
            nipple_prone_x = sub_df['left nipple prone transformed x'].values
            nipple_prone_y = sub_df['left nipple prone transformed y'].values
            nipple_prone_z = sub_df['left nipple prone transformed z'].values

            # Left nipple supine position
            nipple_supine_x = sub_df['left nipple supine x'].values
            nipple_supine_y = sub_df['left nipple supine y'].values
            nipple_supine_z = sub_df['left nipple supine z'].values
        else:
            # Right nipple prone position
            nipple_prone_x = sub_df['right nipple prone transformed x'].values
            nipple_prone_y = sub_df['right nipple prone transformed y'].values
            nipple_prone_z = sub_df['right nipple prone transformed z'].values

            # Right nipple supine position
            nipple_supine_x = sub_df['right nipple supine x'].values
            nipple_supine_y = sub_df['right nipple supine y'].values
            nipple_supine_z = sub_df['right nipple supine z'].values

        sternum_prone_x = sub_df['sternum superior prone transformed x'].values
        sternum_prone_y = sub_df['sternum superior prone transformed y'].values
        sternum_prone_z = sub_df['sternum superior prone transformed z'].values

        sternum_supine_x = sub_df['sternum superior supine x'].values
        sternum_supine_y = sub_df['sternum superior supine y'].values
        sternum_supine_z = sub_df['sternum superior supine z'].values

        # Calculate nipple displacement vector (supine - prone)
        nipple_disp_x = (nipple_supine_x - sternum_supine_x) - (nipple_prone_x - sternum_prone_x)
        nipple_disp_y = (nipple_supine_y - sternum_supine_y) - (nipple_prone_y - sternum_prone_y)
        nipple_disp_z = (nipple_supine_z - sternum_supine_z) - (nipple_prone_z - sternum_prone_z)
        nipple_displacement = np.column_stack((nipple_disp_x, nipple_disp_y, nipple_disp_z))

        # Calculate base points: landmark position relative to prone nipple
        base_x = prone_x - (nipple_prone_x - sternum_prone_x)
        base_y = prone_y - (nipple_prone_y - sternum_prone_y)
        base_z = prone_z - (nipple_prone_z - sternum_prone_z)
        base_points = np.column_stack((base_x, base_y, base_z))

        # Calculate landmark displacement relative to sternum
        lm_disp_x = supine_x - prone_x
        lm_disp_y = supine_y - prone_y
        lm_disp_z = supine_z - prone_z
        lm_displacement = np.column_stack((lm_disp_x, lm_disp_y, lm_disp_z))

        # Calculate vectors: landmark motion relative to nipple
        # Subtract nipple motion to show intrinsic deformation
        ld_vectors_rel_nipple = lm_displacement - nipple_displacement

        # end_x =  supine_x - (nipple_supine_x - sternum_supine_x)
        # end_y =  supine_y - (nipple_supine_y - sternum_supine_y)
        # end_z =  supine_z - (nipple_supine_z - sternum_supine_z)
        # end_points = np.column_stack((end_x, end_y, end_z))
        # vectors_rel_nipple = end_points - base_points

        # Extract DTS values if available
        dts_col = 'Distance to skin (prone) [mm]'
        dts_values = sub_df[dts_col].values if dts_col in sub_df.columns else None

        return base_points, ld_vectors_rel_nipple, dts_values, lm_displacement, nipple_displacement

    # Extract data for both breasts
    base_left, vec_left, dts_left, lm_disp_left, nipple_disp_left = get_nipple_relative_points_and_vectors(left_df, is_left_breast=True)
    base_right, vec_right, dts_right, lm_disp_right, nipple_disp_right = get_nipple_relative_points_and_vectors(right_df, is_left_breast=False)

    # 4. Call the specialized plotting function
    plot_nipple_relative_vectors(
        base_point_left=base_left,
        vector_left=vec_left,
        base_point_right=base_right,
        vector_right=vec_right,
        dts_left=dts_left,
        dts_right=dts_right,
        title=title,
        save_path=save_path,
        use_dts_cmap=use_dts_cmap
    )
    return base_left, base_right, vec_left, vec_right, lm_disp_left, lm_disp_right, nipple_disp_left, nipple_disp_right


def analyze_3d_stability(df):
    # 1. Calculate the 3D Vector Magnitude (Euclidean Distance)
    # Assumes X, Y, Z are aligned to the ribcage
    dx = df['X_supine'] - df['X_prone']
    dy = df['Y_supine'] - df['Y_prone']
    dz = df['Z_supine'] - df['Z_prone']
    df['total_3d_shift'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # 2. Correlate with Prone Depth (DTS)
    rho, pval = stats.spearmanr(df['DTS (prone) [mm]'], df['total_3d_shift'], nan_policy='omit')

    # 3. Categorical Analysis (Superficial <= 10mm vs Deep > 10mm)
    superficial_3d = df[df['DTS (prone) [mm]'] <= 10]['total_3d_shift']
    deep_3d = df[df['DTS (prone) [mm]'] > 10]['total_3d_shift']
    t_stat, p_comp = stats.mannwhitneyu(superficial_3d, deep_3d)

    print(f"--- 3D STABILITY ANALYSIS (Aligned to Ribcage) ---")
    print(f"Correlation (Depth vs Total Shift): rho = {rho:.3f}, p = {pval:.4e}")
    print(f"Superficial 3D Shift Mean: {superficial_3d.mean():.2f} mm")
    print(f"Deep 3D Shift Mean: {deep_3d.mean():.2f} mm")
    print(f"Significant Difference? {'Yes' if p_comp < 0.05 else 'No'} (p = {p_comp:.4e})")


# analyze_3d_stability(df_analysis)


def parse_clock_time(time_str):
    """
    Parse clock time string (e.g., "2:30", "9:00") to hours and degrees.

    Args:
        time_str: Clock position as string (e.g., "2:30", "central")

    Returns:
        clock_hours: Clock position in decimal hours (e.g., 2.5 for "2:30")
        angle_degrees: Angle in degrees (0-360, 0=12 o'clock, clockwise)
    """
    if pd.isna(time_str) or str(time_str).lower() == 'central':
        return np.nan, np.nan

    try:
        # Parse "HH:MM" format
        parts = str(time_str).split(':')
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0

        # Convert to decimal hours
        clock_hours = hours + minutes / 60.0

        # Convert to degrees (12 o'clock = 0°, clockwise)
        # Each hour = 30 degrees
        angle_degrees = (clock_hours % 12) * 30

        return clock_hours, angle_degrees
    except:
        return np.nan, np.nan


def circular_mean_angle(angles_rad):
    """
    Calculate the circular mean of angles (in radians).
    This correctly handles the circular nature of angles (e.g., 350° and 10° average to 0°, not 180°).

    This is critical for polar plots where averaging 350° and 10° should give 0° (or 360°),
    not 180° as arithmetic mean would incorrectly calculate.

    Args:
        angles_rad: Array of angles in radians

    Returns:
        mean_angle_rad: Circular mean angle in radians
    """
    # Convert angles to unit vectors on the unit circle
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))

    # Calculate mean angle from the mean vector
    mean_angle = np.arctan2(sin_mean, cos_mean)

    return mean_angle


def calculate_clock_position(y, z):
    """
    Calculate clock position from nipple-relative coordinates.
    Using standard breast clock convention:
    - 12 o'clock = Superior (positive z)
    - 3 o'clock = Lateral for right breast (positive y), Medial for left breast (negative y)
    - 6 o'clock = Inferior (negative z)
    - 9 o'clock = Medial for right breast (negative y), Lateral for left breast (positive y)

    Args:
        y: Lateral-medial coordinate (relative to nipple)
        z: Inferior-superior coordinate (relative to nipple)

    Returns:
        clock_position: Clock position in hours (1-12)
        angle_degrees: Angle in degrees (0-360, 0=12 o'clock)
    """
    # Calculate angle in radians (atan2 gives -pi to pi)
    angle_rad = np.arctan2(y, z)

    # Convert to degrees (0-360, with 0 at top/12 o'clock)
    angle_degrees = np.degrees(angle_rad)
    if angle_degrees < 0:
        angle_degrees += 360

    # Convert to clock position (12 at top, clockwise)
    # 0 degrees = 12 o'clock, 90 degrees = 3 o'clock, etc.
    clock_position = 12 - (angle_degrees / 30)  # 30 degrees per hour
    if clock_position <= 0:
        clock_position += 12
    if clock_position > 12:
        clock_position -= 12

    return clock_position, angle_degrees


def analyse_clock_position_rotation(df_ave, base_left=None, base_right=None, vec_left=None, vec_right=None, save_dir=None):
    """
    Analyze clock position rotation from prone to supine, focusing on gravity-induced
    lateral displacement. Creates polar plots showing rotation patterns.

    Tests the hypothesis: "We observed a mean clockwise rotation of X hours (Y degrees)
    for left-sided tumors, consistent with gravity-induced lateral displacement in the
    supine position."

    Args:
        df_ave: DataFrame with landmark data including nipple-relative positions
        base_left: Nx3 array of prone landmark positions relative to left nipple (from plot_nipple_relative_landmarks)
        base_right: Nx3 array of prone landmark positions relative to right nipple (from plot_nipple_relative_landmarks)
        vec_left: Nx3 array of displacement vectors for left breast (from plot_nipple_relative_landmarks)
        vec_right: Nx3 array of displacement vectors for right breast (from plot_nipple_relative_landmarks)
        save_dir: Directory to save plots (default: ../output/figs/clock_analysis/)
    """
    print("\n" + "="*80)
    print("CLOCK POSITION ROTATION ANALYSIS")
    print("="*80)

    if save_dir is None:
        save_dir = Path("..") / "output" / "figs" / "v5" / "clock_analysis"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Calculate nipple-relative positions for prone and supine
    df = df_ave.copy()

    # Separate left and right breasts
    df_left = df[df['landmark side (prone)'] == 'LB'].copy()
    df_right = df[df['landmark side (prone)'] == 'RB'].copy()

    # Helper function to compute clock angles and radii from nipple-relative coordinates
    def compute_clock_positions(base_points, vectors):
        """
        Compute clock positions (angle and radius) from nipple-relative coordinates.
        Uses coronal plane projection (X, Z) for 2D clock face.

        Args:
            base_points: Nx3 array of prone positions relative to nipple [X, Y, Z]
            vectors: Nx3 array of displacement vectors [dX, dY, dZ]

        Returns:
            angle_prone, angle_supine, distance_prone, distance_supine, angle_rotation, clock_rotation
        """
        if len(base_points) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # Prone positions (already relative to nipple)
        prone_x = base_points[:, 0]  # X: right-left
        prone_y = base_points[:, 1]  # Y: ant-post
        prone_z = base_points[:, 2]  # Z: inf-sup

        # Supine positions
        supine_x = prone_x + vectors[:, 0]
        supine_y = prone_y + vectors[:, 1]
        supine_z = prone_z + vectors[:, 2]

        # Clock face uses coronal plane (X, Z)
        # Angle: 0° = superior (12 o'clock), positive = clockwise
        # arctan2(x, z) gives angle where 0 = +Z axis, positive toward +X
        angle_prone = np.degrees(np.arctan2(prone_x, prone_z))
        angle_supine = np.degrees(np.arctan2(supine_x, supine_z))

        # Normalize to [0, 360)
        angle_prone = (angle_prone + 360) % 360
        angle_supine = (angle_supine + 360) % 360

        # Distance (radius on coronal plane)
        distance_prone = np.sqrt(prone_x**2 + prone_z**2)
        distance_supine = np.sqrt(supine_x**2 + supine_z**2)

        # Calculate rotation (handle wraparound)
        angle_diff = angle_supine - angle_prone
        # Normalize to [-180, 180]
        angle_diff = np.where(angle_diff > 180, angle_diff - 360, angle_diff)
        angle_diff = np.where(angle_diff < -180, angle_diff + 360, angle_diff)

        clock_rotation = angle_diff / 30.0  # Convert to hours

        return angle_prone, angle_supine, distance_prone, distance_supine, angle_diff, clock_rotation

    # Calculate clock positions for each breast using nipple-relative coordinates
    processed_dfs = {}

    for breast_side, df_orig, base_points, vectors in [
        ('Left', df_left, base_left, vec_left),
        ('Right', df_right, base_right, vec_right)
    ]:
        if len(df_orig) == 0 or base_points is None or len(base_points) == 0:
            continue

        # Work on a copy
        df_subset = df_orig.copy()

        # Ensure we have matching lengths
        if len(df_subset) != len(base_points):
            print(f"Warning: DataFrame length ({len(df_subset)}) doesn't match base_points length ({len(base_points)}) for {breast_side} breast")
            continue

        # Compute clock positions from nipple-relative coordinates
        angle_prone, angle_supine, distance_prone, distance_supine, angle_rotation, clock_rotation = \
            compute_clock_positions(base_points, vectors)

        # Add to dataframe
        df_subset['angle_prone'] = angle_prone
        df_subset['angle_supine'] = angle_supine
        df_subset['distance_prone'] = distance_prone
        df_subset['distance_supine'] = distance_supine
        df_subset['angle_rotation'] = angle_rotation
        df_subset['clock_rotation'] = clock_rotation

        # Convert angles to clock hours for display
        df_subset['clock_prone'] = ((angle_prone / 30.0) % 12)
        df_subset['clock_supine'] = ((angle_supine / 30.0) % 12)

        # Replace 0 with 12 for clock display
        df_subset['clock_prone'] = df_subset['clock_prone'].replace(0, 12)
        df_subset['clock_supine'] = df_subset['clock_supine'].replace(0, 12)

        # Distance change
        df_subset['distance_change'] = distance_supine - distance_prone

        # Remove any NaN rows
        df_clean = df_subset.dropna(subset=['angle_prone', 'angle_supine']).copy()

        # Store processed dataframe
        processed_dfs[breast_side] = df_clean

    # Update df_left and df_right for subsequent steps
    df_left = processed_dfs.get('Left', pd.DataFrame())
    df_right = processed_dfs.get('Right', pd.DataFrame())


    # --- PART 1: CLOCK FREQUENCY ANALYSIS (Count & Visualization) ---
    print("\n" + "-"*80)
    print("CLOCK POSITION FREQUENCY ANALYSIS")
    print("-"*80)

    # We will bin the clock data into half-hour intervals
    # 12:00, 12:30, 1:00, 1:30, etc.
    def round_to_half_hour(h):
        if pd.isna(h):
            return h
        # Round to nearest 0.5
        r = round(h * 2) / 2.0
        # Handle 0 -> 12
        if r == 0:
            r = 12.0
        elif r > 12:
            r = r - 12
        return r

    def format_clock_time(h):
        """Format decimal hours as clock time (e.g., 1.5 -> '1:30')"""
        if pd.isna(h):
            return "N/A"
        whole = int(h)
        frac = h - whole
        if abs(frac - 0.5) < 0.01:
            return f"{whole}:30"
        else:
            return f"{whole}:00"

    for df_subset, breast_side in [(df_left, 'Left'), (df_right, 'Right')]:
        if len(df_subset) == 0:
            continue

        print(f"\n{breast_side} Breast Frequency Distribution (Half-Hour Precision):")

        # Round to half-hour intervals for frequency counting
        hours_prone = df_subset['clock_prone'].apply(round_to_half_hour)
        hours_supine = df_subset['clock_supine'].apply(round_to_half_hour)

        # Create counts
        counts_prone = hours_prone.value_counts().sort_index()
        counts_supine = hours_supine.value_counts().sort_index()

        # Ensure all half-hour intervals 1:00 to 12:30 exist
        half_hour_bins = []
        for h in range(1, 13):
            half_hour_bins.append(float(h))
            half_hour_bins.append(h + 0.5)

        for hh in half_hour_bins:
            if hh not in counts_prone.index:
                counts_prone.loc[hh] = 0
            if hh not in counts_supine.index:
                counts_supine.loc[hh] = 0

        counts_prone = counts_prone.sort_index()
        counts_supine = counts_supine.sort_index()

        # Print Table
        print(f"{'Time':<8} | {'Prone N':<10} | {'Supine N':<10}")
        print("-" * 35)
        for hh in half_hour_bins:
            time_str = format_clock_time(hh)
            print(f"{time_str:<8} | {int(counts_prone.loc[hh]):<10} | {int(counts_supine.loc[hh]):<10}")

        # --- ROSE PLOT (Histogram) with Half-Hour Intervals ---
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Setup bins for rose plot (centered on half-hour intervals)
        # Create bins for all half-hour positions
        theta_centers = []
        widths = []
        radii_prone_plot = []
        radii_supine_plot = []

        for hh in half_hour_bins:
            # Convert clock hours to degrees: 12:00 -> 0°, 1:00 -> 30°, etc.
            # Handle 12 specially (it's at 0°)
            if hh >= 12:
                deg = (hh - 12) * 30
            else:
                deg = hh * 30
            theta_centers.append(np.radians(deg))
            widths.append(np.radians(15))  # 15 degrees width for half-hour bins
            radii_prone_plot.append(counts_prone.loc[hh])
            radii_supine_plot.append(counts_supine.loc[hh])

        # Plot Prone (Blue)
        bars = ax.bar(theta_centers, radii_prone_plot, width=np.radians(15), bottom=0.0,
                     color='blue', alpha=0.3, edgecolor='blue', linewidth=1, label='Prone')

        # Plot Supine (Red)
        bars2 = ax.bar(theta_centers, radii_supine_plot, width=np.radians(12), bottom=0.0,
                      color='red', alpha=0.5, edgecolor='red', linewidth=1, label='Supine')

        # Set tick labels for whole hours only (to avoid clutter)
        ax.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax.set_title(f'{breast_side} Breast: Clock Position Frequency (Half-Hour Bins)',
                    pad=20, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=11)

        # Save
        filename = f"clock_frequency_{breast_side.lower()}_breast_half_hour.png"
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved frequency plot: {save_path}")
        plt.show()
        plt.close()

    # --- PART 2: ROTATION & SHIFT ANALYSIS ---
    # Statistical Analysis
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY (Rotation & Shift)")
    print("-"*80)

    for df_subset, breast_side in [(df_left, 'Left'), (df_right, 'Right')]:
        if len(df_subset) == 0:
            print(f"\n{breast_side} Breast: No data available")
            continue

        print(f"\n{breast_side} Breast (n={len(df_subset)}):")
        print("-"*40)

        # Clock rotation statistics
        mean_rotation_deg = df_subset['angle_rotation'].mean()
        std_rotation_deg = df_subset['angle_rotation'].std()
        mean_rotation_hours = df_subset['clock_rotation'].mean()

        print(f"Mean rotation: {mean_rotation_hours:.2f} hours ({mean_rotation_deg:.1f}°)")
        print(f"Std rotation: {std_rotation_deg:.1f}°")
        print(f"Median rotation: {df_subset['angle_rotation'].median():.1f}°")
        print(f"Range: [{df_subset['angle_rotation'].min():.1f}°, {df_subset['angle_rotation'].max():.1f}°]")

        # Test if rotation is significantly different from zero
        t_stat, p_val = stats.ttest_1samp(df_subset['angle_rotation'], 0)
        print(f"\nTest if rotation ≠ 0: t={t_stat:.3f}, p={p_val:.4e}")

        if p_val < 0.05:
            direction = "clockwise" if mean_rotation_deg > 0 else "counterclockwise"
            print(f"✓ Significant {direction} rotation detected!")
        else:
            print("✗ No significant rotation detected")

        # Distance change statistics
        mean_dist_change = df_subset['distance_change'].mean()
        print(f"\nMean distance change: {mean_dist_change:.2f} mm")
        print(f"Mean prone distance: {df_subset['distance_prone'].mean():.2f} mm")
        print(f"Mean supine distance: {df_subset['distance_supine'].mean():.2f} mm")

        # Test if distance changes
        t_stat_dist, p_val_dist = stats.ttest_rel(df_subset['distance_prone'],
                                                   df_subset['distance_supine'])
        print(f"Test if distance changed: t={t_stat_dist:.3f}, p={p_val_dist:.4e}")

    # Create Polar Plots
    print("\n" + "-"*80)
    print("GENERATING POLAR PLOTS")
    print("-"*80)

    for df_subset, breast_side in [(df_left, 'Left'), (df_right, 'Right')]:
        if len(df_subset) == 0:
            continue

        # Create polar plot
        fig = plt.figure(figsize=(14, 6))

        # Plot 1: Individual trajectories
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')  # 0° at top (12 o'clock)
        ax1.set_theta_direction(-1)  # Clockwise

        # Convert angles to radians for plotting
        theta_prone = np.radians(df_subset['angle_prone'].values)
        theta_supine = np.radians(df_subset['angle_supine'].values)
        r_prone = df_subset['distance_prone'].values
        r_supine = df_subset['distance_supine'].values

        # Plot trajectories
        for i in range(len(df_subset)):
            # Draw line from prone to supine
            ax1.plot([theta_prone[i], theta_supine[i]],
                    [r_prone[i], r_supine[i]],
                    'gray', alpha=0.3, linewidth=1)

        # Plot points
        ax1.scatter(theta_prone, r_prone, c='blue', s=50, alpha=0.6,
                   label='Prone', zorder=5, edgecolors='darkblue')
        ax1.scatter(theta_supine, r_supine, c='red', s=50, alpha=0.6,
                   label='Supine', zorder=5, edgecolors='darkred')

        # Add clock hour labels
        ax1.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax1.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])

        ax1.set_ylabel('Distance from Nipple (mm)', labelpad=30)
        ax1.set_title(f'{breast_side} Breast: Prone→Supine Shift\n(Individual Landmarks)',
                     fontsize=12, pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax1.grid(True, alpha=0.3)

        # Plot 2: Mean shift with confidence
        ax2 = fig.add_subplot(122, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)

        # Calculate mean positions using circular mean for angles
        # CRITICAL: Must use circular mean for angles, not arithmetic mean!
        # Example: mean of 350° and 10° is 0°, not 180°
        theta_prone_rad = np.radians(df_subset['angle_prone'].values)
        theta_supine_rad = np.radians(df_subset['angle_supine'].values)
        mean_theta_prone = circular_mean_angle(theta_prone_rad)
        mean_theta_supine = circular_mean_angle(theta_supine_rad)
        mean_r_prone = df_subset['distance_prone'].mean()
        mean_r_supine = df_subset['distance_supine'].mean()

        # Plot individual points (lighter)
        ax2.scatter(theta_prone, r_prone, c='lightblue', s=30, alpha=0.3, zorder=3)
        ax2.scatter(theta_supine, r_supine, c='lightcoral', s=30, alpha=0.3, zorder=3)

        # Plot mean trajectory
        ax2.plot([mean_theta_prone, mean_theta_supine],
                [mean_r_prone, mean_r_supine],
                'black', linewidth=3, zorder=10)
        ax2.annotate('',
                    xy=(mean_theta_supine, mean_r_supine),  # Arrow head (end point)
                    xytext=(mean_theta_prone, mean_r_prone),  # Arrow tail (start point)
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    zorder=10)

        # Plot mean points
        ax2.scatter([mean_theta_prone], [mean_r_prone], c='blue', s=200,
                   marker='*', edgecolors='darkblue', linewidths=1,
                   label='Mean Prone', zorder=11)
        ax2.scatter([mean_theta_supine], [mean_r_supine], c='red', s=200,
                   marker='*', edgecolors='darkred', linewidths=1,
                   label='Mean Supine', zorder=11)

        ax2.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax2.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax2.set_ylabel('Distance from Nipple (mm)', labelpad=30)

        # Add rotation annotation
        mean_rotation = df_subset['angle_rotation'].mean()
        mean_rotation_hours = df_subset['clock_rotation'].mean()
        direction = "Clockwise" if mean_rotation > 0 else "Counterclockwise"

        title_text = f'{breast_side} Breast: Mean Rotation\n'
        title_text += f'{direction}: {abs(mean_rotation_hours):.2f} hours ({abs(mean_rotation):.1f}°)'
        ax2.set_title(title_text, fontsize=12, pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = f"clock_rotation_{breast_side.lower()}_breast.png"
        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
        plt.close()

    # Create combined comparison plot - Individual points
    # Right breast on LEFT subplot, Left breast on RIGHT subplot
    if len(df_left) > 0 and len(df_right) > 0:
        print("\nGenerating combined polar plot with individual points...")
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

        # LEFT SUBPLOT: RIGHT BREAST
        ax_left.set_theta_zero_location('N')
        ax_left.set_theta_direction(-1)

        theta_prone_r = np.radians(df_right['angle_prone'].values)
        theta_supine_r = np.radians(df_right['angle_supine'].values)
        r_prone_r = df_right['distance_prone'].values
        r_supine_r = df_right['distance_supine'].values

        # Plot all trajectories
        for i in range(len(df_right)):
            ax_left.plot([theta_prone_r[i], theta_supine_r[i]],
                        [r_prone_r[i], r_supine_r[i]],
                        'gray', alpha=0.3, linewidth=1)

        # Plot individual points
        ax_left.scatter(theta_prone_r, r_prone_r, c='lightblue', s=30, alpha=0.3, zorder=3)
        ax_left.scatter(theta_supine_r, r_supine_r, c='lightcoral', s=30, alpha=0.3, zorder=3)

        # Mean trajectory - use circular mean for angles
        mean_theta_prone_r = circular_mean_angle(theta_prone_r)
        mean_theta_supine_r = circular_mean_angle(theta_supine_r)
        mean_r_prone_r = df_right['distance_prone'].mean()
        mean_r_supine_r = df_right['distance_supine'].mean()

        ax_left.plot([mean_theta_prone_r, mean_theta_supine_r],
                    [mean_r_prone_r, mean_r_supine_r],
                    'black', linewidth=2, zorder=10)
        ax_left.annotate('',
                        xy=(mean_theta_supine_r, mean_r_supine_r),
                        xytext=(mean_theta_prone_r, mean_r_prone_r),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2),
                        zorder=10)

        # Plot mean points
        ax_left.scatter([mean_theta_prone_r], [mean_r_prone_r], c='blue', s=200,
                       marker='*', edgecolors='darkblue', linewidths=1,
                       label='Mean Prone', zorder=11)
        ax_left.scatter([mean_theta_supine_r], [mean_r_supine_r], c='red', s=200,
                       marker='*', edgecolors='darkred', linewidths=1,
                       label='Mean Supine', zorder=11)

        ax_left.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_left.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_left.set_ylabel('Distance from Nipple (mm)', labelpad=30, fontsize=11)

        mean_rotation_r = df_right['angle_rotation'].mean()
        mean_rotation_hours_r = df_right['clock_rotation'].mean()
        direction_r = "Clockwise" if mean_rotation_r > 0 else "Counterclockwise"

        ax_left.set_title(
            f'Landmarks in right breast\n{direction_r}: {abs(mean_rotation_hours_r):.2f}h ({abs(mean_rotation_r):.1f}°)',
            fontsize=12, color='black', pad=20)
        ax_left.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, ncol=1)
        ax_left.grid(True, alpha=0.3)

        # RIGHT SUBPLOT: LEFT BREAST
        ax_right.set_theta_zero_location('N')
        ax_right.set_theta_direction(-1)

        theta_prone_l = np.radians(df_left['angle_prone'].values)
        theta_supine_l = np.radians(df_left['angle_supine'].values)
        r_prone_l = df_left['distance_prone'].values
        r_supine_l = df_left['distance_supine'].values

        # Plot all trajectories
        for i in range(len(df_left)):
            ax_right.plot([theta_prone_l[i], theta_supine_l[i]],
                         [r_prone_l[i], r_supine_l[i]],
                         'gray', alpha=0.3, linewidth=1)

        # Plot individual points
        ax_right.scatter(theta_prone_l, r_prone_l, c='lightblue', s=30, alpha=0.3, zorder=3)
        ax_right.scatter(theta_supine_l, r_supine_l, c='lightcoral', s=30, alpha=0.3, zorder=3)

        # Mean trajectory - use circular mean for angles
        mean_theta_prone_l = circular_mean_angle(theta_prone_l)
        mean_theta_supine_l = circular_mean_angle(theta_supine_l)
        mean_r_prone_l = df_left['distance_prone'].mean()
        mean_r_supine_l = df_left['distance_supine'].mean()

        ax_right.plot([mean_theta_prone_l, mean_theta_supine_l],
                      [mean_r_prone_l, mean_r_supine_l],
                      'black', linewidth=2, zorder=10)
        ax_right.annotate('',
                          xy=(mean_theta_supine_l, mean_r_supine_l),
                          xytext=(mean_theta_prone_l, mean_r_prone_l),
                          arrowprops=dict(arrowstyle='->', color='black', lw=2),
                          zorder=10)
        ax_right.scatter([mean_theta_prone_l], [mean_r_prone_l], c='blue', s=200,
                         marker='*', edgecolors='darkblue', linewidths=1,
                         label='Mean Prone', zorder=11)
        ax_right.scatter([mean_theta_supine_l], [mean_r_supine_l], c='red', s=200,
                         marker='*', edgecolors='darkred', linewidths=1,
                         label='Mean Supine', zorder=11)

        ax_right.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_right.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_right.set_ylabel('Distance from Nipple (mm)', labelpad=30, fontsize=11)

        mean_rotation_l = df_left['angle_rotation'].mean()
        mean_rotation_hours_l = df_left['clock_rotation'].mean()
        direction_l = "Clockwise" if mean_rotation_l > 0 else "Counterclockwise"

        ax_right.set_title(
            f'Landmarks in left breast: \n{direction_l}: {abs(mean_rotation_hours_l):.2f}h ({abs(mean_rotation_l):.1f}°)',
            fontsize=12, pad=20)
        ax_right.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, ncol=1)
        ax_right.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_dir / "clock_rotation_comparison_individual.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot (individual): {save_path}")
        plt.show()
        plt.close()

        # ========================================================================
        # Create INDIVIDUAL LANDMARKS ONLY comparison plot (no mean overlay)
        # Right breast on LEFT subplot, Left breast on RIGHT subplot
        # ========================================================================
        print("Generating combined polar plot with individual landmarks only (no mean)...")
        fig_indiv, (ax_left_indiv, ax_right_indiv) = plt.subplots(1, 2, figsize=(14, 6),
                                                                  subplot_kw=dict(projection='polar'))

        # LEFT SUBPLOT: RIGHT BREAST INDIVIDUAL LANDMARKS ONLY
        ax_left_indiv.set_theta_zero_location('N')
        ax_left_indiv.set_theta_direction(-1)

        # Plot all trajectories for right breast
        for i in range(len(df_right)):
            ax_left_indiv.plot([theta_prone_r[i], theta_supine_r[i]],
                               [r_prone_r[i], r_supine_r[i]],
                               'gray', alpha=0.3, linewidth=1)

        # Plot individual points for right breast
        ax_left_indiv.scatter(theta_prone_r, r_prone_r, c='blue', s=50, alpha=0.6,
                              label='Prone', zorder=5, edgecolors='darkblue')
        ax_left_indiv.scatter(theta_supine_r, r_supine_r, c='red', s=50, alpha=0.6,
                              label='Supine', zorder=5, edgecolors='darkred')

        ax_left_indiv.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_left_indiv.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_left_indiv.set_ylabel('Distance from Nipple (mm)', labelpad=30)
        ax_left_indiv.set_title(f'Right Breast: Prone->Supine Shift\n(n={len(df_right)} landmarks)',
                                fontsize=12, pad=20)
        ax_left_indiv.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_left_indiv.grid(True, alpha=0.3)

        # RIGHT SUBPLOT: LEFT BREAST INDIVIDUAL LANDMARKS ONLY
        ax_right_indiv.set_theta_zero_location('N')
        ax_right_indiv.set_theta_direction(-1)

        # Plot all trajectories for left breast
        for i in range(len(df_left)):
            ax_right_indiv.plot([theta_prone_l[i], theta_supine_l[i]],
                                [r_prone_l[i], r_supine_l[i]],
                                'gray', alpha=0.3, linewidth=1)

        # Plot individual points for left breast
        ax_right_indiv.scatter(theta_prone_l, r_prone_l, c='blue', s=50, alpha=0.6,
                               label='Prone', zorder=5, edgecolors='darkblue')
        ax_right_indiv.scatter(theta_supine_l, r_supine_l, c='red', s=50, alpha=0.6,
                               label='Supine', zorder=5, edgecolors='darkred')

        ax_right_indiv.set_xticks(np.radians(np.arange(0, 360, 30)))
        ax_right_indiv.set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        ax_right_indiv.set_ylabel('Distance from Nipple (mm)', labelpad=30)
        ax_right_indiv.set_title(f'Left Breast: Prone->Supine Shift\n(n={len(df_left)} landmarks)',
                                 fontsize=12, pad=20)
        ax_right_indiv.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_right_indiv.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path_indiv = save_dir / "clock_rotation_individual_landmarks_only.png"
        plt.savefig(save_path_indiv, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot (individual landmarks only): {save_path_indiv}")
        plt.show()
        plt.close()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Return summary statistics
    summary = {}
    if len(df_left) > 0:
        summary['left'] = {
            'n': len(df_left),
            'mean_rotation_deg': df_left['angle_rotation'].mean(),
            'mean_rotation_hours': df_left['clock_rotation'].mean(),
            'std_rotation_deg': df_left['angle_rotation'].std(),
            'p_value': stats.ttest_1samp(df_left['angle_rotation'], 0)[1],
            'mean_distance_change': df_left['distance_change'].mean()
        }

    if len(df_right) > 0:
        summary['right'] = {
            'n': len(df_right),
            'mean_rotation_deg': df_right['angle_rotation'].mean(),
            'mean_rotation_hours': df_right['clock_rotation'].mean(),
            'std_rotation_deg': df_right['angle_rotation'].std(),
            'p_value': stats.ttest_1samp(df_right['angle_rotation'], 0)[1],
            'mean_distance_change': df_right['distance_change'].mean()
        }

    return summary



def plot_3panel_anatomical_views(df_ave, save_path=None):
    """
    Create a 3-panel figure showing nipple motion from prone to supine
    in three anatomical views (Coronal, Sagittal, Axial), all relative to sternum.

    Visual encoding (Grammar of Graphics):
    - Color = Subject Identity (unique hue per patient)
    - Shape = Triangle for Nipple
    - Fill Style = State (Hollow: Prone, Filled: Supine)
    - Arrow = Motion trajectory from prone to supine

    Args:
        df_ave: DataFrame with landmark and nipple position data
        save_path: Path to save the figure (default: ../output/figs/nipple_displacement_3panel.png)
    """
    print("\n" + "="*80)
    print("GENERATING 3-PANEL ANATOMICAL VIEWS FIGURE")
    print("="*80)

    # Get unique subjects and assign colors
    unique_subjects = df_ave['VL_ID'].unique()
    n_subjects = len(unique_subjects)
    cmap = plt.cm.get_cmap('tab20', n_subjects)
    subject_colors = {subj: cmap(i) for i, subj in enumerate(unique_subjects)}

    # Separate left and right breasts
    df_left = df_ave[df_ave['landmark side (prone)'] == 'LB'].copy()
    df_right = df_ave[df_ave['landmark side (prone)'] == 'RB'].copy()

    if len(df_left) == 0 and len(df_right) == 0:
        print("No data available")
        return

    # ========================================================================
    # PREPARE DATA: Extract landmark and nipple positions relative to sternum
    # ========================================================================

    def get_positions_rel_sternum(df, side='left'):
        """Extract landmark and nipple positions relative to sternum.
        """
        if len(df) == 0:
            return None

        # Sternum columns
        sternum_prone_x = df['sternum superior prone transformed x'].values
        sternum_prone_y = df['sternum superior prone transformed y'].values
        sternum_prone_z = df['sternum superior prone transformed z'].values
        sternum_supine_x = df['sternum superior supine x'].values
        sternum_supine_y = df['sternum superior supine y'].values
        sternum_supine_z = df['sternum superior supine z'].values

        # Landmark positions (already relative to sternum)
        lm_prone_x = df['landmark ave prone transformed x'].values
        lm_prone_y = df['landmark ave prone transformed y'].values
        lm_prone_z = df['landmark ave prone transformed z'].values
        lm_supine_x = df['landmark ave supine x'].values
        lm_supine_y = df['landmark ave supine y'].values
        lm_supine_z = df['landmark ave supine z'].values

        # Get nipple columns based on side
        if side == 'left':
            nipple_prone_x_raw = df['left nipple prone transformed x'].values
            nipple_prone_y_raw = df['left nipple prone transformed y'].values
            nipple_prone_z_raw = df['left nipple prone transformed z'].values
            nipple_supine_x_raw = df['left nipple supine x'].values
            nipple_supine_y_raw = df['left nipple supine y'].values
            nipple_supine_z_raw = df['left nipple supine z'].values
        else:
            nipple_prone_x_raw = df['right nipple prone transformed x'].values
            nipple_prone_y_raw = df['right nipple prone transformed y'].values
            nipple_prone_z_raw = df['right nipple prone transformed z'].values
            nipple_supine_x_raw = df['right nipple supine x'].values
            nipple_supine_y_raw = df['right nipple supine y'].values
            nipple_supine_z_raw = df['right nipple supine z'].values

        # Calculate nipple positions relative to sternum
        nipple_prone_x = nipple_prone_x_raw - sternum_prone_x
        nipple_prone_y = nipple_prone_y_raw - sternum_prone_y
        nipple_prone_z = nipple_prone_z_raw - sternum_prone_z
        nipple_supine_x = nipple_supine_x_raw - sternum_supine_x
        nipple_supine_y = nipple_supine_y_raw - sternum_supine_y
        nipple_supine_z = nipple_supine_z_raw - sternum_supine_z

        return {
            'vl_ids': df['VL_ID'].values,
            'lm_prone': (lm_prone_x, lm_prone_y, lm_prone_z),
            'lm_supine': (lm_supine_x, lm_supine_y, lm_supine_z),
            'nipple_prone': (nipple_prone_x, nipple_prone_y, nipple_prone_z),
            'nipple_supine': (nipple_supine_x, nipple_supine_y, nipple_supine_z),
            'side': side
        }

    left_data = get_positions_rel_sternum(df_left, 'left')
    right_data = get_positions_rel_sternum(df_right, 'right')

    # ========================================================================
    # CREATE 3-PANEL FIGURE
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define view configurations
    # Each view: (ax_index, x_coord_idx, y_coord_idx, xlabel, ylabel, title)
    views = [
        (0, 0, 2, 'Right-Left (mm)', 'Inf-Sup (mm)', 'Panel A: Coronal View'),      # X, Z
        (1, 1, 2, 'Ant-Post (mm)', 'Inf-Sup (mm)', 'Panel B: Sagittal View'),       # Y, Z
        (2, 0, 1, 'Right-Left (mm)', 'Ant-Post (mm)', 'Panel C: Axial View'),       # X, Y
    ]

    def plot_anatomy_on_ax(ax, data, coord_x_idx, coord_y_idx, subject_colors):
        """Plot nipples with semantic encoding."""
        if data is None:
            return

        vl_ids = data['vl_ids']
        nipple_prone = data['nipple_prone']
        nipple_supine = data['nipple_supine']

        for i, vl_id in enumerate(vl_ids):
            color = subject_colors.get(vl_id, 'gray')

            # Nipple positions
            np_px, np_py = nipple_prone[coord_x_idx][i], nipple_prone[coord_y_idx][i]
            np_sx, np_sy = nipple_supine[coord_x_idx][i], nipple_supine[coord_y_idx][i]

            # Skip if NaN
            if np.isnan(np_px) or np.isnan(np_sx) or np.isnan(np_py) or np.isnan(np_sy):
                continue

            # --- NIPPLES (Triangles) ---
            # Prone: Hollow triangle
            ax.scatter(np_px, np_py, s=80, marker='^', facecolors='none',
                      edgecolors=color, linewidths=1.5, zorder=6, alpha=0.8)
            # Supine: Filled triangle
            ax.scatter(np_sx, np_sy, s=80, marker='^', facecolors=color,
                      edgecolors=color, linewidths=1.5, zorder=6, alpha=0.8)
            # Arrow from prone to supine
            dx, dy = np_sx - np_px, np_sy - np_py
            if abs(dx) > 1 or abs(dy) > 1:
                ax.annotate('', xy=(np_sx, np_sy), xytext=(np_px, np_py),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                          alpha=0.6))

    # Plot each view
    for ax_idx, coord_x_idx, coord_y_idx, xlabel, ylabel, title in views:
        ax = axes[ax_idx]

        # Plot both breasts
        plot_anatomy_on_ax(ax, left_data, coord_x_idx, coord_y_idx, subject_colors)
        plot_anatomy_on_ax(ax, right_data, coord_x_idx, coord_y_idx, subject_colors)

        # Mark sternum origin
        ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, zorder=10)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

        # Formatting
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)

        # Set axis limits
        ax.set_xlim(-250, 250)
        ax.set_ylim(-200, 200)
        ax.set_xticks(np.arange(-250, 251, 50))
        ax.set_yticks(np.arange(-200, 201, 50))

        # Add breast labels
        ax.text(-120, 180, 'Right Breast', fontsize=10, ha='center', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text(120, 180, 'Left Breast', fontsize=10, ha='center', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ========================================================================
    # CREATE DUAL LEGEND
    # ========================================================================

    # Legend 1: Shape/Fill meaning (State)
    from matplotlib.lines import Line2D
    legend_elements_shape = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
               markeredgecolor='gray', markersize=10, markeredgewidth=2,
               label='Nipple (Prone)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markeredgecolor='gray', markersize=10, markeredgewidth=2,
               label='Nipple (Supine)', linestyle='None'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5,
               label='Nipple motion'),
    ]

    # Add shape legend below the figure
    fig.legend(handles=legend_elements_shape, loc='lower center', ncol=3,
              fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    # Legend 2: Subject colors (if not too many subjects)
    if n_subjects <= 10:
        legend_elements_subj = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=subject_colors[subj],
                   markersize=8, label=subj, linestyle='None')
            for subj in sorted(unique_subjects)[:10]
        ]
        axes[2].legend(handles=legend_elements_subj, loc='upper right',
                      fontsize=8, title='Subject', title_fontsize=9,
                      bbox_to_anchor=(1.15, 1.0))

    # Main title
    fig.suptitle('Nipple Displacement: Prone -> Supine (Reference: Sternum)\n' +
                f'n = {n_subjects} subjects',
                fontsize=14, y=1.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save
    if save_path is None:
        save_path = Path("..") / "output" / "figs" / "nipple_displacement_3panel.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved 3-panel figure: {save_path}")
    plt.show()
    plt.close()

    print("\n" + "="*80)
    print("3-PANEL ANATOMICAL VIEWS FIGURE COMPLETE")
    print("="*80)

    return {
        'n_landmarks': len(df_ave),
        'n_subjects': n_subjects,
        'n_left': len(df_left),
        'n_right': len(df_right)
    }





def plot_anatomical_correlation_matrix(df):
    """
    Generates a correlation matrix to analyze the drivers of landmark displacement.
    Calculates the 'Delta' (Change in Distance) for Skin, Ribs, and Nipple.

    This analysis helps explain WHY landmarks moved, based on the biomechanics of
    breast deformation (compression, sagging, and strain).

    Args:
        df: DataFrame with landmark displacement and distance measurements
    """
    print("\n" + "=" * 80)
    print("ANATOMICAL CORRELATION MATRIX ANALYSIS")
    print("=" * 80)
    print("Analyzing biomechanical drivers of landmark displacement...")

    # 1. Calculate Deltas (Supine - Prone)
    # Negative Value = Landmark got CLOSER to the structure (Compression)
    # Positive Value = Landmark moved AWAY from the structure (Expansion/Sag)
    df = df.copy()
    df['Delta_Skin']   = df['Distance to skin (supine) [mm]'] - df['Distance to skin (prone) [mm]']
    df['Delta_Rib']    = df['Distance to rib cage (supine) [mm]'] - df['Distance to rib cage (prone) [mm]']
    df['Delta_Nipple'] = df['Distance to nipple (supine) [mm]'] - df['Distance to nipple (prone) [mm]']

    print(f"\nCalculated Deltas (Supine - Prone):")
    print(f"  Delta_Skin:   Mean = {df['Delta_Skin'].mean():.2f} mm (SD = {df['Delta_Skin'].std():.2f})")
    print(f"  Delta_Rib:    Mean = {df['Delta_Rib'].mean():.2f} mm (SD = {df['Delta_Rib'].std():.2f})")
    print(f"  Delta_Nipple: Mean = {df['Delta_Nipple'].mean():.2f} mm (SD = {df['Delta_Nipple'].std():.2f})")

    # 2. Select Variables for Correlation
    # We correlate the 'Causes' (Anatomical Changes) with the 'Effects' (Displacement Vectors)
    analysis_cols = [
        # --- The Effects (Motion) ---
        'Landmark displacement [mm]',      # Total Magnitude
        'Landmark displacement vector vx', # X: Medial-Lateral
        'Landmark displacement vector vy', # Y: Anterior-Posterior
        'Landmark displacement vector vz', # Z: Superior-Inferior

        # --- The Causes (Biomechanics) ---
        'Delta_Rib',                       # Compression against chest?
        'Delta_Skin',                      # Compression by skin?
        'Delta_Nipple',                    # Tethers to nipple?
        'Distance to rib cage (prone) [mm]'# Initial Depth (Pendulum effect)
    ]

    # Filter for existing columns only
    valid_cols = [c for c in analysis_cols if c in df.columns]
    missing_cols = [c for c in analysis_cols if c not in df.columns]

    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")

    print(f"\nAnalyzing correlations for {len(valid_cols)} variables:")
    for col in valid_cols:
        print(f"  - {col}")

    corr_data = df[valid_cols].dropna()

    if corr_data.empty:
        print("\nError: Not enough data for correlation analysis.")
        return

    print(f"\nValid data points: {len(corr_data)} landmarks")

    # 3. Calculate Correlation (Pearson)
    corr_matrix = corr_data.corr()

    # Print key correlations
    print("\n" + "-" * 80)
    print("KEY CORRELATIONS (with Total Displacement):")
    print("-" * 80)
    displacement_corr = corr_matrix['Landmark displacement [mm]'].sort_values(ascending=False)
    for var, corr_val in displacement_corr.items():
        if var != 'Landmark displacement [mm]':
            if abs(corr_val) > 0.3:  # Moderate or stronger
                strength = "Strong" if abs(corr_val) > 0.5 else "Moderate"
                direction = "positive" if corr_val > 0 else "negative"
                print(f"  {var:40s}: r = {corr_val:+.3f} ({strength} {direction})")

    # 4. Plot Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a mask to hide the upper triangle (optional, reduces visual clutter)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create custom labels for better readability
    label_mapping = {
        'Landmark displacement [mm]': 'Displacement Magnitude',
        'Landmark displacement vector vx': 'Displacement X (Med-Lat)',
        'Landmark displacement vector vy': 'Displacement Y (Ant-Post)',
        'Landmark displacement vector vz': 'Displacement Z (Inf-Sup)',
        'Delta_Rib': 'Δ Distance to Rib',
        'Delta_Skin': 'Δ Distance to Skin',
        'Delta_Nipple': 'Δ Distance to Nipple',
        'Distance to rib cage (prone) [mm]': 'Initial Depth (Prone DTR)'
    }

    # Rename columns and index for display
    display_matrix = corr_matrix.copy()
    display_matrix.columns = [label_mapping.get(col, col) for col in display_matrix.columns]
    display_matrix.index = [label_mapping.get(idx, idx) for idx in display_matrix.index]

    sns.heatmap(display_matrix,
                annot=True,         # Show the correlation coefficient numbers
                fmt=".2f",          # 2 decimal places
                cmap='coolwarm',    # Red (+), Blue (-), White (0)
                center=0,           # Center color map at 0
                square=True,        # Force square cells
                linewidths=0.5,     # Grid lines
                cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"},
                mask=mask,          # Apply the triangular mask
                ax=ax,
                vmin=-1, vmax=1)    # Fix scale from -1 to +1

    ax.set_title('Correlation Matrix: Anatomical Drivers vs. Landmark Displacement\n' +
                'Biomechanical Analysis of Prone -> Supine Motion',
                fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    # Save figure
    save_path = Path("..") / "output" / "figs" / "v5"  / "correlation_matrix_anatomical.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved correlation matrix: {save_path}")

    plt.show()
    plt.close()

    print("\n" + "=" * 80)
    print("CORRELATION MATRIX ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nInterpretation Guide:")
    print("  • Red (+1.0): Strong positive correlation - variables increase together")
    print("  • Blue (-1.0): Strong negative correlation - one increases as other decreases")
    print("  • White (0.0): No correlation")
    print("  • |r| > 0.5: Strong relationship")
    print("  • |r| > 0.3: Moderate relationship")
    print("  • |r| < 0.3: Weak relationship")
    print("\nKey Questions Answered:")
    print("  1. Does initial depth affect displacement? (Pendulum effect)")
    print("  2. Are landmarks compressed toward ribs/skin?")
    print("  3. Does nipple motion tether landmarks?")
    print("  4. Which anatomical factors drive motion in each direction?")

    return corr_matrix



if __name__ == "__main__":
    # Initialize logging - save all terminal output to file
    tee = Tee(LOG_FILE_PATH)
    sys.stdout = tee
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be saved to: {LOG_FILE_PATH}")
    print("=" * 80)

    df_raw, df_ave, df_demo = read_data(EXCEL_FILE_PATH)
    print(df_raw.head())
    print(df_ave.head())

    #%% demographic analysis
    # Select only the necessary columns
    df_subset = df_ave[['VL_ID', 'Age', 'BMI']]

    # Remove duplicates based on 'VL_ID'
    # keep='first' ensures we keep the first entry found for that subject.
    df_subjects = df_subset.drop_duplicates(subset=['VL_ID'], keep='first')

    # Perform analysis on Age and BMI
    # .describe() gives count, mean, std, min, max, and quartiles
    demographics = df_subjects[['Age', 'BMI']].describe()
    # Print the results
    print("Number of unique subjects:", len(df_subjects))
    print("\nAnalysis Summary:")
    print(demographics)

    # Group Age Variables
    # Bins: 0-24, 25-44, 45-64, 65+
    # Note: The bins are set up so (24, 44] means 24 < x <= 44
    age_bins = [0, 24, 44, 64, np.inf]
    age_labels = ['18-24', '25-44', '45-64', '65+']
    df_subjects['Age_Group'] = pd.cut(df_subjects['Age'], bins=age_bins, labels=age_labels)

    # Group BMI Variables
    # Categories: <= 24.9, 25-29.9, >= 30
    bmi_bins = [-np.inf, 24.9, 29.9, np.inf]
    bmi_labels = ['Underweight/Normal', 'Overweight', 'Obese']
    df_subjects['BMI_Group'] = pd.cut(df_subjects['BMI'], bins=bmi_bins, labels=bmi_labels)

    # Check the results (Count participants in each group)
    print("--- Age Group Distribution ---")
    print(df_subjects['Age_Group'].value_counts().sort_index())

    print("\n--- BMI Group Distribution ---")
    print(df_subjects['BMI_Group'].value_counts().sort_index())

    # Preview the new dataframe
    print("\n--- Data Preview ---")
    print(df_subjects.head())


    #%% landmarks characteristics
    landmark_type_raw = df_raw['Landmark type']
    # fibroadenoma_count = len(landmark_type_raw[landmark_type_raw == 'fibroadenoma'])
    # print(f"Count of fibroadenoma: {fibroadenoma_count}")

    total_num_of_landmark = df_ave.shape[0]
    total_num_of_landmark_raw = df_raw.shape[0]/4
    print("Total number of landmarks is: ", total_num_of_landmark)
    # print("Total number of landmarks before post processing is: ", total_num_of_landmark_raw)

    # how many landmarks per volunteer in raw and filtered data
    landmark_counts = df_ave.groupby('VL_ID').size()
    print("Number of landmarks per volunteer:\n", landmark_counts)
    print("Total number of volunteers is: ", landmark_counts.shape[0])
    landmark_counts_raw = df_raw.groupby('VL_ID').size()
    # print("Total number of volunteers before post processing is: ", landmark_counts_raw.shape[0])

    # Count the number of rows for each unique landmark type
    type_counts = df_ave['Landmark type'].value_counts()
    type_proportions = df_ave['Landmark type'].value_counts(normalize=True) * 100

    # print("--- Counts of Each Landmark Type ---")
    # print(type_counts)
    type_summary_table = pd.DataFrame({
        'N': type_counts,
        'Percentage (%)': type_proportions
    }).reset_index().rename(columns={'index': 'Landmark Type'})

    # Ensure percentages are rounded and display the final table
    type_summary_table['Percentage (%)'] = type_summary_table['Percentage (%)'].round(1)
    total_count = type_summary_table['N'].sum()
    total_percentage_display = type_summary_table['Percentage (%)'].sum()

    total_row = pd.DataFrame({
        'Landmark type': ['Total'],
        'N': [total_count],
        'Percentage (%)': [total_percentage_display]
    })
    final_table = pd.concat([type_summary_table, total_row], ignore_index=True)
    print("--- Landmark Type Distribution Summary Table ---")
    print(final_table)


    # ====================landmarks location on the surface (Quadrant/Side)
    # Count by Side
    print("\n--- DISTRIBUTION BY SIDE ---")
    print(df_ave['landmark side (prone)'].value_counts())

    # Count by Quadrant
    print("\n--- DISTRIBUTION BY QUADRANT ---")
    landmark_quadrant_prone = df_ave['Quadrant (prone)'].value_counts()
    landmark_quadrant_supine = df_ave['Quadrant (supine)'].value_counts()
    landmark_quadrant = pd.concat([landmark_quadrant_prone, landmark_quadrant_supine], axis=1, keys=['Prone', 'Supine'])
    print(landmark_quadrant)

    # Cross-tabulation: Which quadrant is most common on which side?
    quad_order = ['UO', 'UI', 'LO', 'LI', 'central']
    shift_table = pd.crosstab(df_ave['Quadrant (prone)'], df_ave['Quadrant (supine)'])
    shift_table = shift_table.reindex(index=quad_order, columns=quad_order, fill_value=0)
    # --- PRINT STATISTICAL TABLES ---
    print("\n" + "=" * 50)
    print("QUADRANT MIGRATION TABLE (Counts with Totals)")
    print("=" * 50)
    # Add totals (margins) to the reindexed table for better readability
    shift_table_totals = shift_table.copy()
    shift_table_totals.loc['Total'] = shift_table_totals.sum()
    shift_table_totals['Total'] = shift_table_totals.sum(axis=1)
    print(shift_table_totals)

    print("\n" + "=" * 50)
    print("MIGRATION PROBABILITIES (Row Percentages)")
    print("Interpretation: % of landmarks starting in [Row] that moved to [Column]")
    print("=" * 50)

    # Calculate Percentages (Row-wise)
    # This represents: Of those starting in the Prone row, what % moved to the Supine column?
    row_sums = shift_table.sum(axis=1)
    pct_table = shift_table.div(row_sums, axis=0).fillna(0) * 100

    # Create a Combined Table
    combined_table = pd.DataFrame(index=quad_order, columns=quad_order)

    for r in quad_order:
        for c in quad_order:
            count = shift_table.loc[r, c]
            pct = pct_table.loc[r, c]
            combined_table.loc[r, c] = f"{count} ({pct:.1f}%)"

    # Add Row Totals column
    # The 'Total' column shows the sum of the Prone row (which is 100% of that row)
    combined_table['Total Prone'] = [f"{int(x)} (100%)" for x in row_sums]

    # Add Column Totals row
    col_sums = shift_table.sum(axis=0)
    grand_total = col_sums.sum()

    # For the bottom row, we show the count and the % of the Grand Total
    col_total_row = []
    for c in quad_order:
        count = col_sums[c]
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        col_total_row.append(f"{count} ({pct:.1f}%)")

    # Bottom right corner (Grand Total)
    col_total_row.append(f"{grand_total} (100%)")

    combined_table.loc['Total Supine'] = col_total_row

    print(combined_table)

    # ===== QUADRANT CHANGES BY LANDMARK TYPE =====
    print("\n" + "=" * 50)
    print("QUADRANT CHANGES BY LANDMARK TYPE")
    print("=" * 50)

    # Create a column indicating if quadrant changed
    df_ave['quadrant_changed'] = df_ave['Quadrant (prone)'] != df_ave['Quadrant (supine)']

    # Group by landmark type and calculate changes
    quadrant_change_by_type = df_ave.groupby('Landmark type').agg({
        'quadrant_changed': ['sum', 'count']
    })

    # Flatten column names
    quadrant_change_by_type.columns = ['Changed', 'Total']
    quadrant_change_by_type['Unchanged'] = quadrant_change_by_type['Total'] - quadrant_change_by_type['Changed']
    quadrant_change_by_type['% Changed'] = (quadrant_change_by_type['Changed'] / quadrant_change_by_type['Total'] * 100).round(1)

    # Reorder columns for better readability
    quadrant_change_by_type = quadrant_change_by_type[['Total', 'Changed', 'Unchanged', '% Changed']]

    # Add totals row
    total_landmarks = quadrant_change_by_type['Total'].sum()
    total_changed = quadrant_change_by_type['Changed'].sum()
    total_unchanged = quadrant_change_by_type['Unchanged'].sum()
    total_pct_changed = (total_changed / total_landmarks * 100).round(1)

    quadrant_change_by_type.loc['Total'] = [total_landmarks, total_changed, total_unchanged, total_pct_changed]

    print("\nLandmark Quadrant Changes Summary by Type:")
    print(quadrant_change_by_type)

    # ===== DETAILED TRANSITIONS FOR CHANGED LANDMARKS =====
    print("\n" + "=" * 50)
    print("SPECIFIC QUADRANT TRANSITIONS (ONLY CHANGED LANDMARKS)")
    print("=" * 50)

    # Overall transitions for all changed landmarks
    df_changed = df_ave[df_ave['quadrant_changed'] == True].copy()

    if len(df_changed) > 0:
        print(f"\n*** OVERALL TRANSITIONS ({len(df_changed)} landmarks changed quadrants) ***")
        overall_transitions = pd.crosstab(
            df_changed['Quadrant (prone)'],
            df_changed['Quadrant (supine)'],
            margins=False
        )
        overall_transitions = overall_transitions.reindex(index=quad_order, columns=quad_order, fill_value=0)

        # Show table
        print("\nTransition Matrix (FROM prone rows -> TO supine columns):")
        print(overall_transitions)

        # Create detailed list of transitions with counts
        print("\nDetailed Transition Counts:")
        transition_list = []
        for from_quad in quad_order:
            for to_quad in quad_order:
                if from_quad != to_quad:  # Only show actual changes
                    count = overall_transitions.loc[from_quad, to_quad] if from_quad in overall_transitions.index and to_quad in overall_transitions.columns else 0
                    if int(count) > 0:
                        transition_list.append({
                            'From': from_quad,
                            'To': to_quad,
                            'Count': int(count),
                            'Percentage': f"{(count/len(df_changed)*100):.1f}%"
                        })

        if transition_list:
            transition_df = pd.DataFrame(transition_list)
            transition_df = transition_df.sort_values('Count', ascending=False)
            print(transition_df.to_string(index=False))
        else:
            print("No quadrant changes detected.")

    # ===== BY LANDMARK TYPE =====
    print("\n" + "=" * 50)
    print("QUADRANT TRANSITIONS BY LANDMARK TYPE")
    print("=" * 50)

    for landmark_type in sorted(df_ave['Landmark type'].unique()):
        df_type = df_ave[df_ave['Landmark type'] == landmark_type]
        df_type_changed = df_type[df_type['quadrant_changed'] == True]

        changed_count = len(df_type_changed)
        total_count = len(df_type)
        pct_changed = round(changed_count / total_count * 100, 1) if total_count > 0 else 0

        print(f"\n{'='*50}")
        print(f"{landmark_type}: {changed_count}/{total_count} ({pct_changed}%) changed quadrants")
        print('='*50)

        if changed_count > 0:
            # Create crosstab for changed landmarks only
            type_transitions = pd.crosstab(
                df_type_changed['Quadrant (prone)'],
                df_type_changed['Quadrant (supine)'],
                margins=False
            )
            type_transitions = type_transitions.reindex(index=quad_order, columns=quad_order, fill_value=0)

            print("\nTransition Matrix (FROM prone rows -> TO supine columns):")
            print(type_transitions)

            # Detailed list for this type
            print("\nSpecific Transitions:")
            type_transition_list = []
            for from_quad in quad_order:
                for to_quad in quad_order:
                    if from_quad != to_quad:
                        count = type_transitions.loc[from_quad, to_quad] if from_quad in type_transitions.index and to_quad in type_transitions.columns else 0
                        if int(count) > 0:
                            type_transition_list.append({
                                'From': from_quad,
                                'To': to_quad,
                                'Count': int(count),
                                '% of Changed': f"{(count/changed_count*100):.1f}%",
                                '% of Total': f"{(count/total_count*100):.1f}%"
                            })

            if type_transition_list:
                type_transition_df = pd.DataFrame(type_transition_list)
                type_transition_df = type_transition_df.sort_values('Count', ascending=False)
                print(type_transition_df.to_string(index=False))
        else:
            print("  → No landmarks of this type changed quadrants")


    # ----- VISUALIZATION ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- ROW 1: SPATIAL FREQUENCY (Quadrant Counts) ---
    # Plot 1: Prone
    ax1 = axes[0]
    sns.countplot(x='Quadrant (prone)', hue='landmark side (prone)', data=df_ave, order=quad_order, palette='viridis', ax=ax1)
    ax1.set_title('Lesion Frequency by Quadrant (PRONE)')
    ax1.set_xlabel('Quadrant')
    ax1.set_ylabel('Count')
    ax1.legend(title='Side')

    # Add counts to Prone bars
    for container in ax1.containers:
        ax1.bar_label(container)

    # Plot 2: Supine
    ax2 = axes[1]
    sns.countplot(x='Quadrant (supine)', hue='landmark side (supine)', data=df_ave, order=quad_order, palette='viridis', ax=ax2)
    ax2.set_title('Lesion Frequency by Quadrant (SUPINE)')
    ax2.set_xlabel('Quadrant')
    ax2.set_ylabel('Count')
    ax2.legend(title='Side')

    # Add counts to Supine bars
    for container in ax2.containers:
        ax2.bar_label(container)

    plt.tight_layout()
    plt.show()


    # Quadrant shift from prone to supine - Stacked Bar Chart
    fig1, ax = plt.subplots(figsize=(8, 6))
    # Plotting the Stacked Bar Chart
    # Use the shift_table directly. The Index (Rows) becomes X-axis, Columns become the Stacks
    shift_table.plot(kind='bar', stacked=True, colormap='Pastel1', ax=ax, edgecolor='black', width=0.7)

    ax.set_title('Quadrant shift from prone to supine', fontsize=14)
    ax.set_xlabel('Starting Quadrant (Prone)', fontsize=12)
    ax.set_ylabel('Number of Landmarks', fontsize=12)

    max_height = shift_table.sum(axis=1).max()
    ax.set_ylim(0, max_height +4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Ending Quadrant (Supine)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)

    row_totals = shift_table.sum(axis=1)

    # Add counts inside the bars
    # Iterate through the "containers" (groups of bar segments)
    for c in ax.containers:
        # Create custom labels for each segment in this container
        labels = []

        # Iterate through every bar segment (rect) in the container
        for i, rect in enumerate(c):
            height = rect.get_height()

            # Only label if height is positive
            if height > 0:
                # 'i' corresponds to the row index (x-axis position)
                total = row_totals.iloc[i]
                pct = (height / total) * 100

                # Format: "Value (Percentage%)"
                label = f"{pct:.1f}%"
                labels.append(label)
            else:
                labels.append("")
        # Apply the custom labels
        ax.bar_label(c, labels=labels, label_type='center', fontsize=8, color='black', weight='regular')
    plt.tight_layout()
    plt.show()


    #%% ===== DISTANCE CHARACTERISTICS =====
    distance_metrics_map = [
        ("DTS (Skin)", "Distance to skin"),
        ("DTN (Nipple)", "Distance to nipple"),
        ("DTR (Rib Cage)", "Distance to rib cage")
    ]

    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ===== TABLE 1: PRONE vs SUPINE COMPARISON =====
    distance_rows = []

    for short_name, excel_prefix in distance_metrics_map:
        prone_col = f"{excel_prefix} (prone) [mm]"
        supine_col = f"{excel_prefix} (supine) [mm]"

        # Get data
        prone_data = df_ave[prone_col].dropna()
        supine_data = df_ave[supine_col].dropna()

        # Prone statistics
        prone_mean = prone_data.mean()
        prone_std = prone_data.std()
        prone_median = prone_data.median()

        # Supine statistics
        supine_mean = supine_data.mean()
        supine_std = supine_data.std()
        supine_median = supine_data.median()

        # Statistical test (paired t-test)
        t_stat, p_value = stats.ttest_rel(prone_data, supine_data)

        # Significance marker
        if p_value < 0.001:
            sig_marker = "***"
        elif p_value < 0.01:
            sig_marker = "**"
        elif p_value < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "ns"

        distance_row = {
            "Metric": short_name,
            "Prone Mean ± SD": f"{prone_mean:.2f} ± {prone_std:.2f}",
            "Prone Median": f"{prone_median:.2f}",
            "Supine Mean ± SD": f"{supine_mean:.2f} ± {supine_std:.2f}",
            "Supine Median": f"{supine_median:.2f}",
            "P-value": f"{p_value:.4e}",
            "Sig.": sig_marker
        }
        distance_rows.append(distance_row)

    # Create DataFrame
    distance_table = pd.DataFrame(distance_rows)
    distance_table.set_index("Metric", inplace=True)

    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY (Prone vs Supine Comparison)")
    print("=" * 80)
    print(distance_table)
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("P-values from paired t-test")
    print("=" * 80)

    # ===== DISTANCE CHANGE SUMMARY (Supine - Prone) =====
    distance_change_rows = []
    dv_differences = []

    for short_name, excel_prefix in distance_metrics_map:
        prone_col = f"{excel_prefix} (prone) [mm]"
        supine_col = f"{excel_prefix} (supine) [mm]"

        # Calculate difference (Supine - Prone)
        diff_col_name = f'diff_{short_name.replace(" ", "_").replace("(", "").replace(")", "")}'
        df_ave[diff_col_name] = df_ave[supine_col] - df_ave[prone_col]
        dv_differences.append(diff_col_name)

        differences = df_ave[diff_col_name].dropna()

        # Difference statistics
        mean_diff = differences.mean()
        std_diff = differences.std()
        median_diff = differences.median()
        min_diff = differences.min()
        max_diff = differences.max()
        range_diff = max_diff - min_diff

        distance_change_row = {
            "Metric": short_name,
            "Mean [mm]": f"{mean_diff:.2f}",
            "SD [mm]": f"{std_diff:.2f}",
            "Median [mm]": f"{median_diff:.2f}",
            "Min [mm]": f"{min_diff:.2f}",
            "Max [mm]": f"{max_diff:.2f}",
            "Range [mm]": f"{range_diff:.2f}",
        }
        distance_change_rows.append(distance_change_row)

    distance_change_table = pd.DataFrame(distance_change_rows)
    distance_change_table.set_index("Metric", inplace=True)

    print("\n" + "=" * 80)
    print("DISTANCE CHANGE SUMMARY (Supine - Prone)")
    print("=" * 80)
    print(distance_change_table)
    print("\nPositive values indicate further distance in supine position")
    print("=" * 80)


    #%% ===== DISPLACEMENT MAGNITUDE SUMMARY =====
    print("\n" + "=" * 80)
    print("DISPLACEMENT MAGNITUDE SUMMARY (Relative to Sternum)")
    print("=" * 80)

    # Get displacement magnitude column (from Excel data)
    disp_col = 'Landmark displacement [mm]'
    disp_rel_nipple_col = 'Landmark displacement relative to nipple [mm]'

    if disp_col in df_ave.columns:
        disp_data = df_ave[disp_col].dropna()

        if len(disp_data) > 0:
            mean_disp = disp_data.mean()
            std_disp = disp_data.std()
            median_disp = disp_data.median()
            min_disp = disp_data.min()
            max_disp = disp_data.max()
            range_disp = max_disp - min_disp

            print(f"\nLandmark Displacement (Sternum Reference):")
            print(f"  N = {len(disp_data)}")
            print(f"  Mean ± SD:    {mean_disp:.2f} ± {std_disp:.2f} mm")
            print(f"  Median:       {median_disp:.2f} mm")
            print(f"  Range:        {min_disp:.2f} - {max_disp:.2f} mm (Range: {range_disp:.2f} mm)")

            # Calculate displacement by breast side
            left_disp = df_ave[df_ave['landmark side (prone)'] == 'LB'][disp_col].dropna()
            right_disp = df_ave[df_ave['landmark side (prone)'] == 'RB'][disp_col].dropna()

            print(f"\n  By Breast Side:")
            print(f"    Left Breast (n={len(left_disp)}):  {left_disp.mean():.2f} ± {left_disp.std():.2f} mm")
            print(f"    Right Breast (n={len(right_disp)}): {right_disp.mean():.2f} ± {right_disp.std():.2f} mm")

            # Statistical comparison between left and right
            if len(left_disp) >= 3 and len(right_disp) >= 3:
                t_stat, p_val = stats.ttest_ind(left_disp, right_disp)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"    Difference: t={t_stat:.3f}, p={p_val:.4e} ({sig})")
        else:
            print("  No displacement data available.")
    else:
        print(f"  Column '{disp_col}' not found in data.")

    # Displacement relative to nipple
    if disp_rel_nipple_col in df_ave.columns:
        disp_nipple_data = df_ave[disp_rel_nipple_col].dropna()

        if len(disp_nipple_data) > 0:
            mean_disp_n = disp_nipple_data.mean()
            std_disp_n = disp_nipple_data.std()
            median_disp_n = disp_nipple_data.median()
            min_disp_n = disp_nipple_data.min()
            max_disp_n = disp_nipple_data.max()

            print(f"\nLandmark Displacement (Nipple Reference):")
            print(f"  N = {len(disp_nipple_data)}")
            print(f"  Mean ± SD:    {mean_disp_n:.2f} ± {std_disp_n:.2f} mm")
            print(f"  Median:       {median_disp_n:.2f} mm")
            print(f"  Range:        {min_disp_n:.2f} - {max_disp_n:.2f} mm")

            # Calculate by breast side
            left_disp_n = df_ave[df_ave['landmark side (prone)'] == 'LB'][disp_rel_nipple_col].dropna()
            right_disp_n = df_ave[df_ave['landmark side (prone)'] == 'RB'][disp_rel_nipple_col].dropna()

            print(f"\n  By Breast Side:")
            print(f"    Left Breast (n={len(left_disp_n)}):  {left_disp_n.mean():.2f} ± {left_disp_n.std():.2f} mm")
            print(f"    Right Breast (n={len(right_disp_n)}): {right_disp_n.mean():.2f} ± {right_disp_n.std():.2f} mm")

    # Nipple displacement summary
    left_nipple_disp_col = 'Left nipple displacement [mm]'
    right_nipple_disp_col = 'Right nipple displacement [mm]'

    if left_nipple_disp_col in df_ave.columns and right_nipple_disp_col in df_ave.columns:
        # Get unique subject nipple displacements (avoid duplicates per subject)
        nipple_data = df_ave.drop_duplicates(subset=['VL_ID'])
        left_nipple_disp = nipple_data[left_nipple_disp_col].dropna()
        right_nipple_disp = nipple_data[right_nipple_disp_col].dropna()

        print(f"\nNipple Displacement (Sternum Reference):")
        print(f"  Left Nipple (n={len(left_nipple_disp)}):  {left_nipple_disp.mean():.2f} ± {left_nipple_disp.std():.2f} mm")
        print(f"  Right Nipple (n={len(right_nipple_disp)}): {right_nipple_disp.mean():.2f} ± {right_nipple_disp.std():.2f} mm")

        # Combined nipple displacement
        all_nipple = pd.concat([left_nipple_disp, right_nipple_disp])
        print(f"  Combined (n={len(all_nipple)}): {all_nipple.mean():.2f} ± {all_nipple.std():.2f} mm")

    print("=" * 80)


    #%% repeated anova for difference in distance to skin, rib cage, and nipples
    SUBJECT_ID = 'VL_ID'
    perform_repeated_measures_analysis(df_ave, SUBJECT_ID, dv_differences)


    #%% GROUP ANALYSIS BASED ON LANDMARK CHARACTERISTICS
    print("\n" + "=" * 80)
    print("GROUP ANALYSIS (Signed Difference: Supine - Prone)")
    print("=" * 80)

    df_diff = df_ave.dropna(subset= dv_differences + ['Quadrant (prone)', 'Landmark type', 'landmark side (prone)']).copy()
    df_diff.rename(columns = {
        'Quadrant (prone)': 'quadrant_prone',
        'Landmark type': 'landmark_type',
        'landmark side (prone)': 'landmark_side_prone'
    }, inplace=True)

    group_vars = ['quadrant_prone', 'landmark_type', 'landmark_side_prone']

    for dv_var_current in dv_differences:
        print(f"\n\n*** Group Analysis for Dependent Variable: {dv_var_current} ***")

        # Loop through all categorical groups
        for group_col in group_vars:
            if group_col == 'landmark_side_prone':
                # Use the two-group test; Landmark Side
                perform_two_group_analysis(df_diff, dv_var_current, group_col)
            else:
                # Use the multi-group test; 1: Breast Quadrant 2: Landmark Type
                perform_group_analysis(df_diff, dv_var_current, group_col)



    #%% How Age and BMI influence the shift in distances
    # Spearman Correlation (Continuous)
    df_analysis = pd.merge(df_ave, df_demo[['VL_ID', 'BMI', 'Age']], on='VL_ID', how='inner', suffixes=('', '_new'))
    for metric in dv_differences:
        model_df = df_analysis[['Age', 'BMI', metric]].dropna()

        rho, pval = stats.spearmanr(df_analysis['BMI'], df_analysis[metric], nan_policy='omit')
        print(f"Correlation between BMI and {metric}: rho={rho:.2f}, p-val={pval:.4e}")
        rho, pval = stats.spearmanr(df_analysis['Age'], df_analysis[metric], nan_policy='omit')
        print(f"Correlation between Age and {metric}: rho={rho:.2f}, p-val={pval:.4e}")

        X = model_df[['Age', 'BMI']]
        X = sm.add_constant(X)  # Adds the intercept term
        y = model_df[metric]
        model = sm.OLS(y, X).fit()
        print(f"\n--- MULTIVARIATE REGRESSION FOR: {metric} ---")
        print(model.summary().tables[1])
    plot_bmi_correlations(df_analysis)

    #%% Effect of a landmark's proximity to the skin on its motion
    investigate_proximity_effect(df_ave)

    # --- Plot show exactly where each landmark started (Prone) and where it ended up (Supine)---
    # # Define a mapping dictionary
    # column_map = {
    #     # --- Positions (Relative to Sternum) ---
    #     'landmark ave prone transformed x': 'X_prone',
    #     'landmark ave prone transformed y': 'Y_prone',
    #     'landmark ave prone transformed z': 'Z_prone',
    #     'landmark ave supine x': 'X_supine',
    #     'landmark ave supine y': 'Y_supine',
    #     'landmark ave supine z': 'Z_supine',
    #
    #     # --- VECTORS: Landmark Relative to Sternum (Displacement) ---
    #     'Landmark displacement vector vx': 'dX_sternum',
    #     'Landmark displacement vector vy': 'dY_sternum',
    #     'Landmark displacement vector vz': 'dZ_sternum',
    #
    #     # --- VECTORS: Landmark Relative to Nipple (Intrinsic Deformation) ---
    #     'Landmark relative to nipple vector vx': 'dX_nipple',
    #     'Landmark relative to nipple vector vy': 'dY_nipple',
    #     'Landmark relative to nipple vector vz': 'dZ_nipple',
    #
    #     'landmark side (prone)': 'Side'
    # }
    #
    # # Rename the columns in your DataFrame
    # df_ave_rename = df_ave.rename(columns=column_map)
    # df_ave_rename['Side'] = df_ave_rename['Side'].replace({'LB': 'Left', 'RB': 'Right'})


    # 2. Run the Vectors Relative to Sternum plotting function
    plot_vectors_rel_sternum(df_ave, color_by='breast',data_type='landmarks', include_dual_sagittal=True)
    plot_vectors_rel_sternum(df_ave, color_by='subject',data_type='landmarks', include_dual_sagittal=True)
    plot_vectors_rel_sternum(df_ave, color_by='dts',data_type='landmarks', include_dual_sagittal=True)
    plot_vectors_rel_sternum(df_ave, color_by='breast',data_type='nipples', include_dual_sagittal=True)
    plot_vectors_rel_sternum(df_ave, color_by='subject',data_type='nipples', include_dual_sagittal=True)
    plot_vectors_rel_sternum(df_ave, color_by='dts',data_type='nipples', include_dual_sagittal=True)
    # plot_sagittal_dual_axes(df_ave, color_by='breast')
    # plot_sagittal_dual_axes(df_ave, color_by='subject')
    # plot_sagittal_dual_axes(df_ave, color_by='dts')

    # plot_vectors_for_vl81(df_ave)
    # 3. Run the Vectors Relative to Nipple plotting function (Intrinsic Deformation)
    # This plots landmark motion after subtracting the movement of the respective nipple.
    print("\n" + "=" * 50)
    print("PLOTTING LANDMARK MOTION RELATIVE TO NIPPLE")
    print("=" * 50)

    # Define save directory
    save_path = Path("..") / "output" /  "figs" / "v5" / "landmark vectors" / "Vectors_rel_nipple"

    base_left, base_right, vec_left, vec_right, lm_disp_left, lm_disp_right, nipple_disp_left, nipple_disp_right =\
        plot_nipple_relative_landmarks(
        df_ave=df_ave,
        vl_id=None,  # None means all subjects
        title="Displacement of landmarks relative to the nipple",
        save_path=save_path,
        use_dts_cmap=True
    )

    # 4. Analyze Clock Position Rotation (Polar Plot Analysis)
    # Tests hypothesis: "We observed a mean clockwise rotation of X hours (Y degrees)
    # for left-sided tumors, consistent with gravity-induced lateral displacement in the supine position."
    print("\n" + "=" * 50)
    print("CLOCK POSITION ROTATION ANALYSIS")
    print("=" * 50)

    clock_summary = analyse_clock_position_rotation(
        df_ave,
        base_left=base_left,
        base_right=base_right,
        vec_left=vec_left,
        vec_right=vec_right
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Clock Position Rotation")
    print("=" * 80)

    if 'left' in clock_summary:
        left_stats = clock_summary['left']
        direction = "clockwise" if left_stats['mean_rotation_deg'] > 0 else "counterclockwise"
        print(f"\nLeft Breast (n={left_stats['n']}):")
        print(f"  Mean rotation: {left_stats['mean_rotation_hours']:.2f} hours ({left_stats['mean_rotation_deg']:.1f}°)")
        print(f"  Direction: {direction}")
        print(f"  Standard deviation: {left_stats['std_rotation_deg']:.1f}°")
        print(f"  Statistical significance: p={left_stats['p_value']:.4e}")
        print(f"  Mean distance change: {left_stats['mean_distance_change']:.2f} mm")

        if left_stats['p_value'] < 0.05:
            print(f"\n  ✓ HYPOTHESIS CONFIRMED: Significant {direction} rotation detected for left-sided landmarks")
            print(f"    This is consistent with gravity-induced lateral displacement in the supine position.")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT CONFIRMED: No significant rotation detected")

    if 'right' in clock_summary:
        right_stats = clock_summary['right']
        direction = "clockwise" if right_stats['mean_rotation_deg'] > 0 else "counterclockwise"
        print(f"\nRight Breast (n={right_stats['n']}):")
        print(f"  Mean rotation: {right_stats['mean_rotation_hours']:.2f} hours ({right_stats['mean_rotation_deg']:.1f}°)")
        print(f"  Direction: {direction}")
        print(f"  Standard deviation: {right_stats['std_rotation_deg']:.1f}°")
        print(f"  Statistical significance: p={right_stats['p_value']:.4e}")
        print(f"  Mean distance change: {right_stats['mean_distance_change']:.2f} mm")

        if right_stats['p_value'] < 0.05:
            print(f"\n  ✓ HYPOTHESIS CONFIRMED: Significant {direction} rotation detected for right-sided landmarks")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT CONFIRMED: No significant rotation detected")

    print("\n" + "=" * 80)

    save_path = Path("..") / "output" /  "figs"  / "v5"
    # plot_3panel_anatomical_views(df_ave, save_path)

    # Correlation matrix analysis
    plot_anatomical_correlation_matrix(df_ave)

    test_partial_correlation(df_ave)

    # Close logging
    print("\n" + "=" * 80)
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output saved to: {LOG_FILE_PATH}")
    print("=" * 80)
    tee.close()
