from pathlib import Path
import pandas as pd
import pingouin as pg
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from pingouin import welch_anova, pairwise_gameshowell, rm_anova, pairwise_ttests
from matplotlib.patches import Circle, Arc
import matplotlib.patches as patches
from plot_nipple_relative_vectors import plot_nipple_relative_vectors


OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v4_2026_01_12.xlsx"


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
        print(f"\n⚠️ RM-ANOVA Skipped: {e}")

    except Exception as e:
        print(f"\n⚠️ RM-ANOVA failed (Switching to Non-Parametric Friedman Test): {e}")

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
    axes[0].set_title('Impact of BMI on Difference in DTN\n(β = -4.01, p < 0.001)', fontsize=14, fontweight='bold')
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
    axes[1].set_title('Impact of BMI on Difference in DTR\n(β = -1.44, p < 0.001)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Body Mass Index (BMI) [kg/m²]', fontsize=12)
    axes[1].set_ylabel('Difference in DTR [mm]', fontsize=12)

    # Clean up the layout
    plt.tight_layout()

    # Save and show
    plt.savefig(output_filename, dpi=300)
    plt.show()
    print(f"Visualization complete. Plot saved as '{output_filename}'")


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


def plot_vectors_rel_sternum(df_ave):
    """
    Plots displacement vectors (Prone -> Supine) for both breasts.
    """
    print("\n--- Plotting Vectors Relative to Sternum ---")

    # 1. copy
    df_subset = df_ave.copy()

    if df_subset.empty:
        print("No data found.")
        return

    # 2. Separate into Left (LB) and Right (RB) breasts
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Helper to extract Base Points (Prone) and Vectors (Supine - Prone)
    def get_points_and_vectors(sub_df):
        if sub_df.empty:
            return np.empty((0, 3)), np.empty((0, 3))

        # Extract Prone (Start/Base points)
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

    # 5. Plotting Loop - Single Plot for Both Breasts
    for plane_name, config in PLANE_CONFIG.items():
        axis_x_idx, axis_y_idx = config['axes']

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(f"{plane_name} Plane: Prone to Supine Displacement (Relative to Sternum)", fontsize=14)

        # Setup axes - standard for all planes
        if True:
            # Default axes setup for Coronal and Axial planes
            ax.set_xlabel(config['xlabel'])
            ax.set_ylabel(config['ylabel'])
            ax.set_xticks(np.arange(-250,251,50))
            ax.set_yticks(np.arange(-250,251,50))

            # For Coronal and Sagittal planes, use y-limit of 100; for Axial, use 200
            if plane_name == 'Coronal':
                ax.set_xlim(lims)
                ax.set_ylim(lims)  # Set y-limit to 100 for Coronal plane
                # Don't enforce equal aspect for Coronal to avoid narrow appearance
                ax.set_aspect('equal', adjustable='box')

            elif plane_name == 'Sagittal':
                # For Sagittal plane, use standard x-limits but y-limit of 100
                ax.set_xlim(lims)
                ax.set_ylim(lims)  # Set y-limit to 100 for Sagittal plane
                ax.set_aspect('equal', adjustable='box')
            else:
                # For Axial plane, use standard limits (200)
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

        # Plot with standard coordinates for all planes
        # Plot Right Breast Vectors
        if len(base_right) > 0:
            ax.quiver(
                base_right[:, axis_x_idx], base_right[:, axis_y_idx],  # X, Y start
                vec_right[:, axis_x_idx], vec_right[:, axis_y_idx],  # U, V components
                angles='xy', scale_units='xy', scale=1,
                color='blue', width=0.003, headwidth=3, label='Right Breast'
            )
            # Plot start points
            ax.scatter(base_right[:, axis_x_idx], base_right[:, axis_y_idx], c='blue', s=20)

            # Add "Right Breast" region label
            right_x_pos = np.mean(base_right[:, axis_x_idx])
            right_y_pos = lims[1] * 0.85
            ax.text(right_x_pos, right_y_pos, 'RIGHT BREAST',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='blue', alpha=0.6)

        # Plot Left Breast Vectors
        if len(base_left) > 0:
            ax.quiver(
                base_left[:, axis_x_idx], base_left[:, axis_y_idx],  # X, Y start
                vec_left[:, axis_x_idx], vec_left[:, axis_y_idx],  # U, V components
                angles='xy', scale_units='xy', scale=1,
                color='green', width=0.003, headwidth=3, label='Left Breast'
            )
            # Plot start points
            ax.scatter(base_left[:, axis_x_idx], base_left[:, axis_y_idx], c='green', s=20)

            # Add "Left Breast" region label
            left_x_pos = np.mean(base_left[:, axis_x_idx])
            left_y_pos = lims[1] * 0.85
            ax.text(left_x_pos, left_y_pos, 'LEFT BREAST',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='green', alpha=0.6)

        # Add legend
        ax.legend(loc='lower right')
        plt.tight_layout()
        filename = f"Vectors_rel_sternum_{plane_name}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved plot: {filename}")
        plt.show()

    # Call the new dual-axis Sagittal plotter
    plot_sagittal_dual_axes(df_ave)


def plot_sagittal_dual_axes(df_ave):
    """
    Creates a detailed dual-plot for the Sagittal plane (Left and Right breasts side-by-side)
    with specific axis configurations:
    - Shared vertical axis at x=0 (Inf-Sup)
    - Two separate x-axis origins
    - Right Breast (Right side): Ant-Post (mm), Blue
    - Left Breast (Left side): Post-Ant (mm), Green
    """
    print("\n--- Plotting Dual Sagittal Axes ---")

    # 1. Use All Data
    df_subset = df_ave.copy()

    if df_subset.empty:
        print("No data found.")

        return

    # 2. Separate into Left (LB) and Right (RB) breasts
    left_df = df_subset[df_subset['landmark side (prone)'] == 'LB']
    right_df = df_subset[df_subset['landmark side (prone)'] == 'RB']

    # 3. Helper to extract Base Points and Vectors
    def get_points_and_vectors(sub_df):
        if sub_df.empty:
            return np.empty((0, 3)), np.empty((0, 3))
        
        prone_x = sub_df['landmark ave prone transformed x'].values
        prone_y = sub_df['landmark ave prone transformed y'].values
        prone_z = sub_df['landmark ave prone transformed z'].values
        base_points = np.column_stack((prone_x, prone_y, prone_z))

        supine_x = sub_df['landmark ave supine x'].values
        supine_y = sub_df['landmark ave supine y'].values
        supine_z = sub_df['landmark ave supine z'].values
        end_points = np.column_stack((supine_x, supine_y, supine_z))

        vectors = end_points - base_points
        return base_points, vectors

    base_left, vec_left = get_points_and_vectors(left_df)
    base_right, vec_right = get_points_and_vectors(right_df)

    # 4. Setup Plot
    # Sagittal Plane: Y (Ant-Post) vs Z (Inf-Sup)
    
    # Create subplots with sharey=True and no horizontal space
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    plt.subplots_adjust(wspace=0.0)

    # Common Settings
    ylim_val = 250
    ax_left.set_ylim(-ylim_val, ylim_val)
    ax_right.set_ylim(-ylim_val, ylim_val)
    
    # Set Y-axis Ticks explicitly to include endpoints
    yticks = np.arange(-250, 251, 50)
    ax_left.set_yticks(yticks)
    ax_right.set_yticks(yticks)
    
    # --- LEFT PLOT (Now RIGHT BREAST) ---
    # Metrics: Ant-Post (mm), Blue
    # Ticks: 150, 100, 50, 0, -50, -100, -150, -200, -250 (Anterior -> Posterior)
    ax_left.set_xlim(150, -250)
    ax_left.set_xticks([150, 100, 50, 0, -50, -100, -150, -200, -250])
    ax_left.set_xlabel("Ant-Post (mm)", color='blue', fontsize=12, fontweight='bold')
    ax_left.set_ylabel("Inf-Sup (mm)", fontsize=12, fontweight='bold')
    
    # Spine Config: Shared Central Axis Effect
    # Move RIGHT spine to data 0
    ax_left.spines['right'].set_position(('data', 0))
    ax_left.spines['left'].set_visible(False)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['bottom'].set_visible(True)
    
    ax_left.spines['right'].set_color('black')
    ax_left.spines['right'].set_linewidth(1)
    
    # Plot Logic for Right Breast (Blue) on Left Plot
    if len(base_right) > 0:
        ax_left.quiver(
            base_right[:, 1], base_right[:, 2],  # Y, Z
            vec_right[:, 1], vec_right[:, 2],
            angles='xy', scale_units='xy', scale=1,
            color='blue', width=0.003, headwidth=3, alpha=0.6
        )
        ax_left.scatter(base_right[:, 1], base_right[:, 2], c='blue', s=10, alpha=0.6)

    # Origin and Grid
    ax_left.plot(0, 0, 'ko', markersize=6, zorder=10) # Origin Dot
    ax_left.grid(True, linestyle='--', alpha=0.5)
    ax_left.set_aspect('equal', adjustable='box')
    ax_left.text(0, ylim_val*0.9, "RIGHT BREAST", ha='center', va='center', color='blue', fontweight='bold')


    # --- RIGHT PLOT (Now LEFT BREAST) ---
    # Metrics: Post-Ant (mm), Green
    # Ticks: -250, -200, ..., 150 (Posterior -> Anterior)
    ax_right.set_xlim(-250, 150)
    ax_right.set_xticks(np.arange(-250, 151, 50))
    ax_right.set_xlabel("Post-Ant (mm)", color='green', fontsize=12, fontweight='bold')
    
    # Move left spine to x=0
    ax_right.spines['left'].set_position(('data', 0))
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['bottom'].set_visible(True)
    
    ax_right.spines['left'].set_color('black')
    ax_right.spines['left'].set_linewidth(1)
    
    # Plot Logic for Left Breast (Green) on Right Plot
    if len(base_left) > 0:
        ax_right.quiver(
            base_left[:, 1], base_left[:, 2],  # Y, Z
            vec_left[:, 1], vec_left[:, 2],
            angles='xy', scale_units='xy', scale=1,
            color='green', width=0.003, headwidth=3, alpha=0.6
        )
        ax_right.scatter(base_left[:, 1], base_left[:, 2], c='green', s=10, alpha=0.6)
        
    # Origin and Grid
    ax_right.plot(0, 0, 'ko', markersize=6, zorder=10) # Origin Dot
    ax_right.grid(True, linestyle='--', alpha=0.5)
    ax_right.set_aspect('equal', adjustable='box')
    ax_right.text(0, ylim_val*0.9, "LEFT BREAST", ha='center', va='center', color='green', fontweight='bold')

    plt.suptitle("Sagittal Plane Dual View: Prone to Supine Displacement (Relative to Sternum)", fontsize=14)
    plt.savefig("Vectors_rel_sternum_sagittal_dual_plot.png", dpi=300)
    print("Saved plot: Vectors_rel_sternum_sagittal_dual_plot.png")
    plt.show()



def plot_anatomical_planes(df, radius=80):
    """
    Plots landmark motion vectors in Axial, Sagittal, and Coronal planes.
    Assumes df has: X_prone, Y_prone, Z_prone, X_supine, Y_supine, Z_supine, and Side.
    """

    # Define Plane Configurations
    # Coronal: X vs Y | Axial: X vs Z | Sagittal: Y vs Z
    PLANE_CONFIGS = {
        'Coronal': {
            'axes': ('X', 'Y'),
            'labels': ('Lateral-Medial (X)', 'Superior-Inferior (Y)'),
            'quads_r': ['UI', 'UO', 'LI', 'LO'], 'quads_l': ['UO', 'UI', 'LO', 'LI']
        },
        'Axial': {
            'axes': ('X', 'Z'),
            'labels': ('Lateral-Medial (X)', 'Anterior-Posterior (Z)'),
            'quads_r': ['Ant-Med', 'Ant-Lat', 'Post-Med', 'Post-Lat'],
            'quads_l': ['Ant-Lat', 'Ant-Med', 'Post-Lat', 'Post-Med']
        },
        'Sagittal': {
            'axes': ('Y', 'Z'),
            'labels': ('Superior-Inferior (Y)', 'Anterior-Posterior (Z)'),
            'quads_r': ['Sup-Ant', 'Inf-Ant', 'Sup-Post', 'Inf-Post'],
            'quads_l': ['Sup-Ant', 'Inf-Ant', 'Sup-Post', 'Inf-Post']
        }
    }

    for plane_name, config in PLANE_CONFIGS.items():
        fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
        fig.suptitle(f"{plane_name} Plane: Landmark Motion (Prone → Supine)", fontsize=16, fontweight='bold')

        ax1_idx, ax2_idx = config['axes']

        for side, ax in zip(['Right', 'Left'], [ax_r, ax_l]):
            # Filter data for this side
            side_df = df[df['landmark side (prone)'] == side].copy()
            if side_df.empty: continue

            # 1. Plot Vectors (Quiver)
            # Origin: Prone coordinates | Vector: Delta (Supine - Prone)
            ax.quiver(
                side_df[f'{ax1_idx}_prone'], side_df[f'{ax2_idx}_prone'],
                side_df[f'{ax1_idx}_supine'] - side_df[f'{ax1_idx}_prone'],
                side_df[f'{ax2_idx}_supine'] - side_df[f'{ax2_idx}_prone'],
                angles='xy', scale_units='xy', scale=1, color='teal', alpha=0.7, width=0.003
            )

            # 2. Plot Prone Points
            ax.scatter(side_df[f'{ax1_idx}_prone'], side_df[f'{ax2_idx}_prone'], color='red', s=15, label='Prone')

            # 3. Formatting
            ax.set_title(f"{side} Breast", loc='left', fontsize=12)
            ax.set_xlabel(config['labels'][0])
            ax.set_ylabel(config['labels'][1])
            ax.axhline(0, color='black', lw=1, ls='--')
            ax.axvline(0, color='black', lw=1, ls='--')

            # Add Boundary Circle & Nipple
            bounding_shape = Circle((0, 0), radius, fill=False, color='gray', lw=1, linestyle=':')
            ax.add_artist(bounding_shape)
            ax.plot(0, 0, 'ko', markersize=8, label='Nipple (0,0)')  # Nipple at origin

            # Quadrant Labels
            q = config['quads_r'] if side == 'Right' else config['quads_l']
            offset = radius * 0.7
            ax.text(offset, offset, q[0], ha='center', va='center', fontweight='bold', alpha=0.5)
            ax.text(-offset, offset, q[1], ha='center', va='center', fontweight='bold', alpha=0.5)
            ax.text(offset, -offset, q[2], ha='center', va='center', fontweight='bold', alpha=0.5)
            ax.text(-offset, -offset, q[3], ha='center', va='center', fontweight='bold', alpha=0.5)

            ax.set_xlim(-radius - 10, radius + 10)
            ax.set_ylim(-radius - 10, radius + 10)
            ax.set_aspect('equal')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


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

def plot_nipple_relative_from_sternum_data(
    base_point, end_point, 
    lm_disp_rel_sternum, 
    nipple_disp_left_vec, nipple_disp_right_vec,
    nipple_pos_prone_l, nipple_pos_prone_r,
    is_left, dts_values=None, 
    title="Nipple Relative Landmarks",
    save_path=None, use_dts_cmap=True
):
    """
    Derives Nipple-relative plotting data from Sternum-relative positions and displacements.
    
    Args:
        base_point: Prone position relative to sternum (X, Y, Z)
        end_point: Supine position relative to sternum (X, Y, Z)
        lm_disp_rel_sternum: Displacement vector relative to sternum (end_point - base_point)
        nipple_disp_left_vec: Nipple displacement relative to sternum (Left)
        nipple_disp_right_vec: Nipple displacement relative to sternum (Right)
        nipple_pos_prone_l: Prone position of left nipple relative to sternum
        nipple_pos_prone_r: Prone position of right nipple relative to sternum
        is_left: Boolean array indicating if landmark belongs to left breast
        dts_values: Depth-to-skin values for coloring
        title: Plot title
        save_path: Path to save the plots
        use_dts_cmap: Whether to use the DTS colormap
    """
    # 1. Origins (landmark position relative to prone nipple)
    # This transforms the sternum-centered prone coordinates to be nipple-centered
    X_left = base_point[is_left.values] - nipple_pos_prone_l
    X_right = base_point[~is_left.values] - nipple_pos_prone_r
    
    # 2. Vectors (landmark motion relative to nipple)
    # This subtracts the whole-breast movement (nipple motion) to show intrinsic deformation
    V_left = lm_disp_rel_sternum[is_left.values] - nipple_disp_left_vec
    V_right = lm_disp_rel_sternum[~is_left.values] - nipple_disp_right_vec
    
    # 3. Extract DTS values per side if provided
    d_left = dts_values[is_left.values] if dts_values is not None else None
    d_right = dts_values[~is_left.values] if dts_values is not None else None
    
    # 4. Call the specialized plotting function
    plot_nipple_relative_vectors(
        base_point_left=X_left,
        vector_left=V_left,
        base_point_right=X_right,
        vector_right=V_right,
        dts_left=d_left,
        dts_right=d_right,
        title=title,
        save_path=save_path,
        use_dts_cmap=use_dts_cmap
    )

def plot_nipple_relative_motion_analysis(df_ave, vl_id=81, save_dir=None, use_dts_cmap=True):
    """
    Extracts landmark and nipple data for a specific subject from the dataframe,
    then calls plot_nipple_relative_from_sternum_data.
    """
    # Filter for subject
    df_sub = df_ave[df_ave['VL_ID'] == vl_id].copy()
    if df_sub.empty:
        print(f"Warning: No data found for subject {vl_id}")
        return
        
    # Identify nipple markers
    nipple_df = df_sub[df_sub['Landmark type'] == 'nipple']
    l_nipple = nipple_df[nipple_df['landmark side (prone)'] == 'LB']
    r_nipple = nipple_df[nipple_df['landmark side (prone)'] == 'RB']
    
    if l_nipple.empty or r_nipple.empty:
        print(f"Warning: Nipple markers not found for subject {vl_id}. Skipping nipple-relative plot.")
        return

    # Columns
    prone_cols = ['landmark ave prone transformed x', 'landmark ave prone transformed y', 'landmark ave prone transformed z']
    supine_cols = ['landmark ave supine x', 'landmark ave supine y', 'landmark ave supine z']
    
    # Prone and supine positions relative to sternum
    n_p_l = l_nipple[prone_cols].values[0]
    n_p_r = r_nipple[prone_cols].values[0]
    n_s_l = l_nipple[supine_cols].values[0]
    n_s_r = r_nipple[supine_cols].values[0]
    
    # Nipple displacements
    nipple_disp_left_vec = n_s_l - n_p_l
    nipple_disp_right_vec = n_s_r - n_p_r
    
    # Non-nipple Landmarks
    df_lm = df_sub[df_sub['Landmark type'] != 'nipple'].copy()
    if df_lm.empty:
        print(f"Warning: No non-nipple landmarks found for subject {vl_id}")
        return
        
    base_point = df_lm[prone_cols].values
    end_point = df_lm[supine_cols].values
    lm_disp_rel_sternum = end_point - base_point
    
    # Metadata
    is_left = df_lm['landmark side (prone)'] == 'LB'
    dts_col = 'Distance to skin (prone) [mm]'
    dts_values = df_lm[dts_col].values if dts_col in df_lm.columns else None
    
    # Sub-path for subject
    save_path = str(Path(save_dir) / f"VL{vl_id}") if save_dir else None
    
    # Perform the call using the requested Sternum-relative inputs
    plot_nipple_relative_from_sternum_data(
        base_point=base_point,
        end_point=end_point,
        lm_disp_rel_sternum=lm_disp_rel_sternum,
        nipple_disp_left_vec=nipple_disp_left_vec,
        nipple_disp_right_vec=nipple_disp_right_vec,
        nipple_pos_prone_l=n_p_l,
        nipple_pos_prone_r=n_p_r,
        is_left=is_left,
        dts_values=dts_values,
        title=f"Landmark Motion Relative to Nipple (VL {vl_id})",
        save_path=save_path,
        use_dts_cmap=use_dts_cmap
    )

if __name__ == "__main__":
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
        print("\nTransition Matrix (FROM prone rows → TO supine columns):")
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

            print("\nTransition Matrix (FROM prone rows → TO supine columns):")
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
    # Define a mapping dictionary
    column_map = {
        # --- Positions (Relative to Sternum) ---
        'landmark ave prone transformed x': 'X_prone',
        'landmark ave prone transformed y': 'Y_prone',
        'landmark ave prone transformed z': 'Z_prone',
        'landmark ave supine x': 'X_supine',
        'landmark ave supine y': 'Y_supine',
        'landmark ave supine z': 'Z_supine',

        # --- VECTORS: Landmark Relative to Sternum (Displacement) ---
        'Landmark displacement vector vx': 'dX_sternum',
        'Landmark displacement vector vy': 'dY_sternum',
        'Landmark displacement vector vz': 'dZ_sternum',

        # --- VECTORS: Landmark Relative to Nipple (Intrinsic Deformation) ---
        'Landmark relative to nipple vector vx': 'dX_nipple',
        'Landmark relative to nipple vector vy': 'dY_nipple',
        'Landmark relative to nipple vector vz': 'dZ_nipple',

        'landmark side (prone)': 'Side'
    }

    # Rename the columns in your DataFrame
    df_ave_rename = df_ave.rename(columns=column_map)
    df_ave_rename['Side'] = df_ave_rename['Side'].replace({'LB': 'Left', 'RB': 'Right'})


    # 2. Run the Vectors Relative to Sternum plotting function
    plot_vectors_rel_sternum(df_ave)

    # 3. Run the Vectors Relative to Nipple plotting function (Intrinsic Deformation)
    # This plots landmark motion after subtracting the movement of the respective nipple.
    print("\n" + "=" * 50)
    print("PLOTTING LANDMARK MOTION RELATIVE TO NIPPLE")
    print("=" * 50)
    
    # Define save directory
    nipple_output_dir = Path("..") / "output" / "nipple_relative"
    
    # We'll plot for a specific subject (e.g., VL 81) as requested in previous iterations
    # but the function can be called for any subject index.
    plot_nipple_relative_motion_analysis(
        df_ave, 
        vl_id=81, 
        save_dir=nipple_output_dir, 
        use_dts_cmap=True
    )




