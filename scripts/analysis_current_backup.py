from pathlib import Path
import pandas as pd
import pingouin as pg
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
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
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color='blue', alpha=0.6)
            elif color_by == 'dts':
                # For DTS coloring: black color for axial and coronal, no text for sagittal
                if plane_name != 'Sagittal':
                    right_x_pos = np.mean(base_right[:, axis_x_idx])
                    right_y_pos = lims[1] * 0.85
                    ax.text(right_x_pos, right_y_pos, 'Right Breast',
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='black', alpha=0.6)
            elif color_by == 'subject':
                # For subject coloring: black color for axial and coronal only
                if plane_name != 'Sagittal':
                    right_x_pos = np.mean(base_right[:, axis_x_idx])
                    right_y_pos = lims[1] * 0.85
                    ax.text(right_x_pos, right_y_pos, 'Right Breast',
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='black', alpha=0.6)

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
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color='green', alpha=0.6)
            elif color_by == 'dts':
                # For DTS coloring: black color for axial and coronal, no text for sagittal
                if plane_name != 'Sagittal':
                    left_x_pos = np.mean(base_left[:, axis_x_idx])
                    left_y_pos = lims[1] * 0.85
                    ax.text(left_x_pos, left_y_pos, 'Left Breast',
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='black', alpha=0.6)
            elif color_by == 'subject':
                # For subject coloring: black color for axial and coronal only
                if plane_name != 'Sagittal':
                    left_x_pos = np.mean(base_left[:, axis_x_idx])
                    left_y_pos = lims[1] * 0.85
                    ax.text(left_x_pos, left_y_pos, 'Left Breast',
                           ha='center', va='center', fontsize=11, fontweight='bold',
                           color='black', alpha=0.6)

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

        save_path = Path("..") / "output" / "figs" / "landmark vectors" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.show()
        plt.close(fig)

    # 7. Dual Sagittal View (optional)
    if include_dual_sagittal:
        plot_sagittal_dual_axes(
            df_ave, color_by=color_by, vl_id=vl_id
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


# ==============================================================================
# SECTION 9: BIOMECHANICAL CORRELATION ANALYSIS
# ==============================================================================

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
        'Landmark displacement [mm]': 'Total Displacement',
        'Landmark displacement vector vx': 'Displacement X (Med-Lat)',
        'Landmark displacement vector vy': 'Displacement Y (Ant-Post)',
        'Landmark displacement vector vz': 'Displacement Z (Inf-Sup)',
        'Delta_Rib': 'Δ Distance to Rib',
        'Delta_Skin': 'Δ Distance to Skin',
        'Delta_Nipple': 'Δ Distance to Nipple',
        'Distance to rib cage (prone) [mm]': 'Initial Depth (Prone)'
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
    save_path = Path("..") / "output" / "figs" / "correlation_matrix_anatomical.png"
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


# ==============================================================================
# SECTION 10: REFERENCE FRAME VALIDATION
# ==============================================================================

def compare_reference_frames(df):
    """
    Comprehensive comparison of landmark displacement calculations using
    different reference frames: Sternum (Absolute) vs Nipple (Relative).

    This analysis answers the critical question: "Does it matter which reference we use?"

    Args:
        df: DataFrame with both sternum-referenced and nipple-referenced displacement data
    """
    print("\n" + "=" * 80)
    print("REFERENCE FRAME COMPARISON: STERNUM vs NIPPLE")
    print("=" * 80)
    print("Analyzing differences between absolute (sternum) and relative (nipple) motion...")

    # Check available columns
    required_cols = {
        'sternum': 'Landmark displacement [mm]',
        'nipple': 'Landmark displacement relative to nipple [mm]'
    }

    missing = [name for name, col in required_cols.items() if col not in df.columns]
    if missing:
        print(f"\n!  Missing data for: {missing}")
        print("Available displacement columns:")
        disp_cols = [col for col in df.columns if 'displacement' in col.lower()]
        for col in disp_cols:
            print(f"  - {col}")
        return

    # 1. Basic Statistics Comparison
    print("\n" + "-" * 80)
    print("DISPLACEMENT MAGNITUDE COMPARISON")
    print("-" * 80)

    sternum_disp = df[required_cols['sternum']].dropna()
    nipple_disp = df[required_cols['nipple']].dropna()

    stats_comparison = pd.DataFrame({
        'Reference Frame': ['Sternum (Absolute)', 'Nipple (Relative)', 'Difference'],
        'Mean [mm]': [
            sternum_disp.mean(),
            nipple_disp.mean(),
            sternum_disp.mean() - nipple_disp.mean()
        ],
        'Std Dev [mm]': [
            sternum_disp.std(),
            nipple_disp.std(),
            sternum_disp.std() - nipple_disp.std()
        ],
        'Median [mm]': [
            sternum_disp.median(),
            nipple_disp.median(),
            sternum_disp.median() - nipple_disp.median()
        ],
        'Min [mm]': [
            sternum_disp.min(),
            nipple_disp.min(),
            sternum_disp.min() - nipple_disp.min()
        ],
        'Max [mm]': [
            sternum_disp.max(),
            nipple_disp.max(),
            sternum_disp.max() - nipple_disp.max()
        ]
    })

    print(stats_comparison.to_string(index=False))

    # 2. Paired Statistical Test
    from scipy import stats as scipy_stats

    # Get paired data (same landmarks)
    paired_df = df[[required_cols['sternum'], required_cols['nipple']]].dropna()

    if len(paired_df) > 0:
        t_stat, p_value = scipy_stats.ttest_rel(
            paired_df[required_cols['sternum']],
            paired_df[required_cols['nipple']]
        )

        print(f"\n📊 Paired t-test (Sternum vs Nipple):")
        print(f"   t-statistic: {t_stat:.3f}")
        print(f"   p-value: {p_value:.4e}")

        if p_value < 0.001:
            print(f"   *** HIGHLY SIGNIFICANT DIFFERENCE (p < 0.001)")
        elif p_value < 0.05:
            print(f"   ** SIGNIFICANT DIFFERENCE (p < 0.05)")
        else:
            print(f"   No significant difference (p ≥ 0.05)")

        # Effect size (Cohen's d)
        diff = paired_df[required_cols['sternum']] - paired_df[required_cols['nipple']]
        cohens_d = diff.mean() / diff.std()
        print(f"   Cohen's d (effect size): {cohens_d:.3f}")

        if abs(cohens_d) > 0.8:
            print(f"   → Large effect size")
        elif abs(cohens_d) > 0.5:
            print(f"   → Medium effect size")
        elif abs(cohens_d) > 0.2:
            print(f"   → Small effect size")
        else:
            print(f"   → Negligible effect size")

    # 3. Correlation between reference frames
    correlation = None
    if len(paired_df) > 0:
        correlation = paired_df[required_cols['sternum']].corr(paired_df[required_cols['nipple']])

        print(f"\n📈 Correlation between reference frames:")
        print(f"   Pearson r: {correlation:.3f}")

        if correlation > 0.9:
            print(f"   → Very strong positive correlation (frames are highly similar)")
        elif correlation > 0.7:
            print(f"   → Strong positive correlation")
        elif correlation > 0.5:
            print(f"   → Moderate positive correlation")
        else:
            print(f"   → Weak correlation (frames give different information)")

    # 4. Clinical Relevance Analysis
    print("\n" + "-" * 80)
    print("CLINICAL RELEVANCE: Which Reference Frame to Use?")
    print("-" * 80)

    mean_diff = sternum_disp.mean() - nipple_disp.mean()

    print(f"\n🏥 Sternum Reference (Absolute Motion):")
    print(f"   Mean displacement: {sternum_disp.mean():.2f} ± {sternum_disp.std():.2f} mm")
    print(f"   Use when:")
    print(f"   • Planning surgery with patient in supine position")
    print(f"   • Need to know WHERE the landmark is in space")
    print(f"   • Correlating with CT/MRI imaging (fixed coordinate system)")
    print(f"   • Designing registration algorithms")

    print(f"\n📍 Nipple Reference (Relative Motion):")
    print(f"   Mean displacement: {nipple_disp.mean():.2f} ± {nipple_disp.std():.2f} mm")
    print(f"   Use when:")
    print(f"   • Landmark is described relative to nipple (clock position)")
    print(f"   • Surgeon uses nipple as anatomical reference")
    print(f"   • Measuring 'true' tissue deformation")
    print(f"   • Isolating landmark motion independent of bulk breast movement")

    print(f"\n🔍 Key Insight:")
    if abs(mean_diff) > 10:
        print(f"   ⚠️  LARGE DIFFERENCE ({abs(mean_diff):.1f} mm) - Reference frame choice MATTERS!")
        print(f"   Sternum-referenced displacement is {abs(mean_diff):.1f} mm {'larger' if mean_diff > 0 else 'smaller'}")
        print(f"   This difference is clinically significant (>10mm)")
    elif abs(mean_diff) > 5:
        print(f"   ⚠️  MODERATE DIFFERENCE ({abs(mean_diff):.1f} mm) - Consider context carefully")
        print(f"   Sternum-referenced displacement is {abs(mean_diff):.1f} mm {'larger' if mean_diff > 0 else 'smaller'}")
    else:
        print(f"   ✓ Small difference ({abs(mean_diff):.1f} mm) - Both frames give similar results")

    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Scatter plot comparison
    ax1 = axes[0]
    if len(paired_df) > 0:
        ax1.scatter(paired_df[required_cols['sternum']],
                   paired_df[required_cols['nipple']],
                   alpha=0.6, s=50, c='steelblue')

        # Add identity line
        max_val = max(paired_df[required_cols['sternum']].max(),
                     paired_df[required_cols['nipple']].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect agreement')

        ax1.set_xlabel('Displacement relative to Sternum [mm]', fontsize=11)
        ax1.set_ylabel('Displacement relative to Nipple [mm]', fontsize=11)
        ax1.set_title('Reference Frame Comparison\n(Each point = one landmark)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

    # Plot 2: Distribution comparison
    ax2 = axes[1]
    positions = [1, 2]
    data_to_plot = [sternum_disp, nipple_disp]

    bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markersize=8))

    ax2.set_xticklabels(['Sternum\n(Absolute)', 'Nipple\n(Relative)'])
    ax2.set_ylabel('Displacement [mm]', fontsize=11)
    ax2.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean values as text
    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        ax2.text(pos, data.max() * 1.05, f'μ={data.mean():.1f}mm',
                ha='center', fontsize=9, fontweight='bold')

    # Plot 3: Bland-Altman plot (agreement analysis)
    ax3 = axes[2]
    if len(paired_df) > 0:
        mean_vals = (paired_df[required_cols['sternum']] + paired_df[required_cols['nipple']]) / 2
        diff_vals = paired_df[required_cols['sternum']] - paired_df[required_cols['nipple']]

        ax3.scatter(mean_vals, diff_vals, alpha=0.6, s=50, c='coral')

        # Add mean difference line
        mean_diff_val = diff_vals.mean()
        ax3.axhline(mean_diff_val, color='blue', linestyle='-', linewidth=2,
                   label=f'Mean diff: {mean_diff_val:.2f}mm')

        # Add ±1.96 SD lines (95% limits of agreement)
        std_diff = diff_vals.std()
        ax3.axhline(mean_diff_val + 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                   label=f'±1.96 SD: ±{1.96*std_diff:.2f}mm')
        ax3.axhline(mean_diff_val - 1.96*std_diff, color='red', linestyle='--', linewidth=1.5)
        ax3.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        ax3.set_xlabel('Mean of Both Measurements [mm]', fontsize=11)
        ax3.set_ylabel('Difference (Sternum - Nipple) [mm]', fontsize=11)
        ax3.set_title('Bland-Altman Plot\n(Agreement Analysis)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = Path("..") / "output" / "figs" / "reference_frame_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison figure: {save_path}")

    plt.show()
    plt.close()

    # 6. Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR YOUR ANALYSIS")
    print("=" * 80)

    if abs(mean_diff) > 10:
        print("\n⚠️  STRONG RECOMMENDATION: Report BOTH reference frames")
        print("\nReason: Large difference indicates they measure different aspects:")
        print("  • Sternum reference → Total spatial displacement (includes nipple motion)")
        print("  • Nipple reference → Relative tissue deformation (excludes nipple motion)")
        print("\nFor your paper, consider reporting:")
        print("  'Landmarks moved X mm relative to sternum (absolute) and Y mm relative")
        print("   to nipple (relative), indicating that Z mm of motion was shared bulk")
        print("   breast displacement.'")
    else:
        print("\n✓ Both reference frames give similar results")
        print("\nYou can choose based on clinical context:")
        print("  • If surgeon uses nipple as reference → Use nipple-referenced values")
        print("  • If correlating with imaging → Use sternum-referenced values")
        print("  • For biomechanics → Consider reporting both")

    print("\n" + "=" * 80)

    return {
        'sternum_mean': sternum_disp.mean(),
        'nipple_mean': nipple_disp.mean(),
        'difference': mean_diff,
        'correlation': correlation,
        'p_value': p_value if len(paired_df) > 0 else None
    }
