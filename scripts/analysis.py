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

OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v2_2025_12_01.xlsx"


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
        print("\n--- PARAMETRIC: Repeated Measures ANOVA Results ---")
        print(aov_rm)

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
            print("--- Pairwise Comparisons (Bonferroni) ---")
            print(post_hoc[['A', 'B', 'T', 'dof', 'p-unc', 'p-corr', 'hedges']])
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

        if p_friedman < ALPHA:
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


def plot_breast_motion_vectors(df, radius=100):
    """
    Plots the motion of landmarks from Prone to Supine position.
    Assumes coordinates are centered at the nipple (0,0).
    """
    # Define the planes and which coordinate columns to use
    planes = {
        'Coronal (Front View)': {'x': 'X', 'y': 'Y', 'label_x': 'Lateral-Medial', 'label_y': 'Sup-Inf'},
        'Axial (Top View)': {'x': 'X', 'y': 'Z', 'label_x': 'Lateral-Medial', 'label_y': 'Ant-Post'},
        'Sagittal (Side View)': {'x': 'Y', 'y': 'Z', 'label_x': 'Sup-Inf', 'label_y': 'Ant-Post'}
    }

    for plane_name, config in planes.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
        fig.suptitle(f'Landmark Displacement: {plane_name}', fontsize=16, fontweight='bold')

        for i, side in enumerate(['Right', 'Left']):
            ax = axes[i]
            side_df = df[df['Side'].str.capitalize() == side]

            if side_df.empty:
                continue

            # Calculate Deltas (Displacement)
            base_x = side_df[f"{config['x']}_prone"]
            base_y = side_df[f"{config['y']}_prone"]
            dx = side_df[f"{config['x']}_supine"] - base_x
            dy = side_df[f"{config['y']}_supine"] - base_y

            # 1. Plot Vectors (Quiver)
            # Teal arrows represent direction and magnitude of movement
            ax.quiver(base_x, base_y, dx, dy, angles='xy', scale_units='xy',
                      scale=1, color='teal', alpha=0.6, width=0.005, label='Movement Vector')

            # 2. Plot Start and End points
            ax.scatter(base_x, base_y, color='red', s=20, label='Prone (Start)', zorder=3)
            ax.scatter(side_df[f"{config['x']}_supine"], side_df[f"{config['y']}_supine"],
                       color='blue', s=10, alpha=0.5, label='Supine (End)')

            # 3. Formatting
            ax.set_title(f"{side} Breast", fontsize=14)
            ax.set_xlabel(config['label_x'])
            ax.set_ylabel(config['label_y'])
            ax.axhline(0, color='black', lw=1, ls='--')
            ax.axvline(0, color='black', lw=1, ls='--')

            # Nipple at origin
            ax.plot(0, 0, 'ko', markersize=8, label='Nipple')

            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_aspect('equal')
            if i == 1: ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
                angles='xy', scale_units='xy', scale=1, color='teal', alpha=0.7, width=0.005
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
    fibroadenoma_count = len(landmark_type_raw[landmark_type_raw == 'fibroadenoma'])
    print(f"Count of fibroadenoma: {fibroadenoma_count}")

    total_num_of_landmark = df_ave.shape[0]
    total_num_of_landmark_raw = df_raw.shape[0]/4
    print("Total number of landmarks is: ", total_num_of_landmark)
    print("Total number of landmarks before post processing is: ", total_num_of_landmark_raw)

    # how many landmarks per volunteer in raw and filtered data
    landmark_counts = df_ave.groupby('VL_ID').size()
    print("Number of landmarks per volunteer:\n", landmark_counts)
    print("Total number of volunteers is: ", landmark_counts.shape[0])
    landmark_counts_raw = df_raw.groupby('VL_ID').size()
    print("Total number of volunteers before post processing is: ", landmark_counts_raw.shape[0])

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


    #%% distance characteristics
    distance_metrics_map = [
        ("DTS (Skin)", "Distance to skin"),
        ("DTN (Nipple)", "Distance to nipple"),
        ("DTR (Rib Cage)", "Distance to rib cage")
    ]
    distance_rows = []

    for short_name, excel_prefix in distance_metrics_map:
        distance_row = {"Metric": short_name}

        for position in ['prone', 'supine']:
            col_name = f"{excel_prefix} ({position}) [mm]"
            prefix = position.title()  # "Prone" or "Supine"

            series = df_ave[col_name].dropna()

            # Calculate Stats
            mean_val = series.mean()
            std_val = series.std()
            median_val = series.median()

            # Add to row dictionary with specific column names
            distance_row[f"{prefix} Mean ± SD"] = f"{mean_val:.2f} ± {std_val:.2f}"
            distance_row[f"{prefix} Median"] = f"{median_val:.2f}"

        distance_rows.append(distance_row)

    # Create DataFrame
    distance_table = pd.DataFrame(distance_rows)

    # Set Metric as index for cleaner display
    distance_table.set_index("Metric", inplace=True)

    #%% --- DIFFERENCE IN DISTANCE TO SKIN, RIB CAGE, AND NIPPLES ---
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY (Side-by-Side Comparison)")
    print("=" * 80)

    # Set pandas display options to ensure columns align nicely
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    print(distance_table)
    print("\n" + "=" * 80)

    # Calculate change in distances (Supine - Prone)
    distance_change_rows = []
    dv_differences = []
    for short_name, excel_prefix in distance_metrics_map:
        prone_col = f"{excel_prefix} (prone) [mm]"
        supine_col = f"{excel_prefix} (supine) [mm]"

        # Calculate difference
        diff_col_name = f'diff_{short_name.replace(" ", "_").replace("(", "").replace(")", "")}'
        df_ave[diff_col_name] = df_ave[supine_col] - df_ave[prone_col]  # Use signed difference
        dv_differences.append(diff_col_name)

        differences = df_ave[diff_col_name]
        # df_ave['diff_magnitude'] = differences

        mean_diff = differences.mean()
        min_diff = differences.min()
        max_diff = differences.max()
        range_diff = max_diff - min_diff

        t_stat, p_val_t = stats.ttest_rel(df_ave[prone_col],df_ave[supine_col])
        # Significance Star
        sig_marker = "*" if p_val_t < 0.05 else "ns"

        distance_change_row = {
            "Metric": short_name,
            "Range [mm]": f"{range_diff:.2f}",
            "Minimum [mm]": f"{min_diff:.2f}",
            "Maximum [mm]": f"{max_diff:.2f}",
            "Mean [mm]": f"{mean_diff:.2f}",
            "P-value (T-test)": f"{p_val_t:.4e}",
            "Sig.": sig_marker
        }
        distance_change_rows.append(distance_change_row)
    distance_change_table = pd.DataFrame(distance_change_rows)
    distance_change_table.set_index("Metric", inplace=True)
    print("\n" + "=" * 80)
    print("DISTANCE CHANGE SUMMARY (Supine - Prone)")
    print("=" * 80)
    print(distance_change_table)




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


    #%% repeated anova for difference in distance to skin, rib cage, and nipples
    SUBJECT_ID = 'VL_ID'
    perform_repeated_measures_analysis(df_ave, SUBJECT_ID, dv_differences)

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
        'landmark ave prone transformed x': 'X_prone',
        'landmark ave prone transformed y': 'Y_prone',
        'landmark ave prone transformed z': 'Z_prone',
        'landmark ave supine x': 'X_supine',
        'landmark ave supine y': 'Y_supine',
        'landmark ave supine z': 'Z_supine',
        'landmark side (prone)': 'Side'
    }

    # Rename the columns in your DataFrame
    df_ave_rename = df_ave.rename(columns=column_map)
    df_ave_rename['Side'] = df_ave_rename['Side'].replace({'LB': 'Left', 'RB': 'Right'})
    plot_breast_motion_vectors(df_ave_rename, radius=250)

