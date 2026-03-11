"""
Landmark Curation & Inter-Observer Variability Analysis

Standalone script that reads from observer_landmarks_comparison.xlsx and produces:
  - A multi-sheet Excel workbook with curation steps and inter-observer metrics
  - A sensitivity sweep plot (cutoff vs retained landmarks)

No project-specific imports (no JSON, mesh, NIfTI, or morphic dependencies).
Designed to be portable to a statistician's project that only has Excel files.

Input:  observer_landmarks_comparison.xlsx  (sheets: 'anthony', 'holly')
Output: landmark_curation_analysis_<date>.xlsx
        figs/sensitivity_sweep_threshold.png
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────
SUPINE_THRESHOLD_MM = 3.0           # Supine Euclidean distance cutoff (mm)
BOOTSTRAP_N = 10_000                # Participant-clustered bootstrap resamples
BOOTSTRAP_CI = 0.95                 # Confidence interval level
SENSITIVITY_MIN = 1.0               # Sweep range start (mm)
SENSITIVITY_MAX = 10.0              # Sweep range end (mm)
SENSITIVITY_STEP = 0.5              # Sweep step size (mm)

INPUT_EXCEL = Path(__file__).parent.parent / "output" / "observer_landmarks_comparison.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
FIGS_DIR = OUTPUT_DIR / "figs"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Loading & Merging
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load anthony and holly sheets from the input Excel file."""
    df_anthony = pd.read_excel(input_path, sheet_name="anthony")
    df_holly = pd.read_excel(input_path, sheet_name="holly")
    return df_anthony, df_holly


def build_raw_all(df_anthony: pd.DataFrame, df_holly: pd.DataFrame) -> pd.DataFrame:
    """
    Sheet 1: Merge all landmarks from both registrars by Subject + Filename.

    Every row is a landmark pair matched by filename (e.g. point.001.json).
    Includes rejected and fibroadenoma landmarks.
    """
    merged = pd.merge(
        df_anthony, df_holly,
        on=["Subject", "Filename"],
        how="outer",
        suffixes=(" (anthony)", " (holly)"),
    )

    column_order = [
        "Subject", "Filename",
        "Landmark Type (anthony)", "Landmark Type (holly)",
        "Status (anthony)", "Status (holly)",
        "Prone X (anthony)", "Prone X (holly)",
        "Prone Y (anthony)", "Prone Y (holly)",
        "Prone Z (anthony)", "Prone Z (holly)",
        "Supine X (anthony)", "Supine X (holly)",
        "Supine Y (anthony)", "Supine Y (holly)",
        "Supine Z (anthony)", "Supine Z (holly)",
    ]
    # Only keep columns that exist (handles missing columns gracefully)
    existing = [c for c in column_order if c in merged.columns]
    # Add any extra columns not in our order
    extra = [c for c in merged.columns if c not in existing]
    return merged[existing + extra].copy()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Curation Steps
# ═══════════════════════════════════════════════════════════════════════════

def build_matched_curated(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Sheet 2: Curate landmarks.

    (a) Verify prone coordinates are identical — flag discrepancies.
    (b) Remove rows where either registrar rejected.
    (c) Verify landmark type names match — flag mismatches.
    """
    df = df_raw.copy()

    # (a) Prone identity check
    prone_cols = [("Prone X", "Prone Y", "Prone Z")]
    prone_mismatch = pd.Series(False, index=df.index)
    for axis in ["Prone X", "Prone Y", "Prone Z"]:
        col_a = f"{axis} (anthony)"
        col_h = f"{axis} (holly)"
        if col_a in df.columns and col_h in df.columns:
            diff = (df[col_a] - df[col_h]).abs()
            prone_mismatch = prone_mismatch | (diff > 1e-6)

    df.insert(2, "Prone Mismatch", prone_mismatch)

    # (b) Remove rejected by either registrar
    status_a = df.get("Status (anthony)", pd.Series("accepted", index=df.index))
    status_h = df.get("Status (holly)", pd.Series("accepted", index=df.index))
    rejected = (status_a == "rejected") | (status_h == "rejected")
    df = df[~rejected].copy()

    # (c) Landmark type match check
    lt_a = df.get("Landmark Type (anthony)", pd.Series(dtype=str))
    lt_h = df.get("Landmark Type (holly)", pd.Series(dtype=str))
    df.insert(3, "Landmark Type Match", lt_a == lt_h)

    return df.reset_index(drop=True)


def build_no_fibroadenoma(df_curated: pd.DataFrame) -> pd.DataFrame:
    """Sheet 3: Remove all fibroadenoma landmarks."""
    df = df_curated.copy()

    lt_a = df.get("Landmark Type (anthony)", pd.Series(dtype=str))
    lt_h = df.get("Landmark Type (holly)", pd.Series(dtype=str))

    is_fibro = (lt_a == "fibroadenoma") | (lt_h == "fibroadenoma")
    return df[~is_fibro].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Supine Distances
# ═══════════════════════════════════════════════════════════════════════════

def build_supine_distances(df_no_fibro: pd.DataFrame) -> pd.DataFrame:
    """
    Sheet 4: Compute supine Euclidean distance + signed Δx, Δy, Δz
    (holly − anthony) for each landmark pair.
    """
    df = df_no_fibro.copy()

    dx = df["Supine X (holly)"] - df["Supine X (anthony)"]
    dy = df["Supine Y (holly)"] - df["Supine Y (anthony)"]
    dz = df["Supine Z (holly)"] - df["Supine Z (anthony)"]
    euclidean = np.sqrt(dx**2 + dy**2 + dz**2)

    df["Supine Δx (holly−anthony)"] = dx
    df["Supine Δy (holly−anthony)"] = dy
    df["Supine Δz (holly−anthony)"] = dz
    df["Supine Euclidean Distance"] = euclidean

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Sensitivity Sweep
# ═══════════════════════════════════════════════════════════════════════════

def build_sensitivity_sweep(
    df_distances: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Sheet 5: Sweep supine cutoff from SENSITIVITY_MIN to SENSITIVITY_MAX.

    Returns:
        df_sweep: Overall sweep table.
        stratified: Dict of landmark_type -> sweep table (if type matters).
    """
    distances = df_distances["Supine Euclidean Distance"].dropna()
    cutoffs = np.arange(SENSITIVITY_MIN, SENSITIVITY_MAX + SENSITIVITY_STEP / 2, SENSITIVITY_STEP)

    rows = []
    for cutoff in cutoffs:
        retained = distances[distances <= cutoff]
        rows.append({
            "Cutoff_mm": round(cutoff, 1),
            "N_retained": len(retained),
            "N_excluded": len(distances) - len(retained),
            "Mean_distance": retained.mean() if len(retained) > 0 else np.nan,
            "Median_distance": retained.median() if len(retained) > 0 else np.nan,
            "SD_distance": retained.std() if len(retained) > 1 else np.nan,
        })

    df_sweep = pd.DataFrame(rows)

    # Stratify by landmark type
    lt_col = "Landmark Type (anthony)"
    stratified = {}
    if lt_col in df_distances.columns:
        for ltype in sorted(df_distances[lt_col].dropna().unique()):
            mask = df_distances[lt_col] == ltype
            dists_type = df_distances.loc[mask, "Supine Euclidean Distance"].dropna()
            type_rows = []
            for cutoff in cutoffs:
                retained = dists_type[dists_type <= cutoff]
                type_rows.append({
                    "Cutoff_mm": round(cutoff, 1),
                    f"N_retained ({ltype})": len(retained),
                    f"N_excluded ({ltype})": len(dists_type) - len(retained),
                    f"Mean_distance ({ltype})": retained.mean() if len(retained) > 0 else np.nan,
                    f"Median_distance ({ltype})": retained.median() if len(retained) > 0 else np.nan,
                })
            stratified[ltype] = pd.DataFrame(type_rows)

    # Merge stratified into main sweep table
    if stratified:
        for ltype, df_type in stratified.items():
            df_sweep = df_sweep.merge(df_type, on="Cutoff_mm", how="left")

    return df_sweep, stratified


def plot_sensitivity_sweep(
    df_sweep: pd.DataFrame,
    stratified: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Save sensitivity sweep plot:
      Top panel:    cutoff vs number of retained landmarks
      Bottom panel: cutoff vs mean/median Euclidean distance among retained
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    cutoffs = df_sweep["Cutoff_mm"]

    # ── Top panel: retained count ──
    ax1.plot(cutoffs, df_sweep["N_retained"], "k-o", markersize=4, label="All types")
    # Stratified lines
    for ltype in sorted(stratified.keys()):
        col = f"N_retained ({ltype})"
        if col in df_sweep.columns:
            ax1.plot(cutoffs, df_sweep[col], "--", markersize=3, label=ltype)

    ax1.axvline(SUPINE_THRESHOLD_MM, color="red", linestyle=":", alpha=0.7,
                label=f"Threshold = {SUPINE_THRESHOLD_MM} mm")
    ax1.set_ylabel("Number of retained landmarks")
    ax1.set_title("Sensitivity Analysis: Supine Distance Cutoff")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: mean/median distance ──
    ax2.plot(cutoffs, df_sweep["Mean_distance"], "b-o", markersize=4, label="Mean")
    ax2.plot(cutoffs, df_sweep["Median_distance"], "g-s", markersize=4, label="Median")
    ax2.axvline(SUPINE_THRESHOLD_MM, color="red", linestyle=":", alpha=0.7,
                label=f"Threshold = {SUPINE_THRESHOLD_MM} mm")
    ax2.set_xlabel("Cutoff (mm)")
    ax2.set_ylabel("Euclidean distance among retained (mm)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sensitivity plot: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Threshold Application
# ═══════════════════════════════════════════════════════════════════════════

def build_threshold(df_distances: pd.DataFrame) -> pd.DataFrame:
    """Sheet 6: Keep only landmarks with supine Euclidean distance ≤ threshold."""
    mask = df_distances["Supine Euclidean Distance"] <= SUPINE_THRESHOLD_MM
    return df_distances[mask].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Inter-Observer Analysis
# ═══════════════════════════════════════════════════════════════════════════

def build_interobserver_per_landmark(df_threshold: pd.DataFrame) -> pd.DataFrame:
    """Sheet 7: Per-landmark inter-observer metrics."""
    cols = [
        "Subject", "Filename",
        "Landmark Type (anthony)", "Landmark Type (holly)",
        "Supine Euclidean Distance",
        "Supine Δx (holly−anthony)",
        "Supine Δy (holly−anthony)",
        "Supine Δz (holly−anthony)",
    ]
    existing = [c for c in cols if c in df_threshold.columns]
    return df_threshold[existing].copy()


def build_interobserver_per_participant(df_threshold: pd.DataFrame) -> pd.DataFrame:
    """Sheet 8: Per-participant summary of inter-observer metrics."""
    grouped = df_threshold.groupby("Subject")

    rows = []
    for subject, grp in grouped:
        dists = grp["Supine Euclidean Distance"].dropna()
        dx = grp["Supine Δx (holly−anthony)"].dropna()
        dy = grp["Supine Δy (holly−anthony)"].dropna()
        dz = grp["Supine Δz (holly−anthony)"].dropna()

        rows.append({
            "Subject": subject,
            "N_landmarks": len(dists),
            "Mean_distance": dists.mean() if len(dists) > 0 else np.nan,
            "Median_distance": dists.median() if len(dists) > 0 else np.nan,
            "SD_distance": dists.std() if len(dists) > 1 else np.nan,
            "Mean_Δx": dx.mean() if len(dx) > 0 else np.nan,
            "Mean_Δy": dy.mean() if len(dy) > 0 else np.nan,
            "Mean_Δz": dz.mean() if len(dz) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def _clustered_bootstrap_mean(
    df_per_participant: pd.DataFrame,
    n_resamples: int,
    ci_level: float,
) -> Dict[str, float]:
    """
    Participant-clustered bootstrap for grand mean of per-participant mean distances.

    Resamples participants (not landmarks), computes grand mean for each resample.
    Returns BCa 95% CI.
    """
    participant_means = df_per_participant["Mean_distance"].dropna().values
    n = len(participant_means)
    if n < 2:
        return {
            "bootstrap_mean": float(participant_means.mean()) if n > 0 else np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
        }

    observed_mean = participant_means.mean()

    rng = np.random.default_rng(42)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(participant_means, size=n, replace=True)
        boot_means[i] = sample.mean()

    # BCa confidence interval
    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_means < observed_mean))

    # Acceleration (jackknife)
    jackknife_means = np.empty(n)
    for i in range(n):
        jackknife_means[i] = np.delete(participant_means, i).mean()
    jack_mean = jackknife_means.mean()
    num = np.sum((jack_mean - jackknife_means) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    alpha = 1.0 - ci_level
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # Adjusted percentiles
    p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

    ci_lower = np.percentile(boot_means, 100 * p_lower)
    ci_upper = np.percentile(boot_means, 100 * p_upper)

    return {
        "bootstrap_mean": float(observed_mean),
        "bootstrap_ci_lower": float(ci_lower),
        "bootstrap_ci_upper": float(ci_upper),
    }


def _fit_mixed_effects(df_threshold: pd.DataFrame) -> Dict[str, float]:
    """
    Mixed-effects model: log(distance) ~ 1 + (1|participant) + (1|landmark_type).

    Uses log-transformed distances with a linear mixed model as a practical
    approximation to a Gamma GLMM.  Falls back gracefully if statsmodels
    is unavailable or the model fails to converge.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  Warning: statsmodels not available. Skipping mixed-effects model.")
        return {
            "mixed_effects_mean": np.nan,
            "mixed_effects_se": np.nan,
            "mixed_effects_ci_lower": np.nan,
            "mixed_effects_ci_upper": np.nan,
            "participant_re_var": np.nan,
            "landmark_type_re_var": np.nan,
        }

    df = df_threshold[["Subject", "Supine Euclidean Distance"]].copy()
    df = df.rename(columns={"Supine Euclidean Distance": "distance"})
    df["log_distance"] = np.log(df["distance"])

    lt_col = "Landmark Type (anthony)"
    if lt_col in df_threshold.columns:
        df["landmark_type"] = df_threshold[lt_col].values
    else:
        df["landmark_type"] = "unknown"

    df = df.dropna(subset=["log_distance"])
    if len(df) < 5:
        return {
            "mixed_effects_mean": np.nan,
            "mixed_effects_se": np.nan,
            "mixed_effects_ci_lower": np.nan,
            "mixed_effects_ci_upper": np.nan,
            "participant_re_var": np.nan,
            "landmark_type_re_var": np.nan,
        }

    try:
        # Nested random effects: participant, then landmark_type within participant
        # statsmodels MixedLM supports one grouping variable; use variance components
        # approach with Subject as group.
        model = smf.mixedlm(
            "log_distance ~ 1",
            data=df,
            groups=df["Subject"],
            re_formula="1",
            vc_formula={"landmark_type": "0 + C(landmark_type)"},
        )
        result = model.fit(reml=True)

        intercept = result.fe_params["Intercept"]
        se = result.bse["Intercept"]

        # Back-transform from log scale (approximate mean on original scale)
        # For log-normal: E[X] = exp(mu + sigma^2/2)
        residual_var = result.scale
        estimated_mean = np.exp(intercept + residual_var / 2)

        ci_lower = np.exp(intercept - 1.96 * se)
        ci_upper = np.exp(intercept + 1.96 * se)

        participant_var = float(result.cov_re.iloc[0, 0])
        # Variance component for landmark type
        lt_var = np.nan
        if hasattr(result, "vcomp") and len(result.vcomp) > 0:
            lt_var = float(result.vcomp[0])

        return {
            "mixed_effects_mean": float(estimated_mean),
            "mixed_effects_se": float(se),
            "mixed_effects_ci_lower": float(ci_lower),
            "mixed_effects_ci_upper": float(ci_upper),
            "participant_re_var": participant_var,
            "landmark_type_re_var": lt_var,
        }
    except Exception as e:
        print(f"  Warning: Mixed-effects model failed: {e}")
        return {
            "mixed_effects_mean": np.nan,
            "mixed_effects_se": np.nan,
            "mixed_effects_ci_lower": np.nan,
            "mixed_effects_ci_upper": np.nan,
            "participant_re_var": np.nan,
            "landmark_type_re_var": np.nan,
        }


def _component_wise_bias(df_per_participant: pd.DataFrame) -> Dict[str, float]:
    """
    Test whether mean Δx, Δy, Δz across participants differ from zero.

    Uses one-sample t-test (or Wilcoxon if n < 20).
    """
    results = {}
    for axis in ["Δx", "Δy", "Δz"]:
        col = f"Mean_{axis}"
        values = df_per_participant[col].dropna().values
        n = len(values)
        results[f"mean_{axis}"] = float(values.mean()) if n > 0 else np.nan
        results[f"sd_{axis}"] = float(values.std()) if n > 1 else np.nan

        if n >= 5:
            if n >= 20:
                t_stat, p_val = stats.ttest_1samp(values, 0)
                results[f"test_{axis}"] = "t-test"
            else:
                t_stat, p_val = stats.wilcoxon(values)
                results[f"test_{axis}"] = "Wilcoxon"
            results[f"p_{axis}"] = float(p_val)
        else:
            results[f"test_{axis}"] = "n<5"
            results[f"p_{axis}"] = np.nan

    return results


def build_interobserver_summary(
    df_per_participant: pd.DataFrame,
    df_threshold: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sheet 9: Overall inter-observer summary.

    Methods:
      (a) Participant-level mean -> grand mean ± SD
      (b) Participant-level median -> grand median
      (c) Participant-clustered bootstrap (BCa CI)
      (d) Mixed-effects model
      + Component-wise bias summary
    """
    part_means = df_per_participant["Mean_distance"].dropna()
    part_medians = df_per_participant["Median_distance"].dropna()

    rows = []

    # (a) Participant-level mean
    rows.append({
        "Method": "Participant-level mean -> grand mean",
        "Estimate": part_means.mean(),
        "SD": part_means.std(),
        "CI_lower": np.nan,
        "CI_upper": np.nan,
        "N_participants": len(part_means),
        "Note": "Mean of per-participant mean distances",
    })

    # (b) Participant-level median
    rows.append({
        "Method": "Participant-level median -> grand median",
        "Estimate": part_medians.median(),
        "SD": np.nan,
        "CI_lower": np.nan,
        "CI_upper": np.nan,
        "N_participants": len(part_medians),
        "Note": "Median of per-participant median distances",
    })

    # (c) Bootstrap
    boot = _clustered_bootstrap_mean(df_per_participant, BOOTSTRAP_N, BOOTSTRAP_CI)
    rows.append({
        "Method": f"Participant-clustered bootstrap ({BOOTSTRAP_N} resamples)",
        "Estimate": boot["bootstrap_mean"],
        "SD": np.nan,
        "CI_lower": boot["bootstrap_ci_lower"],
        "CI_upper": boot["bootstrap_ci_upper"],
        "N_participants": len(part_means),
        "Note": f"{BOOTSTRAP_CI*100:.0f}% BCa CI",
    })

    # (d) Mixed-effects
    me = _fit_mixed_effects(df_threshold)
    rows.append({
        "Method": "Mixed-effects (log-normal approx)",
        "Estimate": me["mixed_effects_mean"],
        "SD": me["mixed_effects_se"],
        "CI_lower": me["mixed_effects_ci_lower"],
        "CI_upper": me["mixed_effects_ci_upper"],
        "N_participants": len(part_means),
        "Note": (f"log(dist) ~ 1 + (1|participant) + (1|landmark_type); "
                 f"RE var(participant)={me['participant_re_var']:.4f}, "
                 f"RE var(landmark_type)={me['landmark_type_re_var']:.4f}"
                 if not np.isnan(me["participant_re_var"]) else "Model did not converge"),
    })

    # Component-wise bias
    bias = _component_wise_bias(df_per_participant)
    for axis in ["Δx", "Δy", "Δz"]:
        p_val = bias[f"p_{axis}"]
        p_str = f"p={p_val:.4f}" if not np.isnan(p_val) else "n/a"
        rows.append({
            "Method": f"Component-wise bias: {axis}",
            "Estimate": bias[f"mean_{axis}"],
            "SD": bias[f"sd_{axis}"],
            "CI_lower": np.nan,
            "CI_upper": np.nan,
            "N_participants": len(df_per_participant),
            "Note": f"{bias[f'test_{axis}']} {p_str} (holly − anthony, per-participant means)",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6b. Bland-Altman Plots & ICC
# ═══════════════════════════════════════════════════════════════════════════

def plot_bland_altman(df_threshold: pd.DataFrame, output_path: Path) -> None:
    """
    Bland-Altman plots for supine Δx, Δy, Δz (holly − anthony).

    Each panel plots (anthony + holly)/2 on x-axis vs (holly − anthony) on y-axis,
    with mean bias and ±1.96 SD limits of agreement.
    """
    axes_info = [
        ("X", "Supine X (anthony)", "Supine X (holly)"),
        ("Y", "Supine Y (anthony)", "Supine Y (holly)"),
        ("Z", "Supine Z (anthony)", "Supine Z (holly)"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (label, col_a, col_h) in zip(axs, axes_info):
        a = df_threshold[col_a].values
        h = df_threshold[col_h].values

        mean_val = (a + h) / 2.0
        diff = h - a

        bias = np.mean(diff)
        sd = np.std(diff, ddof=1)
        loa_upper = bias + 1.96 * sd
        loa_lower = bias - 1.96 * sd

        ax.scatter(mean_val, diff, alpha=0.5, s=20, edgecolors="none")
        ax.axhline(bias, color="red", linestyle="-", linewidth=1.5,
                    label=f"Bias = {bias:.3f} mm")
        ax.axhline(loa_upper, color="grey", linestyle="--", linewidth=1,
                    label=f"+1.96 SD = {loa_upper:.3f}")
        ax.axhline(loa_lower, color="grey", linestyle="--", linewidth=1,
                    label=f"-1.96 SD = {loa_lower:.3f}")
        ax.axhline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_xlabel(f"Mean supine {label} (mm)")
        ax.set_ylabel(f"Difference {label}: holly - anthony (mm)")
        ax.set_title(f"Bland-Altman: Supine {label}")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Bland-Altman plot: {output_path}")


def compute_icc(df_threshold: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICC(3,1) — two-way mixed, single measures, consistency —
    for each supine coordinate axis (X, Y, Z).

    ICC(3,1) is appropriate when the same two fixed raters assess all subjects.

    Formula: ICC(3,1) = (MS_subjects - MS_error) /
                        (MS_subjects + (k-1)*MS_error)
    where k = number of raters (2).
    """
    axes_info = [
        ("X", "Supine X (anthony)", "Supine X (holly)"),
        ("Y", "Supine Y (anthony)", "Supine Y (holly)"),
        ("Z", "Supine Z (anthony)", "Supine Z (holly)"),
    ]

    rows = []
    for label, col_a, col_h in axes_info:
        a = df_threshold[col_a].dropna().values
        h = df_threshold[col_h].dropna().values
        n = len(a)
        k = 2  # number of raters

        if n < 3:
            rows.append({
                "Axis": label,
                "ICC(3,1)": np.nan,
                "95% CI lower": np.nan,
                "95% CI upper": np.nan,
                "N_landmarks": n,
                "Bias (mm)": np.nan,
                "LoA lower (mm)": np.nan,
                "LoA upper (mm)": np.nan,
            })
            continue

        # Stack into n x k matrix
        data = np.column_stack([a, h])  # shape (n, 2)

        # Grand mean
        grand_mean = data.mean()

        # Subject means (row means)
        subject_means = data.mean(axis=1)

        # Rater means (column means)
        rater_means = data.mean(axis=0)

        # Sum of squares
        ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
        ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_error = ss_total - ss_subjects - ss_raters

        # Mean squares
        df_subjects = n - 1
        df_error = (n - 1) * (k - 1)

        ms_subjects = ss_subjects / df_subjects
        ms_error = ss_error / df_error

        # ICC(3,1)
        icc = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error)

        # 95% CI for ICC(3,1) using F-distribution
        # F = MS_subjects / MS_error
        f_val = ms_subjects / ms_error if ms_error > 0 else np.inf
        f_lower = f_val / stats.f.ppf(0.975, df_subjects, df_error)
        f_upper = f_val / stats.f.ppf(0.025, df_subjects, df_error)

        icc_lower = (f_lower - 1) / (f_lower + k - 1)
        icc_upper = (f_upper - 1) / (f_upper + k - 1)

        # Bland-Altman stats for this axis
        diff = h - a
        bias = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)

        rows.append({
            "Axis": label,
            "ICC(3,1)": float(icc),
            "95% CI lower": float(icc_lower),
            "95% CI upper": float(icc_upper),
            "N_landmarks": n,
            "Bias (mm)": float(bias),
            "LoA lower (mm)": float(bias - 1.96 * sd_diff),
            "LoA upper (mm)": float(bias + 1.96 * sd_diff),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Excel Writer
# ═══════════════════════════════════════════════════════════════════════════

def write_output(
    sheets: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Write all sheets to an Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet names have a 31-char limit
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            print(f"  Sheet '{safe_name}': {len(df)} rows, {len(df.columns)} cols")

    print(f"\n  Output saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Landmark Curation & Inter-Observer Variability Analysis")
    print(f"  Input:     {INPUT_EXCEL}")
    print(f"  Threshold: {SUPINE_THRESHOLD_MM} mm")
    print("=" * 70)

    # ── Load ──
    print("\n1. Loading raw data...")
    df_anthony, df_holly = load_raw_data(INPUT_EXCEL)
    print(f"  Anthony: {len(df_anthony)} landmarks, "
          f"{df_anthony['Subject'].nunique()} subjects")
    print(f"  Holly:   {len(df_holly)} landmarks, "
          f"{df_holly['Subject'].nunique()} subjects")

    # ── Sheet 1: Raw All ──
    print("\n2. Building 1_raw_all (merge by filename)...")
    df_raw_all = build_raw_all(df_anthony, df_holly)
    print(f"  Merged: {len(df_raw_all)} rows")

    # ── Sheet 2: Matched & Curated ──
    print("\n3. Building 2_matched_curated...")
    df_curated = build_matched_curated(df_raw_all)
    n_prone_mismatch = df_curated["Prone Mismatch"].sum()
    n_type_mismatch = (~df_curated["Landmark Type Match"]).sum()
    print(f"  After removing rejected: {len(df_curated)} rows")
    print(f"  Prone mismatches flagged: {n_prone_mismatch}")
    print(f"  Landmark type mismatches flagged: {n_type_mismatch}")

    # ── Sheet 3: No Fibroadenoma ──
    print("\n4. Building 3_no_fibroadenoma...")
    df_no_fibro = build_no_fibroadenoma(df_curated)
    n_fibro_removed = len(df_curated) - len(df_no_fibro)
    print(f"  Removed {n_fibro_removed} fibroadenoma rows -> {len(df_no_fibro)} remaining")

    # ── Sheet 4: Supine Distances ──
    print("\n5. Building 4_supine_distances...")
    df_distances = build_supine_distances(df_no_fibro)
    dists = df_distances["Supine Euclidean Distance"].dropna()
    print(f"  Euclidean distance: mean={dists.mean():.2f}, "
          f"median={dists.median():.2f}, SD={dists.std():.2f} mm")

    # ── Sheet 5: Sensitivity Sweep ──
    print("\n6. Building 5_sensitivity_sweep...")
    df_sweep, stratified = build_sensitivity_sweep(df_distances)
    print(f"  Sweep: {len(df_sweep)} cutoff values "
          f"({SENSITIVITY_MIN}–{SENSITIVITY_MAX} mm)")

    plot_path = FIGS_DIR / "sensitivity_sweep_threshold.png"
    plot_sensitivity_sweep(df_sweep, stratified, plot_path)

    # ── Sheet 6: Threshold ──
    print(f"\n7. Building 6_threshold (<= {SUPINE_THRESHOLD_MM} mm)...")
    df_threshold = build_threshold(df_distances)
    n_excluded = len(df_distances) - len(df_threshold)
    print(f"  Retained: {len(df_threshold)}, Excluded: {n_excluded}")

    # ── Sheet 7: Per-Landmark ──
    print("\n8. Building 7_interobserver_per_landmark...")
    df_per_lm = build_interobserver_per_landmark(df_threshold)

    # ── Sheet 8: Per-Participant ──
    print("\n9. Building 8_interobserver_per_participant...")
    df_per_part = build_interobserver_per_participant(df_threshold)
    print(f"  {len(df_per_part)} participants")

    # ── Sheet 9: Summary ──
    print("\n10. Building 9_interobserver_summary...")
    df_summary = build_interobserver_summary(df_per_part, df_threshold)
    for _, row in df_summary.iterrows():
        est = row["Estimate"]
        est_str = f"{est:.3f}" if not np.isnan(est) else "n/a"
        method_ascii = row['Method'].encode('ascii', 'replace').decode('ascii')
        print(f"  {method_ascii}: {est_str}")

    # ── Sheet 10: ICC + Bland-Altman summary ──
    print("\n11. Building 10_icc_bland_altman...")
    df_icc = compute_icc(df_threshold)
    for _, row in df_icc.iterrows():
        icc_val = row["ICC(3,1)"]
        icc_str = f"{icc_val:.4f}" if not np.isnan(icc_val) else "n/a"
        ci_l = row["95% CI lower"]
        ci_u = row["95% CI upper"]
        ci_str = f"[{ci_l:.4f}, {ci_u:.4f}]" if not np.isnan(ci_l) else ""
        print(f"  Supine {row['Axis']}: ICC(3,1) = {icc_str} {ci_str}, "
              f"bias = {row['Bias (mm)']:.3f} mm, "
              f"LoA = [{row['LoA lower (mm)']:.3f}, {row['LoA upper (mm)']:.3f}]")

    # ── Bland-Altman plot ──
    print("\n12. Generating Bland-Altman plots...")
    ba_path = FIGS_DIR / "bland_altman_interobserver.png"
    plot_bland_altman(df_threshold, ba_path)

    # ── Write Excel ──
    timestamp = datetime.now().strftime("%Y_%m_%d")
    output_path = OUTPUT_DIR / f"landmark_curation_analysis_{timestamp}.xlsx"

    sheets = {
        "1_raw_all": df_raw_all,
        "2_matched_curated": df_curated,
        "3_no_fibroadenoma": df_no_fibro,
        "4_supine_distances": df_distances,
        "5_sensitivity_sweep": df_sweep,
        "6_threshold": df_threshold,
        "7_interobserver_per_landmark": df_per_lm,
        "8_interobserver_per_particip": df_per_part,
        "9_interobserver_summary": df_summary,
        "10_icc_bland_altman": df_icc,
    }

    print(f"\n13. Writing Excel...")
    write_output(sheets, output_path)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
