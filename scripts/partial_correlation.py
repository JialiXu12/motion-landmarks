"""
Script for partial correlation analysis

Uses Spearman's rank correlation (ρ) instead of Pearson's (r) because:
- 100% of variables in this dataset are non-normally distributed (Shapiro-Wilk test)
- Presence of outliers in biomechanical measurements
- Non-linear monotonic relationships detected between key variables
- More robust for anatomical/biomechanical data

Spearman's correlation:
- Non-parametric (no normality assumption)
- Robust to outliers
- Captures monotonic relationships (not just linear)
- Appropriate for this New Zealand cohort with higher BMI variability
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def compute_partial_correlation(df, x_var, y_var, control_vars):
    """
    Compute partial correlation between x_var and y_var, controlling for control_vars.

    Uses the regression residual method:
    1. Regress X on control variables, get residuals (X_res)
    2. Regress Y on control variables, get residuals (Y_res)
    3. Correlate X_res with Y_res

    Args:
        df: DataFrame containing the variables
        x_var: First variable (string)
        y_var: Second variable (string)
        control_vars: List of control variable names

    Returns:
        Dictionary with correlation coefficient, p-value, and sample size
    """
    # Prepare data - drop rows with missing values
    vars_needed = [x_var, y_var] + control_vars
    df_clean = df[vars_needed].dropna()

    if len(df_clean) < 10:
        return {'r': np.nan, 'p_value': np.nan, 'n': len(df_clean), 'note': 'Insufficient data'}

    # Extract arrays
    X_raw = df_clean[x_var].values.reshape(-1, 1)
    Y_raw = df_clean[y_var].values.reshape(-1, 1)
    Z = df_clean[control_vars].values

    # Regress X on Z, get residuals
    model_x = LinearRegression()
    model_x.fit(Z, X_raw)
    X_residuals = X_raw - model_x.predict(Z)

    # Regress Y on Z, get residuals
    model_y = LinearRegression()
    model_y.fit(Z, Y_raw)
    Y_residuals = Y_raw - model_y.predict(Z)

    # Compute correlation between residuals
    # Using Spearman's correlation (non-parametric, robust to outliers and non-normality)
    rho, p_value = stats.spearmanr(X_residuals.flatten(), Y_residuals.flatten())

    return {
        'r': rho,  # Using 'r' key for consistency, but this is Spearman's ρ
        'p_value': p_value,
        'n': len(df_clean)
    }


def test_partial_correlation(df_ave):
    """Test the partial correlation analysis with actual data"""
    print("=" * 80)
    print("TESTING PARTIAL CORRELATION ANALYSIS")
    print("=" * 80)


    # Check if required columns exist
    required_cols = ['BMI', 'Age', 'Landmark displacement [mm]',
                     'Distance to rib cage (prone) [mm]']
    missing = [col for col in required_cols if col not in df_ave.columns]

    if missing:
        print(f"\n[ERROR] Missing required columns: {missing}")
        return

    print(f"[OK] All required columns present")

    # Test 1: Simple partial correlation
    print("\n" + "-" * 80)
    print("TEST 1: Partial Correlation (Displacement vs Initial Depth)")
    print("-" * 80)

    x_var = 'Landmark displacement [mm]'
    y_var = 'Distance to rib cage (prone) [mm]'
    control_vars = ['BMI', 'Age']

    # Compute zero-order correlation (using Spearman's)
    df_temp = df_ave[[x_var, y_var]].dropna()
    r_zero, p_zero = stats.spearmanr(df_temp[x_var], df_temp[y_var])

    # Compute partial correlation
    result_partial = compute_partial_correlation(df_ave, x_var, y_var, control_vars)

    print(f"\nVariable 1: {x_var}")
    print(f"Variable 2: {y_var}")
    print(f"Controls: {control_vars}")
    print(f"\nResults:")
    print(f"  Zero-order correlation:  r = {r_zero:.3f}, p = {p_zero:.4e}, n = {len(df_temp)}")
    print(f"  Partial correlation:     r = {result_partial['r']:.3f}, p = {result_partial['p_value']:.4e}, n = {result_partial['n']}")
    print(f"  Change in correlation:   Δr = {result_partial['r'] - r_zero:+.3f}")

    if abs(result_partial['r']) < abs(r_zero) * 0.7:
        print(f"  => BMI/Age confounding detected (>30% reduction)")
    elif abs(result_partial['r']) > abs(r_zero) * 1.3:
        print(f"  => BMI/Age suppression effect (>30% increase)")
    else:
        print(f"  => True biomechanical relationship (minimal change)")

    # Test 2: Multiple comparisons
    print("\n" + "-" * 80)
    print("TEST 2: Multiple Variable Comparisons")
    print("-" * 80)

    # Calculate Deltas
    df_ave['Delta_Rib'] = df_ave['Distance to rib cage (supine) [mm]'] - df_ave['Distance to rib cage (prone) [mm]']

    test_pairs = [
        ('Landmark displacement [mm]', 'Delta_Rib'),
        ('Landmark displacement vector vx', 'Distance to rib cage (prone) [mm]'),
        ('Landmark displacement vector vy', 'Delta_Rib'),
    ]

    print(f"\nTesting {len(test_pairs)} variable pairs...\n")

    for i, (var1, var2) in enumerate(test_pairs, 1):
        if var1 not in df_ave.columns or var2 not in df_ave.columns:
            print(f"{i}. {var1[:30]:30s} vs {var2[:30]:30s} -> SKIPPED (missing column)")
            continue

        df_temp = df_ave[[var1, var2]].dropna()
        if len(df_temp) < 10:
            print(f"{i}. {var1[:30]:30s} vs {var2[:30]:30s} -> SKIPPED (insufficient data)")
            continue

        r_zero, _ = stats.spearmanr(df_temp[var1], df_temp[var2])
        result = compute_partial_correlation(df_ave, var1, var2, control_vars)

        change_pct = ((result['r'] - r_zero) / abs(r_zero) * 100) if abs(r_zero) > 0.01 else 0

        print(f"{i}. r={r_zero:+.3f} -> r_partial={result['r']:+.3f} (Delta={change_pct:+.0f}%)")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nInterpretation:")
    print("  * If partial r << zero-order r: BMI/Age confounding")
    print("  * If partial r ~= zero-order r: True biomechanical effect")
    print("  * If partial r > zero-order r: BMI/Age suppression")


