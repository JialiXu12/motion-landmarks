"""
Analysis: Optimal Stopping Criteria for ICP Alignment

This script analyzes the trade-off between iterations and alignment quality
to determine the optimal stopping criteria.

Key Question:
- 100 iterations: Slightly higher RMSE, slightly lower STD
- 200 iterations: Slightly lower RMSE, slightly higher STD

Which is better?
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_convergence_tradeoff():
    """
    Analyzes the convergence behavior and recommends optimal settings.
    """

    print("=" * 80)
    print("CONVERGENCE ANALYSIS: 100 vs 200 Iterations")
    print("=" * 80)

    print("\n1. UNDERSTANDING THE METRICS:")
    print("-" * 80)
    print("""
RMSE (Root Mean Square Error):
- Measures average alignment error across all inlier points
- Lower RMSE = better average fit
- Sensitive to outliers (squared errors amplify large deviations)

Standard Deviation (STD):
- Measures variability/spread of errors
- Lower STD = more consistent alignment (less variation)
- Indicates robustness - how uniform the fit is across the surface

Trade-off:
┌────────────────┬──────────┬─────────┬────────────────────────────┐
│ Iterations     │ RMSE     │ STD     │ Interpretation             │
├────────────────┼──────────┼─────────┼────────────────────────────┤
│ 100 (Fewer)    │ Higher   │ Lower   │ Stops earlier, more stable │
│ 200 (More)     │ Lower    │ Higher  │ Overfits, less consistent  │
└────────────────┴──────────┴─────────┴────────────────────────────┘
    """)

    print("\n2. WHAT THIS MEANS:")
    print("-" * 80)
    print("""
Scenario: 100 iterations vs 200 iterations

Higher RMSE at 100 iterations:
- Algorithm hasn't fully minimized the objective function
- Still room for improvement in average error
- BUT: May have found a MORE ROBUST solution

Lower STD at 100 iterations:
- Errors are more UNIFORM across the surface
- Less variance = more consistent alignment
- Better generalization (not overfitting to noise)

Lower RMSE at 200 iterations:
- Algorithm has further minimized the objective
- Better average fit to the data
- BUT: May be fitting to noise or outliers

Higher STD at 200 iterations:
- Errors are MORE VARIABLE across the surface
- Some areas fit very well, others fit worse
- Classic sign of OVERFITTING
    """)

    print("\n3. RECOMMENDATION:")
    print("-" * 80)
    print("""
For Medical Imaging Alignment, PREFER:

✓ LOWER STD (more consistent) over slightly lower RMSE
✓ Earlier stopping (100 iterations) if STD is better
✓ Robust alignment that generalizes well

Why?
1. Anatomical stability > fitting noise
2. Consistency across patients matters more than perfect fit on one scan
3. Lower variance = more reliable clinical measurements
4. Overfitting can introduce spurious deformations

RECOMMENDED: Use 100 iterations (or adaptive stopping)
    """)

    print("\n4. IMPROVED STOPPING CRITERIA:")
    print("-" * 80)
    print("""
Instead of fixed iterations, use ADAPTIVE stopping based on:

1. RMSE Convergence: Stop when RMSE change < threshold
   Current: abs(prev_rmse - rmse) < 1e-6
   
2. Rotation Convergence: Stop when rotation change is minimal
   Add: np.linalg.norm(R_delta - np.eye(3)) < 1e-6
   
3. Windowed Convergence: Check last N iterations
   Stop if RMSE improves < threshold for N consecutive iterations
   
4. STD Monitoring: Stop if STD starts increasing
   Indicates potential overfitting
   
5. Early Stopping with Patience:
   - Track best RMSE
   - Stop if no improvement for N iterations
   - Return best solution, not last iteration
    """)

    print("\n5. OPTIMAL CONFIGURATION:")
    print("-" * 80)
    print("""
Recommended parameters for optimal_sternum_fixed_alignment:

max_iterations: 150  (middle ground between 100-200)
convergence_threshold: 1e-5  (slightly relaxed)
patience: 10  (stop if no improvement for 10 iterations)
rotation_threshold: 1e-6  (minimal rotation change)

Additional criteria:
- Monitor both RMSE and STD
- Use early stopping with best-solution tracking
- Consider clinical requirements (consistency vs accuracy)
    """)

    print("\n6. DECISION RULE:")
    print("-" * 80)
    print("""
Choose between two solutions A and B:

IF (RMSE_diff < 0.5 mm):  # Minimal difference
    CHOOSE: Lower STD (more robust)
ELSE IF (RMSE_diff > 2.0 mm):  # Significant difference
    CHOOSE: Lower RMSE (better accuracy)
ELSE:
    COMPUTE: Score = w_rmse * RMSE + w_std * STD
    WHERE: w_rmse = 0.6, w_std = 0.4
    CHOOSE: Lower combined score

For your case (slight differences):
→ CHOOSE 100 iterations (lower STD)
    """)

    return {
        'recommended_iterations': 150,
        'convergence_threshold': 1e-5,
        'patience': 10,
        'rotation_threshold': 1e-6,
        'monitor_std': True,
        'use_early_stopping': True
    }


def create_convergence_plot_example():
    """
    Creates an example plot showing typical convergence behavior.
    """

    # Simulate typical convergence curves
    iterations = np.arange(1, 201)

    # Simulate RMSE (decreasing with diminishing returns)
    rmse = 8.0 * np.exp(-iterations / 30) + 3.5 + 0.1 * np.random.randn(200).cumsum() * 0.01

    # Simulate STD (U-shaped: decreases then increases due to overfitting)
    std = 4.0 * np.exp(-iterations / 40) + 2.5 + 0.05 * (iterations - 100) ** 2 / 1000

    # Simulate rotation change (exponential decay)
    rotation_change = 0.1 * np.exp(-iterations / 20) + 0.0001

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot RMSE
    axes[0].plot(iterations, rmse, 'b-', linewidth=2, label='RMSE')
    axes[0].axvline(100, color='orange', linestyle='--', linewidth=2, label='100 iterations')
    axes[0].axvline(200, color='red', linestyle='--', linewidth=2, label='200 iterations')
    axes[0].set_ylabel('RMSE (mm)', fontsize=12, fontweight='bold')
    axes[0].set_title('Convergence Behavior: RMSE, STD, and Rotation Change', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot STD
    axes[1].plot(iterations, std, 'g-', linewidth=2, label='STD')
    axes[1].axvline(100, color='orange', linestyle='--', linewidth=2, label='100 iterations (optimal)')
    axes[1].axvline(200, color='red', linestyle='--', linewidth=2, label='200 iterations (overfitting)')
    axes[1].set_ylabel('STD (mm)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Highlight the U-shape
    min_std_idx = np.argmin(std)
    axes[1].plot(iterations[min_std_idx], std[min_std_idx], 'r*', markersize=15,
                 label=f'Optimal STD at iteration {min_std_idx+1}')
    axes[1].annotate('Optimal stopping point\n(before overfitting)',
                     xy=(min_std_idx, std[min_std_idx]),
                     xytext=(min_std_idx + 40, std[min_std_idx] + 0.5),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2),
                     fontsize=11, fontweight='bold')

    # Plot rotation change
    axes[2].semilogy(iterations, rotation_change, 'm-', linewidth=2, label='Rotation change')
    axes[2].axvline(100, color='orange', linestyle='--', linewidth=2)
    axes[2].axvline(200, color='red', linestyle='--', linewidth=2)
    axes[2].axhline(1e-6, color='k', linestyle=':', linewidth=2, label='Threshold (1e-6)')
    axes[2].set_xlabel('Iteration', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Rotation Change', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../output/figs/convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Convergence plot saved to: ../output/figs/convergence_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Run analysis
    config = analyze_convergence_tradeoff()

    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION:")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("""
For your observed behavior (100 vs 200 iterations):

✓ USE 100 ITERATIONS (or adaptive stopping around 100-150)

Reason:
- Lower STD indicates more robust, consistent alignment
- Slightly higher RMSE is acceptable trade-off
- Prevents overfitting to noise in individual scans
- Better generalization across subjects

Implementation:
- Update max_iterations to 150 (middle ground)
- Add early stopping with patience=10
- Monitor both RMSE and STD
- Stop when both have converged or STD starts increasing
    """)

    print("\nGenerating convergence plot...")
    try:
        create_convergence_plot_example()
    except Exception as e:
        print(f"Could not create plot: {e}")
        print("(Plot requires matplotlib and display)")

    print("\n" + "=" * 80)
