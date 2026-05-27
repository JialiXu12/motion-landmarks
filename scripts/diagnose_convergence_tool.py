"""
Extract and analyze actual iteration history from alignment results.

This script will help you see the actual RMSE progression to iteration 50.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_iteration_history_from_results():
    """
    If you have alignment results saved, this will plot the actual convergence.
    """

    print("=" * 80)
    print("TO SEE ACTUAL ITERATION HISTORY:")
    print("=" * 80)
    print("""
Add this code to your analysis/main script AFTER alignment completes:

```python
# After alignment
alignment_results = align_prone_to_supine_optimal(...)

# Extract iteration history
history = alignment_results['alignment_info']['iteration_history']

# Convert to arrays for plotting
iterations = [h['iteration'] for h in history]
rmse_vals = [h['rmse'] for h in history]
n_inliers = [h['n_inliers'] for h in history]
rot_change = [h['rotation_change'] for h in history]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# RMSE
axes[0].plot(iterations, rmse_vals, 'b-o', linewidth=2, markersize=4)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('RMSE (mm)')
axes[0].set_title('RMSE Convergence')
axes[0].grid(True, alpha=0.3)

# Inliers
axes[1].plot(iterations, n_inliers, 'g-o', linewidth=2, markersize=4)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Number of Inliers')
axes[1].set_title('Inlier Count')
axes[1].grid(True, alpha=0.3)

# Rotation change
axes[2].semilogy(iterations, rot_change, 'm-o', linewidth=2, markersize=4)
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Rotation Change')
axes[2].set_title('Rotation Matrix Change per Iteration')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figs/actual_convergence.png', dpi=300)
plt.show()

# Print summary
print(f"Total iterations: {len(iterations)}")
print(f"Initial RMSE: {rmse_vals[0]:.4f} mm")
print(f"Final RMSE: {rmse_vals[-1]:.4f} mm")
print(f"RMSE improvement: {rmse_vals[0] - rmse_vals[-1]:.4f} mm")
print(f"Final inliers: {n_inliers[-1]}")

# Check why it stopped
if len(iterations) < 100:
    final_change = abs(rmse_vals[-1] - rmse_vals[-2])
    print(f"\\nStopped at iteration {len(iterations)} due to convergence")
    print(f"Final RMSE change: {final_change:.8f} mm (threshold: 1e-6)")
else:
    print(f"\\nReached max iterations (100)")
```
    """)

    print("\n" + "=" * 80)
    print("QUICK DIAGNOSIS CHECKLIST:")
    print("=" * 80)
    print("""
To determine if iteration 50 was good or bad, check:

1. Final RMSE value:
   ✓ < 3 mm: Excellent alignment
   ✓ 3-5 mm: Good alignment  
   ✓ 5-10 mm: Acceptable alignment
   ✗ > 10 mm: Poor alignment (check data/parameters)

2. Sternum error:
   ✓ < 0.001 mm: Perfect (as expected with our method)
   ✗ > 0.001 mm: Bug in code (should never happen)

3. Inlier fraction:
   ✓ > 0.8: Excellent coverage
   ✓ 0.7-0.8: Good coverage
   ✓ 0.5-0.7: Acceptable coverage
   ✗ < 0.5: Poor overlap or parameter issue

4. Why stopped at iteration 50:
   ✓ RMSE change < 1e-6: Natural convergence (GOOD)
   ✗ Not enough correspondences: Data issue
   ✗ Max iterations: May need more iterations

5. RMSE trend:
   ✓ Smooth exponential decay: Healthy convergence
   ✓ Plateaus early (iter 20-50): Fast convergence
   ✗ Oscillating: Unstable, check parameters
   ✗ Still decreasing rapidly at iter 50: Premature stop
    """)

    print("\n" + "=" * 80)
    print("COMMON SCENARIOS:")
    print("=" * 80)
    print("""
Scenario 1: Ideal Convergence at Iteration 50
---------------------------------------------
Iterations: 50
Initial RMSE: 12.5 mm
Final RMSE: 3.8 mm
RMSE change at iter 50: 0.0000005 mm
Sternum error: 0.0 mm
Inliers: 85%

→ EXCELLENT! Algorithm found optimal solution efficiently.
→ No need for more iterations.


Scenario 2: Premature Convergence at Iteration 50
-------------------------------------------------
Iterations: 50
Initial RMSE: 18.2 mm
Final RMSE: 9.5 mm (still high!)
RMSE change at iter 50: 0.0000008 mm
Sternum error: 0.0 mm
Inliers: 55%

→ Converged but to suboptimal solution
→ Possible fixes:
   - Relax convergence_threshold to 1e-5
   - Increase max_correspondence_distance
   - Check data quality


Scenario 3: Fast Convergence (< 50 iterations)
----------------------------------------------
Iterations: 35
Initial RMSE: 10.1 mm
Final RMSE: 3.2 mm
RMSE change at iter 35: 0.0000007 mm
Sternum error: 0.0 mm
Inliers: 88%

→ EXCELLENT! Algorithm is very efficient
→ No issues
    """)


def create_convergence_diagnostic_function():
    """
    Generate a reusable function to add to your codebase.
    """

    diagnostic_code = '''
def diagnose_convergence(alignment_results):
    """
    Diagnose why alignment stopped and evaluate quality.
    
    Args:
        alignment_results: Dictionary returned by align_prone_to_supine_optimal
    """
    info = alignment_results['alignment_info']
    history = info['iteration_history']
    
    # Extract metrics
    iterations = [h['iteration'] for h in history]
    rmse_vals = [h['rmse'] for h in history]
    n_inliers = [h['n_inliers'] for h in history]
    
    # Summary
    print("\\n" + "="*60)
    print("CONVERGENCE DIAGNOSTIC")
    print("="*60)
    print(f"Iterations completed: {len(iterations)}")
    print(f"Initial RMSE: {rmse_vals[0]:.4f} mm")
    print(f"Final RMSE: {rmse_vals[-1]:.4f} mm")
    print(f"RMSE reduction: {rmse_vals[0] - rmse_vals[-1]:.4f} mm")
    print(f"Final inliers: {n_inliers[-1]} / {info['n_total_source']}")
    print(f"Inlier fraction: {info['inlier_fraction']:.2%}")
    print(f"Sternum error: {info['sternum_error_mm']:.8f} mm")
    
    # Stopping reason
    if len(iterations) < 100:
        final_change = abs(rmse_vals[-1] - rmse_vals[-2]) if len(rmse_vals) > 1 else 0
        print(f"\\nStopped: Converged at iteration {len(iterations)}")
        print(f"Final RMSE change: {final_change:.8f} mm (threshold: 1e-6)")
    else:
        print(f"\\nStopped: Reached max iterations")
    
    # Quality assessment
    print("\\nQuality Assessment:")
    if rmse_vals[-1] < 3:
        print("  ✓ Excellent alignment (RMSE < 3 mm)")
    elif rmse_vals[-1] < 5:
        print("  ✓ Good alignment (RMSE < 5 mm)")
    elif rmse_vals[-1] < 10:
        print("  ✓ Acceptable alignment (RMSE < 10 mm)")
    else:
        print("  ✗ Poor alignment (RMSE > 10 mm) - check data/parameters")
    
    if info['sternum_error_mm'] < 0.001:
        print("  ✓ Sternum perfectly fixed (error < 0.001 mm)")
    else:
        print(f"  ✗ Sternum drift detected ({info['sternum_error_mm']:.6f} mm)")
    
    if info['inlier_fraction'] > 0.8:
        print("  ✓ Excellent point coverage (>80% inliers)")
    elif info['inlier_fraction'] > 0.7:
        print("  ✓ Good point coverage (>70% inliers)")
    elif info['inlier_fraction'] > 0.5:
        print("  ✓ Acceptable point coverage (>50% inliers)")
    else:
        print("  ✗ Poor point coverage (<50% inliers)")
    
    print("="*60)
    
    return {
        'iterations': len(iterations),
        'initial_rmse': rmse_vals[0],
        'final_rmse': rmse_vals[-1],
        'quality': 'excellent' if rmse_vals[-1] < 3 else 'good' if rmse_vals[-1] < 5 else 'acceptable' if rmse_vals[-1] < 10 else 'poor'
    }
'''

    print("\n" + "=" * 80)
    print("REUSABLE DIAGNOSTIC FUNCTION:")
    print("=" * 80)
    print("\nAdd this to your utils.py or analysis.py:\n")
    print(diagnostic_code)
    print("\nUsage:")
    print("```python")
    print("results = align_prone_to_supine_optimal(...)")
    print("diagnose_convergence(results)")
    print("```")


if __name__ == "__main__":
    plot_iteration_history_from_results()
    create_convergence_diagnostic_function()
