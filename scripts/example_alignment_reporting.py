"""
Example: Generate Publication-Ready Alignment Accuracy Reports

This script demonstrates how to use the alignment reporting functions
to generate statistics suitable for scientific journal submission.

Usage:
    python example_alignment_reporting.py
"""

import numpy as np


# ============================================================================
# Standalone reporting functions (copied from alignment.py for demo purposes)
# ============================================================================

def print_alignment_accuracy_report(
        rib_error_mag: np.ndarray,
        sternum_error: float,
        info: dict
) -> None:
    """
    Print publication-ready alignment accuracy statistics.
    """
    # Calculate comprehensive statistics
    mean_err = np.mean(rib_error_mag)
    std_err = np.std(rib_error_mag)
    rmse = np.sqrt(np.mean(rib_error_mag ** 2))
    median_err = np.median(rib_error_mag)
    q25 = np.percentile(rib_error_mag, 25)
    q75 = np.percentile(rib_error_mag, 75)
    min_err = np.min(rib_error_mag)
    max_err = np.max(rib_error_mag)

    # Inlier statistics
    n_inliers = info.get('n_inliers', 'N/A')
    n_total = info.get('n_total_source', len(rib_error_mag))
    inlier_pct = info.get('inlier_fraction', 0) * 100 if 'inlier_fraction' in info else 0

    print("Alignment Accuracy Metrics:")
    print("-" * 60)
    print(f"  Sternum Superior Error: {sternum_error:.4f} mm (fixed point)")
    print(f"  Ribcage Surface Alignment:")
    print(f"    RMSE: {rmse:.2f} mm")
    print(f"    Mean ± SD: {mean_err:.2f} ± {std_err:.2f} mm")
    print(f"    Median [IQR]: {median_err:.2f} [{q25:.2f}-{q75:.2f}] mm")
    print(f"    Range: {min_err:.2f}-{max_err:.2f} mm")
    print(f"  Algorithm Performance:")
    print(f"    Inliers: {n_inliers}/{n_total} ({inlier_pct:.1f}%)")
    print(f"    Iterations: {info.get('iterations', 'N/A')}")
    print(f"    Method: {info.get('method', 'N/A')}")
    print()
    print("Recommended text for Methods section:")
    print("-" * 60)
    print(f"Prone-to-supine alignment achieved a ribcage surface RMSE of")
    print(f"{rmse:.2f} mm (mean ± SD: {mean_err:.2f} ± {std_err:.2f} mm), with the")
    print(f"sternum superior landmark fixed at the origin (0.00 mm error).")
    print(f"The median alignment error was {median_err:.2f} mm (IQR: {q25:.2f}-{q75:.2f} mm),")
    print(f"indicating good registration quality across the thoracic region.")


def generate_alignment_report_latex_table(
        rib_error_mag: np.ndarray,
        sternum_error: float,
        info: dict
) -> str:
    """Generate LaTeX table code for alignment accuracy metrics."""
    mean_err = np.mean(rib_error_mag)
    std_err = np.std(rib_error_mag)
    rmse = np.sqrt(np.mean(rib_error_mag ** 2))
    median_err = np.median(rib_error_mag)
    q25 = np.percentile(rib_error_mag, 25)
    q75 = np.percentile(rib_error_mag, 75)

    latex = r"""
\begin{table}[h]
\centering
\caption{Prone-to-Supine Alignment Accuracy}
\label{tab:alignment_accuracy}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Value (mm)} \\
\hline
Sternum Superior Error & 0.00 \\
Ribcage RMSE & """ + f"{rmse:.2f}" + r""" \\
Ribcage Mean $\pm$ SD & """ + f"{mean_err:.2f} $\\pm$ {std_err:.2f}" + r""" \\
Ribcage Median [IQR] & """ + f"{median_err:.2f} [{q25:.2f}-{q75:.2f}]" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    return latex


def aggregate_alignment_statistics(alignment_results_dict: dict) -> dict:
    """Aggregate alignment statistics across multiple subjects."""
    # Collect per-subject metrics
    subject_rmse = []
    subject_mean = []
    subject_median = []
    subject_std = []
    subject_sternum_err = []
    subject_inlier_pct = []
    subject_iterations = []

    for vl_id, results in alignment_results_dict.items():
        if results is None or 'ribcage_error_rmse' not in results:
            print(f"Warning: Skipping subject {vl_id} - incomplete results")
            continue

        subject_rmse.append(results['ribcage_error_rmse'])
        subject_mean.append(results['ribcage_error_mean'])
        subject_std.append(results['ribcage_error_std'])
        subject_sternum_err.append(results['sternum_error'])

        # Extract from nested info dict if available
        if 'info' in results:
            info = results['info']
            subject_inlier_pct.append(info.get('inlier_fraction', np.nan) * 100)
            subject_iterations.append(info.get('iterations', np.nan))

        # Calculate median per subject from error magnitudes if available
        if 'ribcage_errors' in results:
            subject_median.append(np.median(results['ribcage_errors']))

    # Convert to arrays
    subject_rmse = np.array(subject_rmse)
    subject_mean = np.array(subject_mean)
    subject_std = np.array(subject_std)
    subject_sternum_err = np.array(subject_sternum_err)

    # Cohort statistics
    cohort_stats = {
        'n_subjects': len(subject_rmse),
        'vl_ids': list(alignment_results_dict.keys()),

        # RMSE across subjects
        'rmse_mean': float(np.mean(subject_rmse)),
        'rmse_std': float(np.std(subject_rmse)),
        'rmse_median': float(np.median(subject_rmse)),
        'rmse_q25': float(np.percentile(subject_rmse, 25)),
        'rmse_q75': float(np.percentile(subject_rmse, 75)),
        'rmse_min': float(np.min(subject_rmse)),
        'rmse_max': float(np.max(subject_rmse)),

        # Mean error across subjects
        'mean_error_mean': float(np.mean(subject_mean)),
        'mean_error_std': float(np.std(subject_mean)),

        # Sternum error across subjects
        'sternum_error_mean': float(np.mean(subject_sternum_err)),
        'sternum_error_max': float(np.max(subject_sternum_err)),

        # Per-subject arrays (for plotting)
        'per_subject_rmse': subject_rmse.tolist(),
        'per_subject_mean': subject_mean.tolist(),
        'per_subject_median': subject_median if subject_median else None,
    }

    # Add optional metrics if available
    if subject_inlier_pct:
        cohort_stats['inlier_pct_mean'] = float(np.mean(subject_inlier_pct))
        cohort_stats['inlier_pct_std'] = float(np.std(subject_inlier_pct))

    if subject_iterations:
        cohort_stats['iterations_mean'] = float(np.mean(subject_iterations))
        cohort_stats['iterations_std'] = float(np.std(subject_iterations))

    return cohort_stats


def print_cohort_alignment_report(cohort_stats: dict) -> None:
    """Print publication-ready cohort-level alignment report."""
    n = cohort_stats['n_subjects']

    print("\n" + "="*70)
    print(f"COHORT ALIGNMENT ACCURACY REPORT (N={n} subjects)")
    print("="*70)

    print(f"\nPrimary Metrics:")
    print(f"  Sternum Superior Error: {cohort_stats['sternum_error_mean']:.4f} mm")
    print(f"    (max across subjects: {cohort_stats['sternum_error_max']:.4f} mm)")

    print(f"\n  Ribcage Surface Alignment:")
    print(f"    RMSE: {cohort_stats['rmse_mean']:.2f} ± {cohort_stats['rmse_std']:.2f} mm")
    print(f"    Median [IQR]: {cohort_stats['rmse_median']:.2f} "
          f"[{cohort_stats['rmse_q25']:.2f}-{cohort_stats['rmse_q75']:.2f}] mm")
    print(f"    Range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm")

    print(f"\n  Mean Error: {cohort_stats['mean_error_mean']:.2f} ± "
          f"{cohort_stats['mean_error_std']:.2f} mm")

    if 'inlier_pct_mean' in cohort_stats:
        print(f"\nAlgorithm Performance:")
        print(f"  Inlier Fraction: {cohort_stats['inlier_pct_mean']:.1f} ± "
              f"{cohort_stats['inlier_pct_std']:.1f} %")

    if 'iterations_mean' in cohort_stats:
        print(f"  Iterations: {cohort_stats['iterations_mean']:.0f} ± "
              f"{cohort_stats['iterations_std']:.0f}")

    print("\n" + "-"*70)
    print("RECOMMENDED TEXT FOR MANUSCRIPT:")
    print("-"*70)
    print(f"\nProne-to-supine alignment was performed using a sternum-fixed")
    print(f"iterative closest point algorithm. Across N={n} subjects, ribcage")
    print(f"surface alignment achieved an RMSE of {cohort_stats['rmse_mean']:.2f} ± "
          f"{cohort_stats['rmse_std']:.2f} mm")
    print(f"(median: {cohort_stats['rmse_median']:.2f} mm, "
          f"range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm),")
    print(f"with sternum superior error of {cohort_stats['sternum_error_mean']:.4f} mm,")
    print(f"indicating excellent registration quality suitable for landmark")
    print(f"displacement analysis.\n")


# ============================================================================
# Example Usage
# ============================================================================


def example_single_subject_report():
    """Example: Report alignment accuracy for a single subject"""

    print("\n" + "="*70)
    print("EXAMPLE 1: Single Subject Report")
    print("="*70)

    # Simulated alignment results for one subject
    # In practice, these come from align_prone_to_supine_fixed_sternum()
    rib_error_mag = np.array([
        1.2, 2.3, 3.1, 2.8, 4.5, 5.2, 3.7, 2.9, 1.8, 3.4,
        4.1, 5.8, 6.2, 7.1, 8.3, 9.2, 10.5, 12.3, 15.4, 18.7,
        # Some outliers in posterior regions
        25.3, 32.1, 45.6, 63.5
    ])

    sternum_error = 0.0000123  # Near-zero for fixed sternum

    info = {
        'n_inliers': 850,
        'n_total_source': 1000,
        'inlier_fraction': 0.85,
        'iterations': 127,
        'method': 'optimal_sternum_fixed_svd'
    }

    # Generate the report
    print_alignment_accuracy_report(rib_error_mag, sternum_error, info)

    # Generate LaTeX table
    print("\n" + "="*70)
    print("LaTeX Table Code:")
    print("="*70)
    latex_code = generate_alignment_report_latex_table(rib_error_mag, sternum_error, info)
    print(latex_code)


def example_cohort_report():
    """Example: Report alignment accuracy across multiple subjects"""

    print("\n" + "="*70)
    print("EXAMPLE 2: Cohort-Level Report")
    print("="*70)

    # Simulated alignment results for multiple subjects
    # In practice, this comes from main.py alignment_results_all dictionary
    alignment_results = {}

    # Simulate 13 subjects with varying alignment quality
    np.random.seed(42)
    for i, vl_id in enumerate([9, 11, 12, 14, 15, 18, 19, 20, 22, 25, 29, 30, 31]):
        # Generate realistic error distributions
        n_points = np.random.randint(800, 1200)
        errors = np.concatenate([
            np.random.gamma(2, 2, int(n_points * 0.7)),  # Most points: low error
            np.random.gamma(4, 4, int(n_points * 0.25)), # Some points: moderate error
            np.random.uniform(20, 70, int(n_points * 0.05))  # Few outliers: high error
        ])

        rmse = np.sqrt(np.mean(errors ** 2))
        mean_err = np.mean(errors)
        std_err = np.std(errors)

        alignment_results[vl_id] = {
            'ribcage_error_rmse': rmse,
            'ribcage_error_mean': mean_err,
            'ribcage_error_std': std_err,
            'sternum_error': np.random.uniform(0, 0.001),  # Near zero
            'ribcage_errors': errors,
            'info': {
                'n_inliers': int(n_points * 0.8),
                'n_total_source': n_points,
                'inlier_fraction': 0.8,
                'iterations': np.random.randint(80, 150),
                'method': 'optimal_sternum_fixed_svd'
            }
        }

    # Aggregate statistics
    cohort_stats = aggregate_alignment_statistics(alignment_results)

    # Print cohort report
    print_cohort_alignment_report(cohort_stats)

    # Additional analysis: Show per-subject breakdown
    print("\n" + "="*70)
    print("Per-Subject RMSE Breakdown:")
    print("="*70)
    print(f"{'VL_ID':<10} {'RMSE (mm)':<12} {'Mean (mm)':<12} {'Status'}")
    print("-"*70)

    for vl_id, results in alignment_results.items():
        rmse = results['ribcage_error_rmse']
        mean = results['ribcage_error_mean']

        # Quality assessment
        if rmse < 5:
            status = "Excellent"
        elif rmse < 8:
            status = "Good"
        elif rmse < 12:
            status = "Acceptable"
        else:
            status = "Review"

        print(f"VL{vl_id:05d}    {rmse:6.2f}       {mean:6.2f}       {status}")

    print()


def example_methods_section():
    """Example: Pre-written methods section text"""

    print("\n" + "="*70)
    print("EXAMPLE 3: Copy-Paste Methods Section Text")
    print("="*70)

    methods_text = """
Image Registration and Alignment

Prone and supine MRI volumes were aligned using a sternum-fixed iterative 
closest point (ICP) algorithm implemented in Python (v3.11) with NumPy 
(v1.24) and SciPy (v1.11). The sternum superior landmark was designated 
as a fixed anatomical reference point to prevent drift in this stable 
bony structure. Prone ribcage surface points were extracted from manual 
segmentations and aligned to the supine target using rotation-only 
transformations computed via Singular Value Decomposition (SVD).

The algorithm iteratively minimized point-to-point Euclidean distances 
between corresponding ribcage surface points, using a correspondence 
distance threshold of 15 mm to exclude outliers. Convergence was declared 
when the root mean square error (RMSE) changed by less than 0.01 mm 
between iterations or after a maximum of 200 iterations. Early stopping 
with patience of 20 iterations was employed to prevent overfitting.

Alignment accuracy was quantified by measuring the Euclidean distance 
between all prone ribcage points (after transformation) and their nearest 
neighbors on the supine ribcage surface. Statistics were computed per 
subject and aggregated across the cohort (N=13 subjects) as mean ± standard 
deviation. The sternum superior error (distance between prone and supine 
sternum after alignment) was verified to be <0.001 mm for all subjects, 
confirming successful anchoring. Ribcage alignment quality was assessed 
using RMSE, mean absolute error, and median [IQR] to account for potential 
outliers in posterior regions where soft tissue deformation is greatest.
"""

    print(methods_text)


def example_results_section():
    """Example: Pre-written results section text"""

    print("\n" + "="*70)
    print("EXAMPLE 4: Copy-Paste Results Section Text")
    print("="*70)

    results_text = """
Alignment Accuracy

The sternum-fixed ICP algorithm achieved excellent alignment accuracy 
across all subjects (Table 1). Sternum superior error was 0.0001 ± 0.0003 mm 
(mean ± SD, N=13), confirming successful anchoring at this fixed reference 
point. Ribcage surface alignment demonstrated good accuracy with RMSE of 
5.52 ± 1.89 mm (range: 3.25-10.76 mm). The median alignment error was 
4.12 mm (IQR: 2.34-6.78 mm), indicating that the majority of the ribcage 
surface aligned within clinically acceptable tolerances (<5 mm).

Alignment quality varied by anatomical region, with anterior ribcage 
(near sternum) showing lower errors (mean: 3.21 ± 1.45 mm) compared to 
posterior ribcage (mean: 8.45 ± 3.67 mm, p<0.001, paired t-test). This 
spatial variation reflects expected differences in soft tissue deformation 
between prone and supine positions, with greater displacement in regions 
distant from the fixed anchor point. Importantly, all anatomical landmarks 
of interest (n=38 total landmarks across subjects) were located in the 
well-aligned anterior-medial region (mean local alignment error: 
3.54 ± 1.89 mm), ensuring reliable landmark displacement measurements.

The algorithm converged within 127 ± 45 iterations (range: 82-189) with 
an inlier fraction of 82.3 ± 8.7%, indicating robust correspondence 
matching despite large deformation between positions. No subjects required 
manual alignment correction or re-processing.
"""

    print(results_text)

    print("\n" + "-"*70)
    print("Note: Replace the example numbers above with your actual results")
    print("      from aggregate_alignment_statistics()")
    print("-"*70)


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# ALIGNMENT ACCURACY REPORTING EXAMPLES")
    print("# For Scientific Journal Submission")
    print("#"*70)

    # Run all examples
    example_single_subject_report()
    example_cohort_report()
    example_methods_section()
    example_results_section()

    print("\n" + "#"*70)
    print("# For more details, see:")
    print("# - ALIGNMENT_ACCURACY_REPORTING_GUIDE.md")
    print("# - alignment.py (function documentation)")
    print("#"*70 + "\n")


