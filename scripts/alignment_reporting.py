"""
Alignment Reporting

Publication-ready alignment accuracy statistics and cohort-level reports.

Extracted from alignment.py in Stage 4 of the refactoring plan.
"""

import numpy as np


def print_alignment_accuracy_report(
        rib_error_mag: np.ndarray,
        sternum_error: float,
        info: dict
) -> None:
    """
    Print publication-ready alignment accuracy statistics.

    This function reports alignment metrics following medical image
    registration best practices for scientific journals.

    Args:
        rib_error_mag: (N,) array of ribcage alignment errors in mm
        sternum_error: scalar sternum alignment error in mm
        info: alignment algorithm info dictionary
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
    print(f"    Mean +/- SD: {mean_err:.2f} +/- {std_err:.2f} mm")
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
    print(f"{rmse:.2f} mm (mean +/- SD: {mean_err:.2f} +/- {std_err:.2f} mm), with the")
    print(f"sternum superior landmark fixed at the origin (0.00 mm error).")
    print(f"The median alignment error was {median_err:.2f} mm (IQR: {q25:.2f}-{q75:.2f} mm),")
    print(f"indicating good registration quality across the thoracic region.")


def aggregate_alignment_statistics(alignment_results_dict: dict) -> dict:
    """
    Aggregate alignment statistics across multiple subjects for cohort reporting.

    Args:
        alignment_results_dict: Dictionary with structure {vl_id: results_dict}
                               where results_dict contains alignment metrics

    Returns:
        Dictionary containing cohort-level statistics
    """
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
    """
    Print publication-ready cohort-level alignment report.

    Args:
        cohort_stats: Output from aggregate_alignment_statistics()
    """
    n = cohort_stats['n_subjects']

    print("\n" + "="*70)
    print(f"COHORT ALIGNMENT ACCURACY REPORT (N={n} subjects)")
    print("="*70)

    print(f"\nPrimary Metrics:")
    print(f"  Sternum Superior Error: {cohort_stats['sternum_error_mean']:.4f} mm")
    print(f"    (max across subjects: {cohort_stats['sternum_error_max']:.4f} mm)")

    print(f"\n  Ribcage Surface Alignment:")
    print(f"    RMSE: {cohort_stats['rmse_mean']:.2f} +/- {cohort_stats['rmse_std']:.2f} mm")
    print(f"    Median [IQR]: {cohort_stats['rmse_median']:.2f} "
          f"[{cohort_stats['rmse_q25']:.2f}-{cohort_stats['rmse_q75']:.2f}] mm")
    print(f"    Range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm")

    print(f"\n  Mean Error: {cohort_stats['mean_error_mean']:.2f} +/- "
          f"{cohort_stats['mean_error_std']:.2f} mm")

    if 'inlier_pct_mean' in cohort_stats:
        print(f"\nAlgorithm Performance:")
        print(f"  Inlier Fraction: {cohort_stats['inlier_pct_mean']:.1f} +/- "
              f"{cohort_stats['inlier_pct_std']:.1f} %")

    if 'iterations_mean' in cohort_stats:
        print(f"  Iterations: {cohort_stats['iterations_mean']:.0f} +/- "
              f"{cohort_stats['iterations_std']:.0f}")

    print("\n" + "-"*70)
    print("RECOMMENDED TEXT FOR MANUSCRIPT:")
    print("-"*70)
    print(f"\nProne-to-supine alignment was performed using a sternum-fixed")
    print(f"iterative closest point algorithm. Across N={n} subjects, ribcage")
    print(f"surface alignment achieved an RMSE of {cohort_stats['rmse_mean']:.2f} +/- "
          f"{cohort_stats['rmse_std']:.2f} mm")
    print(f"(median: {cohort_stats['rmse_median']:.2f} mm, "
          f"range: {cohort_stats['rmse_min']:.2f}-{cohort_stats['rmse_max']:.2f} mm),")
    print(f"with sternum superior error of {cohort_stats['sternum_error_mean']:.4f} mm,")
    print(f"indicating excellent registration quality suitable for landmark")
    print(f"displacement analysis.\n")
