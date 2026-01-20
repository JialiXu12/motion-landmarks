"""
Command-line interface for plot_sagittal_dual_axes function

Usage:
    python run_sagittal_dual.py --color breast
    python run_sagittal_dual.py --color subject
    python run_sagittal_dual.py --color dts
    python run_sagittal_dual.py --color dts --subject 9
"""

import argparse
from pathlib import Path
from analysis import read_data, plot_sagittal_dual_axes

# Configuration
OUTPUT_DIR = Path("../output")
EXCEL_FILE_PATH = OUTPUT_DIR / "landmark_results_v4_2026_01_12.xlsx"


def main():
    parser = argparse.ArgumentParser(
        description='Generate sagittal dual-axis plots with different coloring schemes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default breast coloring (blue/green)
  python run_sagittal_dual.py
  
  # Color by subject
  python run_sagittal_dual.py --color subject
  
  # Color by DTS (distance to skin)
  python run_sagittal_dual.py --color dts
  
  # Filter for specific subject with DTS coloring
  python run_sagittal_dual.py --color dts --subject 9
  
  # All three color schemes
  python run_sagittal_dual.py --all
        '''
    )

    parser.add_argument(
        '--color',
        choices=['breast', 'subject', 'dts'],
        default='breast',
        help='Coloring scheme: breast (default), subject, or dts'
    )

    parser.add_argument(
        '--subject',
        type=int,
        default=None,
        help='Filter for specific subject ID (VL_ID)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate plots with all three coloring schemes'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=str(EXCEL_FILE_PATH),
        help=f'Path to Excel data file (default: {EXCEL_FILE_PATH})'
    )

    args = parser.parse_args()

    # Load data
    print("=" * 80)
    print("SAGITTAL DUAL AXES PLOTTER")
    print("=" * 80)
    print(f"\nLoading data from: {args.data}")

    try:
        df_raw, df_ave, df_demo = read_data(Path(args.data))
        print(f"✓ Data loaded successfully")
        print(f"  - Total landmarks: {len(df_ave)}")
        print(f"  - Unique subjects: {df_ave['VL_ID'].nunique()}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return 1

    # Generate plots
    if args.all:
        print("\n" + "=" * 80)
        print("Generating plots with ALL coloring schemes")
        print("=" * 80)

        for color_scheme in ['breast', 'subject', 'dts']:
            print(f"\n>>> Generating plot with '{color_scheme}' coloring...")
            try:
                plot_sagittal_dual_axes(df_ave, color_by=color_scheme, vl_id=args.subject)
                print(f"✓ '{color_scheme}' plot completed")
            except Exception as e:
                print(f"✗ Error with '{color_scheme}' coloring: {e}")
                import traceback
                traceback.print_exc()
                return 1
    else:
        print(f"\n>>> Generating plot with '{args.color}' coloring...")
        if args.subject:
            print(f"    Filtering for subject VL_{args.subject}")

        try:
            plot_sagittal_dual_axes(df_ave, color_by=args.color, vl_id=args.subject)
            print(f"✓ Plot completed successfully")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY ✓")
    print("=" * 80)
    output_dir = Path("..") / "output" / "figs" / "landmark vectors"
    print(f"Output saved to: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
