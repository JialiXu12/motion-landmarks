"""
Test script for plot_convergence_diagram function.
Run this directly to test if the convergence plot is working properly.
"""
import sys
import os

# Add scripts directory and external modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(project_root, 'external', 'breast_metadata_mdv'))
sys.path.insert(0, os.path.join(project_root, 'src', 'morphic'))  # Contains morphic/ package
sys.path.insert(0, os.path.join(project_root, 'src', 'mesh-tools'))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no GUI required

from pathlib import Path
from surface_to_point_alignment import plot_convergence_diagram


def create_mock_info():
    """Create mock alignment info for testing."""
    return {
        'method': 'plane_to_point_icp',
        'iterations': 10,
        'converged': True,
        'iteration_history': [
            {'iteration': 1, 'rmse': 5.0, 'rotation_change': 0.1,
             'angle_x_deg': 1.0, 'angle_y_deg': 2.0, 'angle_z_deg': 0.5, 'total_angle_deg': 2.5},
            {'iteration': 2, 'rmse': 4.5, 'rotation_change': 0.05,
             'angle_x_deg': 1.2, 'angle_y_deg': 2.1, 'angle_z_deg': 0.6, 'total_angle_deg': 2.6},
            {'iteration': 3, 'rmse': 4.2, 'rotation_change': 0.02,
             'angle_x_deg': 1.3, 'angle_y_deg': 2.2, 'angle_z_deg': 0.7, 'total_angle_deg': 2.7},
            {'iteration': 4, 'rmse': 4.0, 'rotation_change': 0.01,
             'angle_x_deg': 1.4, 'angle_y_deg': 2.3, 'angle_z_deg': 0.8, 'total_angle_deg': 2.8},
            {'iteration': 5, 'rmse': 3.9, 'rotation_change': 0.005,
             'angle_x_deg': 1.45, 'angle_y_deg': 2.35, 'angle_z_deg': 0.85, 'total_angle_deg': 2.85},
            {'iteration': 6, 'rmse': 3.85, 'rotation_change': 0.002,
             'angle_x_deg': 1.48, 'angle_y_deg': 2.38, 'angle_z_deg': 0.88, 'total_angle_deg': 2.88},
            {'iteration': 7, 'rmse': 3.82, 'rotation_change': 0.001,
             'angle_x_deg': 1.49, 'angle_y_deg': 2.39, 'angle_z_deg': 0.89, 'total_angle_deg': 2.89},
            {'iteration': 8, 'rmse': 3.80, 'rotation_change': 0.0005,
             'angle_x_deg': 1.495, 'angle_y_deg': 2.395, 'angle_z_deg': 0.895, 'total_angle_deg': 2.895},
            {'iteration': 9, 'rmse': 3.79, 'rotation_change': 0.0002,
             'angle_x_deg': 1.498, 'angle_y_deg': 2.398, 'angle_z_deg': 0.898, 'total_angle_deg': 2.898},
            {'iteration': 10, 'rmse': 3.78, 'rotation_change': 0.0001,
             'angle_x_deg': 1.5, 'angle_y_deg': 2.4, 'angle_z_deg': 0.9, 'total_angle_deg': 2.9},
        ]
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Testing plot_convergence_diagram function")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    save_path = output_dir / "test_convergence_diagram.png"

    print(f"\nBackend: {matplotlib.get_backend()}")
    print(f"Save path: {save_path}")

    # Create mock data
    info = create_mock_info()
    print(f"Created mock data with {len(info['iteration_history'])} iterations")

    # Call the function
    try:
        plot_convergence_diagram(info, save_path=str(save_path))
    except Exception as e:
        print(f"ERROR during plotting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify the file was created
    if save_path.exists():
        size = save_path.stat().st_size
        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Plot saved to {save_path}")
        print(f"File size: {size:,} bytes")
        print("=" * 60)
    else:
        print(f"\n{'=' * 60}")
        print(f"FAILED: Plot was not saved to {save_path}")
        print("=" * 60)
        sys.exit(1)


