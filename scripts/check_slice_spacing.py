"""
Quick utility to report slice spacing (mm) for all supine DICOM scans.

Reads scan.spacing = [pixel_x, pixel_y, slice_thickness] via breast_metadata.Scan.
Run from the project root:
    python scripts/check_slice_spacing.py

Output is printed to stdout and saved to output/slice_spacing.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import external.breast_metadata_mdv.breast_metadata as breast_metadata

# Expected ~0.9 mm based on observed Z-difference quantisation in landmark data.
ROOT_PATH_MRI = Path(r'U:\projects\volunteer_camri\old_data\mri_t2')
POSITION = "supine"
OUTPUT_CSV = Path(__file__).parent.parent / "output" / "slice_spacing.csv"

VL_IDS = [
    9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25, 29, 30, 31,
    32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 89,
]

rows = []
print(f"{'Subject':<12} {'Slice spacing (mm)':>20} {'Pixel X (mm)':>14} {'Pixel Y (mm)':>14}")
print("-" * 65)

for vl_id in VL_IDS:
    vl_id_str = f"VL{vl_id:05d}"
    dicom_dir = ROOT_PATH_MRI / vl_id_str / POSITION

    if not dicom_dir.is_dir():
        print(f"{vl_id_str:<12} {'DIR NOT FOUND':>20}")
        rows.append({"Subject": vl_id_str, "slice_spacing_mm": None,
                     "pixel_x_mm": None, "pixel_y_mm": None, "note": "dir not found"})
        continue

    try:
        scan = breast_metadata.Scan(str(dicom_dir))
        sx, sy, sz = scan.spacing
        print(f"{vl_id_str:<12} {sz:>20.4f} {sx:>14.4f} {sy:>14.4f}")
        rows.append({"Subject": vl_id_str, "slice_spacing_mm": sz,
                     "pixel_x_mm": sx, "pixel_y_mm": sy, "note": ""})
    except Exception as e:
        print(f"{vl_id_str:<12} {'ERROR':>20}  {e}")
        rows.append({"Subject": vl_id_str, "slice_spacing_mm": None,
                     "pixel_x_mm": None, "pixel_y_mm": None, "note": str(e)})

# Summary
valid = [r["slice_spacing_mm"] for r in rows if r["slice_spacing_mm"] is not None]
if valid:
    import statistics
    unique = sorted(set(round(v, 3) for v in valid))
    print()
    print(f"Unique slice spacings found: {unique}")
    print(f"Mean: {statistics.mean(valid):.4f} mm  "
          f"Min: {min(valid):.4f} mm  Max: {max(valid):.4f} mm")

# Save CSV
import csv
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Subject", "slice_spacing_mm",
                                           "pixel_x_mm", "pixel_y_mm", "note"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {OUTPUT_CSV}")
