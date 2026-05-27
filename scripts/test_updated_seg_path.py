"""
Test that the correct supine ribcage segmentation path is selected
for updated-seg subjects vs standard subjects.

Mirrors the constants in main_alignment.py — keep in sync if paths change.
"""

from pathlib import Path

# --- constants mirrored from main_alignment.py ---
SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
SUPINE_RIBCAGE_UPDATED_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage\updated")
UPDATED_SEG_IDS = {20, 22, 31, 44, 46, 54, 64, 68, 70, 72}

UPDATED_IDS = [20, 22, 31, 44, 46, 54, 64, 68, 70, 72]
STANDARD_IDS = [9, 11, 12, 14, 15, 17, 18, 19, 25, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 47, 48, 49, 50]


def get_seg_path(vl_id: int) -> Path:
    """Mirrors the logic in main_alignment.py loop."""
    vl_id_str = f"VL{vl_id:05d}"
    seg_root = SUPINE_RIBCAGE_UPDATED_ROOT if vl_id in UPDATED_SEG_IDS else SUPINE_RIBCAGE_ROOT
    return seg_root / f"rib_cage_{vl_id_str}.nii.gz"


def test_updated_ids_use_updated_root():
    for vl_id in UPDATED_IDS:
        path = get_seg_path(vl_id)
        assert SUPINE_RIBCAGE_UPDATED_ROOT in path.parents, (
            f"VL{vl_id:05d}: expected updated root, got {path}"
        )
    print(f"PASS  {len(UPDATED_IDS)} updated IDs -> updated root")


def test_standard_ids_use_standard_root():
    for vl_id in STANDARD_IDS:
        path = get_seg_path(vl_id)
        assert SUPINE_RIBCAGE_UPDATED_ROOT not in path.parents, (
            f"VL{vl_id:05d}: expected standard root, got {path}"
        )
        assert SUPINE_RIBCAGE_ROOT in path.parents, (
            f"VL{vl_id:05d}: expected standard root, got {path}"
        )
    print(f"PASS  {len(STANDARD_IDS)} standard IDs -> standard root")


def test_updated_ids_set_matches():
    assert UPDATED_SEG_IDS == set(UPDATED_IDS), (
        f"UPDATED_SEG_IDS in main_alignment.py does not match expected set.\n"
        f"  In code:    {sorted(UPDATED_SEG_IDS)}\n"
        f"  Expected:   {sorted(UPDATED_IDS)}"
    )
    print(f"PASS  UPDATED_SEG_IDS set matches expected {sorted(UPDATED_IDS)}")


def print_paths():
    """Print resolved paths for all updated IDs for visual inspection."""
    print("\nResolved segmentation paths:")
    print(f"  Standard root : {SUPINE_RIBCAGE_ROOT}")
    print(f"  Updated root  : {SUPINE_RIBCAGE_UPDATED_ROOT}")
    print()
    all_ids = sorted(set(UPDATED_IDS + STANDARD_IDS[:5]))
    for vl_id in all_ids:
        path = get_seg_path(vl_id)
        tag = "UPDATED" if vl_id in UPDATED_SEG_IDS else "standard"
        print(f"  VL{vl_id:05d} [{tag:7s}] -> {path}")


if __name__ == "__main__":
    test_updated_ids_set_matches()
    test_updated_ids_use_updated_root()
    test_standard_ids_use_standard_root()
    print_paths()
    print("\nAll tests passed.")
