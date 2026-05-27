"""
Fix swapped nipple labels in the original JSON files.

This script identifies subjects with swapped left/right nipple labels
and corrects them directly in the source JSON files.

The issue: In 49 subjects, the "nipple-l" and "nipple-r" labels are swapped.
- "nipple-l" has NEGATIVE X (which is actually the RIGHT side)
- "nipple-r" has POSITIVE X (which is actually the LEFT side)

This script will:
1. Read each JSON file
2. Check if nipple labels are swapped (based on X coordinate relative to sternum)
3. If swapped, swap the nipple-l and nipple-r data and save back to the file
4. Create a backup before modifying
"""
import json
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

# Path to anatomical landmark JSON files
ANATOMICAL_JSON_BASE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results")

# Create backup directory
BACKUP_DIR = Path(r"U:\sandbox\jxu759\volunteer_seg\results_backup_nipple_fix")


def check_and_fix_nipple_swap(json_file: Path, dry_run: bool = True) -> dict:
    """
    Check if nipple labels are swapped in a JSON file and optionally fix them.

    Args:
        json_file: Path to the JSON file
        dry_run: If True, only report what would be changed without modifying files

    Returns:
        dict with status information
    """
    result = {
        'file': str(json_file),
        'status': 'unknown',
        'is_swapped': False,
        'fixed': False,
        'error': None
    }

    if not json_file.exists():
        result['status'] = 'not_found'
        return result

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Find the landmarks
        bodies_dict = data.get('bodies', {})
        landmarks = None
        body_key = None

        if 'Jiali-test' in bodies_dict:
            landmarks = bodies_dict['Jiali-test'].get('landmarks')
            body_key = 'Jiali-test'
        elif 'Ray-Test' in bodies_dict:
            landmarks = bodies_dict['Ray-Test'].get('landmarks')
            body_key = 'Ray-Test'

        if not landmarks:
            result['status'] = 'no_landmarks'
            return result

        # Get nipple and sternum positions
        if 'nipple-l' not in landmarks or 'nipple-r' not in landmarks or 'sternal-superior' not in landmarks:
            result['status'] = 'missing_landmarks'
            return result

        nl = landmarks["nipple-l"]["3d_position"]
        nr = landmarks["nipple-r"]["3d_position"]
        ss = landmarks["sternal-superior"]["3d_position"]

        nipple_left_x = nl['x']
        nipple_right_x = nr['x']
        sternum_x = ss['x']

        left_x_rel = nipple_left_x - sternum_x
        right_x_rel = nipple_right_x - sternum_x

        # Check if swapped: left nipple should be MORE POSITIVE than right nipple
        is_swapped = left_x_rel < right_x_rel

        result['is_swapped'] = is_swapped
        result['left_x_rel'] = left_x_rel
        result['right_x_rel'] = right_x_rel

        if is_swapped:
            result['status'] = 'swapped'

            if not dry_run:
                # Create backup
                backup_path = BACKUP_DIR / json_file.relative_to(ANATOMICAL_JSON_BASE_ROOT)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_file, backup_path)

                # Swap the nipple data
                temp_nipple_l = landmarks["nipple-l"].copy()
                landmarks["nipple-l"] = landmarks["nipple-r"].copy()
                landmarks["nipple-r"] = temp_nipple_l

                # Save the modified data
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)

                result['fixed'] = True
                result['backup_path'] = str(backup_path)
        else:
            result['status'] = 'correct'

        return result

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return result


def main(dry_run: bool = True):
    """
    Main function to check and fix nipple labels in all JSON files.

    Args:
        dry_run: If True, only report what would be changed without modifying files
    """
    print("="*80)
    if dry_run:
        print("DRY RUN - No files will be modified")
    else:
        print("FIXING FILES - Files will be modified (backups will be created)")
    print("="*80)

    # Get list of VL IDs to check
    vl_ids = list(range(9, 100))  # Check VL9 to VL99

    swapped_subjects = []
    correct_subjects = []
    error_subjects = []

    for vl_id in vl_ids:
        vl_id_str = f"{vl_id:05d}"

        # Check both prone and supine
        for position in ['prone', 'supine']:
            anatomical_json_root = ANATOMICAL_JSON_BASE_ROOT / position / "landmarks"
            filename = f"VL{vl_id_str}_skeleton_data_{position}_t2.json"

            # Try primary path
            primary_path = anatomical_json_root / filename
            fallback_path = anatomical_json_root / "combined" / filename

            if primary_path.exists():
                json_path = primary_path
            elif fallback_path.exists():
                json_path = fallback_path
            else:
                continue

            result = check_and_fix_nipple_swap(json_path, dry_run=dry_run)

            if result['status'] == 'swapped':
                swapped_subjects.append({
                    'vl_id': vl_id,
                    'position': position,
                    'left_x_rel': result['left_x_rel'],
                    'right_x_rel': result['right_x_rel'],
                    'fixed': result['fixed']
                })
            elif result['status'] == 'correct':
                correct_subjects.append({'vl_id': vl_id, 'position': position})
            elif result['status'] == 'error':
                error_subjects.append({'vl_id': vl_id, 'position': position, 'error': result['error']})

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Files with SWAPPED labels: {len(swapped_subjects)}")
    print(f"Files with CORRECT labels: {len(correct_subjects)}")
    print(f"Files with ERRORS: {len(error_subjects)}")

    if swapped_subjects:
        print(f"\n{'='*80}")
        print("SWAPPED FILES:")
        print(f"{'='*80}")
        for subj in swapped_subjects:
            fixed_str = " [FIXED]" if subj['fixed'] else ""
            print(f"VL_{subj['vl_id']} ({subj['position']}): 'Left' X rel={subj['left_x_rel']:.1f}, 'Right' X rel={subj['right_x_rel']:.1f}{fixed_str}")

    if error_subjects:
        print(f"\n{'='*80}")
        print("ERRORS:")
        print(f"{'='*80}")
        for subj in error_subjects:
            print(f"VL_{subj['vl_id']} ({subj['position']}): {subj['error']}")

    # Get unique VL IDs that need fixing
    swapped_vl_ids = sorted(set(s['vl_id'] for s in swapped_subjects))
    print(f"\n{'='*80}")
    print(f"UNIQUE SUBJECTS WITH SWAPPED LABELS: {len(swapped_vl_ids)}")
    print(f"VL_IDs: {swapped_vl_ids}")
    print(f"{'='*80}")

    if dry_run and swapped_subjects:
        print("\n⚠️  This was a DRY RUN. To actually fix the files, run with dry_run=False")
        print("   Example: main(dry_run=False)")

    return swapped_subjects, correct_subjects, error_subjects


if __name__ == "__main__":
    # First do a dry run to see what would be changed
    print("\n" + "="*80)
    print("STEP 1: DRY RUN - Checking files...")
    print("="*80 + "\n")
    swapped, correct, errors = main(dry_run=True)

    if swapped:
        print("\n" + "="*80)
        response = input("Do you want to fix these files? (yes/no): ")
        if response.lower() == 'yes':
            print("\nSTEP 2: FIXING FILES...")
            print("="*80 + "\n")
            main(dry_run=False)
            print("\n✅ Files have been fixed! Backups saved to:", BACKUP_DIR)
        else:
            print("\n❌ No changes made.")
