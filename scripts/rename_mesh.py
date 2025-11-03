import os
import shutil
from pathlib import Path

# --- 1. CONFIGURATION ---

# The top-level folder to start searching in (e.g., "folder")
SOURCE_ROOT = Path(r"U:\sandbox\fpan017\meshes\new_workflow\ribcage\new_cases")

# The single folder where all files will be copied
DEST_DIR = Path(r"U:\sandbox\jxu759\volunteer_prone_mesh")

# The exact name of the file you want to find
FILE_TO_FIND = "ribcage.mesh"

# The new suffix to append to the subject ID (e.g., "VL00009" + this suffix)
NEW_SUFFIX = "_ribcage_prone.mesh"


# --- End Configuration ---


def main():
    """
    Finds, copies, and renames ribcage meshes based on their subject folder.
    """

    # 1. Check source and create destination
    if not SOURCE_ROOT.is_dir():
        print(f"Error: Source directory not found at: {SOURCE_ROOT}")
        return

    if not DEST_DIR.exists():
        print(f"Destination not found. Creating: {DEST_DIR}")
        DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for '{FILE_TO_FIND}' in {SOURCE_ROOT}...")

    found_count = 0

    # 2. Loop through all items in the source root
    for item in SOURCE_ROOT.iterdir():

        # Check if the item is a directory (e.g., "VL00009")
        if item.is_dir():

            # This is our subject ID
            subject_id = item.name  # e.g., "VL00009"

            # 3. Construct the full path to the file we're looking for
            source_file_path = item / FILE_TO_FIND

            # 4. Check if that specific file exists
            if source_file_path.exists():

                # 5. Construct the new name and destination path
                new_file_name = f"{subject_id}{NEW_SUFFIX}"
                dest_file_path = DEST_DIR / new_file_name

                try:
                    # 6. Copy the file
                    print(f"  [FOUND] Copying {source_file_path.name} from {subject_id}")
                    print(f"     -> TO: {dest_file_path}")
                    shutil.copy2(source_file_path, dest_file_path)
                    found_count += 1

                except Exception as e:
                    print(f"  [ERROR] Failed to copy {source_file_path}: {e}")

    print(f"\nScript finished. Copied {found_count} file(s).")


if __name__ == "__main__":
    main()