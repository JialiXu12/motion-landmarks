import numpy as np
import SimpleITK as sitk
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Import structures
try:
    from .structures import MRImage, AnatomicalLandmarks, RegistrarData, ScanData, Subject
except ImportError:
    from structures import MRImage, AnatomicalLandmarks, RegistrarData, ScanData, Subject

# --- Helper functions  ---
def _read_single_point_json(json_path: Path, position: str) -> Optional[np.ndarray]:
    """
    Helper to read the legacy nested JSON structure for a single point.
    e.g., {"supine_point": {"point": {"x": 1, "y": 2, "z": 3}}}
    """
    if not json_path.exists():
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    key = f"{position}_point"
    if key in data:
        pt = data[key]['point']
        return np.array([pt['x'], pt['y'], pt['z']])

    # Handle cases where the file exists but the point for the position doesn't
    return None

def _safe_get_metadata(image: sitk.Image, key: str) -> Optional[str]:
    """Safely get metadata from a SimpleITK image."""
    if image.HasMetaDataKey(key):
        return image.GetMetaData(key)
    return None


def _parse_dicom_metadata(image: sitk.Image) -> dict:
    """Extracts age, weight, and height from DICOM tags."""
    metadata = {}

    # (0010, 1010) = Patient's Age
    age_str = _safe_get_metadata(image, "0010|1010")
    if age_str: metadata['age'] = age_str

    # (0010, 1030) = Patient's Height (in meters)
    height_str = _safe_get_metadata(image, "0010|1030")
    if height_str:
        try:
            metadata['height'] = float(height_str)
        except ValueError:
            pass

    # (0010, 1020) = Patient's Weight (in kg)
    weight_str = _safe_get_metadata(image, "0010|1020")
    if weight_str:
        try:
            metadata['weight'] = float(weight_str)
        except ValueError:
            pass

    return metadata


# --- Reader Functions ---

def read_dicom_data(dicom_directory: Path) -> Tuple[MRImage, dict]:
    """
    Reads a DICOM series, extracts metadata, and forces the image
    into 'RAI' orientation using SimpleITK.
    """
    if not dicom_directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_directory}")

    # --- Step 1: Load the image as it is on disk ---
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_directory))
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series found in {dicom_directory}")

    # Read the header of the first file in the series
    first_slice_image = sitk.ReadImage(dicom_names[0])

    # Extract metadata from that first slice's header
    subject_metadata = _parse_dicom_metadata(first_slice_image)

    reader.SetFileNames(dicom_names)
    original_sitk_image = reader.Execute()

    # --- Step 2: Re-orient the image to RAI ---
    sitk_image = sitk.DICOMOrient(original_sitk_image, 'RAI')

    # --- Step 3: Extract metadata from the re-oriented image ---
    image_array = sitk.GetArrayFromImage(sitk_image)  # (Z, Y, X)
    spacing = np.array(sitk_image.GetSpacing())  # (X, Y, Z)
    origin = np.array(sitk_image.GetOrigin())  # (X, Y, Z)
    orientation = np.array(sitk_image.GetDirection()).reshape(3, 3)

    mri_image = MRImage(
        image_array=image_array,
        spacing=spacing,
        origin=origin,
        orientation=orientation
    )

    # subject_metadata = _parse_dicom_metadata(sitk_image)
    return mri_image, subject_metadata


def read_anatomical_landmarks(json_file: Path) -> AnatomicalLandmarks:
    """
    Reads anatomical landmarks from the new single-file JSON structure.
    """
    if not json_file.exists():
        raise FileNotFoundError(f"Anatomical landmark file not found: {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    landmarks = None
    try:
        bodies_dict = data.get('bodies', {})

        if 'Jiali-test' in bodies_dict:
            landmarks = bodies_dict['Jiali-test'].get('landmarks')

        if not landmarks and 'Ray-Test' in bodies_dict:
            landmarks = bodies_dict['Ray-Test'].get('landmarks')

        return AnatomicalLandmarks(
            nipple_left=np.array(landmarks["nipple-l"]),
            nipple_right=np.array(landmarks["nipple-r"]),
            sternum_superior=np.array(landmarks["sternal-superior"]),
            sternum_inferior=np.array(landmarks["sternal-inferior"])
        )
    except KeyError as e:
        print(f"Error parsing anatomical landmarks from {json_file}: Missing key {e}")
        print("This often happens if the 'Jiali-test' key is not present.")
        raise ValueError(
            f"File {json_file} does not match expected JSON structure."
        )


def read_all_registrar_data(
        soft_tissue_root: Path, vl_id_str: str, position: str
) -> Dict[str, RegistrarData]:
    """
    Loads soft tissue landmarks for a single subject from all registrars.
    """
    all_registrars_data = {}

    if not soft_tissue_root.is_dir():
        print(f"Warning: Soft tissue root not found: {soft_tissue_root}")
        return all_registrars_data

    # Loop through each item in the root (e.g., 'registrar_A', 'registrar_B')
    for registrar_name in ["ben_reviewed","holly"]:
        registrar_dir = soft_tissue_root / registrar_name
        if not registrar_dir.is_dir():
            print(f"Warning: Registrar directory not found: {registrar_dir}")
            continue

        # Get the subject-specific folder for this registrar
        subject_landmarks_dir = registrar_dir / vl_id_str
        if not subject_landmarks_dir.is_dir():
            # This registrar may not have data for this subject
            continue

        landmarks_dict = {}
        landmark_counts = defaultdict(int)

        # Loop through all point.XXX.json files
        for filename in sorted(os.listdir(subject_landmarks_dir)):
            if filename.startswith("point.") and filename.endswith(".json"):
                file_path = subject_landmarks_dir / filename

                with open(file_path, 'r') as f:
                    data = json.load(f)

                if data.get("status") == "rejected":
                    continue

                point = _read_single_point_json(file_path, position)
                if point is None:
                    continue

                landmark_type = data.get("type", "unknown").lower().replace(" ", "_")
                landmark_counts[landmark_type] += 1
                key = f"{landmark_type}_{landmark_counts[landmark_type]}"

                landmarks_dict[key] = point

        # If we found landmarks, add this registrar's data
        if landmarks_dict:
            all_registrars_data[registrar_name] = RegistrarData(
                soft_tissue_landmarks=landmarks_dict
            )

    return all_registrars_data


def _load_scan_data(
        vl_id: int,
        position: str,
        dicom_root: Path,
        anatomical_json_base_root: Path,
        soft_tissue_root: Path
) -> Tuple[ScanData, dict]:
    """Loads all data for a single scan (one position)."""
    vl_id_str = f"{vl_id:05d}"

    # 1. Construct all paths
    dicom_dir = dicom_root / f"VL{vl_id_str}" / position
    anatomical_json_root = anatomical_json_base_root / position / "landmarks"
    filename = f"VL{vl_id_str}_skeleton_data_{position}_t2.json"
    primary_path = anatomical_json_root / filename
    fallback_path = anatomical_json_root / "combined" / filename

    if primary_path.exists():
        anatomical_json_path = primary_path
    elif fallback_path.exists():
        anatomical_json_path = fallback_path
    else:
        raise FileNotFoundError(
            f"Anatomical JSON not found for {vl_id_str}:\n"
            f"  - Looked for: {primary_path}\n"
            f"  - Looked for: {fallback_path}"
        )

    # 2. Load DICOM Image and Metadata
    mri_image, subject_meta = read_dicom_data(dicom_dir)

    # 3. Load Anatomical Landmarks
    anatomical_landmarks = read_anatomical_landmarks(anatomical_json_path)

    # 4. Load Soft Tissue Landmarks
    registrar_data = read_all_registrar_data(
        soft_tissue_root, vl_id_str, position
    )

    # 5. Create the ScanData object
    scan_data = ScanData(
        position=position,
        mri_image=mri_image,
        anatomical_landmarks=anatomical_landmarks,
        registrar_data=registrar_data
    )

    return scan_data, subject_meta


# -----------------------------------------------------------------
# 2. NEW main loader function. Call this from main.py
# -----------------------------------------------------------------
def load_subject(
        vl_id: int,
        positions: List[str],  # e.g., ["prone", "supine"]
        dicom_root: Path,
        anatomical_json_base_root: Path,
        soft_tissue_root: Path
) -> Subject:
    """
    Loads all requested scans for a single subject and combines
    them into one Subject object.
    """
    vl_id_str = f"{vl_id:05d}"
    subject = Subject(subject_id=f"VL{vl_id_str}")

    shared_meta_loaded = False

    for position in positions:
        try:
            # Load the data for one scan
            scan_data, subject_meta = _load_scan_data(
                vl_id=vl_id,
                position=position,
                dicom_root=dicom_root,
                anatomical_json_base_root=anatomical_json_base_root,
                soft_tissue_root=soft_tissue_root
            )

            # Add the scan to the subject's dictionary
            subject.scans[position] = scan_data

            # Load the shared metadata (age, etc.) only ONCE
            if not shared_meta_loaded and subject_meta:
                subject.age = subject_meta.get('age')
                subject.weight = subject_meta.get('weight')
                subject.height = subject_meta.get('height')
                shared_meta_loaded = True

        except Exception as e:
            # This allows loading to continue if e.g. "supine" is missing
            print(f"Warning: Failed to load '{position}' for {vl_id_str}. Error: {e}")

    return subject