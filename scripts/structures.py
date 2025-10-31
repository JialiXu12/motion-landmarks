import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MRImage:
    """Holds the 3D MRI image data and its essential metadata."""
    image_array: np.ndarray
    spacing: np.ndarray
    origin: np.ndarray
    orientation: np.ndarray


@dataclass
class AnatomicalLandmarks:
    """Holds the coordinates for the core anatomical landmarks."""
    nipple_left: np.ndarray
    nipple_right: np.ndarray
    sternum_superior: np.ndarray  # Corresponds to sternal-superior
    sternum_inferior: np.ndarray  # Corresponds to sternal-inferior


@dataclass
class RegistrarData:
    """Holds all data associated with a single registrar's annotation."""
    # Using the flexible dictionary
    soft_tissue_landmarks: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ScanData:
    """
    Holds all data specific to one scan position (e.g., "prone").
    """
    position: str
    mri_image: MRImage
    anatomical_landmarks: AnatomicalLandmarks
    registrar_data: Dict[str, RegistrarData] = field(default_factory=dict)


@dataclass
class Subject:
    """
    A single container to hold all data related to one subject.
    This includes shared metadata and a dictionary of scans.
    """
    subject_id: str

    # Shared metadata
    age: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None

    # Position-specific data
    # e.g., {"prone": ScanData(...), "supine": ScanData(...)}
    scans: Dict[str, ScanData] = field(default_factory=dict)