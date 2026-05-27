"""
Diagnostic script to check what's wrong with VL00009 alignment.
"""

import sys
from pathlib import Path

# Try to import and check the ribcage file
vl_id = 9
vl_id_str = f"VL{vl_id:05d}"

SUPINE_RIBCAGE_ROOT = Path(r"U:\sandbox\jxu759\volunteer_seg\results\supine\rib_cage")
supine_seg_file = SUPINE_RIBCAGE_ROOT / f"rib_cage_{vl_id_str}.nii.gz"

print(f"Checking file: {supine_seg_file}")
print(f"File exists: {supine_seg_file.exists()}")

if supine_seg_file.exists():
    print(f"File size: {supine_seg_file.stat().st_size} bytes")

    # Try to load it
    try:
        import SimpleITK as sitk
        print("\nTrying to load with SimpleITK...")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(supine_seg_file))
        itk_mask = reader.Execute()
        print(f"✓ Successfully loaded with SimpleITK")
        print(f"  Image size: {itk_mask.GetSize()}")
        print(f"  Image spacing: {itk_mask.GetSpacing()}")
        print(f"  Image origin: {itk_mask.GetOrigin()}")

        # Try the full readNIFTIImage function
        print("\nTrying breast_metadata.readNIFTIImage...")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import external.breast_metadata_mdv.breast_metadata as breast_metadata

        mask_scan = breast_metadata.readNIFTIImage(
            str(supine_seg_file), 'RAI', swap_axes=True
        )
        print(f"✓ Successfully loaded with readNIFTIImage")
        print(f"  Scan type: {type(mask_scan)}")
        print(f"  Scan values shape: {mask_scan.values.shape if hasattr(mask_scan, 'values') else 'NO VALUES'}")
        print(f"  Scan orientation: {mask_scan.orientation if hasattr(mask_scan, 'orientation') else 'NO ORIENTATION'}")

    except Exception as e:
        print(f"✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ File does not exist!")
    print(f"\nChecking parent directory: {SUPINE_RIBCAGE_ROOT}")
    print(f"Parent exists: {SUPINE_RIBCAGE_ROOT.exists()}")
    if SUPINE_RIBCAGE_ROOT.exists():
        print(f"\nFiles in directory:")
        for f in sorted(SUPINE_RIBCAGE_ROOT.glob("*.nii.gz"))[:10]:
            print(f"  {f.name}")
