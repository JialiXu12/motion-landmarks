import breast_metadata
import os
from tools import landmarks as ld
import morphic
import numpy as np



if __name__ == '__main__':

    prone_masks_path = r'U:\sandbox\jxu759\volunteer_seg\prone_rib_cage_seg_for_John\body'

    for seg in os.listdir(prone_masks_path):
        if seg.endswith('.nii.gz'):
            prone_ribcage_seg_path = os.path.join(prone_masks_path, seg)
            prone_ribcage_mask = breast_metadata.readNIFTIImage(prone_ribcage_seg_path, orientation_flag='RAI', swap_axes=True)
            prone_ribcage_pc = ld.extract_contour_points(prone_ribcage_mask, 20000)

            base_filename = seg.replace('.nii.gz', '')
            output_file = os.path.join(prone_masks_path,'cloud_pts', base_filename)

            save_path = output_file + '.data'
            data = morphic.Data()
            data.values = prone_ribcage_pc
            data.save(save_path)

            np.savetxt(output_file + ".txt", prone_ribcage_pc)