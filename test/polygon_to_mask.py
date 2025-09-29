import os
import numpy as np
import cv2
from pycocotools.coco import COCO

def coco_to_semantic_masks(annotation_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    coco = COCO(annotation_file)

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        width, height = img_info['width'], img_info['height']
        file_name = img_info['file_name']

        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Sort by area descending so larger objects overwrite smaller ones
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        for ann in anns:
            cat_id = ann['category_id']
            seg_mask = coco.annToMask(ann)  # Binary mask for this object
            mask[seg_mask == 1] = cat_id    # Assign category ID to mask

        # Save mask as PNG
        mask_filename = os.path.splitext(file_name)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask to {mask_path}")

# Example usage
# coco_to_semantic_masks(
#     annotation_file="images/test/polygons/train/_annotations.coco.json",
#     output_dir="images/test/polygons/train/masks"
# )