import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# === Set Paths ===
annotation_file = "images/test/polygons/train/_annotations.coco.json"
output_dir = "images/test/polygons/train/masks"

# === Optional: Define colors for categories ===
COLOR_MAP = {
    1: (255, 0, 0),    # Red for camera
    2: (0, 255, 0),    # Green for line
    3: (0, 0, 255),    # Blue for robot
}

# === Load COCO annotations ===
with open(annotation_file, 'r') as f:
    coco = json.load(f)

images = {img['id']: img for img in coco['images']}
annotations = coco['annotations']
anns_by_image = defaultdict(list)
for ann in annotations:
    anns_by_image[ann['image_id']].append(ann)

os.makedirs(output_dir, exist_ok=True)

# === Process each image ===
for img_id, img_info in images.items():
    height, width = img_info['height'], img_info['width']
    file_name = img_info['file_name']
    base_name = os.path.splitext(file_name)[0]

    # Create empty masks
    mask = np.zeros((height, width), dtype=np.uint8)
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for ann in anns_by_image[img_id]:
        category_id = ann['category_id']
        if category_id == 0:
            continue  # skip background category

        segmentation = ann.get('segmentation', [])
        if not segmentation:
            continue

        for seg in segmentation:
            pts = np.array(seg, dtype=np.float32).reshape((-1, 2))
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            pts = pts.astype(np.int32).reshape((-1, 1, 2))

            # Draw grayscale mask
            cv2.fillPoly(mask, [pts], color=category_id)

            # Draw color mask
            color = COLOR_MAP.get(category_id, (255, 255, 255))  # default: white
            cv2.fillPoly(color_mask, [pts], color=color)

    # Save grayscale mask (for training)
    mask_path = os.path.join(output_dir, base_name + ".png")
    cv2.imwrite(mask_path, mask)

    # Save color mask (for inspection)
    color_mask_path = os.path.join(output_dir, "vis_" + base_name + ".png")
    cv2.imwrite(color_mask_path, color_mask)
    
        # Visualize (optional)
    plt.imshow(mask, cmap='jet')
    plt.title(f"Mask for {file_name}")
    plt.colorbar()
    plt.show()

    print(f"Saved: {mask_path}")
    print(f"Saved: {color_mask_path}")
