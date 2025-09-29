import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# === Settings ===
annotation_file = "images/test/polygons/train/_annotations.coco.json"
target_image_filename = "frame_00000_png.rf.8111405ed0856e4e05c17f8319510412.jpg"

CATEGORY_IDS = {
    "camera": 1,
    "line": 2,
    "robot": 3
}

# === Load COCO annotations ===
coco = COCO(annotation_file)

# Find image ID
image_info = next(img for img in coco.loadImgs(coco.getImgIds()) if img['file_name'] == target_image_filename)
image_id = image_info['id']
height, width = image_info['height'], image_info['width']

# Get annotations
anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

# === Generate binary masks for each category ===
masks = {name: np.zeros((height, width), dtype=np.uint8) for name in CATEGORY_IDS}
for ann in anns:
    cat_id = ann['category_id']
    name = next((k for k, v in CATEGORY_IDS.items() if v == cat_id), None)
    if name:
        rle = coco.annToRLE(ann)
        mask = maskUtils.decode(rle)
        masks[name] = np.maximum(masks[name], mask)

# === Find line contour ===
line_mask = (masks["line"] * 255).astype(np.uint8)
contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
line_contour = max(contours, key=cv2.contourArea)

# Fit line
[vx, vy, x0, y0] = cv2.fitLine(line_contour, cv2.DIST_L2, 0, 0.01, 0.01)
direction = np.array([vx[0], vy[0]])
origin = np.array([x0[0], y0[0]])
pt1 = origin + direction * 1000
pt2 = origin - direction * 1000

def closest_point(mask, pt):
    points = np.column_stack(np.where(mask > 0))
    if points.size == 0:
        return None, float('inf')
    dists = np.linalg.norm(points - pt[::-1], axis=1)
    closest = points[np.argmin(dists)][::-1]
    return closest, dists.min()

# Contact checks
pt1_robot, d1r = closest_point(masks["robot"], pt1)
pt2_robot, d2r = closest_point(masks["robot"], pt2)
pt1_cam, d1c = closest_point(masks["camera"], pt1)
pt2_cam, d2c = closest_point(masks["camera"], pt2)

robot_thresh = 10
camera_thresh = 10
pt1_on_robot = d1r < robot_thresh
pt2_on_robot = d2r < robot_thresh
pt1_on_cam = d1c < camera_thresh
pt2_on_cam = d2c < camera_thresh

# Orientation logic
if pt1_on_robot and not pt2_on_robot:
    bottom, top = pt1, pt2
elif pt2_on_robot and not pt1_on_robot:
    bottom, top = pt2, pt1
elif pt1_on_robot and pt2_on_robot:
    bottom, top = (pt1, pt2) if d1c > d2c else (pt2, pt1)
else:
    bottom, top = pt1, pt2  # fallback

# Angle calculation
vec = top - bottom
angle = math.degrees(math.atan2(vec[1], vec[0]))

# === Visualization ===
viz = np.zeros((height, width, 3), dtype=np.uint8)
viz[masks["robot"] > 0] = (128, 0, 0)
viz[masks["camera"] > 0] = (0, 0, 128)
viz[masks["line"] > 0] = (0, 128, 0)

cv2.circle(viz, tuple(np.int32(bottom)), 5, (0, 255, 255), -1)
cv2.circle(viz, tuple(np.int32(top)), 5, (255, 255, 0), -1)
cv2.arrowedLine(viz, tuple(np.int32(bottom)), tuple(np.int32(top)), (0, 255, 255), 2, tipLength=0.05)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
plt.title(f"Line Angle: {angle:.2f}° (from robot to line tip)")
plt.axis('off')
plt.show()
