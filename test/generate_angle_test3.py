import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# === Settings ===
annotation_file = "images/test/polygons/train/_annotations.coco.json"
target_image_filename = "frame_01219_png.rf.5e239accda0dfc99bc61f97ee1028dd5.jpg"

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

# === Find farthest in-mask point along line direction (both directions) ===
def find_edge_point(mask, origin, direction, max_length=1000, step=1):
    last_valid = origin.copy()
    for i in range(1, max_length, step):
        point = origin + direction * i
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
            last_valid = point.copy()
        else:
            break
    return last_valid

pt1 = find_edge_point(masks["line"], origin, direction)
pt2 = find_edge_point(masks["line"], origin, -direction)

# === Centers of mass ===
def get_center(mask):
    coords = np.column_stack(np.where(mask > 0))
    return coords.mean(axis=0)[::-1] if coords.size > 0 else np.array([0, 0])

camera_center = get_center(masks["camera"])
robot_center = get_center(masks["robot"])


# === Compute closest robot distances ===
def closest_point(mask, pt):
    points = np.column_stack(np.where(mask > 0))
    if points.size == 0:
        return None, float('inf')
    dists = np.linalg.norm(points - pt[::-1], axis=1)
    closest = points[np.argmin(dists)][::-1]
    return closest, dists.min()

pt1_robot, d1r = closest_point(masks["robot"], pt1)
pt2_robot, d2r = closest_point(masks["robot"], pt2)

robot_thresh = 10
pt1_on_robot = d1r < robot_thresh
pt2_on_robot = d2r < robot_thresh

# Distance to camera center
d1_cam_center = np.linalg.norm(pt1 - camera_center)
d2_cam_center = np.linalg.norm(pt2 - camera_center)

# Distance to robot center
d1_robot_center = np.linalg.norm(pt1 - robot_center)
d2_robot_center = np.linalg.norm(pt2 - robot_center)

# === Orientation decision logic ===
if pt1_on_robot and not pt2_on_robot:
    bottom, top = pt1, pt2
    condition_text = "Only pt1 on robot"
elif pt2_on_robot and not pt1_on_robot:
    bottom, top = pt2, pt1
    condition_text = "Only pt2 on robot"
elif pt1_on_robot and pt2_on_robot:
    
    # # Check with camera center
    # if d1_cam_center > d2_cam_center:
    #     bottom, top = pt1, pt2
    #     condition_text = "Both on robot; pt1 farther from camera"
    # else:
    #     bottom, top = pt2, pt1
    #     condition_text = "Both on robot; pt2 farther from camera"
        
    # Check with robot center
    if d1_robot_center < d2_robot_center:
        bottom, top = pt1, pt2
        condition_text = "Both on robot; pt1 closer to robot"
    else:
        bottom, top = pt2, pt1
        condition_text = "Both on robot; pt2 closer to robot"
        

else:
    bottom, top = pt1, pt2
    condition_text = "Fallback (none on robot)"

# === Angle computation ===
vec = top - bottom
angle = math.degrees(math.atan2(vec[1], vec[0]))
angle = (angle + 360) % 360

ref_vec_cam = camera_center - bottom
ref_angle_cam = (math.degrees(math.atan2(ref_vec_cam[1], ref_vec_cam[0])) + 360) % 360

ref_vec_rob = robot_center - bottom
ref_angle_rob = (math.degrees(math.atan2(ref_vec_rob[1], ref_vec_rob[0])) + 360) % 360
ref_angle = (ref_angle_cam + ref_angle_rob) / 2

relative_angle = (angle - ref_angle + 360) % 360

# === Print to console ===
print(f"[INFO] Absolute angle: {angle:.2f}°")
print(f"[INFO] Reference angle (bottom -> robot center): {ref_angle_rob:.2f}°")
print(f"[INFO] Reference angle (bottom -> camera center): {ref_angle_cam:.2f}°")
print(f"[INFO] Relative angle: {relative_angle:.2f}°")
print(f"[INFO] Decision logic: {condition_text}")

# === Visualization ===
viz = np.zeros((height, width, 3), dtype=np.uint8)
viz[masks["robot"] > 0] = (128, 0, 0)
viz[masks["camera"] > 0] = (0, 0, 128)
viz[masks["line"] > 0] = (0, 128, 0)

# Draw orientation arrow
cv2.arrowedLine(viz, tuple(np.int32(bottom)), tuple(np.int32(top)), (0, 255, 255), 2, tipLength=0.1)
cv2.circle(viz, tuple(np.int32(bottom)), 6, (0, 255, 255), -1)  # Yellow
cv2.circle(viz, tuple(np.int32(top)), 6, (255, 255, 0), -1)     # Cyan

# Camera center and debug lines
cam_center_pt = tuple(np.int32(camera_center))
cv2.circle(viz, cam_center_pt, 6, (255, 0, 255), -1)  # Magenta
cv2.line(viz, cam_center_pt, tuple(np.int32(bottom)), (255, 0, 255), 1)
# cv2.line(viz, cam_center_pt, tuple(np.int32(pt2)), (255, 0, 255), 1)

# Robot center and debug lines
cv2.circle(viz, tuple(np.int32(robot_center)), 6, (255, 255, 255), -1)
cv2.line(viz, tuple(np.int32(bottom)), tuple(np.int32(robot_center)), (255, 255, 255), 2)

# Text
cv2.putText(viz, condition_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(viz, f"Abs Angle: {angle:.2f}°", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(viz, f"Rel Angle: {relative_angle:.2f}°", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Show
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
plt.title("Line Orientation with Logic & Distance")
plt.axis('off')
plt.show()
