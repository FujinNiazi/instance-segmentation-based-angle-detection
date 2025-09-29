import cv2
import numpy as np
import math

# === Settings ===
mask_path = "images/test/polygons/train/masks/frame_00000_png.rf.279d91176dcff954a5a3bedaec15cf6a.png"
line_category_id = 2  # category_id for 'line'

# === Load mask ===
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
if mask is None:
    raise FileNotFoundError(f"Could not read mask: {mask_path}")

# === Extract binary mask for the line ===
binary_mask = np.uint8(mask == line_category_id) * 255

# === Find non-zero points ===
coords = cv2.findNonZero(binary_mask)
if coords is None or len(coords) < 2:
    print("Not enough points to fit a line.")
    exit()

# === Fit line ===
[vx, vy, x0, y0] = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
direction = np.array([vx[0], vy[0]])  # direction vector

# === Find bottom-most point in the contour (largest y value) ===
bottom_idx = np.argmax(coords[:, 0, 1])  # y is second dim
bottom_point = coords[bottom_idx][0]

# === Vector from fitted line origin to bottom point ===
to_bottom = np.array([bottom_point[0] - x0, bottom_point[1] - y0])
dot_product = np.dot(direction, to_bottom)

# === Flip direction if it's pointing away from the bottom ===
if dot_product < 0:
    direction = -direction

# === Compute angle ===
angle_rad = math.atan2(direction[0], direction[1])
angle_deg = math.degrees(angle_rad)
angle_deg = 360 - (angle_deg % 360)


print(f"Directional angle of line: {angle_deg:.2f} degrees")

# === Visualization ===
color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
pt1 = (int(x0 - direction[0] * 1000), int(y0 - direction[1] * 1000))
pt2 = (int(x0 + direction[0] * 1000), int(y0 + direction[1] * 1000))
cv2.line(color_mask, pt1, pt2, (0, 255, 0), 2)
cv2.circle(color_mask, tuple(bottom_point), 5, (0, 0, 255), -1)

cv2.imshow("Directional Line", color_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
