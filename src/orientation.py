import numpy as np
import cv2
import math

def process_frame_with_masks(frame, masks):
    """
    Inputs:
        - frame: original image (H, W, 3)
        - masks: dict of binary masks { 'camera': HxW, 'line': HxW, 'robot': HxW }
    Returns:
        - angle_info: dict with angle data and decision logic
        - viz: image with visual overlays
    """
    height, width = frame.shape[:2]

    # === Find line contour ===
    line_mask = (masks["line"] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "abs_angle": None,
            "rel_angle": None,
            "ref_angle_cam": None,
            "ref_angle_rob": None,
            "condition": "No line contour found",
            "bottom": None,
            "top": None
        }, frame.copy()

    line_contour = max(contours, key=cv2.contourArea)

    # Fit line
    [vx, vy, x0, y0] = cv2.fitLine(line_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([vx[0], vy[0]])
    origin = np.array([x0[0], y0[0]])

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

    def get_center(mask):
        coords = np.column_stack(np.where(mask > 0))
        return coords.mean(axis=0)[::-1] if coords.size > 0 else np.array([0, 0])

    camera_center = get_center(masks["camera"])
    robot_center = get_center(masks["robot"])

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
    
    pt1_camera, d1r_ = closest_point(masks["camera"], pt1)
    pt2_camera, d2r_ = closest_point(masks["camera"], pt2)

    camera_thresh = 5
    pt1_on_camera = d1r_ < camera_thresh
    pt2_on_camera = d2r_ < camera_thresh
    
    d1_cam_center = np.linalg.norm(pt1 - camera_center)
    d2_cam_center = np.linalg.norm(pt2 - camera_center)
    d1_robot_center = np.linalg.norm(pt1 - robot_center)
    d2_robot_center = np.linalg.norm(pt2 - robot_center)

    # Orientation decision logic
    if pt1_on_robot and not pt2_on_robot:
        bottom, top = pt1, pt2
        condition_text = "Only pt1 on robot"
    elif pt2_on_robot and not pt1_on_robot:
        bottom, top = pt2, pt1
        condition_text = "Only pt2 on robot"
    
    elif pt1_on_camera and not pt2_on_camera:
        bottom, top = pt2, pt1
        condition_text = "Both on robot; pt1 on camera"
        
    elif pt2_on_camera and not pt1_on_camera:
        bottom, top = pt1, pt2
        condition_text = "Both on robot; pt2 on camera"
    elif pt1_on_robot and pt2_on_robot:
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

    # === Visualization ===
    viz = np.zeros((height, width, 3), dtype=np.uint8)
    viz[masks["robot"] > 0] = (128, 0, 0)
    viz[masks["camera"] > 0] = (0, 0, 128)
    viz[masks["line"] > 0] = (0, 128, 0)

    cv2.arrowedLine(viz, tuple(np.int32(bottom)), tuple(np.int32(top)), (0, 255, 255), 2, tipLength=0.1)
    cv2.circle(viz, tuple(np.int32(bottom)), 6, (0, 255, 255), -1)  # Yellow
    cv2.circle(viz, tuple(np.int32(top)), 6, (255, 255, 0), -1)     # Cyan

    # Camera center
    cam_center_pt = tuple(np.int32(camera_center))
    cv2.circle(viz, cam_center_pt, 6, (255, 0, 255), -1)
    cv2.line(viz, cam_center_pt, tuple(np.int32(bottom)), (255, 0, 255), 1)

    # Robot center
    rob_center_pt = tuple(np.int32(robot_center))
    cv2.circle(viz, rob_center_pt, 6, (255, 255, 255), -1)
    cv2.line(viz, tuple(np.int32(bottom)), rob_center_pt, (255, 255, 255), 2)

    # Text
    cv2.putText(viz, condition_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(viz, f"Abs Angle: {angle:.2f}°", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(viz, f"Rel Angle: {relative_angle:.2f}°", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === Final Output ===
    angle_info = {
        "abs_angle": round(angle, 2),
        "rel_angle": round(relative_angle, 2),
        "ref_angle_cam": round(ref_angle_cam, 2),
        "ref_angle_rob": round(ref_angle_rob, 2),
        "condition": condition_text,
        "bottom": bottom.tolist(),  # ensure it's JSON serializable
        "top": top.tolist()
    }

    return angle_info, viz
