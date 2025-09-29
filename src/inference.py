import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from orientation import process_frame_with_masks  
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_predictor(config_path, weights_path, score_thresh=0.2):
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda"  # or "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def run_on_video_or_images(predictor, cfg, input_source, output_dir, class_map):
    
    video_types = ('.mp4', '.avi', '.mov', '.mkv', '.webm')  
    image_types = ('.jpg', '.jpeg', '.png')
    
    if input_source.suffix.lower() in video_types:
        cap = cv2.VideoCapture(str(input_source))
    else:
        cap = None

    is_video = cap is not None

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    angle_data = []
    
    os.makedirs(os.path.join(output_dir, "frames_masked"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames_with_boxes"), exist_ok=True)

    if not is_video:
        img_files = sorted([
            f for f in os.listdir(input_source)
            if f.lower().endswith(image_types)
        ])

    while True:
        # Load frame
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_idx >= len(img_files):
                break
            frame_path = os.path.join(input_source, img_files[frame_idx])
            frame = cv2.imread(frame_path)

        outputs = predictor(frame)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()
        

        # Build masks_dict
        masks_dict = {k: np.zeros(frame.shape[:2], dtype=np.uint8) for k in class_map}
        for cls, mask in zip(pred_classes, pred_masks):
            for name, cid in class_map.items():
                if cls == cid:
                    masks_dict[name] = np.maximum(masks_dict[name], mask.astype(np.uint8))

        angle_info, annotated = process_frame_with_masks(frame, masks_dict)
        
        # NEW: Skip frame if no valid contour was found
        if angle_info.get("top") is None or angle_info.get("bottom") is None:
            print(f"[INFO] No valid contour found in frame {frame_idx}, skipping.")
            frame_idx += 1
            continue

        # === Save orientation mask output ===
        out_path_mask = os.path.join(output_dir, "frames_masked", f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path_mask, annotated)
        
        # === 2. Draw instance predictions + overlay orientation arrow ===
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_img = vis_output.get_image()[:, :, ::-1]

        # # Draw the same arrow as in `annotated` on top of the vis_img
        vis_img = np.ascontiguousarray(vis_img)
        bottom = angle_info.get("bottom")
        top = angle_info.get("top")
        if bottom is not None and top is not None:
            pt1 = tuple(np.int32(bottom))
            pt2 = tuple(np.int32(top))
            cv2.arrowedLine(vis_img, pt1, pt2, (0, 255, 255), 2, tipLength=0.1)
            cv2.circle(vis_img, pt1, 6, (0, 255, 255), -1)
            cv2.circle(vis_img, pt2, 6, (255, 255, 0), -1)

        # Save the visualized frame
        out_path_boxes = os.path.join(output_dir, "frames_with_boxes", f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path_boxes, vis_img)
        
        # Collect angle data
        angle_data.append({
            "frame": frame_idx,
            **angle_info
        })

        print(f"[INFO] Processed frame {frame_idx}")
        frame_idx += 1

    if is_video:
        cap.release()

    return angle_data
