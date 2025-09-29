def run_on_video_or_images(predictor, cfg, input_source, output_dir, class_map):
    video_types = ('.mp4', '.avi', '.mov', '.mkv', '.webm')  
    image_types = ('.jpg', '.jpeg', '.png')
    
    cap = cv2.VideoCapture(input_source) if input_source.endswith(video_types) else None
    is_video = cap is not None

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    angle_data = []
    valid_frames = []  # List to store valid frames for video stitching
    
    os.makedirs(os.path.join(output_dir, "frames_masked"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames_with_boxes"), exist_ok=True)

    if not is_video:
        img_files = sorted([f for f in os.listdir(input_source) if f.lower().endswith(image_types)])

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

        # Build masks_dict for the classes we're interested in
        masks_dict = {k: np.zeros(frame.shape[:2], dtype=np.uint8) for k in class_map}
        for cls, mask in zip(pred_classes, pred_masks):
            for name, cid in class_map.items():
                if cls == cid:
                    masks_dict[name] = np.maximum(masks_dict[name], mask.astype(np.uint8))

        # Check if any relevant class mask is present in masks_dict
        has_valid_mask = False
        for name, mask in masks_dict.items():
            if np.any(mask):  # If any mask of the desired class is present
                has_valid_mask = True
                break

        if not has_valid_mask:
            print(f"[WARNING] No relevant class detections for frame {frame_idx}, skipping save.")
            frame_idx += 1
            continue

        # Process frame with masks
        angle_info, annotated = process_frame_with_masks(frame, masks_dict)

        # Skip frame if no valid contour was found
        if angle_info.get("top") is None or angle_info.get("bottom") is None:
            print(f"[INFO] No valid contour found in frame {frame_idx}, skipping.")
            frame_idx += 1
            continue

        # Save frames only if they have valid detections
        out_path_mask = os.path.join(output_dir, "frames_masked", f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path_mask, annotated)
        
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_img = vis_output.get_image()[:, :, ::-1]

        # Save the visualized frame with boxes
        out_path_boxes = os.path.join(output_dir, "frames_with_boxes", f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path_boxes, vis_img)

        # Store valid frame for stitching video later
        valid_frames.append(annotated)

        # Collect angle data
        angle_data.append({
            "frame": frame_idx,
            **angle_info
        })

        print(f"[INFO] Processed frame {frame_idx}")
        frame_idx += 1

    if is_video:
        cap.release()

    return angle_data, valid_frames  # Return valid frames list for video stitching

