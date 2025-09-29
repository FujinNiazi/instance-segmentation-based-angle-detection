import os
import argparse
from pathlib import Path
from inference import setup_predictor, run_on_video_or_images
from utils import save_angles_to_csv, plot_angle
from utils import save_video_from_frames

# Get the absolute path to the current file's directory
current_dir = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images or video using Detectron2")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--weights", type=str, help="Path to trained weights (.pth)")
    parser.add_argument("--input", type=str, help="Path to input video or image folder")
    parser.add_argument("--output", type=str, help="Path to output directory")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    # === Default Configuration ===
    config_path = args.config or current_dir.parent / "config" / "config.yaml"
    weights_path = args.weights or current_dir.parent / "weights" / "model_final.pth"
    input_source = args.input or current_dir.parent / "input" / "3.mov"  # or "input/images/"
    output_dir = args.output or current_dir.parent / "output" 
    csv_output = current_dir.parent / "output" / "angles.csv"
    plot_output_abs = current_dir.parent / "output" / "absolute_angle_plot.png"
    plot_output_rel = current_dir.parent / "output" / "relative_angle_plot.png"
    plot_unwrapped_abs = current_dir.parent / "output" / "abs_unwrapped_angle_plot.png"
    plot_unwrapped_rel = current_dir.parent / "output" / "rel_unwrapped_angle_plot.png"
    class_map = {
        
        "robot": 0,
        "camera": 1,
        "line": 2
    }

    # === Ensure output directory exists ===
    os.makedirs(output_dir, exist_ok=True)

    # === Setup model ===
    predictor, cfg = setup_predictor(config_path, weights_path)

    # === Run inference & orientation ===
    angle_data = run_on_video_or_images(predictor, cfg, input_source, output_dir, class_map)
    
    
    # === Sanity check ===
    if not angle_data:
        print("[WARNING] No valid angles extracted — Inference not made.")
        return

    # === Save outputs ===
    angle_data = save_angles_to_csv(angle_data, csv_output)
    plot_angle(angle_data, "abs_angle_smooth", "Absolute Angle per Frame", "Absolute Angle (°)", plot_output_abs)
    plot_angle(angle_data, "rel_angle_smooth", "Relative Angle per Frame", "Relative Angle (°)", plot_output_rel)
    plot_angle(angle_data, "abs_angle_unwrapped", "Absolute Unwrapped Angle per Frame", "Absolute Unwrapped Angle (°)", plot_unwrapped_abs)
    plot_angle(angle_data, "rel_angle_unwrapped", "Relative Unwrapped Angle per Frame", "Relative Unwrapped Angle (°)", plot_unwrapped_rel)

    
    # === Save videos ===
    masked_video_path = current_dir.parent / "output" / "video_masked.avi"
    boxes_video_path = current_dir.parent / "output" / "video_with_boxes.avi"

    save_video_from_frames(os.path.join(output_dir, "frames_masked"), masked_video_path, fps=30)
    save_video_from_frames(os.path.join(output_dir, "frames_with_boxes"), boxes_video_path, fps=30)
    
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()
