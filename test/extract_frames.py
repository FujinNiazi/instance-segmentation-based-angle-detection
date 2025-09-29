import cv2
import os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Done! Extracted {frame_count} frames to '{output_folder}'")

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    output_folder = input("Enter the output folder for frames: ").strip()
    extract_frames(video_path, output_folder)
