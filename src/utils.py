import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def smooth_angles(df, column='angle_deg', window_size=4):
    """
    Smooths angle values in degrees using vector averaging.

    Parameters:
        df (pd.DataFrame): DataFrame containing the angle column.
        column (str): Name of the column with raw angles in degrees.
        window_size (int): Rolling window size.

    Returns:
        pd.Series: Smoothed angle values in degrees (0–360).
    """
    angles_rad = np.deg2rad(df[column])
    sin_angle = np.sin(angles_rad)
    cos_angle = np.cos(angles_rad)

    sin_smooth = pd.Series(sin_angle).rolling(window=window_size, center=True, min_periods=1).mean()
    cos_smooth = pd.Series(cos_angle).rolling(window=window_size, center=True, min_periods=1).mean()

    smoothed_angles_rad = np.arctan2(sin_smooth, cos_smooth)
    smoothed_angles_deg = np.rad2deg(smoothed_angles_rad) % 360

    return smoothed_angles_deg

def unwrap_angles(df, column='angle_deg'):
    """
    Unwraps angle values in degrees using unwrap function.
    """
    series = df[column].copy()

    # Handle NaNs before unwrapping
    series = series.interpolate(method='linear', limit_direction='both')

    unwrapped_angles_deg = np.rad2deg(np.unwrap(np.deg2rad(series)))
    return unwrapped_angles_deg

def save_angles_to_csv(angle_data, output_path, window_size=5):
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(angle_data)

    # Apply smoothing and add new columns
    df['abs_angle_smooth'] = smooth_angles(df, column='abs_angle', window_size=window_size)
    df['rel_angle_smooth'] = smooth_angles(df, column='rel_angle', window_size=window_size)
    
    # Apply unwrapping and add new colums
    df['abs_angle_unwrapped'] = unwrap_angles(df, column='abs_angle_smooth')
    df['rel_angle_unwrapped'] = unwrap_angles(df, column='rel_angle_smooth')

    # Update the angle_data list with smoothed values
    angle_data = df.to_dict(orient='records')

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved CSV to {output_path}")

    return angle_data  # Optional: return updated list with smoothed values

def plot_angle(angle_data, key, title, ylabel, output_path):
    frames = [
        entry["frame"]
        for entry in angle_data
        if key in entry and entry[key] is not None
    ]
    angles = [
        entry[key]
        for entry in angle_data
        if key in entry and entry[key] is not None
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(frames, angles, marker='o', color='orange', linewidth=2)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved angle plot to {output_path}")

    
def save_video_from_frames(frame_dir, output_path, fps):
    
    output_path = str(output_path)

    frame_files = sorted([
        f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.png'))
    ])
    if not frame_files:
        print("[WARNING] No frames found for video.")
        return

    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'mp4v' for MP4

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fname in frame_files:
        frame = cv2.imread(os.path.join(frame_dir, fname))
        out.write(frame)

    out.release()
    print(f"[INFO] Saved stitched video to {output_path}")
