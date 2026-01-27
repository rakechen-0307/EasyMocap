# Extract frames from a video file using OpenCV and save them as images

import cv2
import pathlib
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_folder", type=str, help="Folder to save the extracted frames.")
    return parser.parse_args()

def extract_frames(video_path, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = output_folder / f"{frame_count:04d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    args = parse_args()
    output_folder = pathlib.Path(args.output_folder) / "images"
    extract_frames(args.video_path, output_folder)