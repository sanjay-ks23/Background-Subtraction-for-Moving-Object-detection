import cv2
import argparse
import numpy as np
from ebgs.ebgs_algorithm import EBGS
import os
import glob

def process_image_sequence(images_path, output_path, display=True):
    # Handle the 'input' subdirectory structure for the Change Detection 2014 dataset
    input_dir = os.path.join(images_path, 'input')
    if not os.path.isdir(input_dir):
        print(f"Error: 'input' directory not found in {images_path}")
        return

    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    if not image_files:
        print(f"No JPG images found in {input_dir}")
        return

    # Read the first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error reading the first image: {image_files[0]}")
        return
    height, width, _ = first_frame.shape

    ebgs = EBGS(width, height)

    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image {image_file}")
            continue

        foreground_mask = ebgs.process_frame(frame)
        
        if output_path:
            mask_filename = os.path.join(output_path, f"mask_{i:04d}.png")
            cv2.imwrite(mask_filename, foreground_mask)

        if display:
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Foreground Mask', foreground_mask)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    
    if display:
        cv2.destroyAllWindows()


def process_video(video_path, output_path, display=True):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ebgs = EBGS(width, height)

    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        foreground_mask = ebgs.process_frame(frame)

        if output_path:
            mask_filename = os.path.join(output_path, f"mask_{frame_index:04d}.png")
            cv2.imwrite(mask_filename, foreground_mask)
        
        if display:
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Foreground Mask', foreground_mask)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        frame_index += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBGS Algorithm for Background Subtraction")
    parser.add_argument("--video", type=str, help="Path to the video file.")
    parser.add_argument("--images", type=str, help="Path to the directory of images (e.g., a video category from the Change Detection 2014 dataset).")
    parser.add_argument("--output", type=str, help="Path to the output directory for masks. If not provided, masks will not be saved.")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time display of video and masks.")
    args = parser.parse_args()

    display_enabled = not args.no_display

    if args.video:
        process_video(args.video, args.output, display=display_enabled)
    elif args.images:
        process_image_sequence(args.images, args.output, display=display_enabled)
    else:
        print("Please provide a path to a video file or an image directory.")
        print("Creating a dummy video and processing it for demonstration.")
        dummy_video_path = "dummy_video.mp4"
        if not os.path.exists(dummy_video_path):
            width, height = 640, 480
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dummy_video_path, fourcc, 20.0, (width, height))
            background = np.zeros((height, width, 3), dtype=np.uint8)
            square_color = (255, 0, 0)
            for i in range(100):
                frame = background.copy()
                cv2.rectangle(frame, (i*5, 100), (i*5+50, 150), square_color, -1)
                out.write(frame)
            out.release()
        process_video(dummy_video_path, args.output, display=display_enabled)