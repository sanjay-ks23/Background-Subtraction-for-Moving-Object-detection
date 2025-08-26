import cv2
import argparse
from ebgs.ebgs_algorithm import EBGS
import os

def main(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        # Create a dummy video file for demonstration if it doesn't exist
        print("Creating a dummy video file for demonstration purposes.")
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
        
        # Create a simple video with a moving square
        background = np.zeros((height, width, 3), dtype=np.uint8)
        square_color = (255, 0, 0)
        
        for i in range(100):
            frame = background.copy()
            cv2.rectangle(frame, (i*5, 100), (i*5+50, 150), square_color, -1)
            out.write(frame)
        out.release()
        print(f"Dummy video created at {video_path}")


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ebgs = EBGS(width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        foreground_mask = ebgs.process_frame(frame)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', foreground_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBGS Algorithm for Background Subtraction")
    parser.add_argument("--video", type=str, help="Path to the video file.", required=True)
    args = parser.parse_args()
    main(args.video)
