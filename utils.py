import cv2
import os

# Bounding box utilities
def get_center_of_bbox(bbox):
    """Get the center coordinates of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """Calculate the width of a bounding box."""
    return bbox[2] - bbox[0]

def get_foot_position(bbox):
    """Get the foot position (bottom-center) of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

# Distance measuring utilities
def measure_distance(p1, p2):
    """
    Measure Euclidean distance between two points.
    p1 and p2 are tuples of (x, y).
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    """
    Measure the x and y distance separately between two points.
    Returns (x_distance, y_distance).
    """
    return p1[0] - p2[0], p1[1] - p2[1]

# Video utilities
def read_video(video_path):
    """
    Read a video file and return a list of frames with improved error handling.
    """
    # Normalize path
    video_path = os.path.normpath(video_path)
    
    # Check if file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: {video_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")

    # Get video properties for debugging
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- Total frames: {frame_count}")
    print(f"- FPS: {fps}")

    frames = []
    frame_read = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_read += 1
        
        # Print progress every 100 frames
        if frame_read % 100 == 0:
            print(f"Read {frame_read} frames...")

    cap.release()
    
    print(f"Successfully read {len(frames)} frames")
    
    if len(frames) == 0:
        raise ValueError("No frames were read from the video file!")
        
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Save a sequence of image frames as a video file with improved error handling.
    """
    try:
        # Normalize path
        output_video_path = os.path.normpath(output_video_path)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        if not output_video_frames:
            raise ValueError("The output video frames list is empty!")

        # Initialize VideoWriter with codec, frame rate, and frame dimensions
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Changed from XVID to MJPG
        frame_height, frame_width = output_video_frames[0].shape[:2]
        
        print(f"Preparing to write video:")
        print(f"- Output path: {output_video_path}")
        print(f"- Frame dimensions: {frame_width}x{frame_height}")
        print(f"- Number of frames: {len(output_video_frames)}")
        
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (frame_width, frame_height))
        if not out.isOpened():
            raise IOError(f"Failed to create output video file: {output_video_path}")

        # Write frames to the video
        for i, frame in enumerate(output_video_frames, 1):
            out.write(frame)
            if i % 100 == 0:
                print(f"Wrote {i} frames...")

        # Release VideoWriter
        out.release()
        print(f"Successfully saved video to: {output_video_path}")

    except Exception as e:
        print(f"Error saving video: {str(e)}")
        raise