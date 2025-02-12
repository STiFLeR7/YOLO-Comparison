import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the sample video
video_path = 'dataset/Vehicle_Detection_Image_Dataset/sample_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_output = cap.get(cv2.CAP_PROP_FPS)

# Define VideoWriter to save the output
out = cv2.VideoWriter('results/v8_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_output, (frame_width, frame_height))

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLOv8 inference on the frame
    results = model(frame)
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    # Increment frame count
    frame_count += 1

# Calculate FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time

# Output FPS result as an integer
print(f"YOLOv8 Inference FPS: {int(fps)}")

# Release resources
cap.release()
out.release()
