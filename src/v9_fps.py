import cv2
import time
from ultralytics import YOLO

# Load YOLOv9 model
model = YOLO('yolo11n.pt')

# Load video
video_path = 'dataset/Vehicle_Detection_Image_Dataset/sample_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps_input = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'results/v9_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model.predict(frame)
    
    # Annotate frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Increment frame count
    frame_count += 1

# Release resources
cap.release()
out.release()
# cv2.destroyAllWindows()  # Commented out to avoid GUI errors

# Calculate FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time

# Output FPS result as an integer
print(f"YOLOv9 Inference FPS: {int(fps)}")
