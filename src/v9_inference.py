import torch
import cv2
import os
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel  # Allow YOLOv9 to trust this model

# Allowlist DetectionModel for unpickling
add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

# Load YOLOv9 model
model = torch.hub.load(
    'WongKinYiu/yolov9',
    'custom',
    path='D:/YOLO-Comparison/models/yolov9-s.pt',  # Ensure you're using correct YOLOv9 weights
    source='github',
    force_reload=True
)

print("YOLOv9 Model Loaded Successfully!")

# Path to your Top-View Vehicle Detection Dataset
dataset_path = r'D:/YOLO-Comparison/dataset/Vehicle_Detection_Image_Dataset'  # Update path if needed

# Get list of images
image_files = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpg', '.png'))]

# Perform inference on each image
for img_path in image_files[:5]:  # Run inference on first 5 images
    img = cv2.imread(img_path)
    results = model(img)

    # Display the results
    results.show()  # This will pop up the image with bounding boxes
    print(f"Processed: {img_path}")
