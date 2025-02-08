from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Path to your Top-View Vehicle dataset images
dataset_path = r'D:/YOLO-Comparison/dataset/Vehicle_Detection_Image_Dataset'  # Update with your extracted dataset path

# Get a list of images from the dataset
image_files = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpg', '.png'))]

# Perform inference on each image
for img_path in image_files[:5]:  # Limit to first 5 images for quick testing
    results = model.predict(img_path)
    results[0].show()  # Display image with bounding boxes
