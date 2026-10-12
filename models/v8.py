from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Load a small pre-trained YOLOv8 model
model.info()  # Display model details
