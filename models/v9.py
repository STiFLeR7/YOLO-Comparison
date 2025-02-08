import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel  # Import DetectionModel from Ultralytics

# Allowlist DetectionModel for unpickling
add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

# Load YOLOv9 model
model = torch.hub.load(
    'WongKinYiu/yolov9',
    'custom',
    path='D:/YOLO-Comparison/yolo11n.pt',
    source='github',
    force_reload=True
)

print("YOLOv9 Model Loaded Successfully!")
