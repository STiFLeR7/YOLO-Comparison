import torch
model = torch.hub.load('WongKinYiu/yolov9', 'custom', path='D:/YOLO-Comparison/models/yolov9-s.pt', source='github')
print("YOLOv9 Model Loaded Successfully!")
