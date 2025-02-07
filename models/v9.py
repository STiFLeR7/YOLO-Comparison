import torch
import builtins

# Override torch.load to bypass weights_only restriction
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Ensure full model loading
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Load YOLOv9 model
model = torch.hub.load('WongKinYiu/yolov9', 'custom', path='D:/YOLO-Comparison/models/yolov9-s.pt', source='github', force_reload=True)

print("YOLOv9 Model Loaded Successfully!")
