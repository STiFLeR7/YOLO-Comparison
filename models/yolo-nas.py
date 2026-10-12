from super_gradients.training import models
model = models.get("yolo_nas_s", pretrained=True)  # Loads small YOLO-NAS
model.info()
