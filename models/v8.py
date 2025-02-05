from super_gradients.training import models

# Load YOLO-NAS model
model = models.get("yolo_nas_s", pretrained=True)

# Run inference on an image
predictions = model.predict("sample.jpg")
predictions.show()
