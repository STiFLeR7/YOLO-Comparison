from ultralytics import YOLO
import os

def visualize_predictions(model_path, image_dir, output_dir):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Run inference on the test images
    results = model.predict(source=image_dir, save=True, save_txt=False, conf=0.25)

    # Move results to output directory
    os.makedirs(output_dir, exist_ok=True)
    for result in results:
        image_name = os.path.basename(result.path)
        output_path = os.path.join(output_dir, image_name)
        result.save(output_path)
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    model_path = 'results/v8/train/weights/best.pt'
    image_dir = 'dataset/Vehicle_Detection_Image_Dataset/valid/images'
    output_dir = 'results/v8/visualizations'
    visualize_predictions(model_path, image_dir, output_dir)
