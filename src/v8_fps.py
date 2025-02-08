from ultralytics import YOLO
import time
import os

def measure_fps(model_path, image_folder):
    model = YOLO(model_path)
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    total_images = len(image_files)
    
    start_time = time.time()
    for img in image_files:
        model(img)
    end_time = time.time()

    total_time = end_time - start_time
    fps = total_images / total_time

    print(f"Processed {total_images} images in {total_time:.2f} seconds.")
    print(f"FPS: {fps:.2f}")

    return fps

if __name__ == "__main__":
    model_path = 'results/v8/train/weights/best.pt'
    image_folder = 'D:/YOLO-Comparison/dataset/Vehicle_Detection_Image_Dataset/train/images'  # Use the same dataset folder for both models
    measure_fps(model_path, image_folder)
