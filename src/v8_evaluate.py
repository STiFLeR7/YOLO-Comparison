from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import time


def plot_loss_graph(metrics, save_path):
    """Plot training loss graph."""
    if 'train/box_loss' in metrics and 'train/cls_loss' in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['epoch'], metrics['train/box_loss'], label='Box Loss')
        plt.plot(metrics['epoch'], metrics['train/cls_loss'], label='Class Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    else:
        print("Loss metrics not found in the results.")


def save_metrics_as_txt(metrics, save_path):
    """Save metrics to a text file."""
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {str(value)}\n")
    print(f"Metrics saved to {save_path}")


def evaluate_model():
    """Train and evaluate YOLOv8 model."""
    start_time = time.time()

    # Load and train the YOLOv8 model
    model = YOLO('yolov8n.pt')
    results = model.train(data='D:/YOLO-Comparison/dataset/Vehicle_Detection_Image_Dataset/data.yaml', epochs=50, imgsz=640, project='results/v8')

    end_time = time.time()
    training_time = end_time - start_time

    # Ensure directory exists
    os.makedirs('results/v8', exist_ok=True)

    # Extract and save metrics
    metrics = results if isinstance(results, dict) else vars(results)

    save_metrics_as_txt(metrics, 'results/v8/metrics.txt')

    print(f"Training completed in {training_time / 3600:.3f} hours.")
    print(f"Final Model Accuracy: {metrics.get('metrics/precision', 'N/A')}")

    # Plot and save the loss graph
    plot_loss_graph(metrics, 'results/v8/loss_graph.png')


if __name__ == '__main__':
    evaluate_model()
