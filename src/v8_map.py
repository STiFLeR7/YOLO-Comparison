from ultralytics import YOLO
from prettytable import PrettyTable

def evaluate_map(model_path, dataset_yaml):
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, save=True)

    # Extract available mAP values
    map_50 = results.box.map50  # mAP at IoU=0.5
    map_50_95 = results.box.map  # mAP averaged over IoU=0.5 to 0.95

    # Try to extract mAP@0.75, handle if not available
    try:
        map_75 = results.box.maps[5]  # mAP at IoU=0.75 (6th value)
    except IndexError:
        map_75 = 'N/A'  # Not available if the array is too short

    # Create and display table
    table = PrettyTable()
    table.field_names = ["IoU Threshold", "mAP Value"]
    table.add_row(["0.5", f"{map_50:.4f}"])
    table.add_row(["0.75", f"{map_75}" if map_75 != 'N/A' else "N/A"])
    table.add_row(["0.5:0.95", f"{map_50_95:.4f}"])
    
    print("\nModel Evaluation Results (mAP at Different IoU Thresholds):")
    print(table)

if __name__ == '__main__':
    model_path = 'D:/YOLO-Comparison/results/v8/train/weights/best.pt'  # Update with your model path
    dataset_yaml = 'D:/YOLO-Comparison/dataset/Vehicle_Detection_Image_Dataset/data.yaml'    # Update with your dataset YAML
    evaluate_map(model_path, dataset_yaml)
