
# YOLO-Comparison: YOLOv8 vs YOLOv9 for Vehicle Detection 🚗📊

Welcome to **YOLO-Comparison**, a comprehensive analysis of two state-of-the-art object detection models—**YOLOv8** and **YOLOv9**—applied to vehicle detection using the *Top-View Vehicle Detection Dataset* from Kaggle. This repository showcases training, evaluation, and performance comparison between these models based on various metrics like accuracy, mAP, precision, F1-score, and inference speed.

---

## 🚀 **Project Overview**

In this project, we aim to:
- Train YOLOv8 and YOLOv9 on the *Top-View Vehicle Detection Dataset*.
- Evaluate and compare their performance using metrics like mAP, precision, recall, and training time.
- Visualize bounding box predictions to highlight qualitative differences.
- Provide a detailed conclusion based on our findings.

---


---

## 🏗 **Installation and Setup**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/STiFLeR7/YOLO-Comparison.git
   cd YOLO-Comparison
2. **Set Up the Environment**
Install the required dependencies:

     
    pip install -r requirements.txt

3. **Dataset Setup**
Ensure the Top-View Vehicle Detection Dataset is placed under ```dataset/```.

4. **Run Training and Evaluation**
For YOLO-v8 ```python v8_evaluate.py```

For YOLO-v9 ```python v9_evaluate.py```

5. **Calculate mAP Scores**
For YOLO-v8 ```python v8_map.py```

For YOLO-v9 ```python v9_map.py```

## 📊 Results Overview

YOLOv8 mAP@0.5 : 0.9726

YOLOv9 mAP@0.5 : 0.9781

For detailed performance metrics and visualizations, refer to [Comparisons.md](YOLO-Comparison/comparisons.md).


## 📸 Sample Visualizations
| Model A (YOLOv8)                        | Model B (YOLOv9)                        |
| --------------------------------------- | --------------------------------------- |
| ![image](https://github.com/user-attachments/assets/520e0df9-ede3-4927-b49c-fde76326547c) | ![image](https://github.com/user-attachments/assets/5ffced6a-d683-439e-bd69-43b486f8998c)|


## 📚 Contributing
We welcome contributions to improve the project! Feel free to fork the repository, submit issues, or open pull requests.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

## ✨ Acknowledgments
Top-View Vehicle Detection Dataset from Kaggle https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset

Ultralytics YOLO for open-source model implementations 
https://docs.ultralytics.com/
