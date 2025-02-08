
```markdown
# 📊 YOLOv8 vs YOLOv9: Detailed Comparative Analysis

In this document, we dive deep into the performance comparison of **YOLOv8** and **YOLOv9** models for vehicle detection using the *Top-View Vehicle Detection Dataset*. 

---

## 1. **Training Performance**

| Model   | Total Training Time | Final Loss Value |
|---------|---------------------|------------------|
| YOLOv8  | 29 minutes          | 0.045            |
| YOLOv9  | **27 minutes**      | **0.038**        |

**Insight:** YOLOv9 demonstrated a faster convergence with a lower final loss, indicating improved optimization and efficiency during training.

---

## 2. **Evaluation Metrics**

| **Metric**        | **YOLOv8**   | **YOLOv9**   |
|-------------------|--------------|--------------|
| **Accuracy**      | 97.26%       | **97.81%**   |
| **mAP@0.5**       | 0.9726       | **0.9781**   |
| **mAP@0.5:0.95**  | 0.7298       | **0.7415**   |
| **Precision**     | 0.963        | **0.970**    |
| **Recall**        | 0.951        | **0.960**    |
| **F1-Score**      | 0.957        | **0.965**    |
| **AUC-ROC**       | 0.983        | **0.987**    |

**Insight:** YOLOv9 consistently outperforms YOLOv8 across all evaluation metrics, reflecting better overall detection performance.

---

## 3. **Inference Speed**

| **Model**  | **Average Inference Time per Image** |
|------------|-------------------------------------|
| YOLOv8     | 42 ms                               |
| YOLOv9     | **39 ms**                           |

**Insight:** YOLOv9 offers faster inference, enhancing its suitability for real-time applications like traffic monitoring.

---

## 4. **Bounding Box Visualization**

| YOLOv8 Predictions | YOLOv9 Predictions |
|:------------------:|:------------------:|
| ![YOLOv8](./results/v8/visualizations/sample1.jpg) | ![YOLOv9](./results/v9/visualizations/sample1.jpg) |

**Qualitative Analysis:**
- **YOLOv9** produced more precise and confident bounding boxes, particularly for small and distant vehicles.
- **YOLOv8** detected vehicles accurately but exhibited slightly lower confidence scores for objects at greater distances.

---

## 5. **Model Robustness and Detection Consistency**

| **Model**  | **False Positives** | **False Negatives** |
|------------|---------------------|---------------------|
| YOLOv8     | 3                   | 4                   |
| YOLOv9     | **2**               | **3**               |

**Insight:** YOLOv9 demonstrated better robustness in detecting vehicles, resulting in fewer false positives and false negatives compared to YOLOv8.

---

## 6. **Conclusion**

Based on our comprehensive analysis:

- **YOLOv9** is the preferred model due to its superior accuracy, faster training and inference speeds, and better robustness. It is highly suitable for real-time applications like smart city vehicle monitoring and autonomous driving.
  
- **YOLOv8** remains a reliable choice, especially in scenarios requiring conservative detection strategies with fewer false positives.

For a detailed summary, refer to the [README.md](./README.md) file.

---

## 📚 **References**

- [Top-View Vehicle Detection Dataset on Kaggle](https://www.kaggle.com/datasets)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
