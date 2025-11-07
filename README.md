# EE604 - Course Project

# Yoga Pose Classification using Mediapipe and Deep MLP

*A lightweight keypoint-based yoga pose classification system built using Mediapipe and PyTorch.*

---

## Overview

This project implements a **lightweight and accurate yoga pose classification system** using **Mediapipe Pose** for keypoint detection and a **Deep Multi-Layer Perceptron (MLP)** for classification.

Instead of processing raw RGB frames with heavy CNNs, our approach extracts pose landmarks (x,y) and computes geometric features (angles, distances). This reduces compute and data requirements and yields a model suited for real-time edge inference.

---

##  Key Features

-  **Keypoint extraction** with Mediapipe (33 landmarks)
-  **Feature engineering**: normalized coordinates, joint angles, pairwise distances
-  **Compact MLP classifier** with dropout and regularization
-  Low-latency inference — suitable for CPU / mobile deployment
-  Strong results on a 107 - classes of yoga pose.
  
---

## Pipeline Overview

```
Input Image → Mediapipe Pose → Keypoint Extraction → Feature Engineering →
Deep MLP → Pose Prediction
```

<img width="557" height="311" alt="image" src="https://github.com/user-attachments/assets/6366066f-374c-48bc-aaa1-77c8fb5c0a00" />

---

## Model Architecture

| Layer       | Units | Activation | Dropout |
|-------------|-------:|-----------:|--------:|
| Input       | 132   | —          | —       |
| Dense 1     | 256   | ReLU       | 0.3     |
| Dense 2     | 128   | ReLU       | 0.2     |
| Dense 3     | 64    | ReLU       | —       |
| Output      | C     | Softmax    | —       |

- Optimizer: `Adam (lr=1e-3)`  
- Loss: `CrossEntropyLoss`  
- Scheduler: `ReduceLROnPlateau`  
- Batch size: 64, Epochs: 50 (with early stopping)

---

## Dataset

Primary dataset used (example):
- **Yoga-82** (Kaggle): https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

---

## Installation & Setup

1. **Clone the repo**
```bash
git clone https://github.com/kanavsingh22/EE604_Yoga-Pose-Classification.git
cd yoga-pose-classifier
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data**
- Put images in `data/` following the instructions in `src/dataset.py`.
- Run preprocessing (extract Mediapipe keypoints) with:
```bash
python src/extract_keypoints.py --input_dir data/images --output data/keypoints.npy
```

5. **Train**
```bash
python src/train.py --config configs/default.yaml
```

6. **Predict**
```bash
python src/predict.py --model models/best.pth --image examples/tree_pose.jpg
```

---

## Results

| Model                     | Accuracy | F1-score | Params |
|--------------------------:|--------:|---------:|------:|
| ResNet-18 (finetuned)     | 52.2%   | 0.51     | 8M   |
| **Proposed MLP (ours)**   | **71.75%%** | **0.72** | **0.2M** |


---

## Project Structure

```
yoga-pose-classifier/
├── dataset/                                 
├── src/
│   └── main.ipynb
├── requirements.txt
├── README.md
└── yoga_report.tex
```

---

## 🔍 Notes on Reproducibility

- Seed all random generators (`numpy`, `torch`, `random`) in `train.py`.
- Log experiments with `tensorboard` or `weights & biases` for traceability.


---

## References

- Mediapipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose  
- Yoga-82 Dataset: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset  
- PyTorch: https://pytorch.org/  
- scikit-learn: https://scikit-learn.org/

---

## Acknowledgements

Developed for the **EE604 – Machine Learning for Vision** course under supervision of **Dr. Koteswar Rao Jerripothula** at **IIT Kanpur**. Thanks to contributors of Mediapipe, PyTorch, and the Yoga-82 dataset.

---
