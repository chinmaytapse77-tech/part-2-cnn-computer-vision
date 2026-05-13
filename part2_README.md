# Part 2: Computer Vision Problem Formulation and CNN Prototype

## Overview
This project formulates a **multi-class image classification** problem on a synthetic manufacturing defect dataset and builds a CNN prototype to classify product surface images into four categories: `dent`, `normal`, `scratch`, and `stain`.

## Dataset
- **Source:** [Masai Project Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Images:** 480 total — 120 per class (perfectly balanced)
- **Original Size:** 96×96 RGB PNG | **Resized to:** 64×64 RGB
- **Classes:** `dent`, `normal`, `scratch`, `stain`

## Project Structure
```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── images/
│   ├── dent/
│   ├── normal/
│   ├── scratch/
│   └── stain/
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    ├── confusion_matrix.png
    ├── class_distribution.png
    └── sample_images.png
```

## Tasks Completed

### Task 1: Problem Identification — Image Classification
The dataset has 4 labeled folders with one label per image and no spatial annotations, making this a **multi-class image classification** problem. Object detection or segmentation would require bounding boxes or pixel-level masks, which are absent here.

### Task 2: Dataset Exploration
- 4 classes, 120 images each — perfectly balanced (no imbalance handling needed)
- Images: 96×96 RGB PNG
- Visual inspection shows distinct textures for each defect type

### Task 3: Image Preprocessing
- Resized to 64×64 pixels (consistent CNN input)
- Normalized pixel values to [0, 1]
- Stratified 80/20 train-test split, then 15% of train for validation
- Data augmentation: rotation, shift, horizontal flip, zoom

### Task 4: CNN Architecture
```
Input (64x64x3)
→ Conv2D(32, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
→ Conv2D(64, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
→ Conv2D(128, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
→ Flatten
→ Dense(128, ReLU) → Dropout(0.4)
→ Dense(4, Softmax)
```
- **Loss:** Categorical Cross-Entropy | **Optimizer:** Adam (lr=0.001)

### Task 5: Model Training and Evaluation
- **Test Accuracy:** ~61% (random baseline = 25%)
- `stain` and `normal` classified well; `dent` is the hardest class
- Overfitting observed — training accuracy (~96%) >> validation accuracy
- **Improvement:** Transfer learning (MobileNet/ResNet) would significantly boost performance

### Task 6: CNN Concept Explanation

**What is Convolution?**
A filter (e.g. 3×3) slides across the image computing dot products, generating feature maps that highlight where specific patterns appear. Multiple filters detect edges, textures, and shapes.

**Why is Pooling Used?**
MaxPooling reduces spatial dimensions (faster training, fewer parameters) and introduces translation invariance — a shifted scratch is still recognized as a scratch.

**Why is ReLU Commonly Used?**
ReLU (`max(0, x)`) is cheap to compute, avoids vanishing gradients in deep networks, and creates sparse activations that act as a natural regularizer.

**Why are CNNs Better than Dense Networks for Images?**
Dense networks flatten images (losing spatial structure) and have too many parameters. CNNs preserve 2D spatial relationships via local connectivity and weight sharing, making them far more efficient and effective for visual data.

### Task 7: Business Use Case — Manufacturing Quality Inspection
CNN-based visual inspection systems can be deployed on production lines to classify product surfaces in real time. Benefits include: 1000s of inspections per hour, consistent accuracy without fatigue, automatic rejection of defective items, and full traceability of all decisions. Companies like BMW, Samsung, and Foxconn already use similar systems.

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd part-2-cnn-computer-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset: images/ folder with dent/, normal/, scratch/, stain/ subfolders

# 4. Run notebook
jupyter notebook notebook.ipynb
```
