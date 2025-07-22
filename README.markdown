# Face Mask Detection Model

## Overview
This project develops a deep learning model to detect whether a person is wearing a face mask using the MobileNetV2 architecture. The model is trained on a dataset of images and achieves high accuracy in classifying images into two categories: "Mask Detected" and "No Mask Detected". The implementation utilizes TensorFlow and Keras for model building and training.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy
- MobileNetV2 (pre-trained weights)

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Download the dataset and place it in a folder named `Dataset`.  
[https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data)


3. Install the required dependencies:
   ```
   pip install tensorflow keras matplotlib numpy
   ```

## Dataset
- The model is trained on a dataset stored in the `Dataset/` directory.
- Images are resized to 224x224 pixels.
- The dataset is split into training and validation sets using a 80/20 ratio with data augmentation (shear, zoom, and horizontal flip).

## Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet, with top layers removed).
- **Custom Layers**: 
  - GlobalAveragePooling2D
  - Dense layer with 128 units and ReLU activation
  - Output Dense layer with 1 unit and sigmoid activation
- The base model is frozen during training to leverage pre-trained features.

## Training
- **Epochs**: 10
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

### Training and Validation Accuracy
![Training and Validation Accuracy](Images/Training%20&%20Validation%20Accuracy.png)

### Training and Validation Loss
![Training and Validation Loss](Images/Training%20&%20Validation%20Loss.png)

### Performance Metrics
- **Training Accuracy**: 99.57%
- **Validation Accuracy**: 98.87%

## Usage
1. Train the model by running the script:
   ```
   python code.py
   ```
   This will save the trained model as `FaceMask_model.h5` and generate accuracy/loss plots.

2. Perform predictions on new images using the prediction script:
   ```
   python predict.py
   ```
   Replace `Test Images/test1.jpg` with the path to the test image.

### Sample Predictions
- **Mask Detected (100.00% confidence)**
 
 
  ![Mask Detected](Images/Figure_2.png)


- **No Mask Detected (99.99% confidence)** 
 
 
  ![No Mask Detected](Images/Figure_1.png)

## Notes
- Ensure the dataset directory and test images are correctly configured.
- The model may require customization of the `DATASET_DIR` and `img_path` variables based on the local file structure.
- The code includes a warning about custom mask layers requiring configuration overrides, which should be addressed for serialization purposes.