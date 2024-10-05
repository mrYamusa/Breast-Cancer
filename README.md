# Breast Ultrasound Image Classification

## Overview
This project implements a convolutional neural network (CNN) for classifying breast ultrasound images into three categories: benign, malignant, and normal. The model leverages transfer learning using a pre-trained EfficientNet architecture from TensorFlow Hub.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Breast cancer is a significant health issue worldwide, and early detection is crucial for effective treatment. This project aims to automate the classification of breast ultrasound images, aiding in the diagnostic process.

## Dataset
The dataset used for this project is the Breast Ultrasound Images Dataset, which contains images labeled as:
- **Benign**
- **Malignant**
- **Normal**

You can download the dataset from [this link](/kaggle/input/breast-ultrasound-images-dataset).

## Technologies Used
- Python
- TensorFlow
- Keras
- TensorFlow Hub
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-ultrasound-image-classification.git
   cd breast-ultrasound-image-classification
   ```

2. Install the required packages:
   ```bash
   tensorflow==2.12.0
   tensorflow-hub==0.12.0
   numpy==1.23.4
   pandas==1.5.1
   scikit-learn==1.1.3
   matplotlib==3.6.0
   tf-keras==0.1.0  
   ```

## Usage
To run the image classification, make sure to adjust any file paths in the script to point to the dataset.

## Model Training
The model is trained using the following parameters:
- Batch Size: 32
- Number of Epochs: 100
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

TensorBoard was used for monitoring the training process. To view TensorBoard logs, run:
```bash
tensorboard --logdir logs/
```

## Results
After training, the model achieves an accuracy of approximately 88% on the validation set. Predictions can be visualized alongside true labels for evaluation.

### Example Predictions
Here are some example predictions from the model:

![Example Prediction](https://github.com/user-attachments/assets/609e63b2-5c1c-450d-9ae1-8ff50ea03953)

## Conclusion
This project demonstrates the potential of using deep learning for medical image classification. Future work could include exploring data augmentation techniques, hyperparameter tuning, and evaluating on a separate test set for better generalization.
