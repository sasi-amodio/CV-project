# Human Action Recognition in Still Images Using Machine and Deep Learning Models

This project aims to recognize human actions in still images using two different approaches: one based on traditional machine learning with an SVM classifier, and the other utilizing a pre-trained deep learning model (GoogleNet) with transfer learning.

## Overview

The dataset consists of images of seven human actions:  
1. Interacting with the computer  
2. Photographing  
3. Playing an instrument  
4. Riding a bike  
5. Riding a horse  
6. Running  
7. Walking

Two models were developed:
1. **Action_Recognition_handcrafted.m**: Using a pre-processing stage with SVM (Support Vector Machine).
2. **Action_Recognition_DL.m**: Based on transfer learning with the pre-trained GoogleNet network.

## SVM-Based Classifier with Pre-Processing

In this system, images are pre-processed using the following steps:
1. **Smoothing**: Applying Gaussian smoothing.
2. **Resizing**: Resizing the images to 256x256 pixels.
3. **Histogram Equalization**: Normalizing illumination effects.

Features are extracted using **Local Binary Patterns (LBP)**, and an exhaustive search is performed using grid search to find the best hyperparameters for the SVM classifier.

### Hyperparameter Optimization
- **Cell Size**: Defines the number of cells for feature extraction.
- **SVM Kernel**: Types of SVM kernel functions (`poly`, `linear`, `rbf`).
- **SVM C**: Regularization parameter that controls the trade-off between achieving a low training error and a low testing error.


The best model achieved the following performance:
- **Training Accuracy**: 100%
- **Validation Accuracy**: 42.54%
- **Test Accuracy**: 40.22%


## GoogleNet-Based Classifier with Tranfer Learning

For the second system, GoogleNet (a pre-trained deep learning model) was used and fine-tuned for the action recognition task. The last layers of GoogleNet was replaced and the first 110 layers was frozen to prevent overfitting on the small dataset.

The best model achieved the following performance:
- Training Accuracy: 98.12%
- Validation Accuracy: 84.29%
- Test Accuracy: 82.86%
