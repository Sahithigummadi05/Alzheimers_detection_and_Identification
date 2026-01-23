Overview
This project aims to detect and classify Alzheimer’s Disease using structural MRI scans from the OASIS (Open Access Series of Imaging Studies) dataset. By utilizing a 3D Convolutional Neural Network (CNN) built with TensorFlow and Keras, the model learns to identify patterns in brain atrophy—specifically in gray matter—to distinguish between healthy brains and those showing signs of cognitive impairment.

Dataset
Source: OASIS-1 VBM (Voxel-Based Morphometry) dataset.

Subjects: 416 subjects.

Target Label: Clinical Dementia Rating (CDR).

Preprocessing:

MRI images are smoothed using a Gaussian filter.

Individual slices are resized to 50x50 pixels.

The final input volume for each subject is 50x50x91 (X, Y, and slice count).

Model Architecture
The project employs a 3D Convolutional Neural Network to capture spatial information across the three dimensions of the brain scans:

Convolutional Layer 1: 3x3x3 filter, 32 feature maps, followed by ReLU activation and 3D Max Pooling.

Convolutional Layer 2: 3x3x3 filter, 64 feature maps, followed by ReLU activation and 3D Max Pooling.

Fully Connected Layer: 1024 neurons with a Dropout rate of 0.8 to prevent overfitting.

Output Layer: Softmax activation for 2-class classification (e.g., Non-Demented vs. Demented).

File Descriptions
CNN.py
The core script for data ingestion and model training. It handles:

Fetching data using nilearn.

Converting CDR scores into categorical labels.

The TensorFlow training loop, which logs loss and accuracy for each of the 1000 epochs to output.txt.

accuracy.py
A utility script for post-training analysis and visualization. It performs the following:

Parses out.txt to extract epoch-wise loss and accuracy metrics.

Unhappiness Score: Calculates a custom metric based on the average loss across training iterations.

Visualizations: Generates plots for the loss function score and accuracy percentage over time, saving them as loss.png and accuracy.png.

Requirements
Python 3.x

TensorFlow / Keras

Nilearn

OpenCV (cv2)

Matplotlib

Scikit-learn

Usage
Train the model: Run python CNN.py. This will download the dataset (if not present), train the 3D CNN, and save the metrics to a text file.

Generate Plots: Run python accuracy.py to process the training logs and view the performance graphs.
