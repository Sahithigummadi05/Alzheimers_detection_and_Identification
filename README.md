# Alzheimer's Detection and Identification System

This repository contains a Deep Learning project designed to detect and classify Alzheimer's Disease from MRI scans. The system utilizes a Convolutional Neural Network (CNN) to analyze medical imaging data and provides a user friendly web interface for real time predictions.

##  Project Overview

Early detection of Alzheimer's is crucial for effective treatment. This project automates the classification of MRI images into four stages of dementia. It includes the complete source code for training the model, processing data, and deploying the solution as a web application.

### Classifications
The model is trained to identify the following classes:
1.  **Non-Demented**
2.  **Very Mild Demented**
3.  **Mild Demented**
4.  **Moderate Demented**

---

##  File Descriptions

Here is a detailed breakdown of the files included in this repository:

### 1. `train.py`
This is the core training script.
* **Function:** Loads the dataset, initializes the neural network from `model.py`, and trains it against the MRI data.
* **Key Features:** Handles data augmentation, compiles the model with appropriate optimizers/loss functions, and saves the final trained weights for deployment.

### 2. `model.py`
This file defines the architecture of the Deep Learning model.
* **Architecture:** Implements a Convolutional Neural Network (CNN) specifically tuned for image classification.
* **Layers:** Includes Convolutional layers for feature extraction, Pooling layers for down-sampling, and Dense layers for the final classification.

### 3. `loader.py`
A utility script for data management.
* **Function:** Handles the preprocessing of input images before they are fed into the model.
* **Operations:** Resizes images to the required input shape, normalizes pixel values, and converts images into NumPy arrays compatible with the model.

### 4. `app.py`
The backend application script (Flask/Python).
* **Function:** Serves as the bridge between the web interface and the trained model.
* **Workflow:** Receives an image from the user, passes it to `loader.py` for processing, runs it through the trained model, and returns the prediction to the browser.

### 5. `index.html`
The frontend user interface.
* **Function:** A clean HTML page that allows users to upload MRI scans easily.
* **Features:** Provides a drag-and-drop or file selection area and displays the prediction results dynamically.

---

##  Installation & Usage

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
bash
git clone [https://github.com/Sahithigummadi05/Alzheimers_detection_and_Identification.git](https://github.com/Sahithigummadi05/Alzheimers_detection_and_Identification.git)

2. Install Dependencies
Ensure you have Python installed, then install the required libraries:

Bash
pip install numpy tensorflow keras flask pillow
3. Run the Application
Launch the web server:

Bash
python app.py
The terminal will verify that the server is running.

Open your web browser and go to http://127.0.0.1:5000/.

4. Using the System
Click the Upload button on the web page.

Select an MRI image file from your computer.

View the immediate classification result on the screen.
cd Alzheimers_detection_and_Identification
View the immediate classification result on the screen.
