# Audio-based Parkinson's Disease Classification


This project implements and evaluates two deep learning models, a 2D Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN), for the classification of Parkinson's disease from audio recordings. The goal is to distinguish between healthy individuals and patients with Parkinson's based on their voice.

## Table of Contents

* [File Structure](#file-structure)
* [About the Project](#about-the-project)
* [Key Features](#key-features)
* [Models Implemented](#models-implemented)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Dependencies](#dependencies)

## File Structure

The project is organized as follows. The audio data (`.wav` or `.mp3` files) should be placed in the `healthy` and `patient` subdirectories.

```
Parkinson-s-Disease-Audio-Classification/
│
├── Datasets/
│   └── Audio/
│       ├── healthy/
│       │   ├── healthy_001.wav
│       │   ├── healthy_002.wav
│       │   └── ...
│       │
│       └── patient/
│           ├── patient_001.wav
│           ├── patient_002.wav
│           └── ...
│
├── Audio Classification Models.py
├── requirements.txt
├── audio_cnn_model.pth         (Generated after training)
└── audio_rnn_model.pth         (Generated after training)
```

## About The Project

This repository contains a Python script for a complete pipeline to train and evaluate deep learning models on audio data for Parkinson's disease classification. It includes steps for data loading, feature extraction, audio augmentation, model definition, training with best practices like learning rate scheduling and early stopping, and comprehensive evaluation.

## Key Features

* **Audio Preprocessing**: Extracts features suitable for different neural network architectures:
    * **Mel Spectrograms** for the CNN.
    * **MFCCs** (including delta and delta-delta) for the RNN.
* **Data Augmentation**: Includes time stretching, pitch shifting, and adding noise to improve model generalization.
* **Two Model Architectures**: Implements both a 2D-CNN for image-like representation of audio and a Bidirectional LSTM-based RNN to capture temporal patterns.
* **Comprehensive Training Loop**: Features a robust training function with a learning rate scheduler (`ReduceLROnPlateau`) and early stopping to prevent overfitting.
* **In-depth Evaluation**: Calculates and visualizes key metrics including:
    * Accuracy, Precision, Recall, and F1-Score
    * Area Under the Curve (AUC)
    * Confusion Matrix
    * ROC Curve

## Models Implemented

1.  **2D Convolutional Neural Network (AudioCNN)**: This model treats the Mel spectrogram of an audio file as an image. It uses a series of 2D convolutional and max-pooling layers to learn hierarchical features from the spectrogram for classification.

2.  **Recurrent Neural Network (AudioRNN)**: This model uses a Bidirectional Long Short-Term Memory (LSTM) network to process sequences of MFCC features. By processing the audio data sequentially, it can learn temporal dependencies in speech that might be indicative of Parkinson's disease.

## Getting Started

To get a local copy up and running, follow these simple steps.

1.  Clone the repo:
    ```
    git clone [https://github.com/Promico-Git/Audio-Based-Parkinson-s-Disease-Classification.git](https://github.com/Promico-Git/Audio-Based-Parkinson-s-Disease-Classification.git)
    ```

2.  Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1.  Organize your dataset according to the [File Structure](#file-structure) section above.

2.  Run the script from your terminal:
    ```
    python "Audio Classification Models.py"
    ```
The script will automatically handle the data splitting, training, and evaluation for both models, printing the results and saving the trained model weights (`audio_cnn_model.pth` and `audio_rnn_model.pth`).

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* Librosa
* Matplotlib
* OpenCV-Python
* Scikit-learn
* Glob
