# Capuchin Audio Detection (Deep Learning)

This project builds a deep learning pipeline to detect Capuchin bird calls from forest audio recordings using Convolutional Neural Networks (CNNs) and spectrogram analysis.

## Overview

The system processes raw audio files, converts them into spectrograms, and trains a CNN model to classify whether a Capuchin call is present. It then performs sliding-window inference on long forest recordings and outputs detection counts in a CSV file.

## Features

- Audio preprocessing with TensorFlow
- Spectrogram generation (STFT)
- CNN-based binary classification
- Sliding window prediction on long recordings
- Post-processing and grouping of detections
- Export of results to `results.csv`

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Google Colab
- Git & GitHub

## How It Works

1. Load and preprocess audio clips
2. Convert audio to spectrograms
3. Train CNN model on labeled clips
4. Run inference on forest recordings
5. Group consecutive detections
6. Export results to CSV

## Output

The final output is a `results.csv` file containing:

- Recording filename
- Number of detected Capuchin calls

## Author

Built as a deep learning audio classification project using TensorFlow.
