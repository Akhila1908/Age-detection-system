# Age-detection-system

A real-time Age Estimation System built using MediaPipe and OpenCV Deep Learning (Caffe).
This project predicts the approximate age of a person in real time from a webcam feed or an uploaded image.
It features a modern Tkinter GUI, and a smooth averaging system for stable age prediction across frames.

# Features

🧍 Detects and tracks faces using MediaPipe

🧠 Predicts Age and Gender using OpenCV deep learning models

📷 Works with both live webcam feed and uploaded images

📊 Smooths predictions over multiple frames for stability

🖥️ Interactive Tkinter GUI with buttons for image upload and live capture

✅ Real-time results displayed on screen with bounding boxes

✅ Completely offline — no cloud dependencies

# Key Technologies

MediaPipe — face detection (fast and lightweight, by Google)

OpenCV DNN (Caffe) — for age prediction using the pre-trained age_net.caffemodel

Tkinter GUI — simple interface for upload/start/stop webcam

NumPy — for averaging and smoothing age predictions

# How it works

Face Detection:
MediaPipe detects face bounding boxes in images or video frames.

Face Cropping:
The detected region is cropped and resized to the required model input size (227×227).

Age Prediction:
The pre-trained AgeNet model predicts the probability distribution across 8 predefined age ranges.
The project converts these ranges to expected (mean) age for easier readability.

Result Display:
The predicted age is displayed on the image (and in the GUI label).
For webcam mode, results are averaged across the last few frames for smooth prediction.

# Demo Screenshots

<img width="826" height="502" alt="Screenshot 2025-10-27 110647" src="https://github.com/user-attachments/assets/be265def-45c8-4db4-ae6b-9265cc828871" />

<img width="820" height="500" alt="Screenshot 2025-10-27 111328" src="https://github.com/user-attachments/assets/87b7cb53-abd7-4006-ae0e-12e69117669d" />

<img width="821" height="499" alt="Screenshot 2025-10-27 111453" src="https://github.com/user-attachments/assets/037ccca8-8452-4ef5-b4e9-88a1d1c205bc" />



