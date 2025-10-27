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

<img width="826" height="502" alt="Screenshot 2025-10-27 110647" src="https://github.com/user-attachments/assets/995a5a96-8eba-4031-a51c-ff9632ca97b4" />

<img width="826" height="502" alt="Screenshot 2025-10-27 110647" src="https://github.com/user-attachments/assets/ba3152cd-1592-4e67-9a3f-fdb817c96cf4" />

