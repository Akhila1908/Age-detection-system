import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

# === Paths to your Age Model ===
age_proto = r"C:\Users\akhil\Downloads\age_deploy.prototxt"
age_model = r"C:\Users\akhil\Downloads\age_net.caffemodel"

# === Load Age Model ===
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

# === MediaPipe Setup ===
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# === Constants ===
MODEL_MEAN_VALUES = (78.4263, 87.7689, 114.8958)
AGE_MIDPOINTS = np.array([1,5,10,17.5,28.5,40.5,50.5,80.0], dtype=float)

# === Helper Function: Age Prediction ===
def predict_age(face_bgr):
    face_resized = cv2.resize(face_bgr, (227, 227))
    blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()[0]
    preds = preds / (preds.sum() + 1e-12)
    expected_age = float(np.dot(preds, AGE_MIDPOINTS))
    return int(round(expected_age))

# === GUI Application ===
class AgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 AI Age Detector (MediaPipe + OpenCV)")
        self.root.geometry("1100x750")
        self.root.resizable(False, False)

        # --- Control Buttons ---
        self.control_frame = tk.Frame(self.root, bg="#ffffff", highlightbackground="#999", highlightthickness=1)
        self.control_frame.place(x=20, y=20, width=1060, height=60)

        self.create_button("📁 Upload Image", self.upload_image, "#4CAF50", 20, 12)
        self.create_button("🎥 Start Webcam", self.start_webcam, "#2196F3", 240, 12)
        self.create_button("🛑 Stop Webcam", self.stop_webcam, "#E91E63", 460, 12)

        self.label_info = tk.Label(self.control_frame, text="Age: --", font=("Arial Rounded MT Bold", 16),
                                   bg="#ffffff", fg="#333")
        self.label_info.place(x=850, y=12, width=150, height=35)

        # --- Image Canvas ---
        self.image_canvas = tk.Label(root, bd=6, relief="ridge", bg="#F7F7F7")
        self.image_canvas.place(x=100, y=100, width=900, height=620)

        self.cap = None
        self.age_history = []

    def create_button(self, text, command, color, x, y):
        btn = tk.Button(self.control_frame, text=text, command=command,
                        bg=color, fg="white", font=("Arial", 13, "bold"),
                        relief="flat", cursor="hand2", activebackground="#444")
        btn.place(x=x, y=y, width=180, height=35)
        return btn

    # --- Image Upload ---
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            return
        frame = self.process_frame(img)
        self.show_on_canvas(frame)

    # --- Webcam Start ---
    def start_webcam(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            return
        self.update_frame()

    # --- Webcam Stop ---
    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # --- Frame Update Loop ---
    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = self.process_frame(frame)
        self.show_on_canvas(frame)
        self.root.after(80, self.update_frame)

    # --- Face Detection + Age Prediction ---
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        smoothed_age = "--"
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                age = predict_age(face_crop)
                self.age_history.append(age)
                if len(self.age_history) > 10:
                    self.age_history.pop(0)
                smoothed_age = int(np.mean(self.age_history))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {smoothed_age}y", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.label_info.config(text=f"Age: {smoothed_age}y" if smoothed_age != "--" else "Age: --")
        return frame

    # --- Display on Canvas ---
    def show_on_canvas(self, frame):
        frame_resized = self.resize_for_canvas(frame)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_canvas.imgtk = img_tk
        self.image_canvas.config(image=img_tk)

    def resize_for_canvas(self, frame):
        h, w = frame.shape[:2]
        scale = min(900 / w, 600 / h)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = AgeApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_webcam(), root.destroy()))
    root.mainloop()
