import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# === Paths: update to your models ===
face_proto = r"C:\Users\akhil\Downloads\opencv_face_detector.pbtxt"
face_model = r"C:\Users\akhil\Downloads\opencv_face_detector_uint8.pb"
age_proto = r"C:\Users\akhil\Downloads\age_deploy.prototxt"
age_model = r"C:\Users\akhil\Downloads\age_net.caffemodel"

# === Load models ===
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

MODEL_MEAN_VALUES = (78.4263, 87.7689, 114.8958)
AGE_MIDPOINTS = np.array([1,5,10,17.5,28.5,40.5,50.5,80.0], dtype=float)

# Helper: CLAHE preprocessing
def preprocess_face_clahe(face_bgr):
    h, w = face_bgr.shape[:2]
    if min(h,w) < 80:
        scale = 80.0 / min(h,w)
        face_bgr = cv2.resize(face_bgr, (0,0), fx=scale, fy=scale)
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    ycrcb[:,:,0] = y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Predict age
def age_predict_from_face(face_bgr, do_flip=True):
    face_proc = preprocess_face_clahe(face_bgr)
    face_resized = cv2.resize(face_proc, (227,227))
    blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()[0]
    if do_flip:
        face_flipped = cv2.flip(face_resized, 1)
        blob_f = cv2.dnn.blobFromImage(face_flipped, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob_f)
        preds_f = age_net.forward()[0]
        preds = (preds + preds_f) / 2.0
    probs = preds.astype(np.float32)
    probs /= (probs.sum() + 1e-12)
    expected_age = float(np.dot(probs, AGE_MIDPOINTS))
    return int(round(expected_age))

# Main GUI Class
class AgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 AI Age Predictor")
        self.root.geometry("1100x750")
        self.root.resizable(False, False)

        # --- Gradient Canvas ---
        self.canvas_bg = tk.Canvas(root, width=1100, height=750, highlightthickness=0)
        self.canvas_bg.pack(fill="both", expand=True)
        self.gradient_colors = ["#74EBD5", "#ACB6E5", "#FFDEE9", "#B5FFFC"]
        self.color_index = 0
        self._animate_gradient()

        # --- Control Frame ---
        self.control_frame = tk.Frame(self.root, bg="#ffffff", highlightbackground="#888", highlightthickness=1)
        self.control_frame.place(x=20, y=18, width=1060, height=68)

        # Buttons
        self.create_button("📁 Upload Image", self.upload_image, "#4CAF50", 30, 20)
        self.create_button("🎥 Start Webcam", self.start_webcam, "#2196F3", 240, 20)
        self.create_button("🛑 Stop Webcam", self.stop_webcam, "#E91E63", 450, 20)

        # Age Label
        self.label_info = tk.Label(self.control_frame, text="Age: --", font=("Arial Rounded MT Bold", 16),
                                   bg="#ffffff", fg="#333")
        self.label_info.place(x=850, y=18, width=150, height=35)

        # Image area
        self.image_canvas = tk.Label(root, bd=6, relief="ridge", bg="#F7F7F7")
        self.image_canvas.place(x=100, y=120, width=900, height=600)

        self.cap = None
        self.age_history = []

    # Gradient animation
    def _animate_gradient(self):
        c1 = self.gradient_colors[self.color_index % len(self.gradient_colors)]
        c2 = self.gradient_colors[(self.color_index + 1) % len(self.gradient_colors)]
        self.draw_gradient(self.canvas_bg, c1, c2)
        self.color_index += 1
        self.root.after(2000, self._animate_gradient)

    def draw_gradient(self, canvas, color1, color2):
        width, height = int(canvas["width"]), int(canvas["height"])
        r1,g1,b1 = self.hex_to_rgb(color1)
        r2,g2,b2 = self.hex_to_rgb(color2)
        for i in range(height):
            r = int(r1 + (r2 - r1)*(i/height))
            g = int(g1 + (g2 - g1)*(i/height))
            b = int(b1 + (b2 - b1)*(i/height))
            color = f"#{r:02x}{g:02x}{b:02x}"
            canvas.create_line(0, i, width, i, fill=color)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

    def create_button(self, text, command, color, x, y):
        btn = tk.Button(self.control_frame, text=text, command=command,
                        bg=color, fg="white", font=("Arial",13,"bold"),
                        relief="flat", cursor="hand2", activebackground="#333")
        btn.place(x=x, y=y, width=180, height=35)
        return btn

    # Process frame (with smoothing)
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
        face_net.setInput(blob)
        detections = face_net.forward()
        smoothed_age = "--"
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf < 0.6: continue
            x1, y1 = int(detections[0,0,i,3]*w), int(detections[0,0,i,4]*h)
            x2, y2 = int(detections[0,0,i,5]*w), int(detections[0,0,i,6]*h)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            age = age_predict_from_face(face_crop)
            self.age_history.append(age)
            if len(self.age_history) > 10:
                self.age_history.pop(0)
            smoothed_age = int(np.mean(self.age_history))
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"Age: {smoothed_age}y",(x1+2,y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        if smoothed_age != "--":
            self.label_info.config(text=f"Age: {smoothed_age}y")
        return frame

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return
        img = cv2.imread(file_path)
        if img is None: return
        output = self.process_frame(img)
        self.show_on_canvas(output)

    def start_webcam(self):
        if self.cap is not None: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            return
        self.update_webcam_loop()

    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_webcam_loop(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: return
        frame = self.process_frame(frame)
        self.show_on_canvas(frame)
        self.root.after(80, self.update_webcam_loop)

    def show_on_canvas(self, frame):
        frame_resized = self.resize_for_canvas(frame)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_canvas.imgtk = img_tk
        self.image_canvas.config(image=img_tk)

    def resize_for_canvas(self, frame):
        h, w = frame.shape[:2]
        scale = min(900/w, 600/h)
        return cv2.resize(frame, (int(w*scale), int(h*scale)))

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = AgeApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_webcam(), root.destroy()))
    root.mainloop()
