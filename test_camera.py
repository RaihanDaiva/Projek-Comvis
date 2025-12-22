import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models.video import r3d_18
from collections import deque
import threading
import time
import gc
from ultralytics import YOLO
import os
# --- CONFIG LINUX/WAYLAND ---
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# --- KONFIGURASI ---
CAMERA_SOURCE = 0
MODEL_PATH = "/home/han/Documents/Kuliah/S5/Comvis/Dataset Video/r3d_18_cheating_best_camera_fixed.pth"  # Sesuaikan nama file model Anda
CLIP_LEN = 32
CLASSES = ["cheating", "not_cheating"]

# --- GLOBAL VARIABLES ---
frame_buffer = deque(maxlen=CLIP_LEN)
current_pred_label = "Waiting for Person..."
current_confidence = "0%"
current_color = (100, 100, 100)
keep_running = True
last_detected_box = None

# --- SETUP DEVICE (FORCE CPU) ---
device = "cpu"
print(f"Device yang dipilih: {device}")

# 1. Load YOLO (Deteksi Orang)
print("Loading YOLOv8...")
yolo_model = YOLO("yolov8n.pt")

# 2. Load R3D-18 (Klasifikasi Aksi)
print("Loading R3D-18 Action Model...")
action_model = r3d_18(weights=None)
action_model.fc = nn.Linear(action_model.fc.in_features, 2)

try:
    # Load checkpoint ke CPU
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    new_state = {}
    for k, v in checkpoint.items():
        if k in action_model.state_dict():
            target_shape = action_model.state_dict()[k].shape
            if target_shape != v.shape:
                if v.shape[2] == 1 and target_shape[2] == 3:
                    v = v.repeat(1, 1, 3, 1, 1) / 3.0
            new_state[k] = v
    action_model.load_state_dict(new_state)
    action_model = action_model.to(device)
    action_model.eval()
    print("✅ Model Action R3D siap di CPU!")
except Exception as e:
    print(f"❌ Error loading R3D model: {e}")
    exit()

# --- BACKGROUND THREAD (ANALISIS AKSI) ---
def action_recognition_worker():
    global current_pred_label, current_confidence, current_color, keep_running
    
    print("🧠 Action Recognition Thread Berjalan...")
    
    while keep_running:
        if len(frame_buffer) == CLIP_LEN:
            try:
                clip_data = list(frame_buffer)
                np_clip = np.array(clip_data)
                np_clip = np_clip.transpose(3, 0, 1, 2) / 255.0
                
                # Input ke CPU
                tensor_clip = torch.tensor(np_clip, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = action_model(tensor_clip)
                    probs = torch.softmax(output, dim=1)
                    score, pred_idx = torch.max(probs, 1)
                    
                    label = CLASSES[pred_idx.item()]
                    conf = score.item() * 100

                # Update Global Variables
                if label == "cheating":
                    current_pred_label = "CHEATING DETECTED!"
                    current_color = (0, 0, 255)
                else:
                    current_pred_label = "NORMAL ACTIVITY"
                    current_color = (0, 255, 0)
                
                current_confidence = f"{conf:.1f}%"
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"Error di thread prediksi: {e}")
        else:
            time.sleep(0.1)

# --- HELPER: Crop ---
def crop_person(frame, box):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# --- MAIN THREAD ---
def run_hybrid_system():
    # --- PERBAIKAN DI SINI: Tambahkan current_color, dll ke global ---
    global keep_running, frame_buffer, last_detected_box, current_color, current_pred_label, current_confidence
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("❌ Kamera error.")
        return

    t = threading.Thread(target=action_recognition_worker)
    t.daemon = True
    t.start()

    print("🎥 Sistem Mulai. YOLO & R3D berjalan di CPU.")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Force YOLO ke CPU
        results = yolo_model(frame, classes=[0], verbose=False, conf=0.5, device='cpu')
        
        person_found = False
        largest_box = None
        max_area = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = [x1, y1, x2, y2]
                    person_found = True

        if person_found:
            last_detected_box = largest_box
            person_img, (bx1, by1, bx2, by2) = crop_person(frame, largest_box)
            
            if person_img.size > 0:
                r3d_input = cv2.resize(person_img, (112, 112))
                r3d_input = cv2.cvtColor(r3d_input, cv2.COLOR_BGR2RGB)
                frame_buffer.append(r3d_input)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), current_color, 2)
                label_text = f"{current_pred_label} ({current_confidence})"
                cv2.putText(frame, label_text, (bx1, by1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)
        else:
            # Karena sudah deklarasi global, ini aman sekarang
            current_color = (100, 100, 100)

        # UI Info
        frame_count += 1
        if time.time() - fps_start_time > 1:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()

        cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)} (CPU Mode)", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        status_msg = "Tracking..." if person_found else "No Person"
        cv2.putText(frame, status_msg, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if person_found else (0, 0, 255), 1)

        cv2.imshow("Hybrid Detection (CPU)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            keep_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")

if __name__ == "__main__":
    run_hybrid_system()