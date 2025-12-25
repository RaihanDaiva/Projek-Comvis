import os
# --- CONFIG LINUX/WAYLAND ---
os.environ["QT_QPA_PLATFORM"] = "xcb" 

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models.video import r3d_18
from collections import deque
import threading
import time
from ultralytics import YOLO

# --- KONFIGURASI INPUT ---
VIDEO_PATH = "/run/media/han/HDD RAIHAN/Real Dataset Cheating Comvis/Preprocessing/dataset_clips_split/test/not_cheating/not_cheating_00145.mp4"

# Konfigurasi Model
MODEL_PATH = "/home/han/Documents/Kuliah/S5/Comvis/Projek/r3d_18_cheating_best_without_TL_v2.pth"
# MODEL_PATH = "/home/han/Documents/Kuliah/S5/Comvis/Projek/r3d_18_cheating_best_v3.pth"
CLIP_LEN = 32
CLASSES = ["cheating", "not_cheating"]

# --- KONFIGURASI DELAY (YANG DITAMBAHKAN) ---
# Semakin besar angkanya, video semakin lambat.
# 0.03 = ~30 FPS (Standar)
# 0.1  = Lambat (Memberi waktu model R3D untuk berpikir)
PLAYBACK_DELAY = 0.05  

# --- GLOBAL VARIABLES (MULTI-PERSON) ---
track_data = {} 
keep_running = True
lock = threading.Lock() 

# --- SETUP DEVICE ---
device = "cuda"
print(f"Device yang dipilih: {device}")

# 1. Load YOLO
print("Loading YOLOv8...")
# Pastikan path ini benar di laptop Anda
yolo_model = YOLO("/home/han/Documents/Kuliah/S5/Comvis/Dataset Video/yolov8n.pt")

# 2. Load R3D-18
print("Loading R3D-18 Action Model...")
action_model = r3d_18(weights=None)
action_model.fc = nn.Linear(action_model.fc.in_features, 2)

try:
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: File model '{MODEL_PATH}' tidak ditemukan!")
        exit()

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

# --- BACKGROUND THREAD ---
def action_recognition_worker():
    global track_data, keep_running
    print("🧠 Multi-Person Analysis Thread Berjalan...")
    
    while keep_running:
        with lock:
            current_ids = list(track_data.keys())

        if not current_ids:
            time.sleep(0.1)
            continue

        processed_any = False

        for track_id in current_ids:
            with lock:
                if track_id not in track_data: continue
                user_buffer = list(track_data[track_id]['buffer'])
            
            if len(user_buffer) == CLIP_LEN:
                try:
                    processed_any = True
                    np_clip = np.array(user_buffer)
                    np_clip = np_clip.transpose(3, 0, 1, 2) / 255.0
                    
                    tensor_clip = torch.tensor(np_clip, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = action_model(tensor_clip)
                        probs = torch.softmax(output, dim=1)
                        score, pred_idx = torch.max(probs, 1)
                        
                        label = CLASSES[pred_idx.item()]
                        conf = score.item() * 100

                    with lock:
                        if track_id in track_data:
                            if label == "cheating":
                                track_data[track_id]['label'] = "CHEATING!"
                                track_data[track_id]['color'] = (0, 0, 255) 
                            else:
                                track_data[track_id]['label'] = "NORMAL"
                                track_data[track_id]['color'] = (0, 255, 0) 
                            
                            track_data[track_id]['conf'] = f"{conf:.1f}%"

                except Exception as e:
                    print(f"Error pada ID {track_id}: {e}")

        # Delay kecil di thread worker agar tidak memakan CPU 100%
        if processed_any:
            time.sleep(0.1) 
        else:
            time.sleep(0.05)

# --- HELPER: Crop ---
def crop_person(frame, box):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# --- MAIN LOOP ---
def run_multi_person_system():
    global keep_running, track_data
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: File video tidak ditemukan di: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka file video.")
        return

    t = threading.Thread(target=action_recognition_worker)
    t.daemon = True
    t.start()

    print(f"🎥 Memutar Video: {VIDEO_PATH}")
    print("Tekan 'q' untuk berhenti.")
    
    fps_start_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Video selesai.")
            keep_running = False
            break

        # --- 1. YOLO TRACKING ---
        results = yolo_model.track(frame, classes=[0], persist=True, verbose=False, device='cpu', tracker="bytetrack.yaml")
        
        # --- 2. UPDATE BUFFER ---
        current_frame_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                current_frame_ids.append(track_id)
                
                with lock:
                    if track_id not in track_data:
                        track_data[track_id] = {
                            'buffer': deque(maxlen=CLIP_LEN),
                            'label': 'Analyzing...',
                            'conf': '0%',
                            'color': (255, 255, 0) 
                        }

                person_img, (bx1, by1, bx2, by2) = crop_person(frame, box)

                if person_img.size > 0:
                    r3d_input = cv2.resize(person_img, (112, 112))
                    r3d_input = cv2.cvtColor(r3d_input, cv2.COLOR_BGR2RGB)
                    
                    with lock:
                        track_data[track_id]['buffer'].append(r3d_input)
                        
                        # Ambil info untuk display
                        lbl = track_data[track_id]['label']
                        cnf = track_data[track_id]['conf']
                        clr = track_data[track_id]['color']

                    # --- VISUALISASI ---
                    
                    # Hanya gambar jika label bukan 'Analyzing...', 
                    # atau jika Anda ingin melihat kotak kuning (analyzing), hapus "if lbl ==..." ini.
                    if lbl == 'Analyzing...':
                        # Opsional: Tetap gambar kotak kuning agar tahu orang terdeteksi
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                        cv2.putText(frame, "Analyzing...", (bx1, by1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), clr, 2)
                        text = f"ID:{track_id} | {lbl} {cnf}"
                        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(frame, (bx1, by1 - 20), (bx1 + t_size[0], by1), clr, -1)
                        cv2.putText(frame, text, (bx1, by1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 3. UI DASHBOARD ---
        frame_count += 1
        if time.time() - fps_start_time > 1:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()

        cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)} | People: {len(current_frame_ids)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Multi-Person Cheating Detection", frame)

        # --- TAMBAHAN DELAY ---
        # Ini menahan loop agar video tidak diputar terlalu cepat
        time.sleep(PLAYBACK_DELAY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            keep_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")

if __name__ == "__main__":
    run_multi_person_system()