import os
import json
import threading
import time
import shutil # مسؤولة عن حذف المجلدات
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify

import cv2
import numpy as np
from PIL import Image as PILImage
import imagehash
import yaml
import torch
from ultralytics import YOLO

app = Flask(__name__)

# --------------------------
# Settings & Initialization
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR = Path.cwd() / "project_data"
TRAIN_DIR = PROJECT_DIR / "train"
HASH_DB = PROJECT_DIR / "hash_db.json"
WEIGHTS_PATH = PROJECT_DIR / "weights"
INITIAL_WEIGHTS = "yolov8n.pt"

HASH_SIZE = 64
SIMILARITY_THRESHOLD = 0.85
HAMMING_THRESHOLD = int((1 - SIMILARITY_THRESHOLD) * HASH_SIZE + 0.5)
TARGET_IMAGES_FOR_TRAIN = 500

ANIMALS = [
    'cat', 'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'sheep', 'lion', 'tiger', 'monkey', 'wolf', 'deer', 'pig', 'goat', 
    'camel', 'donkey'
]

new_animals_file = PROJECT_DIR / "new_animals.txt"
if new_animals_file.exists():
    with open(new_animals_file, "r") as f:
        new_animals = [line.strip() for line in f if line.strip()]
    ANIMALS.extend(new_animals)
    ANIMALS = list(set(ANIMALS))

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(WEIGHTS_PATH, exist_ok=True)
if not HASH_DB.exists():
    with open(HASH_DB, "w") as f:
        json.dump({}, f)

# Global variables
global_model = None
is_training = False
training_status = "Ready"

def init_model():
    global global_model
    if WEIGHTS_PATH.exists():
        pt_files = list(WEIGHTS_PATH.glob("*.pt"))
        if pt_files:
            latest_weights = max(pt_files, key=os.path.getmtime)
            print(f"Startup: Loaded custom weights: {latest_weights}")
            global_model = YOLO(str(latest_weights))
            return
    print(f"Startup: Loaded base weights: {INITIAL_WEIGHTS}")
    global_model = YOLO(INITIAL_WEIGHTS)

init_model()

# --------------------------
# Database & Hash Helpers
# --------------------------
def load_hash_db():
    with open(HASH_DB, "r") as f: return json.load(f)

def save_hash_db(db):
    with open(HASH_DB, "w") as f: json.dump(db, f, indent=2)

def image_hash_for_pil(pil_img): return imagehash.average_hash(pil_img)

def hamming_distance_from_str(h1_str, h2_str):
    return imagehash.hex_to_hash(h1_str) - imagehash.hex_to_hash(h2_str)

def ensure_label_dirs(label): 
    img_dir = TRAIN_DIR / label / "images"
    lbl_dir = TRAIN_DIR / label / "labels"
    coord_dir = TRAIN_DIR / label / "coords"
    for p in (img_dir, lbl_dir, coord_dir): os.makedirs(p, exist_ok=True)
    return img_dir, lbl_dir, coord_dir

def count_images_for_label(label):
    img_dir, _, _ = ensure_label_dirs(label)
    return len(list(img_dir.glob("*.jpg")))

# --------------------------
# Processing logic
# --------------------------
def save_multiple_detections(frame_bgr, detections, label_name, saved_count):
    img_dir, lbl_dir, _ = ensure_label_dirs(label_name)
    h_img, w_img, _ = frame_bgr.shape
    filename_base = f"auto_{int(time.time())}_{saved_count}"
    img_path = img_dir / f"{filename_base}.jpg"
    label_path = lbl_dir / f"{filename_base}.txt"

    cv2.imwrite(str(img_path), frame_bgr)
    with open(label_path, "w") as f:
        for det in detections:
            x1, y1, x2, y2 = det['box']
            dw, dh = 1.0 / w_img, 1.0 / h_img
            x_center = (x1 + x2) / 2.0 * dw
            y_center = (y1 + y2) / 2.0 * dh
            w, h = (x2 - x1) * dw, (y2 - y1) * dh
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    return img_path

def is_similar_image(pil_crop, label):
    db = load_hash_db()
    h_str = str(image_hash_for_pil(pil_crop))
    for existing_h in db.get(label, []):
        if hamming_distance_from_str(h_str, existing_h) <= HAMMING_THRESHOLD:
            return True
    return False

def add_hash_for_image(pil_crop, label):
    db, h_str = load_hash_db(), str(image_hash_for_pil(pil_crop))
    lst = db.get(label, [])
    lst.append(h_str)
    db[label] = lst
    save_hash_db(db)

def run_detection_on_frame(model, frame_bgr, conf=0.25):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, imgsz=640, conf=conf, iou=0.35, device=DEVICE, verbose=False)
    out = []
    res = results[0]
    if res.boxes is None: return out
        
    for box in res.boxes:
        cls_id = int(box.cls.cpu().numpy()[0]) if box.cls is not None else 0
        score = float(box.conf.cpu().numpy()[0])
        if score < 0.3: continue
        
        label_name = res.names.get(cls_id, str(cls_id))
        if label_name not in ANIMALS: continue
            
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        h, w, _ = frame_bgr.shape
        x1, x2 = max(0, min(w-1, x1)), max(0, min(w-1, x2))
        y1, y2 = max(0, min(h-1, y1)), max(0, min(h-1, y2))
        
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0: continue
            
        crop_pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        out.append({"label": label_name, "box": (x1, y1, x2, y2), "score": score, "crop_pil": crop_pil, "crop_bgr": crop})
    return out

def process_and_save(detections, frame_bgr, override_label=None):
    if not detections: return
    label = override_label if override_label else detections[0]["label"]
    
    if is_similar_image(detections[0]["crop_pil"], label): return

    img_path = save_multiple_detections(frame_bgr, detections, label, int(time.time() * 1000) % 10000)
    if img_path:
        for det in detections: add_hash_for_image(det["crop_pil"], label)
    return True

# --------------------------
# Training Logic
# --------------------------
def train_background_task():
    global is_training, training_status, global_model
    is_training = True
    training_status = "Training in progress..."
    try:
        labels = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
        if not labels: raise ValueError("No label folders found.")
        
        train_list = []
        for lbl in labels:
            for img in (TRAIN_DIR / lbl / "images").glob("*.jpg"):
                train_list.append(str(img.resolve()))
                
        train_images_txt = PROJECT_DIR / "train_images.txt"
        with open(train_images_txt, "w") as f: f.write("\n".join(train_list))
        
        data_yaml_path = PROJECT_DIR / "data_auto.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump({"train": str(train_images_txt), "val": str(train_images_txt), "names": {i: name for i, name in enumerate(labels)}}, f)

        model = YOLO(INITIAL_WEIGHTS)
        name = f"custom_{int(time.time())}"
        model.train(data=str(data_yaml_path), epochs=20, imgsz=640, name=name, device=DEVICE)

        runs_dir = Path("runs") / "detect" / name / "weights" / "best.pt"
        if runs_dir.exists():
            dest = WEIGHTS_PATH / f"{name}_best.pt"
            dest.write_bytes(runs_dir.read_bytes())
            global_model = YOLO(str(dest)) 
            training_status = f"Training finished! Model updated to {name}_best.pt"
        else:
            training_status = "Training failed. best.pt not found."
    except Exception as e:
        training_status = f"Error: {str(e)}"
    finally:
        is_training = False

# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        results = run_detection_on_frame(global_model, frame)
        for det in results:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['label']} {det['score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if results: process_and_save(results, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = run_detection_on_frame(global_model, frame)
    process_and_save(results, frame)
    return jsonify({"message": f"Processed image. Found {len(results)} animals."})

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    files = request.files.getlist('files')
    animal_name = request.form.get('animal_name')
    
    if not files or not animal_name:
        return jsonify({"error": "Missing files or animal name"}), 400

    if animal_name not in ANIMALS:
        ANIMALS.append(animal_name)
        with open(new_animals_file, "a") as f:
            f.write(animal_name + "\n")

    saved_count = 0
    for file in files:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None: continue
        
        results = run_detection_on_frame(global_model, frame)
        if results:
            was_saved = process_and_save(results, frame, override_label=animal_name)
            if was_saved:
                saved_count += 1

    return jsonify({"message": f"Folder processed. Extracted valid training crops from {saved_count} images."})

@app.route('/delete_animal', methods=['POST'])
def delete_animal():
    data = request.json
    animal = data.get('animal')
    if not animal:
        return jsonify({"error": "No animal specified"}), 400

    # حذف المجلد من ملفات التدريب
    animal_dir = TRAIN_DIR / animal
    if animal_dir.exists():
        shutil.rmtree(animal_dir)

    # حذفه من قاعدة بيانات التكرار (Hash DB)
    db = load_hash_db()
    if animal in db:
        del db[animal]
        save_hash_db(db)

    return jsonify({"message": f"تم حذف جميع بيانات '{animal}' بنجاح."})

@app.route('/start_training', methods=['POST'])
def start_training():
    global is_training
    if is_training: return jsonify({"status": "Already training!"})
    threading.Thread(target=train_background_task, daemon=True).start()
    return jsonify({"status": "Training started in background."})

@app.route('/training_status', methods=['GET'])
def get_training_status():
    stats = {lbl: count_images_for_label(lbl) for lbl in ANIMALS if count_images_for_label(lbl) > 0}
    return jsonify({"status": training_status, "is_training": is_training, "stats": stats})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)