import os
import json
import threading
import time
import shutil
import base64
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_file

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
UPLOAD_FOLDER = PROJECT_DIR / "uploads"
PROGRESS_FILE = PROJECT_DIR / "progress.json"
INITIAL_WEIGHTS = "yolov8n.pt"

HASH_SIZE = 64
SIMILARITY_THRESHOLD = 0.85
HAMMING_THRESHOLD = int((1 - SIMILARITY_THRESHOLD) * HASH_SIZE + 0.5)

TOTAL_EPOCHS = 20

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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not HASH_DB.exists():
    with open(HASH_DB, "w") as f: json.dump({}, f)
if not PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, "w") as f: json.dump({}, f)

# Global variables
global_model = None
is_training = False
training_status = "Ready"
current_video_path = None
current_run_dir = None
current_weights_path = None

def init_model():
    global global_model, current_weights_path
    if WEIGHTS_PATH.exists():
        pt_files = list(WEIGHTS_PATH.glob("*.pt"))
        if pt_files:
            latest_weights = max(pt_files, key=os.path.getmtime)
            current_weights_path = str(latest_weights)
            global_model = YOLO(current_weights_path)
            return
    current_weights_path = None
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
    for p in (img_dir, lbl_dir): os.makedirs(p, exist_ok=True)
    return img_dir, lbl_dir

def count_images_for_label(label):
    img_dir, _ = ensure_label_dirs(label)
    return len(list(img_dir.glob("*.jpg")))

# --------------------------
# Data Saving & Augmentation
# --------------------------
def save_multiple_detections(frame_bgr, detections, label_name, saved_count, augment=False):
    img_dir, lbl_dir = ensure_label_dirs(label_name)
    h_img, w_img, _ = frame_bgr.shape
    
    filename_base = f"img_{int(time.time()*1000)}_{saved_count}"
    cv2.imwrite(str(img_dir / f"{filename_base}.jpg"), frame_bgr)
    
    with open(lbl_dir / f"{filename_base}.txt", "w") as f:
        for det in detections:
            x1, y1, x2, y2 = det['box']
            dw, dh = 1.0 / w_img, 1.0 / h_img
            xc, yc = (x1 + x2)/2.0 * dw, (y1 + y2)/2.0 * dh
            w, h = (x2 - x1) * dw, (y2 - y1) * dh
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    if augment:
        frame_flipped = cv2.flip(frame_bgr, 1)
        filename_aug = f"{filename_base}_aug"
        cv2.imwrite(str(img_dir / f"{filename_aug}.jpg"), frame_flipped)
        
        with open(lbl_dir / f"{filename_aug}.txt", "w") as f:
            for det in detections:
                x1, y1, x2, y2 = det['box']
                dw, dh = 1.0 / w_img, 1.0 / h_img
                xc, yc = (x1 + x2)/2.0 * dw, (y1 + y2)/2.0 * dh
                w, h = (x2 - x1) * dw, (y2 - y1) * dh
                xc_flipped = 1.0 - xc 
                f.write(f"0 {xc_flipped:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return True

def process_and_save(detections, frame_bgr, override_label=None, augment=False):
    if not detections: return False
    label = override_label if override_label else detections[0]["label"]
    
    db = load_hash_db()
    h_str = str(image_hash_for_pil(detections[0]["crop_pil"]))
    for existing_h in db.get(label, []):
        if hamming_distance_from_str(h_str, existing_h) <= HAMMING_THRESHOLD:
            return False

    if save_multiple_detections(frame_bgr, detections, label, int(time.time()*1000)%10000, augment):
        lst = db.get(label, [])
        lst.append(h_str)
        db[label] = lst
        save_hash_db(db)
        return True
    return False

def run_detection_on_frame(model, frame_bgr, conf=0.3):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, imgsz=640, conf=conf, iou=0.35, device=DEVICE, verbose=False)
    out = []
    if results[0].boxes is None: return out
        
    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy()[0]) if box.cls is not None else 0
        score = float(box.conf.cpu().numpy()[0])
        label_name = results[0].names.get(cls_id, str(cls_id))
        
        if label_name not in ANIMALS: continue
            
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        h, w, _ = frame_bgr.shape
        x1, x2 = max(0, min(w-1, x1)), max(0, min(w-1, x2))
        y1, y2 = max(0, min(h-1, y1)), max(0, min(h-1, y2))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0: continue
            
        out.append({
            "label": label_name, "box": (x1, y1, x2, y2), 
            "score": score, "crop_pil": PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        })
    return out

# --------------------------
# Training YOLO Callbacks
# --------------------------
current_batch_count = 0

def on_train_epoch_start(trainer):
    global current_batch_count
    current_batch_count = 0

def on_train_batch_end(trainer):
    global current_batch_count
    current_batch_count += 1
    try: total = len(trainer.train_loader)
    except: total = 1
        
    prog = {"epoch": trainer.epoch + 1, "batch": current_batch_count, "total_batches": total}
    with open(PROGRESS_FILE, "w") as f: json.dump(prog, f)

def train_background_task():
    global is_training, training_status, global_model, current_run_dir, current_weights_path
    is_training = True
    training_status = "Preparing Dataset..."
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

        with open(PROGRESS_FILE, "w") as f: json.dump({"epoch": 1, "batch": 0, "total_batches": 0}, f)

        model = YOLO(INITIAL_WEIGHTS)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)

        name = f"custom_{int(time.time())}"
        current_run_dir = Path("runs") / "detect" / name
        
        training_status = "Training in progress..."
        model.train(data=str(data_yaml_path), epochs=TOTAL_EPOCHS, imgsz=640, name=name, device=DEVICE, workers=0)

        best_pt = current_run_dir / "weights" / "best.pt"
        if best_pt.exists():
            dest = WEIGHTS_PATH / f"{name}_best.pt"
            dest.write_bytes(best_pt.read_bytes())
            current_weights_path = str(dest)
            global_model = YOLO(current_weights_path) 
            training_status = f"Training Finished!"
        else:
            training_status = "Training failed. best.pt not found."
    except Exception as e:
        training_status = f"Error: {str(e)}"
    finally:
        is_training = False
        with open(PROGRESS_FILE, "w") as f: json.dump({}, f)

# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def index(): return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        results = run_detection_on_frame(global_model, frame)
        for det in results:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            accuracy = int(det['score'] * 100)
            label_text = f"{det['label'].capitalize()} {accuracy}%"
            cv2.putText(frame, label_text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if results: process_and_save(results, frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_playback_frames():
    global current_video_path
    if not current_video_path or not os.path.exists(current_video_path): return
    cap = cv2.VideoCapture(current_video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        results = run_detection_on_frame(global_model, frame)
        for det in results:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add accuracy as a percentage for video overlay
            accuracy = int(det['score'] * 100)
            label_text = f"{det['label'].capitalize()} {accuracy}%"
            cv2.putText(frame, label_text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if results: process_and_save(results, frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_playback_feed')
def video_playback_feed(): return Response(generate_video_playback_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_path
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    video_path = UPLOAD_FOLDER / "temp_video.mp4"
    file.save(str(video_path))
    current_video_path = str(video_path)
    return jsonify({"message": "Video uploaded successfully and ready for playback."})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Process a single image and return detailed message"""
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    augment = request.form.get('augment') == 'true'
    npimg = np.frombuffer(request.files['file'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = run_detection_on_frame(global_model, frame)
    found_details = []
    
    for det in results:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Accuracy percentage e.g. 89%
        accuracy = int(det['score'] * 100)
        label_text = f"{det['label'].capitalize()} {accuracy}%"
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        found_details.append(label_text)
    
    process_and_save(results, frame, augment=augment)
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Build response message listing detected animals
    if found_details:
        msg = f"✅ Processed successfully! Found: {', '.join(found_details)}"
    else:
        msg = "❌ No target animals detected in this image."
        
    return jsonify({"message": msg, "image": base64.b64encode(buffer).decode('utf-8')})

@app.route('/upload_folder_chunk', methods=['POST'])
def upload_folder_chunk():
    """Process folder images one by one and return detection summary"""
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    animal_name = request.form.get('animal_name')
    augment = request.form.get('augment') == 'true'
    
    if animal_name not in ANIMALS:
        ANIMALS.append(animal_name)
        with open(new_animals_file, "a") as f: f.write(animal_name + "\n")

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    saved_crops = 0
    found_details = []
    
    if frame is not None:
        results = run_detection_on_frame(global_model, frame)
        for det in results:
            accuracy = int(det['score'] * 100)
            found_details.append(f"{det['label'].capitalize()} {accuracy}%")
            
        if process_and_save(results, frame, override_label=animal_name, augment=augment):
            saved_crops = 2 if augment else 1

    # Compose detected animal names and accuracy for response
    detail_msg = ", ".join(found_details) if found_details else "None"
    return jsonify({"status": "ok", "saved_crops": saved_crops, "found": detail_msg})

@app.route('/delete_animal', methods=['POST'])
def delete_animal():
    animal = request.json.get('animal')
    if not animal:
        return jsonify({"error": "No animal specified"}), 400

    # 1. Remove hash entries from DB first to keep UI in sync
    db = load_hash_db()
    if animal in db:
        del db[animal]
        save_hash_db(db)

    # 2. Remove folder and images; ignore Windows permission locked-file errors
    animal_dir = TRAIN_DIR / animal
    if animal_dir.exists():
        shutil.rmtree(animal_dir, ignore_errors=True)

    return jsonify({"message": f"Deleted all data for '{animal}'."})

@app.route('/start_training', methods=['POST'])
def start_training():
    global is_training
    if is_training: return jsonify({"status": "Already training!"})
    threading.Thread(target=train_background_task, daemon=True).start()
    return jsonify({"status": "Training started."})

@app.route('/training_status', methods=['GET'])
def get_training_status():
    stats = {lbl: count_images_for_label(lbl) for lbl in ANIMALS if count_images_for_label(lbl) > 0}
    current_epoch, current_batch, total_batches = 0, 0, 0

    if is_training and PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
                current_epoch = data.get("epoch", 0)
                current_batch = data.get("batch", 0)
                total_batches = data.get("total_batches", 0)
        except: pass

    return jsonify({
        "status": training_status, 
        "is_training": is_training, 
        "stats": stats,
        "current_epoch": current_epoch,
        "total_epochs": TOTAL_EPOCHS,
        "current_batch": current_batch,
        "total_batches": total_batches,
        "has_model": current_weights_path is not None
    })

@app.route('/download_model', methods=['GET'])
def download_model():
    if current_weights_path and os.path.exists(current_weights_path):
        return send_file(current_weights_path, as_attachment=True)
    return "No model found", 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)