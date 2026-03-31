"""
Function: Animal detection using YOLO model, saving clips + coordinates, avoiding similar elements (>85% similarity)
The model is automatically retrained when 500 new images are collected for a class, and after user approval, the new model is used.
Interface: Tkinter with options: image | video | live, button to select external training folder
"""

import os
import json
import threading
import time
from pathlib import Path
from tkinter import (
    Tk, Label, Button, filedialog, messagebox, StringVar, Frame
)
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk, Image as PILImage
import imagehash
import yaml
import torch
from ultralytics import YOLO

# --------------------------
# Automatic device detection
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Project settings (editable)
# --------------------------
PROJECT_DIR = Path.cwd() / "project_data"
TRAIN_DIR = PROJECT_DIR / "train"      # Will contain: train/<animal_label>/images... and labels...
HASH_DB = PROJECT_DIR / "hash_db.json" # Hash database to avoid duplicates
WEIGHTS_PATH = PROJECT_DIR / "weights" # Place to save new weights

# DEFAULT fallback model
INITIAL_WEIGHTS = "yolov8n.pt"

# Thresholds
HASH_SIZE = 64   # Hash length (imagehash average_hash uses default 8x8 = 64 bits)
SIMILARITY_THRESHOLD = 0.85  # Keep only if similarity < 0.85

# We calculate hamming_threshold based on HASH_SIZE
HAMMING_THRESHOLD = int((1 - SIMILARITY_THRESHOLD) * HASH_SIZE + 0.5)

# List of allowed animals to detect
ANIMALS = [
    'cat', 'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'sheep', 'lion', 'tiger', 'monkey', 'wolf', 'deer', 'pig', 'goat', 
    'camel', 'donkey'
]

# Reading new animals from the file
new_animals_file = PROJECT_DIR / "new_animals.txt"
if new_animals_file.exists():
    with open(new_animals_file, "r") as f:
        new_animals = [line.strip() for line in f if line.strip()]
    ANIMALS.extend(new_animals)
    ANIMALS = list(set(ANIMALS))  # Remove duplicates

TARGET_IMAGES_FOR_TRAIN = 500

# Initialize directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(WEIGHTS_PATH, exist_ok=True)
if not HASH_DB.exists():
    with open(HASH_DB, "w") as f:
        json.dump({}, f)

# --------------------------
# Database & Hash Helpers
# --------------------------
def load_hash_db():
    with open(HASH_DB, "r") as f:
        return json.load(f)

def save_hash_db(db):
    with open(HASH_DB, "w") as f:
        json.dump(db, f, indent=2)

def image_hash_for_pil(pil_img):
    return imagehash.average_hash(pil_img)

def hamming_distance_from_str(h1_str, h2_str):
    h1 = imagehash.hex_to_hash(h1_str)
    h2 = imagehash.hex_to_hash(h2_str)
    return (h1 - h2)

# --------------------------
# Image and coordinates saving management
# --------------------------
def save_multiple_detections(frame_bgr, detections, label_name, saved_count):
    img_dir, lbl_dir, _ = ensure_label_dirs(label_name)
    h_img, w_img, _ = frame_bgr.shape
    
    # Unique filename for this frame
    filename_base = f"auto_{int(time.time())}_{saved_count}"
    img_path = img_dir / f"{filename_base}.jpg"
    label_path = lbl_dir / f"{filename_base}.txt"

    # Save the full image once
    cv2.imwrite(str(img_path), frame_bgr)

    # Write one line for every single deer found in this image
    with open(label_path, "w") as f:
        for det in detections:
            x1, y1, x2, y2 = det['box']
            
            # Convert to YOLO format (normalized 0.0 to 1.0)
            dw = 1.0 / w_img
            dh = 1.0 / h_img
            x_center = (x1 + x2) / 2.0 * dw
            y_center = (y1 + y2) / 2.0 * dh
            w = (x2 - x1) * dw
            h = (y2 - y1) * dh
            
            # '0' is the class index for your specific animal folder
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    return img_path


def ensure_label_dirs(label): 
    img_dir = TRAIN_DIR / label / "images"
    lbl_dir = TRAIN_DIR / label / "labels"
    coord_dir = TRAIN_DIR / label / "coords"
    for p in (img_dir, lbl_dir, coord_dir):
        os.makedirs(p, exist_ok=True)
    return img_dir, lbl_dir, coord_dir

# --------------------------
# Duplicate check (hash db)
# --------------------------
def is_similar_image(pil_crop, label):
    db = load_hash_db()
    h = image_hash_for_pil(pil_crop)
    h_str = str(h)
    
    existing = db.get(label, [])
    for existing_h in existing:
        dist = hamming_distance_from_str(h_str, existing_h)
        if dist <= HAMMING_THRESHOLD:
            return True
    return False

def add_hash_for_image(pil_crop, label):
    db = load_hash_db()
    h = image_hash_for_pil(pil_crop)
    h_str = str(h)
    
    lst = db.get(label, [])
    lst.append(h_str)
    db[label] = lst
    save_hash_db(db)

# --------------------------
# Counter for new images
# --------------------------
def count_images_for_label(label):
    img_dir, _, _ = ensure_label_dirs(label)
    return len(list(img_dir.glob("*.jpg")))

# --------------------------
# Training the model
# --------------------------
def train_model_using_folder(train_folder, external_data_yaml=None, epochs=20, imgsz=640, model_save_name=None, callback=None):
    if external_data_yaml:
        data_yaml = external_data_yaml
    else:
        labels = sorted([p.name for p in (Path(train_folder)).iterdir() if p.is_dir()])
        if not labels:
            raise ValueError("No label folders found in train folder.")
        
        dataset_dir = Path(train_folder)
        class_list = labels
        data_yaml_path = PROJECT_DIR / "data_auto.yaml"
        
        train_list = []
        for lbl in labels:
            img_dir = dataset_dir / lbl / "images"
            for img in img_dir.glob("*.jpg"):
                train_list.append(str(img.resolve()))
        
        train_images_txt = PROJECT_DIR / "train_images.txt"
        with open(train_images_txt, "w") as f:
            f.write("\n".join(train_list))
        
        data_yaml_content = {
            "train": str(train_images_txt.resolve()),
            "val": str(train_images_txt.resolve()),  # Using train as val for simplicity in this loop
            "names": {i: name for i, name in enumerate(class_list)}
        }
        
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_yaml_content, f)
        
        data_yaml = str(data_yaml_path.resolve())

    if callback:
        callback("Loading base model...")
    
    # Always start training from a base model or the current best? 
    # Usually better to start from base (pretrained) for robustness, 
    # but could fine-tune current. Here we use INITIAL_WEIGHTS (base) to avoid drift.
    model = YOLO(INITIAL_WEIGHTS) 

    name = model_save_name or f"custom_{int(time.time())}"
    
    if callback:
        callback(f"Starting training: epochs={epochs}, imgsz={imgsz} ...")
    
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, name=name, device=DEVICE)
    
    # Locate the best.pt
    runs_dir = Path("runs") / "detect" / name / "weights"
    best = runs_dir / "best.pt"
    
    if best.exists():
        dest = WEIGHTS_PATH / f"{name}_best.pt"
        dest.write_bytes(best.read_bytes())
        if callback:
            callback(f"Training finished. New weights saved: {dest}")
        return str(dest)
    else:
        # Fallback search if 'runs' structure varies
        possible_bests = list(Path("runs").glob("**/best.pt"))
        if possible_bests:
             # Take the most recent one
            best = max(possible_bests, key=os.path.getmtime)
            dest = WEIGHTS_PATH / f"{name}_best.pt"
            dest.write_bytes(best.read_bytes())
            if callback:
                callback(f"Training finished (found via glob). New weights: {dest}")
            return str(dest)

        if callback:
            callback("Training completed but could not find best.pt in runs/..")
        return None

# --------------------------
# Detection on Frame
# --------------------------
def run_detection_on_frame(model, frame_bgr, conf=0.25, device=None):
    device = device or DEVICE
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(frame_rgb, imgsz=640, conf=conf, iou=0.35, device=device, verbose=False)
    
    out = []
    res = results[0]
    boxes = res.boxes
    if boxes is None:
        return out
        
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.cpu().numpy()[0]) if box.cls is not None else 0
        score = float(box.conf.cpu().numpy()[0])
        if score < 0.3:
            continue
        xyxy = box.xyxy.cpu().numpy()[0]
        
        label_name = res.names.get(cls_id, str(cls_id))
        if label_name in ANIMALS:
            print(f"[DETECTED] {label_name.upper()} | Confidence: {score:.2f}")
        
        x1, y1, x2, y2 = map(int, xyxy)
        h, w, _ = frame_bgr.shape
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0: continue
            
        crop_pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        out.append({
            "label": label_name,
            "box": (x1, y1, x2, y2),
            "score": score,
            "crop_pil": crop_pil,
            "crop_bgr": crop
        })
    
    # Filter only animals we care about
    # Note: If we're trained a custom model, the labels might just be 'deer' etc.
    # We allow the detection if it is in our target ANIMALS list.
    out = [det for det in out if det['label'] in ANIMALS]
    return out

# --------------------------
# GUI: Tkinter
# --------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Animal Detector & Auto Trainer")
        self.status = StringVar(value="Ready")
        
        # ---------------------------------------------------------
        #  CRITICAL FIX: AUTO-LOAD LATEST TRAINED MODEL STARTUP
        # ---------------------------------------------------------
        found_custom_model = False
        if WEIGHTS_PATH.exists():
            # Find all .pt files in project_data/weights
            pt_files = list(WEIGHTS_PATH.glob("*.pt"))
            if pt_files:
                # Sort by modification time (newest first)
                latest_weights = max(pt_files, key=os.path.getmtime)
                print(f"Startup: Found custom trained weights: {latest_weights}")
                try:
                    self.model = YOLO(str(latest_weights))
                    self.current_weights = str(latest_weights)
                    found_custom_model = True
                    self.status.set(f"Loaded Custom Model: {latest_weights.name}")
                except Exception as e:
                    print(f"Error loading custom weights: {e}")

        if not found_custom_model:
            print(f"Startup: No custom weights found. Loading base: {INITIAL_WEIGHTS}")
            self.model = YOLO(INITIAL_WEIGHTS)
            self.current_weights = INITIAL_WEIGHTS
            self.status.set(f"Loaded Base Model: {INITIAL_WEIGHTS}")
        # ---------------------------------------------------------

        self.frame = Frame(root)
        self.frame.pack(padx=8, pady=8)

        btn_image = Button(self.frame, text="Open Image", width=20, command=self.open_image)
        btn_video = Button(self.frame, text="Open Video", width=20, command=self.open_video)
        btn_live = Button(self.frame, text="Open Live Camera", width=20, command=self.open_live)
        btn_process_folder = Button(self.frame, text="Process Ready Folder", width=20, command=self.process_ready_folder)

        btn_image.grid(row=0, column=0, padx=4, pady=4)
        btn_video.grid(row=0, column=1, padx=4, pady=4)
        btn_live.grid(row=0, column=2, padx=4, pady=4)
        btn_process_folder.grid(row=1, column=0, columnspan=3, pady=4)

        self.canvas_label = Label(root)
        self.canvas_label.pack()

        self.status_label = Label(root, textvariable=self.status)
        self.status_label.pack(pady=4)

        self._imgtk = None
        self.detection_window = None
        self.detection_canvas = None
        self.detection_imgtk = None
        self.stop_stream = False

    def set_status(self, txt):
        self.status.set(txt)
        self.root.update_idletasks()

    def process_ready_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images of one animal")
        if not folder:
            return
        
        animal_name = Path(folder).name
        
        # Add the new animal if not present
        if animal_name not in ANIMALS:
            ANIMALS.append(animal_name)
            with open(new_animals_file, "a") as f:
                f.write(animal_name + "\n")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder).glob(ext))
            
        num_images = len(image_files)
        if num_images == 0:
            messagebox.showerror("No images", "No image files found in the selected folder.")
            return

        resp = messagebox.askyesno("Confirm Processing", f"Found {num_images} images for animal: {animal_name}\nDo you want to detect/save crops from these?")
        if resp:
            self.perform_detection_on_folder(folder, animal_name, image_files)

    def perform_detection_on_folder(self, folder, animal_name, image_files):
        self.set_status("Processing images...")
        saved_images_count = 0
        
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None: continue
            
            # 1. Get all detections for this specific image
            results = run_detection_on_frame(self.model, frame)
            
            if not results:
                continue

            # 2. Prepare paths for this image
            img_dir, lbl_dir, _ = ensure_label_dirs(animal_name)
            filename_base = f"{int(time.time())}_{saved_images_count}"
            save_path = img_dir / f"{filename_base}.jpg"
            label_path = lbl_dir / f"{filename_base}.txt"

            # 3. Write ALL detections to ONE text file
            h_img, w_img, _ = frame.shape
            with open(label_path, "w") as f:
                for det in results:
                    # Calculate YOLO normalized coordinates
                    x1, y1, x2, y2 = det['box']
                    dw = 1.0 / w_img
                    dh = 1.0 / h_img
                    x_center = (x1 + x2) / 2.0 * dw
                    y_center = (y1 + y2) / 2.0 * dh
                    w = (x2 - x1) * dw
                    h = (y2 - y1) * dh
                    
                    # Write a new line for every deer found
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            # 4. Save the image once
            cv2.imwrite(str(save_path), frame)
            saved_images_count += 1

        self.set_status(f"Saved {saved_images_count} images with all detected animals.")
        
        if saved_images_count > 0:
            if messagebox.askyesno("Train", "Detections complete. Train now?"):
                self.train_on_animal(animal_name)
    def train_on_animal(self, animal_name):
        train_root = TRAIN_DIR
        data_yaml = None
        
        def training_thread():
            self.set_status("Training started... This may take a while.")
            try:
                # Start training
                new_weights = train_model_using_folder(
                    str(train_root), 
                    external_data_yaml=data_yaml, 
                    epochs=20, 
                    imgsz=640, 
                    model_save_name=None, 
                    callback=self.set_status
                )
                
                if new_weights:
                    # -----------------------------------------------------
                    # CRITICAL FIX: SWAP MODEL IMMEDIATELY AFTER TRAINING
                    # -----------------------------------------------------
                    self.set_status("Training done. Loading new weights...")
                    
                    # Unload old model (Python garbage collection helps here)
                    self.model = None 
                    
                    # Load new model
                    self.model = YOLO(new_weights)
                    self.current_weights = new_weights
                    
                    self.set_status(f"Model Updated: {Path(new_weights).name}")
                    messagebox.showinfo("Training finished", f"Success!\nThe app is now using the new model:\n{Path(new_weights).name}")
                else:
                    self.set_status("Training ended (No weights found).")
                    messagebox.showwarning("Training ended", "Training ended but no weights found.")
                    
            except Exception as e:
                self.set_status(f"Error: {str(e)}")
                messagebox.showerror("Training error", str(e))
            finally:
                if self.status.get().startswith("Training started"):
                    self.set_status("Ready")

        t = threading.Thread(target=training_thread, daemon=True)
        t.start()

    # ------------------
    # Display & Input
    # ------------------
    def display_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im_pil = PILImage.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(im_pil.resize((800, 450)))
        self.canvas_label.config(image=imgtk)
        self._imgtk = imgtk

    def open_detection_window(self):
        if self.detection_window is None or not self.detection_window.winfo_exists():
            self.detection_window = tk.Toplevel(self.root)
            self.detection_window.title("Detection Results")
            self.detection_canvas = tk.Label(self.detection_window)
            self.detection_canvas.pack()
            self.detection_imgtk = None
            self.detection_window.bind('<Key>', self.on_key_press)
            self.detection_window.focus_set()
        self.detection_window.lift()

    def on_key_press(self, event):
        if event.char == 'q':
            self.stop_stream = True
            if self.detection_window:
                self.detection_window.destroy()
                self.detection_window = None

    def display_in_detection_window(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im_pil = PILImage.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(im_pil.resize((800, 450)))
        self.detection_canvas.config(image=imgtk)
        self.detection_imgtk = imgtk

    def draw_detections_on_frame(self, frame_bgr, detections):
        # This loop goes through EVERY detection found by the model
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            score = det['score']
            
            # Draw the rectangle for THIS specific deer
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add the text label above THIS specific box
            cv2.putText(frame_bgr, f"{label} {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path: return
        
        self.set_status("Processing image...")
        frame = cv2.imread(path)
        if frame is None: return

        results = run_detection_on_frame(self.model, frame)
        
        frame_with_boxes = frame.copy()
        self.draw_detections_on_frame(frame_with_boxes, results)
        
        self.open_detection_window()
        self.display_in_detection_window(frame_with_boxes)
        
        self.process_detections_and_save(results, frame)
        self.set_status("Done processing image.")

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path: return
        
        cap = cv2.VideoCapture(path)
        self.stop_stream = False
        idx = 0
        self.open_detection_window()
        
        while cap.isOpened() and not self.stop_stream:
            ret, frame = cap.read()
            if not ret: break
            
            results = run_detection_on_frame(self.model, frame)
            
            frame_with_boxes = frame.copy()
            self.draw_detections_on_frame(frame_with_boxes, results)
            self.display_in_detection_window(frame_with_boxes)
            
            if results:
                self.process_detections_and_save(results, frame)
            
            idx += 1
            self.root.update()
        
        cap.release()
        self.set_status("Video finished")

    def open_live(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not open camera.")
            return
            
        self.stop_stream = False
        self.open_detection_window()
        
        while cap.isOpened() and not self.stop_stream:
            ret, frame = cap.read()
            if not ret: break
            
            results = run_detection_on_frame(self.model, frame)
            
            frame_with_boxes = frame.copy()
            self.draw_detections_on_frame(frame_with_boxes, results)
            self.display_in_detection_window(frame_with_boxes)
            
            if results:
                self.process_detections_and_save(results, frame)
                
            self.root.update()
            
        cap.release()
        self.set_status("Live stream stopped")

    def process_detections_and_save(self, detections, frame_bgr):
        if not detections:
            return

        # Use the label from the first detection for folder organization
        label = detections[0]["label"]
        
        # Check similarity for the first object to avoid saving identical video frames
        if is_similar_image(detections[0]["crop_pil"], label):
            return

        # Call the new multi-box saving function
        img_path = save_multiple_detections(frame_bgr, detections, label, int(time.time()))
        
        if img_path:
            for det in detections:
                add_hash_for_image(det["crop_pil"], label)
            
            cnt = count_images_for_label(label)
            self.set_status(f"Saved {len(detections)} deer in one image. Total: {cnt}")
            
            # Trigger training when target is reached
            if cnt > 0 and cnt % TARGET_IMAGES_FOR_TRAIN == 0:
                self.ask_and_train_for_label(label)

    def ask_and_train_for_label(self, label):
        resp = messagebox.askyesno("Train new model?", f"There are {count_images_for_label(label)} images for label '{label}'. Do you want to train a new model now?")
        if resp:
            self.train_on_animal(label)

    def stop(self):
        self.stop_stream = True

# --------------------------
# Main
# --------------------------
def main():
    print(f"Using device: {DEVICE}")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    
    root = Tk()
    app = App(root)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            app.stop()
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
