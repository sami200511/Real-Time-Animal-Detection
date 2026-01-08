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
from PIL import Image, ImageTk
import imagehash
from PIL import Image as PILImage

# Ultralytics YOLO
from ultralytics import YOLO

import torch
import yaml

# Automatic device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Project settings (editable)
# --------------------------
PROJECT_DIR = Path.cwd() / "project_data"
TRAIN_DIR = PROJECT_DIR / "train"      # Will contain: train/<animal_label>/images... and labels...
HASH_DB = PROJECT_DIR / "hash_db.json" # Hash database to avoid duplicates
WEIGHTS_PATH = PROJECT_DIR / "weights" # Place to save new weights
INITIAL_WEIGHTS = "yolov8n.pt"         # Or the name of the default model file available to you

# Thresholds
HASH_SIZE = 64   # Hash length (imagehash average_hash uses default 8x8 = 64 bits)
SIMILARITY_THRESHOLD = 0.85  # Keep only if similarity < 0.85
# We calculate hamming_threshold based on HASH_SIZE
HAMMING_THRESHOLD = int((1 - SIMILARITY_THRESHOLD) * HASH_SIZE + 0.5)

# List of allowed animals to detect
ANIMALS = [
    'cat', 'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'sheep',
    'lion', 'tiger', 'monkey', 'wolf', 'deer','pig', 'goat', 'camel', 'donkey'
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

# Load hash database (dictionary: label -> list of hashes as hex strings)
def load_hash_db():
    with open(HASH_DB, "r") as f:
        return json.load(f)

def save_hash_db(db):
    with open(HASH_DB, "w") as f:
        json.dump(db, f, indent=2)

# Calculate hash for PIL image
def image_hash_for_pil(pil_img):
    return imagehash.average_hash(pil_img)  # returns ImageHash object

# Convert to savable values
def hash_to_str(h):
    return str(h)

# Measure similarity via Hamming distance from imagehash hashes
def hamming_distance_from_str(h1_str, h2_str):
    # imagehash string like "ffffffff..." or "a3b..."
    # Safer convert to ImageHash then use - operator
    h1 = imagehash.hex_to_hash(h1_str)
    h2 = imagehash.hex_to_hash(h2_str)
    return (h1 - h2)

# --------------------------
# Image and coordinates saving management
# --------------------------
def ensure_label_dirs(label):
    img_dir = TRAIN_DIR / label / "images"
    lbl_dir = TRAIN_DIR / label / "labels"
    coord_dir = TRAIN_DIR / label / "coords"  # Optional: copy JSON for coordinates
    for p in (img_dir, lbl_dir, coord_dir):
        os.makedirs(p, exist_ok=True)
    return img_dir, lbl_dir, coord_dir

def save_detection_crop(img_bgr, bbox_xyxy, label_name, idx):
    img_dir, lbl_dir, coord_dir = ensure_label_dirs(label_name)
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    filename = f"{int(time.time())}_{idx}.jpg"
    img_path = img_dir / filename
    cv2.imwrite(str(img_path), crop)

    h, w, _ = img_bgr.shape
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width_n = (x2 - x1) / w
    height_n = (y2 - y1) / h

    label_txt = lbl_dir / (img_path.stem + ".txt")
    with open(label_txt, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width_n:.6f} {height_n:.6f}\n")

    coord_file = coord_dir / (img_path.stem + ".json")
    coord_data = {
        "image": str(img_path),
        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
        "normalized": {"xc": x_center, "yc": y_center, "w": width_n, "h": height_n},
        "label": label_name
    }
    with open(coord_file, "w") as f:
        json.dump(coord_data, f, indent=2)

    return img_path

# --------------------------
# Duplicate check (hash db)
# --------------------------
def is_similar_image(pil_crop, label):
    db = load_hash_db()
    h = image_hash_for_pil(pil_crop)
    h_str = h.__str__()  # hex representation
    existing = db.get(label, [])
    for existing_h in existing:
        dist = hamming_distance_from_str(h_str, existing_h)
        if dist <= HAMMING_THRESHOLD:
            return True
    return False

def add_hash_for_image(pil_crop, label):
    db = load_hash_db()
    h = image_hash_for_pil(pil_crop)
    h_str = h.__str__()
    lst = db.get(label, [])
    lst.append(h_str)
    db[label] = lst
    save_hash_db(db)

# --------------------------
# Counter for new images for 'label'
# --------------------------
def count_images_for_label(label):
    img_dir, _, _ = ensure_label_dirs(label)
    return len(list(img_dir.glob("*.jpg")))

# --------------------------
# Training the model (using ultralytics YOLO.train)
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
            "val": str(train_images_txt.resolve()),
            "names": class_list
        }
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_yaml_content, f)
        data_yaml = str(data_yaml_path.resolve())

    if callback:
        callback("Loading base model...")
    model = YOLO(INITIAL_WEIGHTS)

    name = model_save_name or f"custom_{int(time.time())}"
    if callback:
        callback(f"Starting training: epochs={epochs}, imgsz={imgsz} ...")
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, name=name, device=DEVICE)
    runs_dir = Path("runs") / "detect" / name / "weights"
    best = runs_dir / "best.pt"
    if best.exists():
        dest = WEIGHTS_PATH / f"{name}_best.pt"
        dest.write_bytes(best.read_bytes())
        if callback:
            callback(f"Training finished. New weights saved: {dest}")
        return str(dest)
    else:
        if callback:
            callback("Training completed but could not find best.pt in runs/..")
        return None

# --------------------------
# Example function for predicting objects on a single frame and returning the clips
# --------------------------
def run_detection_on_frame(model, frame_bgr, conf=0.25, device=None):
    device = device or DEVICE
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, imgsz=640, conf=conf, device=device)
    out = []
    res = results[0]
    boxes = res.boxes
    if boxes is None:
        return out
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.cpu().numpy()[0]) if box.cls is not None else 0
        score = float(box.conf.cpu().numpy()[0])
        xyxy = box.xyxy.cpu().numpy()[0]
        label_name = res.names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, xyxy)
        h, w, _ = frame_bgr.shape
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        crop = frame_bgr[y1:y2, x1:x2]
        crop_pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        out.append({
            "label": label_name,
            "box": (x1, y1, x2, y2),
            "score": score,
            "crop_pil": crop_pil,
            "crop_bgr": crop
        })
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
        self.model = YOLO(INITIAL_WEIGHTS)
        self.current_weights = INITIAL_WEIGHTS

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
        folder = filedialog.askdirectory(title="Select folder with images of one animal (folder name = animal name)")
        if not folder:
            return
        animal_name = Path(folder).name
        # Add the new animal if not present
        if animal_name not in ANIMALS:
            ANIMALS.append(animal_name)
            # Write to the file
            with open(new_animals_file, "a") as f:
                f.write(animal_name + "\n")
        # Count the images
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder).glob(ext))
        num_images = len(image_files)
        if num_images == 0:
            messagebox.showerror("No images", "No image files found in the selected folder.")
            return
        # Ask using messagebox
        resp = messagebox.askyesno("Confirm Processing", f"Found {num_images} images for animal: {animal_name}\nDo you want to detect animals in these images?")
        if resp:
            self.perform_detection_on_folder(folder, animal_name, image_files)

    def perform_detection_on_folder(self, folder, animal_name, image_files):
        self.set_status("Processing images...")
        saved_count = 0
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            results = run_detection_on_frame(self.model, frame)
            # Filter for the specified animal only
            for det in results:
                # Treat all detections as animal_name
                label = animal_name  # Treat as animal_name
                crop_pil = det["crop_pil"]
                crop_bgr = det["crop_bgr"]
                # check similarity
                if is_similar_image(crop_pil, label):
                    continue
                # save
                img_path_saved = save_detection_crop(frame, det["box"], label, saved_count)
                if img_path_saved:
                    add_hash_for_image(crop_pil, label)
                    saved_count += 1
            # If no detections were saved, save the full crop of the image as animal_name
            if saved_count == 0:
                # Save the full crop of the image
                h, w = frame.shape[:2]
                bbox_full = (0, 0, w, h)
                crop_pil_full = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not is_similar_image(crop_pil_full, animal_name):
                    img_path_saved = save_detection_crop(frame, bbox_full, animal_name, saved_count)
                    if img_path_saved:
                        add_hash_for_image(crop_pil_full, animal_name)
                        saved_count += 1
        self.set_status(f"Processed {len(image_files)} images, saved {saved_count} crops.")
        # Ask about training
        train_confirm = messagebox.askyesno("Train Model", f"Detection complete. Saved {saved_count} crops.\nDo you want to train the model on these images?")
        if train_confirm:
            self.train_on_animal(animal_name)

    def train_on_animal(self, animal_name):
        # Use TRAIN_DIR as train_root
        train_root = TRAIN_DIR
        data_yaml = None  # Will be created automatically
        # run training
        def training_thread():
            self.set_status("Training started...")
            try:
                new_weights = train_model_using_folder(str(train_root), external_data_yaml=data_yaml, epochs=20, imgsz=640, model_save_name=None, callback=self.set_status)
                if new_weights:
                    self.set_status("Loading new weights...")
                    self.model = YOLO(new_weights, device=DEVICE)
                    self.current_weights = new_weights
                    messagebox.showinfo("Training finished", f"Training finished and new model loaded:\n{new_weights}")
                else:
                    messagebox.showwarning("Training ended", "Training ended but no weights found.")
            except Exception as e:
                messagebox.showerror("Training error", str(e))
            finally:
                self.set_status("Ready")
        t = threading.Thread(target=training_thread, daemon=True)
        t.start()

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
            # Bind key presses
            self.detection_window.bind('<Key>', self.on_key_press)
            self.detection_window.focus_set()  # To make the window receive key presses
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
        """Draw boxes and texts on the frame"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            score = det['score']
            # Draw the green box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the text
            cv2.putText(frame_bgr, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.set_status("Processing image...")
        frame = cv2.imread(path)
        # detect
        results = run_detection_on_frame(self.model, frame)
        # Draw boxes on a copy of the frame
        frame_with_boxes = frame.copy()
        self.draw_detections_on_frame(frame_with_boxes, results)
        # Display in detection window
        self.open_detection_window()
        self.display_in_detection_window(frame_with_boxes)
        self.process_detections_and_save(results, frame)
        self.set_status("Done processing image.")

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        self.stop_stream = False
        idx = 0
        self.open_detection_window()  # Open the window once
        while cap.isOpened() and not self.stop_stream:
            ret, frame = cap.read()
            if not ret:
                break
            results = run_detection_on_frame(self.model, frame)
            # Draw boxes
            frame_with_boxes = frame.copy()
            self.draw_detections_on_frame(frame_with_boxes, results)
            self.display_in_detection_window(frame_with_boxes)
            if results:
                self.process_detections_and_save(results, frame)
            idx += 1
            # small delay to allow UI refresh
            self.root.update()
            if idx % 5 == 0:
                # let user break
                pass
        cap.release()
        self.set_status("Video finished")

    def open_live(self):
        cap = cv2.VideoCapture(0)  # default camera
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Could not open camera.")
            return
        self.stop_stream = False
        self.open_detection_window()  # Open the window once
        while cap.isOpened() and not self.stop_stream:
            ret, frame = cap.read()
            if not ret:
                break
            results = run_detection_on_frame(self.model, frame)
            # Draw boxes
            frame_with_boxes = frame.copy()
            self.draw_detections_on_frame(frame_with_boxes, results)
            self.display_in_detection_window(frame_with_boxes)
            if results:
                self.process_detections_and_save(results, frame)
            self.root.update()
        cap.release()
        self.set_status("Live stream stopped")

    def process_detections_and_save(self, detections, frame_bgr):
        saved_any = False
        for i, det in enumerate(detections):
            label = det["label"]
            crop_pil = det["crop_pil"]
            crop_bgr = det["crop_bgr"]
            # check similarity
            if is_similar_image(crop_pil, label):
                # too similar: skip
                continue
            # save
            img_path = save_detection_crop(frame_bgr, det["box"], label, i)
            if img_path:
                add_hash_for_image(crop_pil, label)
                saved_any = True
                # count
                cnt = count_images_for_label(label)
                if cnt % TARGET_IMAGES_FOR_TRAIN == 0:
                    # reached multiple of threshold -> ask user to train
                    self.ask_and_train_for_label(label)
        if saved_any:
            self.set_status("Saved new images.")

    def ask_and_train_for_label(self, label):
        # ask user
        resp = messagebox.askyesno("Train new model?", f"There are {count_images_for_label(label)} images for label '{label}'. Do you want to train a new model now?")
        if not resp:
            return
        # Use project train dir
        train_root = TRAIN_DIR
        data_yaml = None

        # run training in background thread to keep UI responsive
        def training_thread():
            self.set_status("Training started...")
            try:
                new_weights = train_model_using_folder(str(train_root), external_data_yaml=data_yaml, epochs=20, imgsz=640, model_save_name=None, callback=self.set_status)
                if new_weights:
                    # load new model
                    self.set_status("Loading new weights...")
                    self.model = YOLO(new_weights)
                    self.current_weights = new_weights
                    messagebox.showinfo("Training finished", f"Training finished and new model loaded:\n{new_weights}")
                else:
                    messagebox.showwarning("Training ended", "Training ended but no weights found.")
            except Exception as e:
                messagebox.showerror("Training error", str(e))
            finally:
                self.set_status("Ready")

        t = threading.Thread(target=training_thread, daemon=True)
        t.start()

    def stop(self):
        self.stop_stream = True

# --------------------------
# Main
# --------------------------
def main():
    print(f"Using device: {DEVICE}")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
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
