# 🐾 Animal Detection & Auto-Training System (YOLOv8)

A real-time animal detection and continuous learning system built with **YOLOv8**, **OpenCV**, and **Tkinter**.

The application detects animals from images, videos, or live camera feeds, automatically collects high-quality training samples, avoids duplicates using perceptual hashing, and re-trains the model when enough new data is collected—all through a graphical interface.

---

## 🚀 Key Features

### Real-Time Animal Detection
* Supports images, videos, and live webcam input.
* Uses **YOLOv8** for fast and accurate inference.

### Automated Dataset Collection
* **Saves detected animal crops automatically.**
* Stores:
    * Cropped images
    * YOLO label files
    * Bounding box coordinates (JSON)

### Duplicate & Near-Duplicate Filtering
* Uses **perceptual image hashing**.
* Rejects images with **>85% similarity**.
* Ensures clean, diverse training data.

### Active Learning Pipeline
* Automatically tracks collected samples per class.
* **When 500 new images are reached:**
    * Prompts the user for approval.
    * Retrains the model.
    * Loads the new weights automatically.

### Expandable Animal Classes
* Built-in animal list (cat, dog, horse, cow, etc.).
* New animals can be added dynamically via folders.
* Persisted across sessions.

### User-Friendly GUI
* Built with **Tkinter**.
* Simple controls for:
    * Image / Video / Live camera.
    * External training folder processing.
    * Training confirmation dialogs.

### Automatic Hardware Selection
* Uses **CUDA** if available.
* Falls back to **CPU** automatically.

---

## 🧠 System Workflow

1.  Detect animals from image / video / webcam.
2.  Crop detections and store them per class.
3.  Check similarity against existing samples.
4.  Save only **unique** images.
5.  Track image count per class.
6.  Trigger **retraining** at 500 samples.
7.  Load new model weights automatically.

---

## 📁 Project Structure

```text
project_data/
│
├── train/
│   ├── cat/
│   │   ├── images/
│   │   ├── labels/
│   │   └── coords/
│   └── dog/
│       ├── images/
│       ├── labels/
│       └── coords/
│
├── weights/
│   └── custom_<timestamp>_best.pt
│
├── hash_db.json
├── new_animals.txt
├── data_auto.yaml
└── train_images.txt
```

## 🖥️ GUI Options

| Option | Description |
| :--- | :--- |
| **Open Image** | Detect animals in a single image |
| **Open Video** | Process a full video file |
| **Open Live Camera** | Real-time webcam detection |
| **Process Ready Folder** | Import images for a new animal class |

## ⚙️ Requirements

Install dependencies before running:

```bash
pip install ultralytics opencv-python pillow imagehash torch pyyaml numpy
```

**Optional (GPU support):**
* NVIDIA CUDA
* Compatible PyTorch build

## ▶️ How to Run

```bash
python animal_trainer.py
```

**The application will:**
1. Load the base YOLOv8 model
2. Launch the graphical interface
3. Automatically select GPU or CPU

## 🧪 Training Logic
* Training starts only after user confirmation
* **Uses:**
    * Auto-generated `data.yaml`
    * YOLOv8 training pipeline
* **Best model is saved to:** `project_data/weights/`

## 📊 Data Quality Controls
* Perceptual hashing to eliminate duplicates
* Hamming distance thresholding
* Prevents overfitting and dataset pollution

## 🔐 Safe Shutdown
* Graceful exit confirmation
* Clean termination of camera/video streams

## 🏗️ Technologies Used
* YOLOv8 (Ultralytics)
* OpenCV
* PyTorch
* Tkinter
* NumPy
* Pillow
* ImageHash
* YAML / JSON

## 📌 Use Cases
* Wildlife monitoring
* Smart surveillance systems
* Dataset generation for ML research
* Continuous learning computer vision systems
* Edge AI prototyping

## 📜 License
This project is provided for educational and research purposes.  
You are free to modify and extend it.
