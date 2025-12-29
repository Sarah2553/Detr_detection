# 🐶🐱🦓 Real-Time Animal Detection in Videos using DETR

This project demonstrates **real-time animal detection in videos** using **DETR (DEtection TRansformer)**.

DETR is an **end-to-end object detection model** developed by **Facebook AI** that replaces traditional detection pipelines (such as Faster R-CNN with anchors and Non-Maximum Suppression) with a **single Transformer-based architecture**.  
It formulates object detection as a **set prediction problem** using a **Transformer encoder–decoder** and **bipartite matching loss**, enabling clean detection without post-processing heuristics.

---

## 🔑 Key Features

- Transformer-based architecture for direct set prediction of object classes and bounding boxes  
- Eliminates anchors and post-processing heuristics like **Non-Maximum Suppression (NMS)**  
- Processes video datasets **frame by frame**  
- Displays **real-time FPS**  
- Draws **colored bounding boxes** per animal class with confidence scores  
- Maintains **per-animal detection counts** saved to a CSV file  
- Estimates **visibility duration per animal** (in seconds)  
- Modular and easy to customize (classes, thresholds, speed)  
- Saves **cropped images** of detected animals for further analysis  

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sarah2553/Detr_detection.git
cd Detr_detection
```

---

### 2️⃣ Create and Activate a Virtual Environment
We recommend using **venv** to avoid dependency conflicts:
```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

---

### 3️⃣ Install Required Python Packages

You can install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Make sure you have **PyTorch** installed. If you have a GPU, install the corresponding CUDA version from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

---

### 4️⃣ Download/Prepare Video Data

- Place your video files (e.g., `.mp4`, `.avi`) in a directory, e.g., `videos/`.
- You can use any video – the detection code will process every frame.

---

### 5️⃣ Run Real-Time Animal Detection

Execute the main detection script as follows:

```bash
python video_detr_detection.py --video_path path/to/your/video.mp4
```

#### Common Options:
- `--video_path`: Path to the input video file.
- `--output_dir`: Directory to save detection results, CSVs, and cropped images.
- `--conf_thresh`: Minimum confidence score to display/buffer detections (default: 0.7).
- `--filter_classes`: List of animal classes to detect (optional).
- `--save_crops`: Save cropped animal images (default: True).


```

---

### 6️⃣ Results

- **FPS, per-class bounding boxes, and animal counts** are printed/displayed during playback.
- **Detection CSV** (`detection_stats.csv`) summarizes counts, duration, and details.
- **Cropped images** are stored per class for further analysis.

---
## 📦 Dataset & Video Files

Due to **GitHub file size limitations**, large video files (`.mp4`, `.avi`) used for testing and demonstration **are not included in this repository**.

GitHub restricts individual file sizes to **100 MB**, and this project uses larger video files for real-time detection.

### How to use your own videos:
- Download or record any animal video (`.mp4`, `.avi`)
- Place it inside a local folder (e.g. `videos/`)
- Run the detection script with: python detect_animals.py --video_path path/to/your/video.mp4

---
## 🤗 Credits

- Facebook AI Research for DETR ([paper](https://arxiv.org/abs/2005.12872), [official repo](https://github.com/facebookresearch/detr))
- PyTorch Ecosystem
- Open-source contributors




