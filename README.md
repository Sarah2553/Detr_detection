# Real-Time Animal Detection in Videos using DETR 🐶🐱🦓
This project demonstrates real-time animal detection in videos using DETR (DEtection TRansformer).

DETR is an end-to-end object detection model developed by Facebook AI that replaces traditional object detection pipelines (such as Faster R-CNN with anchors and NMS) with a single Transformer-based architecture. It formulates object detection as a set prediction problem using a transformer encoder–decoder and bipartite matching loss. 

## 🔑 Key Features
Transformer-based architecture for direct set prediction of object classes and bounding boxes

Eliminates anchors and post-processing heuristics like Non-Maximum Suppression

Processes video datasets frame by frame

Displays real-time FPS

Draws colored bounding boxes per animal class with confidence scores

Maintains per-animal detection counts saved to a CSV file

Estimates detection duration per animal (in seconds)

Modular and easy to customize (thresholds, classes, speed)

Saves cropped images of detected animals for further analysis

## 🛠️ Installation 

1️⃣ Clone the Repository
```bash 
git clone https://github.com/Sarah2553/Detr_detection.git
cd detr-video-detection

2️⃣ Create and Activate a Virtual Environment
python -m venv .venv


Windows

.venv\Scripts\activate


Linux / macOS

source .venv/bin/activate

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

▶️ Running the Project

Place your input video inside the video/ directory, then run:
bash
python video_detr_detection.py