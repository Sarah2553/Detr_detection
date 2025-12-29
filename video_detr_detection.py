import torch
import cv2
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
import time
import numpy as np
from tqdm import tqdm
import os

# -------------------------------
# 1️⃣ Load DETR model
# -------------------------------
print("📦 Loading DETR model...")
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# COCO Classes
CLASSES = [
    'N/A','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird',
    'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','N/A',
    'backpack','umbrella','N/A','N/A','handbag','tie','suitcase','frisbee','skis',
    'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','N/A','wine glass','cup','fork','knife',
    'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog',
    'pizza','donut','cake','chair','couch','potted plant','bed','N/A','dining table',
    'N/A','N/A','toilet','N/A','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','N/A','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush'
]

ANIMAL_IDS = [16,17,18,19,20,21,22,23,24,25]

# Colors per animal for boxes
COLORS = {
    "cat": (255,0,0), "dog": (0,255,0), "horse": (0,0,255),
    "sheep": (255,255,0), "cow": (255,0,255), "elephant": (0,255,255),
    "bear": (128,0,128), "zebra": (0,128,128), "giraffe": (128,128,0), "bird": (255,128,0)
}

# -------------------------------
# 2️⃣ Video Setup
# -------------------------------
video_path = "video/video.mp4v"
output_video = "result_output.mp4"
output_csv = "detection_stats.csv"
save_crops = True
crops_dir = "crops"
os.makedirs(crops_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Cannot open video"); exit()

w,h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

speed_factor = 1
fps_fast = fps 
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps_fast, (w,h))

# Detection settings
SKIP_FRAMES = 1
INFERENCE_SIZE = 360

# Statistics
animal_stats = defaultdict(int)
current_detect = []
frame_id = 0
start_time = time.time()

# -------------------------------
# Fullscreen window
# -------------------------------
cv2.namedWindow("DETR Video Processing", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("DETR Video Processing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n🚀 Processing video...\n")
for _ in tqdm(range(frames), desc="Analyzing"):
    ret, frame = cap.read()
    if not ret: break
    frame_id += 1

    if frame_id % SKIP_FRAMES == 0:
        r = INFERENCE_SIZE / max(h,w)
        resized = cv2.resize(frame, (int(w*r), int(h*r)))
        img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_model = model(img)

        prob = out_model['pred_logits'].softmax(-1)[0,:,:-1]
        keep = prob.max(-1).values > 0.70
        boxes = out_model['pred_boxes'][0][keep].cpu()
        labels = prob[keep].argmax(-1).cpu()
        confidences = prob[keep].max(-1).values.cpu()

        current_detect = []
        for i,(box,label) in enumerate(zip(boxes,labels)):
            if label.item() in ANIMAL_IDS:
                name = CLASSES[label.item()]
                confidence = confidences[i].item()
                current_detect.append((box.tolist(), name, confidence))
                animal_stats[name] += 1

                # Save cropped image
                if save_crops:
                    xc, yc, bw, bh = box
                    x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
                    x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
                    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                    cv2.imwrite(f"{crops_dir}/{name}_{frame_id}.jpg", crop)

    # Draw boxes and confidence
    for box, name, confidence in current_detect:
        xc, yc, bw, bh = box
        x1,y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
        x2,y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
        color = COLORS.get(name, (0,255,0))
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{name} {confidence:.2f}", (x1, y1-10), 0, 0.7, color, 2)

    # Draw live FPS
    fps_live = frame_id / (time.time()-start_time)
    cv2.putText(frame,f"FPS: {fps_live:.1f}", (10,25),0,0.7,(0,255,255),2)

    cv2.imshow("DETR Video Processing", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# -------------------------------
# 3️⃣ Export Results
# -------------------------------
if animal_stats:
    df = pd.DataFrame(list(animal_stats.items()), columns=["Animal","Count"])
    df["Estimated_time(seconds)"] = ((df["Count"]*SKIP_FRAMES)/fps).round(2)
    df.to_csv(output_csv, index=False)
    print("\n📊 Detection Summary:\n", df, "\n")
    print(f"💾 CSV saved → {output_csv}")
    print(f"🎥 Output video → {output_video}")
else:
    print("\n⚠ No animals detected")
