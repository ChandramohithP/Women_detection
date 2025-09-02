"""
realtime_person_gender.py

Production-style real-time person detection + tracking + gender classification.

Features:
 - Uses YOLOv8n (ultralytics) for fast person detection (GPU if available, CPU optimized).
 - Falls back to OpenCV DNN YOLOv3-tiny if ultralytics unavailable (requires weights/cfg).
 - Simple IoU-based multi-object tracker with persistent IDs and last_seen timestamps.
 - Batch gender classification using HuggingFace AutoImageProcessor + AutoModelForImageClassification.
 - Per-track voting buffer to stabilize gender labels; female bias multiplier available.
 - Minimum crop size checks, robust preprocessing (preserve aspect ratio), CLAHE for low light.
 - Aggregates counts per unique ID; smooths counters across short window to avoid flicker.
 - Fullscreen OpenCV visualization: bounding boxes, ID, gender + confidence, stable counters.
 - Logging & defensive error handling.

Configure parameters in the CONFIG block.
"""

import time
import math
import sys
import traceback
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------
# CONFIG - tweak these values
# ----------------------------
# Model paths / model names
USE_ULTRALYTICS = True         # if True tries YOLOv8n via ultralytics; else uses OpenCV DNN yolov3-tiny
YOLOV8_NAME = "yolov8n.pt"     # ultralytics will auto-download if not present
# If fallback to cv2.dnn is used, set the following:
YOLO3_TINY_WEIGHTS = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\weights\yolov3-tiny.weights"
YOLO3_TINY_CFG     = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\cfg\yolov3-tiny.cfg"
YOLO3_NAMES        = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\lib\coco.names"

# HuggingFace gender model directory (must contain config, model and processor)
GENDER_MODEL_PATH = r"D:\gd_model\gender-classification"

# Runtime parameters (safe defaults)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTION_CONF_THRESHOLD = 0.35   # keep detections above this confidence
NMS_IOU_THRESHOLD = 0.45
MIN_CROP_AREA = 1600              # skip tiny crops (px^2)
CLASSIFY_EVERY_N_FRAMES = 2       # classify every N frames (batch mode)
FEMALE_BIAS = 1.5                 # multiply female probability by this factor before argmax
TRACK_MAX_AGE = 1.8               # seconds to keep a track without updates
IOU_MATCH_THRESHOLD = 0.35        # IoU threshold for matching detection->track
VOTE_BUFFER_SIZE = 7              # per-track vote buffer length for stable label
STABLE_VOTE_MIN = 3               # minimum same votes in buffer to be stable
COUNTER_SMOOTH_WINDOW = 5        # frames to average counters for display
BATCH_MAX_CROPS = 12              # max crops to batch-classify at once

# Visualization config
WINDOW_NAME = "Production Person→Gender (press q to quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Debug / logging
VERBOSE = False
# ---------------------------- END CONFIG ----------------------------

# ----------------------------
# Utility / helper functions
# ----------------------------
def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def safe_rect(frame_w, frame_h, x, y, w, h):
    """Clamp box to frame and return ints (x,y,w,h)"""
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    w = int(max(1, min(w, frame_w - x1)))
    h = int(max(1, min(h, frame_h - y1)))
    return x1, y1, w, h

def iou(boxA, boxB):
    # box: [x,y,w,h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0

def resize_preserve_aspect(img, target_short=224):
    """Resize so shorter side == target_short, keep aspect, then center-crop to square target_short x target_short."""
    h, w = img.shape[:2]
    if min(h, w) == 0:
        return None
    scale = target_short / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    # center crop to target_short x target_short
    cy, cx = nh // 2, nw // 2
    half = target_short // 2
    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    crop = resized[y1:y1 + target_short, x1:x1 + target_short]
    if crop.shape[0] != target_short or crop.shape[1] != target_short:
        crop = cv2.resize(crop, (target_short, target_short))
    return crop

def clahe_enhance(bgr_roi):
    """Apply CLAHE on L channel to help low-light frames."""
    try:
        lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr_roi

# ----------------------------
# Tracker (IoU based) - lightweight production tracker
# ----------------------------
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int,int,int,int]           # x,y,w,h
    last_seen: float
    history_votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_BUFFER_SIZE))
    gender: Optional[str] = None
    gender_conf: float = 0.0
    stable: bool = False

class IoUTracker:
    def __init__(self, max_age=TRACK_MAX_AGE, iou_threshold=IOU_MATCH_THRESHOLD):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def step(self, detections: List[Tuple[int,int,int,int]]):
        """
        detections: list of boxes [x,y,w,h] for this frame
        returns matched_pairs list of (track_id, det_idx) and unmatched det_idx list
        """
        now = time.time()
        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        # Build IoU matrix and greedily match
        iou_rows = []
        for tid, tr in self.tracks.items():
            iou_row = [iou(tr.bbox, d) for d in detections]
            iou_rows.append((tid, iou_row))

        # Greedy
        used_dets = set()
        for tid, iou_row in sorted(iou_rows, key=lambda x: -max(x[1]) if x[1] else 0):
            best_j = -1
            best_val = 0
            for j, val in enumerate(iou_row):
                if val > best_val and j not in used_dets:
                    best_val = val; best_j = j
            if best_val >= self.iou_threshold and best_j >= 0:
                matches.append((tid, best_j))
                used_dets.add(best_j)
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(best_j)

        # Update matched tracks
        for tid, j in matches:
            self.tracks[tid].bbox = detections[j]
            self.tracks[tid].last_seen = now

        # Create new tracks for unmatched detections
        for j in list(unmatched_dets):
            self.tracks[self.next_id] = Track(track_id=self.next_id, bbox=detections[j], last_seen=now)
            matches.append((self.next_id, j))
            self.next_id += 1

        # Remove old tracks
        to_delete = []
        for tid, tr in list(self.tracks.items()):
            if now - tr.last_seen > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return matches, list(unmatched_dets)

# ----------------------------
# Classifier wrapper (batch mode)
# ----------------------------
class GenderClassifier:
    def __init__(self, model_path: str, device: str = DEVICE, female_bias: float = FEMALE_BIAS):
        try:
            self.device = device
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.labels = self.model.config.id2label
            self.female_bias = female_bias
            print(f"[INFO] Loaded gender model from {model_path}; device={self.device}; labels={self.labels}")
        except Exception as e:
            print("[ERROR] Failed to load gender model:", e)
            raise

    def predict_batch(self, crops: List[np.ndarray]) -> List[Tuple[str,float]]:
        """crops: list of BGR images (person ROIs). returns list of (label, conf)."""
        if len(crops) == 0:
            return []
        # Preprocess crops: enhance + square resize preserving aspect
        processed = []
        for c in crops:
            # skip very small
            h,w = c.shape[:2]
            if h*w < MIN_CROP_AREA:
                processed.append(None)
                continue
            c2 = clahe_enhance(c)
            square = resize_preserve_aspect(c2, target_short=224)
            if square is None:
                processed.append(None)
            else:
                processed.append(square)

        # Prepare batch for valid images
        valid_idx = [i for i, p in enumerate(processed) if p is not None]
        if not valid_idx:
            # return Unknown for all
            return [("unknown", 0.0) for _ in crops]

        images = [processed[i] for i in valid_idx]
        inputs = self.processor(images=images, return_tensors="pt")
        # move tensors to device if model is on device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        # apply female bias and renormalize per-row
        female_idx = next((k for k, v in self.labels.items() if v.lower() == "female"), None)
        if female_idx is not None:
            probs[:, female_idx] *= self.female_bias
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums

        results_valid = []
        for p in probs:
            idx = int(np.argmax(p))
            results_valid.append((self.labels[idx], float(p[idx])))

        # map back to original crops list
        results = []
        vi = 0
        for i in range(len(crops)):
            if i in valid_idx:
                results.append(results_valid[vi])
                vi += 1
            else:
                results.append(("unknown", 0.0))
        return results

# ----------------------------
# Main real-time pipeline
# ----------------------------
def main_loop(video_src=0):
    # initialize detector
    use_ultralytics = False
    detector = None
    yolo_model = None
    try:
        if USE_ULTRALYTICS:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(YOLOV8_NAME)  # loads yolov8n or file if exists
                use_ultralytics = True
                print("[INFO] Using ultralytics YOLOv8 detector:", YOLOV8_NAME)
            except Exception as e:
                print("[WARN] ultralytics not available or failed – falling back to OpenCV DNN. Reason:", e)
                use_ultralytics = False

        if not use_ultralytics:
            # fallback to OpenCV DNN yolov3-tiny
            print("[INFO] Loading OpenCV DNN YOLOv3-tiny weights (fallback).")
            net = cv2.dnn.readNet(YOLO3_TINY_WEIGHTS, YOLO3_TINY_CFG)
            # Force CPU backend to avoid CUDA backend errors in many pip builds
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open(YOLO3_NAMES, "r") as f:
                yolo_names = [l.strip() for l in f.readlines()]
            detector = {"type": "cv2-dnn", "net": net, "names": yolo_names}
        else:
            detector = {"type": "ultralytics", "model": yolo_model}

    except Exception as e:
        print("[FATAL] Detector initialization failed:", e)
        traceback.print_exc()
        sys.exit(1)

    # classifier init
    try:
        classifier = GenderClassifier(GENDER_MODEL_PATH, device=DEVICE, female_bias=FEMALE_BIAS)
    except Exception as e:
        print("[FATAL] Gender classifier failed to initialize:", e)
        traceback.print_exc()
        sys.exit(1)

    # tracker
    tracker = IoUTracker(max_age=TRACK_MAX_AGE, iou_threshold=IOU_MATCH_THRESHOLD)

    # counters smoothing
    counter_buffer = deque(maxlen=COUNTER_SMOOTH_WINDOW)

    # camera
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera:", video_src)
        return

    # try to set a lower resolution for speed (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_id = 0
    last_classify_frame = -999
    print("[INFO] Starting main loop. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed; breaking.")
                break
            frame_id += 1
            H, W = frame.shape[:2]

            detections = []
            # --- DETECTION ---
            if detector["type"] == "ultralytics":
                # ultralytics returns list of boxes in format (xyxy)
                try:
                    results = detector["model"].predict(source=frame, imgsz=640, conf=DETECTION_CONF_THRESHOLD, verbose=False)
                    # results is a list; take first
                    res = results[0]
                    # res.boxes.xyxy (tensor), res.boxes.conf, res.boxes.cls
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.array([])
                    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.array([])
                    clss = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes, "cls") else np.array([])
                    for b, c, cl in zip(boxes_xyxy, confs, clss):
                        if cl != 0:  # only person class (coco class 0)
                            continue
                        x1, y1, x2, y2 = map(int, b.tolist())
                        w = x2 - x1; h = y2 - y1
                        detections.append([x1, y1, w, h, float(c)])
                except Exception as e:
                    print("[WARN] ultralytics predict error:", e)
            else:
                # OpenCV DNN path
                net = detector["net"]
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
                net.setInput(blob)
                # get unconnected output layer names robustly
                try:
                    outs = net.forward(net.getUnconnectedOutLayersNames())
                except Exception:
                    # older OpenCV versions expect list of layer names
                    outs = net.forward()
                # outs is list of arrays
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        if scores.size == 0:
                            continue
                        class_id = int(np.argmax(scores))
                        conf = float(scores[class_id])
                        if class_id == 0 and conf >= DETECTION_CONF_THRESHOLD:
                            cx = int(detection[0] * W)
                            cy = int(detection[1] * H)
                            w = int(detection[2] * W)
                            h = int(detection[3] * H)
                            x = cx - w // 2
                            y = cy - h // 2
                            detections.append([int(x), int(y), int(w), int(h), conf])

            # Apply NMS to detections (x,y,w,h,conf)
            boxes_np = np.array([d[:4] for d in detections]) if len(detections) > 0 else np.zeros((0,4))
            scores_np = np.array([d[4] for d in detections]) if len(detections) > 0 else np.zeros((0,))
            keep_idx = []
            if len(boxes_np) > 0:
                # convert to x1,y1,x2,y2 for NMS
                x1 = boxes_np[:,0]; y1 = boxes_np[:,1]; x2 = boxes_np[:,0] + boxes_np[:,2]; y2 = boxes_np[:,1] + boxes_np[:,3]
                # clamp
                x1 = np.clip(x1, 0, W-1); y1 = np.clip(y1, 0, H-1); x2 = np.clip(x2, 0, W-1); y2 = np.clip(y2, 0, H-1)
                rects = np.stack([x1,y1,x2,y2], axis=1).astype(np.int32)
                # Use OpenCV NMSBoxes if available
                try:
                    idxs = cv2.dnn.NMSBoxes(rects.tolist(), scores_np.tolist(), DETECTION_CONF_THRESHOLD, NMS_IOU_THRESHOLD)
                    if isinstance(idxs, (list,tuple,np.ndarray)):
                        if isinstance(idxs, np.ndarray):
                            idxs = idxs.flatten().tolist()
                        else:
                            # might be list of lists
                            idxs = [int(i[0]) if isinstance(i, (list,tuple,np.ndarray)) else int(i) for i in idxs]
                    keep_idx = idxs
                except Exception:
                    # fallback naive NMS
                    order = np.argsort(-scores_np)
                    kept = []
                    for ii in order:
                        keep = True
                        for kk in kept:
                            iou_val = iou(boxes_np[ii], boxes_np[kk])
                            if iou_val > NMS_IOU_THRESHOLD:
                                keep = False; break
                        if keep: kept.append(ii)
                    keep_idx = kept

            kept_boxes = [detections[i] for i in keep_idx] if len(keep_idx) > 0 else []

            # Tracker matching (IoU-based)
            det_boxes_only = [[b[0], b[1], b[2], b[3]] for b in kept_boxes]
            matches, unmatched = tracker.step(det_boxes_only)

            # Build crops for classification on selected frames (batch)
            crops_for_class = []
            crop_track_ids = []
            classify_now = (frame_id - last_classify_frame) >= CLASSIFY_EVERY_N_FRAMES
            if classify_now:
                last_classify_frame = frame_id

            for (tid, det_idx) in matches:
                box = kept_boxes[det_idx]
                x, y, w, h = safe_rect(W, H, box[0], box[1], box[2], box[3])
                area = w * h
                if area < MIN_CROP_AREA:
                    # too small; skip classification but update track bbox
                    continue
                if classify_now:
                    crop = frame[y:y+h, x:x+w].copy()
                    crops_for_class.append(crop)
                    crop_track_ids.append(tid)

            # Batch classify
            results = classifier.predict_batch(crops_for_class) if len(crops_for_class) > 0 else []

            # assign results back to tracks
            for (tid, (label, conf)) in zip(crop_track_ids, results):
                tr = tracker.tracks.get(tid)
                if tr is None:
                    continue
                tr.history_votes.append(label)
                tr.gender_conf = conf
                # majority vote
                cnt = Counter(tr.history_votes)
                if len(cnt) > 0:
                    maj_label, maj_cnt = cnt.most_common(1)[0]
                    tr.gender = maj_label
                    tr.stable = maj_cnt >= STABLE_VOTE_MIN

            # Compose display: iterate tracks
            male_total = 0
            female_total = 0
            display_frame = frame.copy()
            for tid, tr in tracker.tracks.items():
                x, y, w, h = tr.bbox
                x, y, w, h = safe_rect(W, H, x, y, w, h)
                # draw box and label; stability affects thickness
                if tr.gender is None:
                    color = (180,180,180)
                    label = f"ID{tid} ?"
                else:
                    g = str(tr.gender).lower()
                    if "female" in g or "woman" in g:
                        color = (200,50,200)
                        female_total += 1 if tr.stable else 0
                    elif "male" in g or "man" in g:
                        color = (50,150,255)
                        male_total += 1 if tr.stable else 0
                    else:
                        color = (180,180,180)
                    label = f"ID{tid} {tr.gender[:1]} {tr.gender_conf:.2f}" if tr.gender_conf else f"ID{tid} {tr.gender[:1]}"

                thickness = 2 if tr.stable else 1
                cv2.rectangle(display_frame, (x,y), (x+w, y+h), color, thickness)
                cv2.putText(display_frame, label, (x, max(12, y-6)), FONT, 0.5, color, 2)

            # smoothing counts
            counter_buffer.append((male_total, female_total))
            m_avg = int(np.mean([c[0] for c in counter_buffer])) if counter_buffer else 0
            f_avg = int(np.mean([c[1] for c in counter_buffer])) if counter_buffer else 0

            # draw counters UI (top-left)
            cv2.rectangle(display_frame, (0,0), (300,80), (0,0,0), -1)
            cv2.putText(display_frame, f"👨 Males (stable): {m_avg}", (10,28), FONT, 0.7, (0,200,255), 2)
            cv2.putText(display_frame, f"👩 Females (stable): {f_avg}", (10,60), FONT, 0.7, (200,30,200), 2)

            # FPS
            now = time.time()
            fps = frame_id / (now - start_time) if (now - start_time) > 0 else 0.0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (W-120, 28), FONT, 0.6, (200,200,0), 2)

            # Show fullscreen window
            cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt: exiting.")
    except Exception as e:
        print("[ERROR] Unexpected error in main loop:", e)
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = time.time()
    main_loop(0)
