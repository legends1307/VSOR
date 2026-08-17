#!/usr/bin/env python
"""
Runs RankSaliencyNetwork on 3 consecutive frames (t-1, t, t+1) and saves the
ranked detections for frame t.

Usage:
    python tools/smoke_visualize.py
    python tools/smoke_visualize.py --weights "" --config data/config.yaml   # random weights
"""
import argparse
import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)

import numpy as np
import cv2
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="data/config_rvsod.yaml")
ap.add_argument("--weights", default="external_sources_outputs/model_0099999.pth")
ap.add_argument("--clip", default="data/RVSOD/test/img/actioncliptest00035")
ap.add_argument("--out", default="tools/smoke_prediction.jpg")
ap.add_argument("--frame", type=int, default=3, help="index of the middle (t) frame")
ap.add_argument("--score-thresh", type=float, default=0.6, help="same filter evaluation/inference.py uses")
args = ap.parse_args()

clip_dir = os.path.join(path, args.clip)
out_path = os.path.join(path, args.out)

cfg = get_cfg()
cfg.merge_from_file(os.path.join(path, args.config))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
torch.manual_seed(0)
model = build_model(cfg)

if args.weights:
    weights_path = os.path.join(path, args.weights) if not os.path.isabs(args.weights) else args.weights
    DetectionCheckpointer(model).resume_or_load(weights_path, resume=False)
    print("loaded checkpoint:", weights_path)
else:
    print("random weights")
model.eval()

files = sorted(os.listdir(clip_dir))
mid = args.frame
chosen = files[mid - 1: mid + 2]
print("frames (t-1, t, t+1):", chosen)

# native resolution -- neither the train mapper nor davis_val resizes, so the
# model has only ever seen native RVSOD frames
frames = [cv2.imread(os.path.join(clip_dir, f)) for f in chosen]
to_tensor = lambda im: torch.as_tensor(np.ascontiguousarray(im.transpose(2, 0, 1)))
batched_inputs = [[{"image": to_tensor(im)} for im in frames]]

with torch.no_grad():
    out = model(batched_inputs)

roi = out["roi_results"][0]
print(f"detected {len(roi)} instances")

canvas = frames[1].copy()
h, w = canvas.shape[:2]

if len(roi) == 0:
    cv2.imwrite(out_path, canvas)
    print("no detections, wrote plain frame:", out_path)
    sys.exit(0)

rank = out["rank_result"][0].squeeze(-1).cpu().numpy()
boxes = roi.pred_boxes.tensor.cpu().numpy()
scores = roi.scores.cpu().numpy()
masks = roi.pred_masks.cpu().numpy()

keep = scores > args.score_thresh
boxes, scores, masks, rank = boxes[keep], scores[keep], masks[keep], rank[keep]
print(f"{keep.sum()} detections above score>{args.score_thresh}")

order = np.argsort(-rank)
palette = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]

drawn = 0
for idx in order:
    if drawn >= 6:
        break
    x0, y0, x1, y1 = boxes[idx].astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        continue
    color = palette[drawn % len(palette)]
    drawn += 1

    mask = cv2.resize(masks[idx, 0], (x1 - x0, y1 - y0)) > 0.5
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[y0:y1, x0:x1] = mask
    overlay = canvas.copy()
    overlay[mask_full > 0] = color
    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    label = f"rank#{drawn} score={scores[idx]:.2f} r={rank[idx]:.1f}"
    cv2.putText(canvas, label, (x0, max(y0 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    print(f"  rank#{drawn}: box={boxes[idx].astype(int).tolist()} score={scores[idx]:.3f} rank_score={rank[idx]:.3f}")

cv2.imwrite(out_path, canvas)
print(f"drew {drawn}/{len(roi)} boxes -> {out_path}")
