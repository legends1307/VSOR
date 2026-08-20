#!/usr/bin/env python
"""
Convert RVSOD's raw annotations into the .pkl schema that
detectron2/data/my_dataset_mapper.py (training) and davis_val (eval) expect:

    list[dict], each dict:
        file_name: str
        width, height: int
        annotations: list[dict]     -- one per SALIENT instance
            bbox: [x, y, w, h]       (XYWH_ABS)
            segmentation: [[x1,y1,...]]
            is_person: int
        rank: list[int]             -- ordinal 1..N, HIGHER = MORE SALIENT
                                       (verified: loss.py wants score to rise with rank)

Single source, no .mat: ranking saliency masks/img/<clip>/<clip>_NN.png,
HxW greyscale. Distinct non-zero grey level = distinct saliency level
(brighter = more salient); connected components within a level = instances.
Same method as testing_script.py, validated there on real frames.

Verified: greyscale level tracks eye-fixation density at r=+0.986, 99.4%
pairwise agreement (testing_script_report.txt) -- matches the paper's
"ranks assigned based on the distribution of fixation points" (S4.1).

Known limitation: instances split into disconnected blobs by occlusion (same
level, not touching) become two instances -- measured at ~84% frame-level
instance-count fidelity vs a .mat-based ground truth check. Also is_person
is absent from RVSOD; defaults to 0.

Usage:
    python tools/dataset_prep/build_rvsod_pkl.py --split test --out data/RVSOD_pkl/test.pkl
"""
import argparse
import json
import os
import pickle
from collections import Counter

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_AREA = 20


def convert_frame(jpg_path, png_path):
    im = cv2.imread(jpg_path)
    if im is None:
        return None, "unreadable jpg"
    h, w = im.shape[:2]

    grey = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if grey is None:
        return None, "missing greyscale png"
    if grey.ndim == 3:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    if grey.shape != (h, w):
        gh, gw = grey.shape
        if abs((gw / gh) - (w / h)) / (w / h) > 0.02:
            return None, f"shape mismatch: png {grey.shape} vs jpg {(h, w)}"
        grey = cv2.resize(grey, (w, h), interpolation=cv2.INTER_NEAREST)

    objs = []
    for level in sorted(int(v) for v in np.unique(grey) if v > 0):
        num, labels, stats, _ = cv2.connectedComponentsWithStats((grey == level).astype(np.uint8), 8)
        for i in range(1, num):
            x, y, cw, ch, area = stats[i]
            if area < MIN_AREA:
                continue
            comp = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            polygons = [c.reshape(-1).tolist() for c in contours if c.shape[0] >= 3]
            if not polygons:
                continue
            objs.append({"bbox": [int(x), int(y), int(cw), int(ch)],
                        "segmentation": polygons, "is_person": 0, "level": level})

    if not objs:
        return None, "no salient instances"

    # brightest first (most salient), but rank VALUE must be highest for the
    # most salient instance: relation_head/loss.py is minimized when the
    # predicted score increases with the rank value. Equal greyscale levels
    # share a rank.
    objs.sort(key=lambda o: o["level"], reverse=True)
    levels_desc = sorted({o["level"] for o in objs}, reverse=True)
    rank_of = {lv: len(levels_desc) - i for i, lv in enumerate(levels_desc)}
    ranks = [rank_of[o["level"]] for o in objs]
    for o in objs:
        del o["level"]

    return {"file_name": jpg_path, "width": w, "height": h,
            "annotations": objs, "rank": ranks}, None


def build_split(split):
    img_root = os.path.join(REPO_ROOT, "data", "RVSOD", split, "img")
    png_root = os.path.join(REPO_ROOT, "data", "RVSOD", split, "ranking saliency masks", "img")

    dataset, skipped = [], {}
    for clip in sorted(os.listdir(img_root)):
        clip_dir = os.path.join(img_root, clip)
        if not os.path.isdir(clip_dir):
            continue
        for fname in sorted(os.listdir(clip_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            png_path = os.path.join(png_root, clip, fname[:-4] + ".png")
            if not os.path.exists(png_path):
                skipped["missing .png"] = skipped.get("missing .png", 0) + 1
                continue
            try:
                entry, reason = convert_frame(os.path.join(clip_dir, fname), png_path)
            except Exception as e:
                entry, reason = None, f"exception: {type(e).__name__}: {e}"
            if entry is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            dataset.append(entry)
    return dataset, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test", "validation"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    dataset, skipped = build_split(args.split)

    n_inst = [len(d["annotations"]) for d in dataset]
    dist = Counter(n_inst)
    lines = [
        f"split           : {args.split}",
        f"rank source     : greyscale png only (no .mat)",
        f"frames converted: {len(dataset)}",
        f"skipped         : {json.dumps(skipped)}",
        f"instances/frame : min={min(n_inst)} max={max(n_inst)} mean={sum(n_inst)/len(n_inst):.2f}",
        "instance-count distribution:",
    ] + [f"    {k} instance(s): {dist[k]} ({100*dist[k]/len(dataset):.1f}%)" for k in sorted(dist)]
    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dataset, f)
    print("wrote", args.out)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write(report + f"\nwrote: {args.out}\n")
        print("wrote", args.report)


if __name__ == "__main__":
    main()
