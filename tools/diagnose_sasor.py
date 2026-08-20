#!/usr/bin/env python
"""
Decompose SA-SOR into its two independent failure modes.

SA-SOR penalizes BOTH bad detection and bad ranking: GT instances that no
prediction matches at IoU>=thresh get rank_index=0, which drags the
correlation down even if the ranking of the matched instances is perfect.
A single SA-SOR number can't tell you which one is hurting you.

This reports:
  1. detection recall @IoU0.5  -- what fraction of GT instances got matched
  2. SA-SOR on FULLY-MATCHED images only -- pure ranking quality, detection
     failures removed from the picture
  3. SA-SOR overall -- what eval_checkpoint.py reports

Reading it:
  low recall + high fully-matched SA-SOR -> your DETECTOR is the bottleneck
      (stage 1 undertrained / score threshold too strict)
  high recall + low fully-matched SA-SOR -> your RANKING HEAD is the
      bottleneck (stage 2 undertrained / graph or loss issue)

Usage:
    python tools/diagnose_sasor.py --config data/config_rvsod.yaml \
        --weights output/rvsod_stage2/model_final.pth --device cuda --limit 300
"""
import argparse
import os
import sys

import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
sys.path.insert(0, os.path.join(path, "tools"))


def calc_iou(a, b):
    inter = (a + b >= 2).astype(np.float32).sum()
    union = (a + b >= 1).astype(np.float32).sum()
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score-thresh", type=float, default=0.6,
                    help="must match evaluation/inference.py's hardcoded 0.6")
    args = ap.parse_args()

    import pickle
    import cv2
    import torch
    import pandas as pd
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data.davis import davis_val

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.DEVICE = args.device
    if args.limit:
        data = pickle.load(open(cfg.DATASETS.TEST[0], "rb"))[: args.limit]
        pickle.dump(data, open("/tmp/diag_subset.pkl", "wb"))
        cfg.DATASETS.TEST = ("/tmp/diag_subset.pkl",)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(args.weights, resume=False)
    print("loaded", args.weights)

    dataset = davis_val(cfg, False)

    recalls, p_full, p_all = [], [], []
    n_fail = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                out = model([item])
                gt_masks = item[1]["gt_masks"]
                gt_ranks = item[1]["gt_rank"]
                shape = item[1]["image_shape"]

                roi = out["roi_results"][0]
                scores = roi.scores.cpu().numpy()
                keep = scores > args.score_thresh
                boxes = roi.pred_boxes.tensor.cpu().numpy()[keep]
                masks = roi.pred_masks.cpu().numpy()[keep]
                rank = out["rank_result"][0].squeeze(-1).cpu().numpy()[keep]

                segmaps = np.zeros([len(masks), shape[0], shape[1]])
                for j in range(len(masks)):
                    x0, y0, x1, y1 = boxes[j].astype(int)
                    x1 = max(x1, x0 + 1)
                    y1 = max(y1, y0 + 1)
                    m = cv2.resize(masks[j, 0], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)
                    segmaps[j, max(y0, 0):y1, max(x0, 0):x1] = m[
                        max(-y0, 0):, max(-x0, 0):][:y1 - max(y0, 0), :x1 - max(x0, 0)]
                segmaps = (segmaps > 0.5).astype(np.uint8)

                if len(gt_ranks) < 2:
                    continue

                # greedy IoU match GT -> prediction
                matched_score = []
                used = set()
                for g in gt_masks:
                    best, best_j = 0.0, -1
                    for j in range(len(segmaps)):
                        if j in used:
                            continue
                        v = calc_iou(g, segmaps[j])
                        if v > best:
                            best, best_j = v, j
                    if best >= args.iou:
                        used.add(best_j)
                        matched_score.append(float(rank[best_j]))
                    else:
                        matched_score.append(None)

                n_matched = sum(s is not None for s in matched_score)
                recalls.append(n_matched / len(gt_masks))

                gt_index = np.array([sorted(gt_ranks).index(a) + 1 for a in gt_ranks])

                # overall: unmatched -> 0 (the SA-SOR penalty)
                got = [s for s in matched_score if s is not None]
                order = {v: k + 1 for k, v in enumerate(sorted(got))}
                rank_index = np.array([0 if s is None else order[s] for s in matched_score])
                if pd.Series(rank_index).var() != 0:
                    c = pd.Series(gt_index).corr(pd.Series(rank_index), method="pearson")
                    if not np.isnan(c):
                        p_all.append(c)

                # fully-matched images only: pure ranking
                if n_matched == len(gt_masks) and len(gt_masks) >= 2:
                    ri = np.array([order[s] for s in matched_score])
                    if pd.Series(ri).var() != 0:
                        c = pd.Series(gt_index).corr(pd.Series(ri), method="pearson")
                        if not np.isnan(c):
                            p_full.append(c)
            except Exception:
                n_fail += 1
            print(f"\r{i+1}/{len(dataset)}", end="", flush=True)

    print()
    print(f"\nfailed images: {n_fail}")
    print(f"multi-instance images scored: {len(p_all)}")
    print(f"fully-matched images: {len(p_full)}")
    print()
    print(f"1. detection recall @IoU{args.iou}:        {np.mean(recalls):.4f}")
    print(f"2. SA-SOR, fully-matched only (ranking):  {np.mean(p_full) if p_full else float('nan'):.4f}")
    print(f"3. SA-SOR, overall:                       {np.mean(p_all) if p_all else float('nan'):.4f}")
    print()
    print("low recall + high (2) -> detector bottleneck (stage 1 / score threshold)")
    print("high recall + low (2) -> ranking bottleneck (stage 2 / graph / loss)")


if __name__ == "__main__":
    main()
