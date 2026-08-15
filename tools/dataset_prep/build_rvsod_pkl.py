#!/usr/bin/env python
"""
Convert RVSOD's raw annotation format into the .pkl schema
detectron2/data/my_dataset_mapper.py (and davis_val, for eval) expect:

    list[dict], each dict:
        file_name: str                 -- path to the jpg frame
        width, height: int
        annotations: list[dict]        -- one per salient instance
            bbox: [x, y, w, h]          (XYWH_ABS)
            segmentation: [[x1,y1,x2,y2,...]]   (single polygon, list-of-lists)
            is_person: int              (0/1 -- NOT derivable from RVSOD, see below)
        rank: list[int]                -- one integer per instance, same order as annotations.
                                           HIGHER = MORE SALIENT (see convention note below);
                                           only relative order matters to the loss, not scale.

Source data (per split: train/test/validation):
    data/RVSOD/<split>/img/<clip>/<clip>_NN.jpg
    data/RVSOD/<split>/ranking saliency masks/mat/<clip>/<clip>_NN.mat
        -> key "img": HxW uint8 array, per-pixel INSTANCE id.
           0 = background, 1..N = one integer per salient instance.

RANK CONVENTION (verified empirically, do not change without re-checking):
    RVSOD's .mat instance IDs run MOST-salient-first: ID 1 is the most salient
    instance, ID N the least. Verified against the dataset's own eye-fixation
    data ("eye fixation data/mat/<clip>.mat", FixationPerFrame): counting
    fixation points landing inside each instance mask over 400 multi-instance
    test frames gives corr(instance_id, fixation_count) = -0.40, and "higher
    ID => more fixations" holds for only 24% of instance pairs (i.e. it is
    reliably the reverse).

    The training loss (RelationLossComputation.loss_compute) is minimized when
    the predicted saliency score INCREASES with the `rank` value -- verified
    directly: monotonically-increasing scores give loss 0.22 vs 1.72 for
    decreasing. So `rank` must be higher-is-more-salient, and we invert the
    raw ID accordingly:  rank = (max_id + 1) - instance_id.

    (An earlier version of this script wrote `rank = instance_id` unchanged,
    which trained the model to rank saliency backwards. Because eval uses the
    same labels, SA-SOR was still self-consistent -- but the model's actual
    predictions were semantically inverted. Any .pkl built before this fix
    must be regenerated.)

Known limitation: is_person is not present anywhere in RVSOD. We default it
to 0 for every instance. It's an auxiliary prior signal for the relation
head (CompressPersonFeature / person_probs), not something the core ranking
loss depends on -- so this is a simplification, not a correctness bug, but
worth knowing if results look off for person-heavy clips.

Usage:
    source .venv-smoke/bin/activate
    python tools/dataset_prep/build_rvsod_pkl.py --split test  --out /tmp/rvsod_test.pkl
    python tools/dataset_prep/build_rvsod_pkl.py --split train --out /tmp/rvsod_train.pkl
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import scipy.io as sio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mask_to_polygons(mask):
    """Binary mask -> list of polygons (list of flat [x1,y1,x2,y2,...] lists),
    same approach as my_dataset_mapper.py's rle_to_polygon (cv2.findContours)."""
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polygons = []
    for c in contours:
        if c.shape[0] < 3:  # degenerate, cv2 needs >=3 points for a polygon
            continue
        polygons.append(c.reshape(-1).tolist())
    return polygons


def convert_frame(jpg_path, mat_path):
    """Returns one dataset dict, or None if this frame should be skipped."""
    rank_map = sio.loadmat(mat_path)["img"]  # HxW, 0=bg, 1..N=instance id & rank
    if rank_map.ndim != 2:
        return None, f"unexpected mat 'img' ndim={rank_map.ndim} shape={rank_map.shape}"
    im = cv2.imread(jpg_path)
    if im is None:
        return None, "unreadable jpg"
    h, w = im.shape[:2]
    if rank_map.shape != (h, w):
        # Some clips' rank .mat is at full source resolution while img/ was
        # downscaled -- same aspect ratio, different overall scale. Recover
        # these with a nearest-neighbor resize (preserves integer instance
        # IDs; any other interpolation would blend IDs into invalid values).
        mh, mw = rank_map.shape
        aspect_mat, aspect_jpg = mw / mh, w / h
        if abs(aspect_mat - aspect_jpg) / aspect_jpg > 0.02:  # >2% off -> genuinely mismatched, skip
            return None, f"shape mismatch: mat {rank_map.shape} vs jpg {(h, w)}"
        rank_map = cv2.resize(rank_map, (w, h), interpolation=cv2.INTER_NEAREST)

    instance_ids = sorted(v for v in np.unique(rank_map) if v != 0)
    if not instance_ids:
        return None, "0 salient instances"
    max_instance_id = max(instance_ids)

    annotations = []
    ranks = []
    for inst_id in instance_ids:
        inst_mask = (rank_map == inst_id)
        ys, xs = np.nonzero(inst_mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        polygons = mask_to_polygons(inst_mask)
        if not polygons:
            continue
        annotations.append({
            "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],  # XYWH_ABS
            "segmentation": polygons,
            "is_person": 0,  # not derivable from RVSOD, see module docstring
        })
        # Invert: RVSOD IDs are most-salient-first (ID 1 = most salient), but
        # the ranking loss wants higher `rank` = more salient. See the
        # RANK CONVENTION note in the module docstring for the verification.
        ranks.append(int(max_instance_id) + 1 - int(inst_id))

    if not annotations:
        return None, "all instances degenerate after polygon extraction"

    return {
        "file_name": jpg_path,
        "width": w,
        "height": h,
        "annotations": annotations,
        "rank": ranks,
    }, None


def build_split(split):
    img_root = os.path.join(REPO_ROOT, "data", "RVSOD", split, "img")
    mat_root = os.path.join(REPO_ROOT, "data", "RVSOD", split, "ranking saliency masks", "mat")

    clips = sorted(os.listdir(img_root))
    dataset = []
    skipped = {}
    for clip in clips:
        clip_img_dir = os.path.join(img_root, clip)
        clip_mat_dir = os.path.join(mat_root, clip)
        if not os.path.isdir(clip_img_dir):
            continue
        for fname in sorted(os.listdir(clip_img_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            jpg_path = os.path.join(clip_img_dir, fname)
            mat_path = os.path.join(clip_mat_dir, fname[:-4] + ".mat")
            if not os.path.exists(mat_path):
                skipped["missing .mat"] = skipped.get("missing .mat", 0) + 1
                continue
            try:
                entry, reason = convert_frame(jpg_path, mat_path)
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
    ap.add_argument("--limit", type=int, default=None, help="only convert first N clips (debug)")
    args = ap.parse_args()

    dataset, skipped = build_split(args.split)

    print(f"split={args.split}: {len(dataset)} frames converted")
    if skipped:
        print("skipped:", skipped)
    if dataset:
        n_inst = [len(d["annotations"]) for d in dataset]
        print(f"instances per frame: min={min(n_inst)} max={max(n_inst)} "
              f"mean={sum(n_inst)/len(n_inst):.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dataset, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
