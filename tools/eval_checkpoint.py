#!/usr/bin/env python
"""
Evaluate a trained checkpoint on RVSOD test data: Spearman rank correlation,
MAE, F-measure (via evaluation/inference.py -- the same code plain_test_net.py
and periodic in-training eval use).

Full data/RVSOD_pkl/test.pkl is 3152 images -- at CPU speeds (~1-2s/image for
inference alone) that's over an hour. --limit lets you run a quick subset
for a sanity check; drop it (or set to 0) for a real full-test-set number,
which you'd normally only do on a GPU anyway.

Usage:
    source .venv-smoke/bin/activate
    python tools/eval_checkpoint.py --config data/config_rvsod.yaml \
        --weights output/rvsod_stage2/model_final.pth --limit 30
"""
import argparse
import os
import pickle
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
sys.path.insert(0, os.path.join(path, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.DEVICE = args.device

    if args.limit:
        full_path = cfg.DATASETS.TEST[0]
        data = pickle.load(open(full_path, "rb"))[: args.limit]
        limited_path = "/tmp/eval_test_subset.pkl"
        pickle.dump(data, open(limited_path, "wb"))
        cfg.DATASETS.TEST = (limited_path,)
        print(f"using a {len(data)}-image subset of {full_path} for speed")

    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(args.weights, resume=False)
    print(f"loaded {args.weights}")

    from evaluation.inference import inference
    # r_corre is computed by evaluation/spearman_correlation.py, whose own
    # internal naming ("Sprman"/rank_evalu) is a misnomer -- it's IoU-matched,
    # Pearson-correlated rank agreement with a penalty for unmatched
    # instances, which is exactly SA-SOR as defined in the paper this metric
    # comes from (Liu et al., TPAMI 2021, reference [2] -- verified against
    # their official code release, byte-for-byte the same implementation).
    r_corre, m_f, _r_map = inference(cfg, model)
    sasor_all = m_f["sasor_all"]

    print("\n=== Results ===")
    print(f"SA-SOR (Original, excl. single-instance):        {r_corre:.4f}")
    print(f"SA-SOR (All, single-instance included as 0):      {sasor_all:.4f}")
    print(f"SA-SOR (Normalized Original, (x+1)/2):            {(r_corre+1)/2:.4f}")
    print(f"SA-SOR (Normalized All, (x+1)/2):                 {(sasor_all+1)/2:.4f}"
          "   <- directly comparable to the paper's 'Ours' row: 0.603 on RVSOD")
    print(f"MAE (SOD-style, binary -- matches this repo's mae_fmeasure_2.py): {m_f['mae']:.4f}"
          "   <- paper's 'Ours' row: 0.0698 on RVSOD")
    print(f"F-measure: {m_f['f_measure']:.4f}   (not in the paper's Table 1, extra info)")


if __name__ == "__main__":
    main()
