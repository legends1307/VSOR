#!/usr/bin/env python
"""
Evaluate a checkpoint and PERSIST everything to txt for reporting.

Writes, under --out-dir (default results/eval/):
    <ckpt>__<split>.txt   full per-split report (metrics + config + params)
    summary.tsv           one appended row per (checkpoint, split) run

Usage:
    python tools/eval_checkpoint.py --config data/config_rvsod.yaml \
        --weights output/rvsod_stage2/model_final.pth --device cuda

    # all three splits, full sets
    python tools/eval_checkpoint.py --config data/config_rvsod.yaml \
        --weights output/rvsod_stage2/model_final.pth --device cuda \
        --splits train,test,validation

    # quick sanity subset
    ... --splits test --limit 50
"""
import argparse
import datetime
import os
import pickle
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
sys.path.insert(0, os.path.join(path, "tools"))


def param_report(model):
    """total / trainable / frozen params, plus a top-level module breakdown."""
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [f"total parameters      : {tot:,}",
             f"trainable parameters  : {tr:,}",
             f"frozen parameters     : {tot - tr:,}",
             "",
             "per top-level module (trainable / total):"]
    for name, mod in model.named_children():
        m_tot = sum(p.numel() for p in mod.parameters())
        m_tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if m_tot:
            lines.append(f"    {name:<22} {m_tr:>12,} / {m_tot:>12,}")
    return "\n".join(lines), tot, tr


def cfg_report(cfg):
    s = cfg.SOLVER
    return "\n".join([
        f"META_ARCHITECTURE     : {cfg.MODEL.META_ARCHITECTURE}",
        f"MASK_ON               : {cfg.MODEL.MASK_ON}",
        f"BACKBONE              : {cfg.MODEL.BACKBONE.NAME}",
        f"ROI_HEADS.NUM_CLASSES : {cfg.MODEL.ROI_HEADS.NUM_CLASSES}",
        f"SCORE_THRESH_TEST     : {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}",
        f"RELATION_HEAD         : {cfg.MODEL.RELATION_HEAD.NAME}",
        f"  Relation_Unit_Nums  : {cfg.MODEL.RELATION_HEAD.Relation_Unit_Nums}",
        f"  POOLER_EXPAND_RATIO : {cfg.MODEL.RELATION_HEAD.POOLER_EXPAND_RATIO}",
        f"  MLP_HEAD_DIM        : {cfg.MODEL.RELATION_HEAD.MLP_HEAD_DIM}",
        f"SOLVER.OPTIMIZER      : {getattr(s, 'OPTIMIZER', 'ADAM')}",
        f"SOLVER.BASE_LR        : {s.BASE_LR}",
        f"SOLVER.STEPS          : {s.STEPS}",
        f"SOLVER.MAX_ITER       : {s.MAX_ITER}",
        f"SOLVER.IMS_PER_BATCH  : {s.IMS_PER_BATCH}",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", default="test",
                    help="comma-separated: test,train,validation")
    ap.add_argument("--pkl-dir", default="data/RVSOD_pkl")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--tag", default="", help="optional label for this run")
    args = ap.parse_args()

    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    import torch

    out_dir = os.path.join(path, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(args.weights))[0]
    summary_path = os.path.join(out_dir, "summary.tsv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("timestamp\tcheckpoint\tsplit\tn_images\tn_scored\t"
                    "sasor_original\tsasor_all\tsasor_norm_original\tsasor_norm_all\t"
                    "mae\ttrainable_params\ttag\n")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(path, args.config))
        cfg.MODEL.DEVICE = args.device

        pkl = os.path.join(path, args.pkl_dir, f"{split}.pkl")
        if not os.path.exists(pkl):
            print(f"!! skipping {split}: {pkl} not found")
            continue
        n_total = len(pickle.load(open(pkl, "rb")))
        if args.limit:
            data = pickle.load(open(pkl, "rb"))[: args.limit]
            pkl = f"/tmp/eval_{split}_subset.pkl"
            pickle.dump(data, open(pkl, "wb"))
            n_total = len(data)
        cfg.DATASETS.TEST = (pkl,)
        cfg.freeze()

        model = build_model(cfg)
        model.eval()
        DetectionCheckpointer(model).resume_or_load(args.weights, resume=False)
        params_txt, _, n_trainable = param_report(model)

        print(f"\n### evaluating {ckpt_name} on {split} ({n_total} images) ###")
        from evaluation.inference import inference
        # r_corre comes from evaluation/spearman_correlation.py -- IoU-matched,
        # Pearson-correlated rank agreement with an unmatched-instance penalty,
        # i.e. SA-SOR. RAW, range [-1,1]; zyf-815's own code reports this
        # (variant='original'). Normalized rows are convenience only.
        r_corre, m_f, _ = inference(cfg, model)
        sasor_all = m_f.get("sasor_all", float("nan"))
        n_scored = m_f.get("n_scored", "?")

        body = "\n".join([
            "=" * 68,
            f"checkpoint : {args.weights}",
            f"config     : {args.config}",
            f"split      : {split}   ({n_total} images"
            + (f", --limit {args.limit}" if args.limit else "") + ")",
            f"device     : {args.device}",
            f"timestamp  : {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"tag        : {args.tag or '-'}",
            "=" * 68,
            "",
            "--- METRICS ---",
            f"SA-SOR (Original, excl. single-instance) : {r_corre:.4f}   <- what zyf-815's code reports",
            f"SA-SOR (All, single-instance counted 0)  : {sasor_all:.4f}",
            f"SA-SOR (Normalized Original, (x+1)/2)    : {(r_corre+1)/2:.4f}",
            f"SA-SOR (Normalized All, (x+1)/2)         : {(sasor_all+1)/2:.4f}",
            f"MAE (SOD-style, binarized)               : {m_f['mae']:.4f}",
            f"F-measure                                : {m_f['f_measure']:.4f}  (hardcoded 0 upstream; not computed)",
            "",
            "paper Table 1 'Ours' on RVSOD: SA-SOR 0.603, MAE 0.0698",
            "(the paper never states WHICH SA-SOR variant it reports -- see",
            " evaluation/spearman_correlation.py docstring)",
            "",
            "--- MODEL / TRAINABLE PARAMETERS ---",
            params_txt,
            "",
            "--- CONFIG ---",
            cfg_report(cfg),
            "",
        ])
        print(body)

        rpt = os.path.join(out_dir, f"{ckpt_name}__{split}.txt")
        with open(rpt, "w") as f:
            f.write(body)
        with open(summary_path, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{ckpt_name}\t"
                    f"{split}\t{n_total}\t{n_scored}\t{r_corre:.4f}\t{sasor_all:.4f}\t"
                    f"{(r_corre+1)/2:.4f}\t{(sasor_all+1)/2:.4f}\t{m_f['mae']:.4f}\t"
                    f"{n_trainable}\t{args.tag or '-'}\n")
        print(f"wrote {rpt}")
        print(f"appended {summary_path}")

        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None


if __name__ == "__main__":
    main()
