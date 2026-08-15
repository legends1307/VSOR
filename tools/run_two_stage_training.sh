#!/usr/bin/env bash
# Runs the paper's full two-stage recipe (section 4.1) end to end:
#   stage 1: SalientDetector, COCO-pretrained init, SGD          (data/config_stage1.yaml)
#   stage 2: RankSaliencyNetwork, init from stage 1's output, Adam  (data/config_rvsod.yaml)
#
# Both stages are resumable (`--resume`, once per invocation of this script)
# -- if interrupted, just re-run this script; each `plain_train_net_our.py`
# call will pick up from its own output dir's last_checkpoint rather than
# restarting, and stage 2 won't re-run if stage 1's model_final.pth already
# exists (see the check below) -- it'll re-launch stage 2 directly.
#
# Needs a GPU for any real run -- see the "how long would this take" estimate
# from earlier in the project notes. CPU works for a smoke-scale run (e.g.
# `SOLVER.MAX_ITER 5` -- config overrides are a bare trailing positional,
# argparse.REMAINDER, NOT a --opts-prefixed flag) to sanity-check the
# pipeline, not for the full 500k/200k iteration counts below.
#
# Usage:
#   source .venv-smoke/bin/activate
#   bash tools/dataset_prep/fetch_pretrained.sh   # once
#   bash tools/run_two_stage_training.sh
#   bash tools/run_two_stage_training.sh SOLVER.MAX_ITER 5   # smoke-scale run
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Auto-detect: configs default MODEL.DEVICE to cuda (correct for a real GPU
# run). On a CPU-only machine that crashes ("Torch not compiled with CUDA
# enabled") unless overridden -- so detect it here instead of relying on
# remembering to pass MODEL.DEVICE cpu (or NOT pass it) depending on which
# machine you're on.
DEVICE_OPTS=()
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "(no CUDA GPU detected -- adding MODEL.DEVICE cpu)"
    DEVICE_OPTS=(MODEL.DEVICE cpu)
fi

STAGE1_CFG="data/config_stage1.yaml"
STAGE2_CFG="data/config_rvsod.yaml"
# NOTE: plain_train_net_our.py's checkpointer uses args.output_dir (the CLI
# --output_dir flag) for where it saves checkpoints, NOT cfg.OUTPUT_DIR from
# the yaml (that field is unused for this purpose here) -- and
# default_argument_parser() auto-generates an incrementing default
# (./Rank_Saliency/Models/RVSOR(1), RVSOR(2), ...) if --output_dir isn't
# passed explicitly, which would make the path below unpredictable. Passing
# --output_dir explicitly keeps this script's checkpoint-handoff logic correct.
STAGE1_OUT="./output/rvsod_stage1"
STAGE2_OUT="./output/rvsod_stage2"
STAGE1_FINAL="$STAGE1_OUT/model_final.pth"

if [ ! -f "$STAGE1_FINAL" ]; then
    echo "=== Stage 1: detector pretraining (SGD, COCO init) ==="
    python tools/plain_train_net_our.py --config-file "$STAGE1_CFG" --resume \
        --output_dir "$STAGE1_OUT" "${DEVICE_OPTS[@]}" "$@"
    if [ ! -f "$STAGE1_FINAL" ]; then
        echo "Stage 1 finished but $STAGE1_FINAL wasn't produced -- check SOLVER.MAX_ITER" \
             "was actually reached (a partial/interrupted run only writes periodic" \
             "checkpoints like model_0009999.pth, not model_final.pth)." >&2
        exit 1
    fi
else
    echo "=== Stage 1 already complete ($STAGE1_FINAL exists), skipping ==="
fi

echo "=== Stage 2: graph module + full fine-tune (Adam, init from stage 1) ==="
python tools/plain_train_net_our.py --config-file "$STAGE2_CFG" --resume \
    --output_dir "$STAGE2_OUT" "${DEVICE_OPTS[@]}" \
    MODEL.WEIGHTS "$STAGE1_FINAL" "$@"
