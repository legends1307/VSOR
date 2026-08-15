#!/usr/bin/env bash
# Downloads pretrained checkpoints referenced by data/config_stage1.yaml and
# tools/smoke_visualize.py, to a local path -- bypasses this fork's
# detectron2:// PathManager resolution, which doesn't work against the
# pip-installed (much newer) fvcore in .venv-smoke (see: AssertionError
# "Checkpoint detectron2://... not found!" if you try MODEL.WEIGHTS with the
# detectron2:// prefix directly).
set -euo pipefail
mkdir -p /tmp/pretrained

echo "Fetching ImageNet-pretrained ResNet-50 backbone (MSRA/R-50, ~102MB)..."
curl -sL --max-time 180 -o /tmp/pretrained/R-50.pkl \
  "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl"

echo "Fetching COCO-pretrained Mask R-CNN R50-FPN 3x (~178MB)..."
curl -sL --max-time 180 -o /tmp/pretrained/mask_rcnn_R_50_FPN_3x_coco.pkl \
  "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

ls -la /tmp/pretrained/
