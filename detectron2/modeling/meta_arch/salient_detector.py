# Stage-1 (detector-only) architecture for the paper's two-stage recipe
# (section 4.1): "We adopt the training strategy outlined in [3], utilizing
# the pre-trained Mask R-CNN learned on the MS-COCO 2017 training split...
# the box head solely dedicated to classifying objects and non-objects."
#
# RankSaliencyNetwork (rcnn.py) always runs the relation head and always
# needs a 3-frame (before, current, after) input -- there's no way to get
# "just the detector" out of it. This is a plain single-frame Mask R-CNN
# (backbone -> FPN -> RPN -> ROI heads), reusing this fork's own backbone/
# proposal_generator/roi_heads builders so it stays consistent with stage 2's
# architecture (same ResNet-FPN, same StandardROIHeads w/ person head), just
# without bottom_up_fuse/PAnetFPN/relation_head, none of which the detector
# branch needs.
#
# To reuse the existing triplet-shaped data loader (build_rank_saliency_train_loader)
# without writing a separate single-frame loader, this model accepts the same
# batched_inputs shape ([[before, current, after]]) and simply ignores the
# before/after frames -- wasteful (loads 3 images, uses 1) but zero new
# data-loading code, and this is a one-time pretraining stage, not the
# expensive stage-2 loop.
import torch
from torch import nn

from detectron2.structures import ImageList

from ..backbone import build_backbone
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["SalientDetector"]


@META_ARCH_REGISTRY.register()
class SalientDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _current_frame_only(batched_inputs):
        # batched_inputs is [[before, current, after]] (triplet loader output);
        # the detector-only stage only needs the current frame.
        return [batched_inputs[0][1]]

    def forward(self, batched_inputs, its=0):
        if not self.training:
            return self.inference(batched_inputs)

        batched_input_cur = self._current_frame_only(batched_inputs)
        images = self.preprocess_image(batched_input_cur)
        gt_instances = [x["instances"].to(self.device) for x in batched_input_cur]

        _, features_fpn = self.backbone(images.tensor)

        proposals, proposal_losses = self.proposal_generator(images, features_fpn, gt_instances)
        _, detector_losses, _ = self.roi_heads(images, features_fpn, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        batched_input_cur = self._current_frame_only(batched_inputs)
        images = self.preprocess_image(batched_input_cur)

        _, features_fpn = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features_fpn, None)
        results, _, _ = self.roi_heads(images, features_fpn, proposals)

        return {"roi_results": results}
