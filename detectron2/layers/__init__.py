# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .batch_norm import FrozenBatchNorm2d, get_norm, NaiveSyncBatchNorm
# SMOKE-TEST PATCH: the compiled detectron2._C extension (CUDA/C++ custom ops)
# is not built in this environment. DeformConv / rotated-ROIAlign are unused by
# RankSaliencyNetwork, so make them optional instead of a hard import-time failure.
try:
    from .deform_conv import DeformConv, ModulatedDeformConv
except ImportError:
    DeformConv = ModulatedDeformConv = None
from .mask_ops import paste_masks_in_image
from .nms import batched_nms, batched_nms_rotated, nms, nms_rotated
from .roi_align import ROIAlign, roi_align
try:
    from .roi_align_rotated import ROIAlignRotated, roi_align_rotated
except ImportError:
    ROIAlignRotated = roi_align_rotated = None
from .shape_spec import ShapeSpec
from .wrappers import BatchNorm2d, Conv2d, ConvTranspose2d, cat, interpolate, Linear

__all__ = [k for k in globals().keys() if not k.startswith("_")]
