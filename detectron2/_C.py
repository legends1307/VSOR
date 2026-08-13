# SMOKE-TEST STUB
# The real `detectron2._C` is a compiled CUDA/C++ extension (deform conv,
# rotated ROIAlign/NMS/box-iou, ...) built via setup.py against a specific
# torch/CUDA ABI. It isn't built in this environment. RankSaliencyNetwork's
# forward pass doesn't touch any of these ops (it only uses plain ROIAlign,
# which has been repointed at torchvision in layers/roi_align.py), so this
# stub only needs to satisfy import-time attribute lookups. If any of these
# are actually *called*, that's a signal the code path genuinely needs the
# real compiled extension.


def __getattr__(name):
    def _unavailable(*args, **kwargs):
        raise NotImplementedError(
            f"detectron2._C.{name} was called, but detectron2._C is a smoke-test "
            "stub (no compiled extension in this environment). This op "
            "(DeformConv / rotated ROIAlign / rotated NMS / box_iou_rotated) "
            "needs the real detectron2 C++/CUDA extension built."
        )
    return _unavailable
