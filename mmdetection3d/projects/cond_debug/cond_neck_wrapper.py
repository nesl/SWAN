import torch
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class CondNeckWrapperPythonIf(BaseModule):
    """Intentional Python control-flow based on tensor value.
    This is expected to FAIL or trace incorrectly in ONNX export.
    """
    def __init__(self, neck, delta=0.1, init_cfg=None):
        super().__init__(init_cfg)
        self.neck = MODELS.build(neck)
        self.delta = float(delta)

    def forward(self, feats):
        outs = self.neck(feats)

        # SECONDFPN usually returns tuple of feature maps
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        x0 = outs[0]

        # ---- INTENTIONALLY BAD: tensor -> Python bool ----
        if x0.reshape(-1)[0] > 0:
            x0 = x0 + self.delta
        else:
            x0 = x0 + 0.0
        # -----------------------------------------------

        outs[0] = x0
        return tuple(outs)