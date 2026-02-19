

_base_ = ['/workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_Multicorrupt_test.py']

model = dict(
    controller = dict(
        type='ConvLayerController',
        layer_budget=10
        # Use defaults for now
    )
)