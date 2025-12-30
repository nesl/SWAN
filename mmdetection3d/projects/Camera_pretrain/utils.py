from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from timm.models.layers import DropPath
from torch import nn
@HOOKS.register_module()
class CameraAnealHook(Hook):
    def after_train_epoch(self, runner):
        model = runner.model
        # unwrap DDP if necessary
        if hasattr(model, 'module'):
            model = model.module

        model.epoch += 1
        model.sample_idx = 0

        runner.logger.info(f"Reset sample_idx; now epoch={model.epoch}")



@HOOKS.register_module()
class DropPathHook(Hook):
    priority = 'NORMAL'
    def __init__(self, start_rate=0.0, end_rate=0.4, total_epochs=50):
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.total_epochs = total_epochs
        self.debug = False

    def before_train_epoch(self, runner):
        progress = runner.epoch / self.total_epochs
        current_rate = self.start_rate + (self.end_rate - self.start_rate) * progress
        print(f"Setting drop path rate to {current_rate} at epoch {runner.epoch}")
        # Update all DropPath modules
        def update_drop_path(module):
            if hasattr(module, "drop_path_rate"):
                module.drop_path_rate = current_rate
                if self.debug:
                    print(f"Updated drop_path_rate to {current_rate}")
            if hasattr(module, "drop_path"):
                if(isinstance(module.drop_path, float)):
                    module.drop_path = current_rate
                    if self.debug:
                        print(f"Updated drop_path float to {current_rate}")
                elif isinstance(module.drop_path, nn.Module):
                    module.drop_path.drop_prob = current_rate
                    if self.debug:
                        print(f"Updated drop_path timm to {current_rate}")
                else:
                    print(f"Unknown drop_path type: {type(module.drop_path)}")
        runner.model.apply(update_drop_path)