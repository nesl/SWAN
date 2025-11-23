from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class UpdateEpochHook(Hook):
    def after_train_epoch(self, runner):
        model = runner.model
        # unwrap DDP if necessary
        if hasattr(model, 'module'):
            model = model.module

        model.epoch += 1
        model.sample_idx = 0

        runner.logger.info(f"Reset sample_idx; now epoch={model.epoch}")