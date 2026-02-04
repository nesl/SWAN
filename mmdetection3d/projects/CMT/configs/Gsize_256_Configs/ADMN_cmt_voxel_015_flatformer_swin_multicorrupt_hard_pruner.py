

_base_ = ['./ADMN_cmt_voxel_015_flatformer_swin_multicorrupt_soft_pruner.py']

model = dict(
    use_hard_pruning=True,
)
