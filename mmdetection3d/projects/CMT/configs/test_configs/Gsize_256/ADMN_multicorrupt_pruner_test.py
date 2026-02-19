_base_ = ['./ADMN_multicorrupt_test.py']


model = dict(
    enable_pruning=True,
    use_hard_pruning=True
)