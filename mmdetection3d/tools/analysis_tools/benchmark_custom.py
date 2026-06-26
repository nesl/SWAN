# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import numpy as np
import torch
from mmengine import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint
from mmengine.config import  DictAction
from mmdet3d.registry import MODELS
from tools.misc.fuse_conv_bn import fuse_module
from torch.utils.flop_counter import FlopCounterMode

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Whether to use automatic mixed precision inference')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope('mmdet3d')
    
    # build config and set cudnn_benchmark
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.env_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    model.to(get_device())
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 50
    pure_inf_time = 0
    # benchmark with several samples and take the average
    total_flops = []
    for i, data in enumerate(dataloader):
        if i > 550:
            break

        if i > num_warmup:
            with FlopCounterMode(display=True) as fcm:
                with torch.no_grad():
                    model.test_step(data)
                total_flops.append(fcm.get_total_flops())
        else:
            with torch.no_grad():
                with autocast(enabled=args.amp):
                    model.test_step(data)



        print("Average Flops", np.mean(total_flops))
        
    np.save('flops.npy', total_flops)

if __name__ == '__main__':
    main()
