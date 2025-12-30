# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import torch

def plot_results_lidar(result, index, out_dir):
    SMALL_SIZE = 200
    MEDIUM_SIZE = 300
    BIGGER_SIZE = 400

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    import numpy as np
    extent = result["point_cloud_range"][::3] + result["point_cloud_range"][1::3]

    vx, vy, vz = result["voxel_shape"]

    x_range = np.arange(extent[0] + vx / 2, extent[1] - vx / 2, vx - 1e-16)
    y_range = np.arange(extent[2] + vy / 2, extent[3] - vy / 2, vy - 1e-16)
    X, Y = np.meshgrid(x_range, y_range)

    # if result["occupied_bev"] is not None:
        # store["occupied_bev"][i] = result["occupied_bev"][0].detach().cpu().numpy().astype(np.int8).T
    """batch_size = result["occupied_bev"].shape[0]
        # vmin, vmax = -1, 5
        # cticks = [-1, 0, 1, 2, 3, 4, 5]
        for b in range(batch_size):
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(200, 100))
            occ_bev =result["occupied_bev"][b].detach().cpu().numpy().T

            # Even bounds give a contour-like effect:
            bounds = np.linspace(-1.5, 5.5, 8)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=7)
            cMap = colors.ListedColormap(
                ["w", 'limegreen', 'darkgreen', "orangered", "darkred", "gold",  "darkgoldenrod"])
            pcm = ax1.pcolormesh(X, Y, occ_bev, norm=norm, cmap=cMap)
            pcm2 = ax2.pcolormesh(X[50:150, 100:], Y[50:150, 100:], occ_bev[50:150, 100:], norm=norm, cmap=cMap)
            cb = fig.colorbar(pcm, orientation='vertical')
            cb.ax.set_xticks(cticks, ['Empty', 'True Unmasked ', 'False Unmasked', 'True Masked ', 'False Masked', 'False Fake', 'True Fake'])

            #im = plt.imshow(occ_bev, extent=extent, vmin=vmin, vmax=vmax)
            plt.suptitle(f"Occupied prediction, Datapoint {i}, batch {b}")
            plt.savefig(f"occ_pred_{i}_{b}.png")
            plt.close()
            data_dict = {
                "n_points": (occ_bev > -1).sum(),
                "TN": ((occ_bev == 0) | (occ_bev == 2)).sum(),
                "FP": ((occ_bev == 1) | (occ_bev == 3)).sum(),
                "FN": (occ_bev == 4).sum(),
                "TP": (occ_bev == 5).sum(),
                "sample": i*batch_size+b,
            }
            data_dict["FPR"] = data_dict["FP"]/(data_dict["TN"]+data_dict["FP"])
            data_dict["FNR"] = data_dict["FN"]/(data_dict["TP"]+data_dict["FN"])
            data_dict["Recall"] = data_dict["TP"]/(data_dict["TP"]+data_dict["FN"])  # TPR
            data_dict["Precision"] = data_dict["TP"]/(data_dict["TP"]+data_dict["FP"])
            data_dict["Accuracy"] = (data_dict["TP"] + data_dict["TN"]) / data_dict["n_points"]
            occ_data.append(data_dict)"""

    # if result["gt_num_points_bev"] is not None:
        # store["gt_num_points_bev"][i] = result["gt_num_points_bev"][0].detach().cpu().numpy().T
    """batch_size = result["gt_num_points_bev"].shape[0]
        for b in range(batch_size):
            fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(200, 200))
            gt_num_points_bev = result["gt_num_points_bev"][b].detach().cpu().numpy().T


            diff_num_points_bev = result["diff_num_points_bev"][b].detach().cpu().numpy().T
            pred_num_points = gt_num_points_bev - diff_num_points_bev

            vmin = gt_num_points_bev[gt_num_points_bev != 0].min()
            assert X.shape == gt_num_points_bev.shape
            pcm = ax1.pcolor(X, Y, gt_num_points_bev,
                                norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                cmap='PuBu_r', shading='auto')
            ax1.set_title("Ground truth")

            pcm2 = ax2.pcolor(X, Y, pred_num_points,
                                norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                cmap='PuBu_r', shading='auto')
            ax2.set_title("Predicted")
            pcm3 = ax3.pcolor(X[80:120, 100:140], Y[80:120, 100:140], gt_num_points_bev[80:120, 100:140],
                                norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                cmap='PuBu_r', shading='auto')
            pcm4 = ax4.pcolor(X[80:120, 100:140], Y[80:120, 100:140], pred_num_points[80:120, 100:140],
                                norm=colors.LogNorm(vmin=vmin, vmax=gt_num_points_bev.max()),
                                cmap='PuBu_r', shading='auto')
            fig.colorbar(pcm, extend='max')
            plt.title(f"Number of points per voxel BEV, Datapoint {i}, batch {b}")
            plt.savefig(f"gt_num_points_bev{i}_{b}.png")
            plt.close()"""
    # if result["diff_num_points_bev"] is not None:
        # store["diff_num_points_bev"][i] = result["diff_num_points_bev"][0].detach().cpu().numpy().T
    """batch_size = result["diff_num_points_bev"].shape[0]
        for b in range(batch_size):
            fig = plt.figure(figsize=(100, 100))
            diff_num_points_bev = result["diff_num_points_bev"][b].detach().cpu().numpy().T
            diff_num_points_bev = np.abs(diff_num_points_bev)
            vmin = diff_num_points_bev[diff_num_points_bev != 0].min()
            vmax = diff_num_points_bev.max()
            assert X.shape == diff_num_points_bev.shape
            pcm = plt.pcolor(X, Y, diff_num_points_bev,
                                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                cmap='PuBu_r', shading='auto')
            fig.colorbar(pcm, extend='both')
            plt.title(f"Diff in predicted number of points per voxel BEV, Datapoint {i}, batch {b}")
            plt.savefig(f"diff_num_points_bev{i}_{b}.png")
            plt.close()"""
    if result["points"] is not None:
        # xticks = np.arange(result["point_cloud_range"][0], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][0])
        xticks_large = np.arange(-50, 50 + 0.000001, 0.5*16)
        xticks_small = np.arange(0, 15 + 0.000001, 0.5)
        xmask = xticks_small % 5 == 0
        xlabels = [round(xticks_small[i], 2) if xmask[i] else None for i in range(xticks_small.size)]
        # xmask = xticks % 10 == 0
        # xmask = np.diff((xticks / 10).astype(int), append=0.0) > 0
        # xlabels = [round(xticks[i], 2) if xmask[i] else None for i in range(xticks.size)]

        # yticks = np.arange(result["point_cloud_range"][1], result["point_cloud_range"][3] + 0.000001, step=result["voxel_shape"][1])
        yticks_large = np.arange(-50, 50 + 0.000001, 0.5*16)
        yticks_small = np.arange(-7.5, 7.5 + 0.000001, 0.5)
        ymask = yticks_small % 5 == 0
        ylabels = [round(yticks_small[i], 2) if ymask[i] else None for i in range(yticks_small.size)]
        # ymask = np.diff((yticks / 10).astype(int), append=0.0) > 0
        # ymask = xticks % 10 == 0
        # ylabels = [round(yticks[i], 2) if ymask[i] else None for i in range(yticks.size)]

        batch = result["points_batch"]
        gt_batch = result["gt_points_batch"]
        batch_size = int(result["gt_points_batch"].max().item()) + 1
        for b in range(batch_size):
            points = result["points"][torch.where(batch == b)].detach().cpu().numpy()
            gt_points = result["gt_points"][torch.where(gt_batch == b)].detach().cpu().numpy()

            cmin = min(points[:, 2].min(), gt_points[:, 2].min())
            cmax = min(points[:, 2].max(), gt_points[:, 2].max())
            color = (points[:, 2] - cmin)/(cmax - cmin)
            gt_color = (gt_points[:, 2] - cmin)/(cmax - cmin)
            gt_mask = (gt_points[:, 0] > 0) & (gt_points[:, 0] < 15) & (gt_points[:, 1] > -7.5) & (gt_points[:, 1] < 7.5)
            p_mask = (points[:, 0] > 0) & (points[:, 0] < 15) & (points[:, 1] > -7.5) & (points[:, 1] < 7.5)

            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(200, 200))
            ax1.scatter(gt_points[:, 0], gt_points[:, 1], s=60, c=gt_color, label="GT")
            ax1.set_title("Ground truth")
            ax1.set_xticks(xticks_large)
            ax1.set_yticks(yticks_large)
            ax1.grid()
            ax2.scatter(points[:, 0], points[:, 1], s=60, c=color, label="Predicted")
            ax2.set_title("Predicted")
            ax2.set_xticks(xticks_large)
            ax2.set_yticks(yticks_large)
            ax2.grid()
            ax3.scatter(gt_points[gt_mask][:, 0], gt_points[gt_mask][:, 1], s=45*45, c=gt_color[gt_mask], label="GT")
            ax3.set_title("Ground truth")
            ax3.set_xticks(xticks_small, xlabels)
            ax3.set_yticks(yticks_small, ylabels)
            ax3.grid()
            ax4.scatter(points[p_mask][:, 0], points[p_mask][:, 1], s=45*45, c=color[p_mask], label="Predicted")
            ax4.set_title("Predicted")
            ax4.set_xticks(xticks_small, xlabels)
            ax4.set_yticks(yticks_small, ylabels)
            ax4.grid()
            f.suptitle(f"Predicted point locations, Datapoint {index}, batch {b}")
            plt.savefig(f"{out_dir}/chamf/chamf_points_bev{index}_{b}.png")
            plt.close()

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        default='viz')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args





def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    model = runner.model
    print(model.load_state_dict(torch.load(args.checkpoint)['state_dict'], strict=False))
    from pathlib import Path
    Path('viz/chamf').mkdir(parents=True, exist_ok=True)
    Path('viz/cam').mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(runner.test_dataloader):
        preprocess_out = model.data_preprocessor(sample)
        batch_input, batch_samples = preprocess_out['inputs'], preprocess_out['data_samples']
        test_output = model.test_pretrain(batch_input, batch_samples, args.show_dir, i)
        plot_results_lidar(test_output, i, 'viz')


    


if __name__ == '__main__':
    main()
