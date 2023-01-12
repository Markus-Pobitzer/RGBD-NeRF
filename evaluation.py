import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from datetime import datetime

from run_nerf_helpers import mse2psnr, img2mse
from load_llff import load_llff_data
from load_blender import load_blender_data
import lpips
from piqa import SSIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

zero = torch.tensor(0., device=device).float()
# Depth metrics
abs_relative_distance = lambda x, y_gt: torch.where(y_gt > 0, (x - y_gt).abs() / y_gt,
                                                    zero).sum() / torch.count_nonzero(y_gt)
mean_squared_relative_distance = lambda x, y_gt: torch.where(y_gt > 0, (x - y_gt) ** 2 / y_gt,
                                                             zero).sum() / torch.count_nonzero(y_gt)
mse = lambda x, y: ((x - y) ** 2).sum() / torch.count_nonzero(y)
rmse_linear = lambda x, y: mse(x, y).sqrt()
rmse_log = lambda x, y: (
        torch.where(y > 0, (x.log() - y.log()) ** 2, zero).sum() / torch.count_nonzero(y)).sqrt()

ssim = SSIM().to(device)
lpips = lpips.LPIPS(net='vgg').to(device)

basedir = "/cluster/scratch/pobmarku/"
testset_numb = "testset_050000"
blender_dirs = ["nerf_rgbd-0501-0829-33/DS-RGB-D-Ship-5", "nerf_rgbd-0501-0735-35/DS-RGB-D-Ship-6",
                "nerf_rgbd-0501-0950-04/DS-RGB-D-Ship-9", "nerf_rgbd-0601-1606-19/DS-RGB-D-Ship-12",
                "nerf_rgbd-0601-1142-29/DS-RGB-D-Ship-15", "nerf_rgbd-0501-1813-56/DS-RGB-D-Ship-20",

                "nerf_rgbd-0601-0727-21/DS-RGB-Ship-5", "nerf_rgbd-0501-0949-56/DS-RGB-Ship-6",
                "nerf_rgbd-0601-1024-17/DS-RGB-Ship-9", "nerf_rgbd-0601-1605-51/DS-RGB-Ship-12",
                "nerf_rgbd-0601-1142-25/DS-RGB-Ship-15", "nerf_rgbd-0501-1814-00/DS-RGB-Ship-20"]
blender_data = "../data/ship_d"
blender_far = 6.0

llff_dirs = ["nerf_rgbd-3012-0759-13/DS-RGB-D-Fern-2", "nerf_rgbd-3012-1053-51/DS-RGB-D-Fern-3",
             "nerf_rgbd-3112-0742-19/DS-RGB-D-Fern-5", "nerf_rgbd-3112-0746-03/DS-RGB-D-Fern-6",
             "nerf_rgbd-0601-1539-40/DS-RGB-D-Fern-12",

             "nerf_rgbd-0401-1144-42/DS-RGB-Fern-2", "nerf_rgbd-0401-1133-40/DS-RGB-D-Fern-3",
             "nerf_rgbd-0401-1137-25/DS-RGB-D-Fern-5", "nerf_rgbd-0401-1135-06/DS-RGB-D-Fern-6",
             "nerf_rgbd-0601-1539-35/DS-RGB-Fern-12"]
llff_data = "../data/nerf_llff_data/fern_depth"
llff_far = 1.0

regnerf_dirs = ["regnerf/llff3/fern", "regnerf/llff6/fern", "regnerf/llff9/fern"]


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def evaluate_blender_data(gt_imgs, gt_depth, imgs, depths):
    return mse2psnr(img2mse(imgs, gt_imgs)).item(), ssim(imgs, gt_imgs).item(), lpips(imgs, gt_imgs).mean().item(), \
           abs_relative_distance(depths, gt_depth).item(), mean_squared_relative_distance(depths, gt_depth).item(), \
           mse(depths, gt_depth).item(), rmse_linear(depths, gt_depth).item(), 0  # ,rmse_log(depths, gt_depth).item()


def evaluate_blender():
    images, poses, depths, render_poses, hwf, i_split = load_blender_data(blender_data, half_res=True,
                                                                          testskip=8, load_depth=True, far=6.0)
    print('Loaded blender', images.shape, render_poses.shape, hwf, blender_data)
    i_train, i_val, i_test = i_split

    depths = depths[i_test]
    images = images[i_test]

    mask = np.where(images[..., -1:] == 0, False, True)
    # Set depth to -1 where we do not have depth information (i.e. background) s.t. we can skip those points
    # whn computing the loss
    depths = depths * images[..., -1:] + 0 * (1. - images[..., -1:])

    # cv2.imwrite("test.png", depths[2] / far * 255.)
    images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device).float()
    depths = torch.from_numpy(depths).to(device).float()

    print("=================== Blender Dataset ===================")
    print("Experiment Name", ";", "PSNR", ";", "SSIM", ";", "LPIPS", ";", "abs relative difference depth", ";",
          "sqr relative difference depth", ";", "MSE Depth", ";", "RMSE (linear) Depth", ";", "RMSE (log) Depth")
    for experiment in blender_dirs:
        experiment_name = experiment[23:]
        experiment_dir = os.path.join(basedir, experiment, testset_numb)

        all_imgs = []
        all_depths = []
        for i in range(len(i_test)):
            fname = os.path.join(experiment_dir, str(i).zfill(3) + '.png')
            all_imgs.append(imageio.imread(fname))
            dname = os.path.join(experiment_dir, str(i).zfill(3) + '_depth.png')
            all_depths.append(imageio.imread(dname))
        all_imgs = (np.array(all_imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        all_imgs = np.array(all_imgs)

        all_depths = np.where(mask, np.expand_dims(all_depths, axis=-1), 0.)
        all_depths = np.array(all_depths)
        shape = all_depths.shape
        all_depths = (blender_far * all_depths.reshape((shape[0], shape[1], shape[2], 1)) / np.max(all_depths)).astype(
            np.float32)

        # print(experiment_name, all_imgs.shape, all_depths.shape)
        all_imgs = torch.from_numpy(all_imgs).permute(0, 3, 1, 2).to(device).float()
        all_depths = torch.from_numpy(all_depths).to(device).float()
        psnr, ssim, lpips, depth_abs, depth_sqr, depth_mse, depth_rmse_linear, depth_rmse_log = \
            evaluate_blender_data(images, depths, all_imgs, all_depths)
        print(experiment_name, ";", psnr, ";", ssim, ";", lpips, ";", depth_abs, ";", depth_sqr, ";", depth_mse, ";",
              depth_rmse_linear, ";", depth_rmse_log)


def evaluate_llff():
    bd_factor = 0.75
    llff_hold = 8
    images, poses, bds, render_poses, i_test, depth_maps = load_llff_data(llff_data, factor=8,
                                                                          recenter=True, bd_factor=bd_factor,
                                                                          spherify=False,
                                                                          load_depth=True)
    images = images[::llff_hold]
    depths = depth_maps[::llff_hold] / np.max(depth_maps)
    numb_imgs = len(images)

    # print("LLFF Data", images.shape, depths.shape)

    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device).float()
    depths = torch.from_numpy(depths).to(device)

    print("=================== LLFF Dataset ===================")
    print("Experiment Name", ";", "PSNR", ";", "SSIM", ";", "LPIPS", ";", "abs relative difference depth", ";",
          "sqr relative difference depth", ";", "MSE Depth", ";", "RMSE (linear) Depth", ";", "RMSE (log) Depth")
    for experiment in llff_dirs:
        experiment_name = experiment[23:]
        experiment_dir = os.path.join(basedir, experiment, testset_numb)

        all_imgs = []
        all_depths = []
        for i in range(numb_imgs):
            fname = os.path.join(experiment_dir, str(i).zfill(3) + '.png')
            all_imgs.append(imread(fname)[..., :3] / 255.)
            dname = os.path.join(experiment_dir, str(i).zfill(3) + '_depth.png')
            all_depths.append(imread(dname))
        all_imgs = np.array(all_imgs).astype(np.float32)
        all_imgs = np.array(all_imgs)

        all_depths = np.asarray(all_depths)
        # Rescale if bd_factor is provided
        # sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
        # all_depths = all_depths * sc
        all_depths = all_depths / np.max(all_depths)

        # print(experiment_name, all_imgs.shape, all_depths.shape)
        all_imgs = torch.from_numpy(all_imgs).permute(0, 3, 1, 2).to(device).float()
        all_depths = torch.from_numpy(all_depths).to(device).float()

        psnr, ssim, lpips, depth_abs, depth_sqr, depth_mse, depth_rmse_linear, depth_rmse_log = \
            evaluate_blender_data(images, depths, all_imgs, all_depths)
        print(experiment_name, ";", psnr, ";", ssim, ";", lpips, ";", depth_abs, ";", depth_sqr, ";", depth_mse, ";",
              depth_rmse_linear, ";", depth_rmse_log)


def evaluate_rgnerf():
    bd_factor = 0.75
    llff_hold = 8
    images, poses, bds, render_poses, i_test, depth_maps = load_llff_data(llff_data, factor=8,
                                                                          recenter=True, bd_factor=bd_factor,
                                                                          spherify=False,
                                                                          load_depth=True)
    images = images[::llff_hold]
    depths = depth_maps[::llff_hold] / np.max(depth_maps)
    numb_imgs = len(images)

    # print("LLFF Data", images.shape, depths.shape)

    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device).float()
    depths = torch.from_numpy(depths).to(device)

    print("=================== RegNeRF Dataset ===================")
    print("Experiment Name", ";", "PSNR", ";", "SSIM", ";", "LPIPS", ";", "abs relative difference depth", ";",
          "sqr relative difference depth", ";", "MSE Depth", ";", "RMSE (linear) Depth", ";", "RMSE (log) Depth")
    for experiment in regnerf_dirs:
        experiment_name = experiment[8:]
        experiment_dir = os.path.join(basedir, experiment)

        all_imgs = []
        for i in range(numb_imgs):
            fname = os.path.join(experiment_dir, "color_" + str(i).zfill(3) + '.png')
            all_imgs.append(imread(fname)[..., :3] / 255.)
        all_imgs = np.array(all_imgs).astype(np.float32)
        all_imgs = np.array(all_imgs)

        # print(experiment_name, all_imgs.shape, all_depths.shape)
        all_imgs = torch.from_numpy(all_imgs).permute(0, 3, 1, 2).to(device).float()

        psnr, ssim, lpips, depth_abs, depth_sqr, depth_mse, depth_rmse_linear, depth_rmse_log = \
            evaluate_blender_data(images, zero, all_imgs, zero)
        print(experiment_name, ";", psnr, ";", ssim, ";", lpips, ";", depth_abs, ";", depth_sqr, ";", depth_mse, ";",
              depth_rmse_linear, ";", depth_rmse_log)


if __name__ == '__main__':
    # evaluate_llff()
    # evaluate_blender()
    evaluate_rgnerf()
