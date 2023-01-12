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

import wandb

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data, load_colmap_depth, load_colmap_llff
from load_dtu import load_dtu_data
from load_blender import load_blender_data
from load_4dor import load_4dor_data

from loss import SigmaLoss

from data import RayDataset
from torch.utils.data import DataLoader

from utils.generate_renderpath import generate_renderpath
import cv2

# import time

# concate_time, iter_time, split_time, loss_time, backward_time = [], [], [], [], []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(2)
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """
    Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, depths=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # B x 8
    if depths is not None:
        rays = torch.cat([rays, depths.reshape(-1, 1)], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []
    depths = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], retraw=True, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depth = depth.cpu().numpy()
        depths.append(depth)
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            print("max:", np.nanmax(depth))
            # depth = depth / 5 * 255
            # depth_color = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
            # depth_color[np.isnan(depth_color)] = 0
            # imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth_color)
            imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth)
            np.savez(os.path.join(savedir, '{:03d}.npz'.format(i)), rgb=rgb.cpu().numpy(), disp=disp.cpu().numpy(),
                     acc=acc.cpu().numpy(), depth=depth)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, depths


def render_test_ray(rays_o, rays_d, hwf, ndc, near, far, use_viewdirs, N_samples, network, network_query_fn, **kwargs):
    H, W, focal = hwf
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    z_vals = z_vals.reshape([rays_o.shape[0], N_samples])

    rgb, sigma, depth_maps, weights = sample_sigma(rays_o, rays_d, viewdirs, network, z_vals, network_query_fn)

    return rgb, sigma, z_vals, depth_maps, weights


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.alpha_model_path is None:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        print('Alpha model reloading from', args.alpha_model_path)
        ckpt = torch.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                             input_ch=input_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(
                device)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                                  alpha_model=alpha_model).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.sigma_loss:
        render_kwargs_train['sigma_loss'] = SigmaLoss(args.N_samples, args.perturb, args.raw_noise_std)

    ##########################

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                sigma_loss=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    if network_fn is not None:
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)
    else:
        # rgb_map, disp_map, acc_map = None, None, None
        # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        # noise = 0
        # alpha = network_query_fn(pts, viewdirs, network_fine.alpha_model)[...,3]
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                         pytest=pytest)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                         pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if sigma_loss is not None and ray_batch.shape[-1] > 11:
        depths = ray_batch[:, 8]
        ret['sigma_loss'] = sigma_loss.calculate_loss(rays_o, rays_d, viewdirs, near, far, depths, network_query_fn,
                                                      network_fine)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def save_depth_differences(abs_depth_diff, testsavedir, far=None, granularity=255):
    """
        Input:
            abs_depth_diff: List of Torch tensors. An entry in the List corresponds to the absolute difference of two
            depth maps.
            testsavedir: Path to where the output images are saved
            granularity: How many interpolation steps between 0 (blue) and 1 (red) in the ouput image
        Saves the absolute difference of two depth maps in a blue-red scaled image. Where blue corresponds to 0 and red
        to the max depth difference in the abs_depth_diff input.
    """
    for depth_index in range(abs_depth_diff.size(0)):
        depth_diff = abs_depth_diff[depth_index].cpu()

        # from https://stackoverflow.com/questions/64071648/converting-grey-image-to-blue-and-red-image
        img = depth_diff.numpy()
        img = np.expand_dims(img, -1)

        print("Min: ", img.min(), "Max:", img.max())

        # Just do define min and max errors
        img[0][0][0] = 0.
        if far is not None:
            img[0][1][0] = far

        print("After redefining. Min: ", img.min(), "Max:", img.max())

        interpolation = np.linspace(0, 100, granularity)
        # find some percentiles for grayscale range of src image
        percentiles = np.percentile(img, interpolation)

        # define the same count of values to further interpolation
        targets = np.geomspace(10, 255, granularity)

        # use interpolation from percentiles to targets for blue and red
        r = np.interp(img, percentiles, targets).astype(np.uint8)
        g = np.zeros_like(img)
        b = np.interp(img, percentiles, targets[::-1]).astype(np.uint8)

        # print("Img size: ", img.shape, " b-channel", b.shape, "g-channel", g.shape, "r-channel", r.shape)

        filename = os.path.join(testsavedir, 'depth_diff_{:03d}.png'.format(depth_index))
        # merge channels to BGR image
        # result = cv2.merge([b, g, r])
        result = np.concatenate((b, g, r), axis=2)
        # print("Final result: ", result.shape)
        cv2.imwrite(filename, result)
        print("Depth difference saved at:", filename)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the train set instead of render_poses path')
    parser.add_argument("--render_mypath", action='store_true',
                        help='render the test path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # debug
    parser.add_argument("--debug", action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                        help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                        help="normalize depth before calculating loss")
    parser.add_argument("--depth_rays_prop", type=float, default=0.5,
                        help="Proportion of depth rays.")

    # Added parameters
    parser.add_argument("--use_depth", action='store_true',
                        help='loads the ground truth depth maps and uses them for training')
    parser.add_argument("--no_test_view", action='store_true',
                        help='loads all the available data as training data for llff')
    parser.add_argument("--wandb", action='store_true', help="Use Weights and Biases for loggin")
    parser.add_argument("--quantitative_evaluation", action='store_true', help="Perform a quantitative evaluation if"
                                                                               "render_only is set to True")
    parser.add_argument("--numb_imgs_train", type=int, default=-1,
                        help='choose numb_imgs_train images as training images from all possible training images.')
    parser.add_argument("--rgb_lambda", type=float, default=1,
                        help="Lambda of image used for loss.")
    parser.add_argument("--entity", type=str, default="",
                        help='WandB entity name')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    if not args.render_only:
        args.basedir += datetime.now().strftime('-%d%m-%H%M-%S')
    basedir = args.basedir
    if args.numb_imgs_train > 0:
        args.expname += "-" + str(args.numb_imgs_train)
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # print("Sigma Loss", args.sigma_loss)

    if args.wandb:
        os.environ[
            'WANDB_START_METHOD'] = 'thread'  # hack: https://github.com/wandb/client/issues/1771#issuecomment-859670559
        wandb.init(
            project="RGBD-NeRF",
            resume=False,
            entity=args.entity,
            name=args.expname,
            id=wandb.util.generate_id(),
            # id=opt.wandb_name,
            dir=args.basedir,
            save_code=False
        )

    depths = []
    if args.dataset_type == 'colmap_llff':
        train_imgs, test_imgs, train_poses, test_poses, render_poses, depth_gts, bds = load_colmap_llff(args.datadir)
        poses = np.concatenate([train_poses, test_poses], axis=0)
        images = np.concatenate([train_imgs, test_imgs], axis=0)
        hwf = train_poses[0, :3, -1]
        train_poses = train_poses[:, :3, :4]
        test_poses = test_poses[:, :3, :4]
        poses = poses[:, :3, :4]
        print('Loaded colmap llff', images.shape, render_poses.shape, hwf, args.datadir)
        i_train = list(range(train_poses.shape[0]))
        i_test = list(range(train_poses.shape[0], poses.shape[0]))
        i_val = i_test
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    elif args.dataset_type == 'llff':
        if args.colmap_depth:
            depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
        images, poses, bds, render_poses, i_test, depth_maps = load_llff_data(args.datadir, args.factor,
                                                                              recenter=True, bd_factor=.75,
                                                                              spherify=args.spherify,
                                                                              load_depth=args.use_depth)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0 or args.no_test_view:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        if args.use_depth:
            print("Rescaling depth to range 0", far)
            depth_maps = depth_maps / np.max(depth_maps) * far
    elif args.dataset_type == 'dtu':
        images, poses, hwf = load_dtu_data(args.datadir)
        print('Loaded DTU', images.shape, poses.shape, hwf, args.datadir)
        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        near = 0.1
        far = 5.0
        if args.colmap_depth:
            depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
    elif args.dataset_type == 'blender':
        near = 2.
        far = 6.

        images, poses, depths, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res,
                                                                              args.testskip, args.use_depth, far=far)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        max_depth = far

        if args.white_bkgd:
            if args.use_depth:
                # Make sure to set depth = far everywhere where we do not have depth information (i.e. no background)
                # depths = depths * images[..., -1:] + far * (1. - images[..., -1:])

                # Set depth to -1 where we do not have depth information (i.e. background) s.t. we can skip those points
                # whn computing the loss
                depths = depths * images[..., -1:] + -1 * (1. - images[..., -1:])

            # cv2.imwrite("test.png", depths[2] / far * 255.)
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        if args.use_depth:
            depth_maps = depths.squeeze()
            print('Loaded depth maps', depth_maps.shape)
    elif args.dataset_type == 'FourDOR':
        near = 0.1
        far = 5.0

        images, poses, hwf, depth_maps, poses_depth, hwf_depth = load_4dor_data(args.datadir)

        # render_poses = generate_renderpath(np.array(poses)[3:4], hwf[2], sc=1)
        render_poses = poses

        print('Loaded 4dor', images.shape, poses.shape, render_poses.shape, hwf, args.datadir)
        i_train = [0, 1, 2, 3, 4]
        i_val = [5]
        i_test = [5]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    if args.numb_imgs_train > 0:
        # Choose args.numb_imgs_train training images from our possible training images to train from.
        len_train = len(i_train) - 1
        i_train = i_train[np.linspace(0, len_train, args.numb_imgs_train, dtype=int)]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
    elif args.render_mypath:
        # render_poses = generate_renderpath(np.array(poses[i_test]), focal)
        render_poses = generate_renderpath(np.array(poses[i_test])[3:4], focal, sc=1)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    print("Searching pretrained models in: ", f)
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
                if args.use_depth:
                    depth_maps = depth_maps[i_test]
            elif args.render_train:
                images = images[i_train]
                if args.use_depth:
                    depth_maps = depth_maps[i_train]
            else:
                # Default is smoother render_poses path
                images = None

            if args.render_test:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
            elif args.render_train:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
            else:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if args.render_test_ray:
                # rays_o, rays_d = get_rays(H, W, focal, render_poses[0])
                index_pose = i_train[0]
                rays_o, rays_d = get_rays_by_coord_np(H, W, focal, poses[index_pose, :3, :4],
                                                      depth_gts[index_pose]['coord'])
                rays_o, rays_d = torch.Tensor(rays_o).to(device), torch.Tensor(rays_d).to(device)
                rgb, sigma, z_vals, depth_maps, weights = render_test_ray(rays_o, rays_d, hwf,
                                                                          network=render_kwargs_test['network_fine'],
                                                                          **render_kwargs_test)
                # sigma = sigma.reshape(H, W, -1).cpu().numpy()
                # z_vals = z_vals.reshape(H, W, -1).cpu().numpy()
                # np.savez(os.path.join(testsavedir, 'rays.npz'), rgb=rgb.cpu().numpy(), sigma=sigma.cpu().numpy(), z_vals=z_vals.cpu().numpy())
                # visualize_sigma(sigma[0, :].cpu().numpy(), z_vals[0, :].cpu().numpy(), os.path.join(testsavedir, 'rays.png'))
                for k in range(20):
                    visualize_weights(weights[k * 100, :].cpu().numpy(), z_vals[k * 100, :].cpu().numpy(),
                                      os.path.join(testsavedir, f'rays_weights_%d.png' % k))
                print("colmap depth:", depth_gts[index_pose]['depth'][0])
                print("Estimated depth:", depth_maps[0].cpu().numpy())
                print(depth_gts[index_pose]['coord'])
            else:
                # print("Max GT Images:", torch.max(images).item())
                rgbs, disps, estimated_depth = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images,
                                          savedir=testsavedir, render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
                disps[np.isnan(disps)] = 0
                print('Depth stats', np.mean(disps), np.max(disps), np.percentile(disps, 95))
                imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30,
                                 quality=8)

                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device).float()
                if args.quantitative_evaluation:
                    import lpips
                    from piqa import SSIM
                    # psnr = PSNR().cuda()
                    ssim = SSIM().cuda()
                    # lpips = LPIPS().cuda()
                    lpips = lpips.LPIPS(net='vgg').to(device)
                    rgbs = torch.from_numpy(rgbs).clip(0, 1).permute(0, 3, 1, 2).to(device).float()
                    print("Max rgbs Image:", torch.max(rgbs).item())
                    print("PSNR;SSIM;LPIPS")
                    # print("PSNR:", mse2psnr(img2mse(rgbs, images)).item())
                    # print("SSIM:", ssim(rgbs, images).item())
                    # print("LPIPS:", lpips(rgbs, images).mean().item())
                    print(mse2psnr(img2mse(rgbs, images)).item(), ";", ssim(rgbs, images).item(), ";",
                          lpips(rgbs, images).mean().item())

                    if args.use_depth:
                        depths_eval = torch.from_numpy(depth_maps).to(device)
                        estimated_depth = torch.as_tensor(estimated_depth).to(device)
                        print("depth map: ", depths_eval.shape, " device: ", depths_eval.device)
                        print("estimated_depth map: ", estimated_depth.shape, " device: ", estimated_depth.device)

                        print("Far: ", far, "depth_maps max:", depth_maps.max(), "estimated_depth max:",
                              estimated_depth.max())
                        # abs_depth_diff = torch.abs(depths_eval / depths_eval.max() - estimated_depth / estimated_depth.max())
                        abs_depth_diff = torch.abs(depths_eval - estimated_depth) / far
                        # abs_depth_diff = (depths_eval - estimated_depth) ** 2
                        # abs_depth_diff = ((depths_eval - estimated_depth) / estimated_depth) ** 2
                        print("depth diff:")
                        save_depth_differences(abs_depth_diff, testsavedir, far=far)

            return

    # Prepare raybatch tensor if batching random rays
    if not args.colmap_depth:
        N_rgb = args.N_rand
    else:
        N_depth = int(args.N_rand * args.depth_rays_prop)
        N_rgb = args.N_rand - N_depth
    use_batching = not args.no_batching
    N_rand = args.N_rand
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        if args.debug:
            print('rays.shape:', rays.shape)
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, np.newaxis]], 1)  # [N, ro+rd+rgb, H, W, 3]
        if args.debug:
            print('rays_rgb.shape:', rays_rgb.shape)

        if args.use_depth:
            print('get depth rays from depth maps')
            if args.debug:
                print('depth_maps.shape:', depth_maps.shape)
                # print('depth_maps[:, None, None].shape:', depth_maps[:, np.newaxis, np.newaxis].shape)
                print('np.expand_dims(depth_maps, axis=(1, 4)).shape:', np.expand_dims(depth_maps, axis=(1, 4)).shape)
            # depth_maps [N, H, W]
            # depth_maps[:, np.newaxis, np.newaxis].shape [N, 1, 1, H, W]
            # rays_d = np.transpose(depth_maps[:, np.newaxis, np.newaxis], [0, 1, 3, 4, 2])  # [N, 1, H, W, 1]
            rays_d = np.expand_dims(depth_maps, axis=(1, 4))  # [N, 1, H, W, 1]
            rays_d = np.repeat(rays_d, 3, axis=4)  # [N, 1, H, W, 3]
            rays_rgb = np.concatenate([rays_rgb, rays_d], 1)  # [N, ro+rd+rgb+depth, H, W, 3]

            print("Rays with depth:", rays_rgb.shape)

            max_depth = np.max(rays_d)
            print("Min depth value:", np.min(rays_d), "Max depth value:", max_depth)

        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3], ro+rd+rgb+depth if use_depth
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        if args.use_depth:
            rays_rgb = np.reshape(rays_rgb, [-1, 4, 3])  # [(N-1)*H*W, ro+rd+rgb+depth, 3]
        else:
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)

        print("Final Rays before shuffeling:", rays_rgb.shape)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        rays_depth = None

        if args.colmap_depth:
            print('get depth rays')
            rays_depth_list = []
            for i in i_train:
                rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i, :3, :4], depth_gts[i]['coord']),
                                      axis=0)  # 2 x N x 3
                # print(rays_depth.shape)
                rays_depth = np.transpose(rays_depth, [1, 0, 2])
                depth_value = np.repeat(depth_gts[i]['depth'][:, None, None], 3, axis=2)  # N x 1 x 3
                weights = np.repeat(depth_gts[i]['error'][:, None, None], 3, axis=2)  # N x 1 x 3
                rays_depth = np.concatenate([rays_depth, depth_value, weights], axis=1)  # N x 4 x 3
                rays_depth_list.append(rays_depth)

            rays_depth = np.concatenate(rays_depth_list, axis=0)
            print('rays_weights mean:', np.mean(rays_depth[:, 3, 0]))
            print('rays_weights std:', np.std(rays_depth[:, 3, 0]))
            print('rays_weights max:', np.max(rays_depth[:, 3, 0]))
            print('rays_weights min:', np.min(rays_depth[:, 3, 0]))
            print('rays_depth.shape:', rays_depth.shape)
            rays_depth = rays_depth.astype(np.float32)
            print('shuffle depth rays')
            np.random.shuffle(rays_depth)

            max_depth = np.max(rays_depth[:, 3, 0])
        print('done')
        i_batch = 0

    if args.debug:
        return
    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    zero = torch.tensor([0.], dtype=torch.float).to(device)
    if use_batching:
        # rays_rgb = torch.Tensor(rays_rgb).to(device)
        # rays_depth = torch.Tensor(rays_depth).to(device) if rays_depth is not None else None
        raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size=N_rgb, shuffle=True, num_workers=0))
        raysDepth_iter = iter(DataLoader(RayDataset(rays_depth), batch_size=N_depth, shuffle=True,
                                         num_workers=0)) if rays_depth is not None else None
    else:
        depths = torch.Tensor(depths).to(device)
    # if args.use_depth:
    #    depths = torch.Tensor(depths).to(device)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(bansedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        target_depth_sigma = None
        # Sample random ray batch
        if use_batching:
            # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            try:
                batch = next(raysRGB_iter).to(device)
            except StopIteration:
                raysRGB_iter = iter(DataLoader(RayDataset(rays_rgb), batch_size=N_rgb, shuffle=True, num_workers=0))
                batch = next(raysRGB_iter).to(device)
            batch = torch.transpose(batch, 0, 1)  # [ro+rd+rgb, (N-1)*H*W, 3]
            depth_s = None
            if args.use_depth:
                batch_rays, target_s, target_depth_sigma = batch[:2], batch[2], batch[3]
                # print("Target depth:", target_depth_sigma.size())
                target_depth_sigma = target_depth_sigma[..., 0]
                # print("Target depth:", target_depth_sigma.size())
            else:
                batch_rays, target_s = batch[:2], batch[2]

            if args.colmap_depth:
                # batch_depth = rays_depth[i_batch:i_batch+N_rand]
                try:
                    batch_depth = next(raysDepth_iter).to(device)
                except StopIteration:
                    raysDepth_iter = iter(
                        DataLoader(RayDataset(rays_depth), batch_size=N_depth, shuffle=True, num_workers=0))
                    batch_depth = next(raysDepth_iter).to(device)
                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth = batch_depth[:2]  # 2 x B x 3
                target_depth = batch_depth[2, :, 0]  # B
                target_depth_sigma = torch.cat([torch.zeros(batch_rays.size()[1]).to(device), target_depth])
                ray_weights = batch_depth[3, :, 0]

            # i_batch += N_rand
            # if i_batch >= rays_rgb.shape[0] or (args.colmap_depth and i_batch >= rays_depth.shape[0]):
            #     print("Shuffle data after an epoch!")
            #     rand_idx = torch.randperm(rays_rgb.shape[0])
            #     rays_rgb = rays_rgb[rand_idx]
            #     if args.colmap_depth:
            #         rand_idx = torch.randperm(rays_depth.shape[0])
            #         rays_depth = rays_depth[rand_idx]
            #     i_batch = 0


        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                if args.use_depth:
                    target_depth_im = depths[img_i]
                    target_depth = target_depth_im[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
                    target_depth_sigma = target_depth

        #####  Core optimization loop  #####
        # timer_0 = time.perf_counter()

        if args.colmap_depth:
            N_batch = batch_rays.shape[1]
            # batch_rays.size() [2, N_rand, 3]
            batch_rays = torch.cat([batch_rays, batch_rays_depth], 1)  # (2, 2 * N_rand, 3)

        # timer_concate = time.perf_counter()

        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                               verbose=i < 10, retraw=True, depths=target_depth_sigma,
                                               **render_kwargs_train)
        # timer_iter = time.perf_counter()

        if args.colmap_depth and not args.depth_with_rgb:
            # _, _, _, depth_col, extras_col = render(H, W, focal, chunk=args.chunk, rays=batch_rays_depth,
            #                                     verbose=i < 10, retraw=True, depths=target_depth,
            #                                     **render_kwargs_train)
            rgb = rgb[:N_batch, :]
            disp = disp[:N_batch]
            acc = acc[:N_batch]
            depth, depth_col = depth[:N_batch], depth[N_batch:]
            extras = {x: extras[x][:N_batch] for x in extras}
            extras_col = {x: extras[x][N_batch:] for x in extras}

        elif args.colmap_depth and args.depth_with_rgb:
            depth_col = depth

        # timer_split = time.perf_counter()

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        depth_loss = 0
        if args.depth_loss:
            # depth_loss = img2mse(depth_col, target_depth)
            if args.use_depth:
                # Estimated depth
                depth_col = depth
                # print("depth_col:", depth_col.size())
                if not use_batching:
                    target_depth = target_depth.view(target_depth.size()[0])
                else:
                    target_depth = target_depth_sigma

                target_depth = torch.where(target_depth > 0, target_depth, zero)
                depth_col = torch.where(target_depth > 0, depth_col, zero)

                # print("depth", torch.max(depth).item(), torch.min(depth).item(), torch.mean(depth).item())
                # print("target", torch.max(target_depth).item(), torch.min(target_depth).item(), torch.mean(target_depth).item())
                # print(depth)
                # print(target_depth)
            if args.weighted_loss:
                if not args.normalize_depth:
                    depth_loss = torch.mean(((depth_col - target_depth) ** 2) * ray_weights)
                else:
                    depth_loss = torch.mean((((depth_col - target_depth) / max_depth) ** 2) * ray_weights)
            elif args.relative_loss:
                depth_loss = torch.mean(((depth_col - target_depth) / target_depth) ** 2)
            else:
                depth_loss = img2mse(depth_col, target_depth)
        sigma_loss = 0
        if args.sigma_loss:
            if args.use_depth:
                sigma_loss = extras['sigma_loss'].mean()
            else:
                sigma_loss = extras_col['sigma_loss'].mean()
            # print(sigma_loss)
        trans = extras['raw'][..., -1]
        loss = img_loss + args.depth_lambda * depth_loss + args.sigma_lambda * sigma_loss
        psnr = mse2psnr(img_loss)

        # timer_loss = time.perf_counter()

        if 'rgb0' in extras and not args.no_coarse:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        if args.wandb:
            wandb.log({'loss': loss.item()}, step=global_step)
            wandb.log({'psnr': psnr.item()}, step=global_step)
            if isinstance(depth_loss, torch.Tensor):
                depth_loss = depth_loss.item()
            wandb.log({'simple depth': depth_loss}, step=global_step)
            if isinstance(sigma_loss, torch.Tensor):
                sigma_loss = sigma_loss.item()
            wandb.log({'sigma': sigma_loss}, step=global_step)

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train[
                                                                                               'network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train[
                                                                                                   'network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if args.i_video > 0 and i % args.i_video == 0 and i > 0:
            hwf_local = hwf
            if args.dataset_type == 'FourDOR':
                hwf_local = [400, 400, hwf[2]]
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, _ = render_path(render_poses, hwf_local, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.nanmax(disps)), fps=30, quality=8)

            if args.wandb:
                wandb.log(
                    {"video_rgb": wandb.Video(moviebase + 'rgb.mp4', fps=4, format="gif")})
                wandb.log(
                    {"video_depth": wandb.Video(moviebase + 'disp.mp4', fps=4, format="gif")})
            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0 and len(i_test) > 0:
            hwf_local = hwf
            if args.dataset_type == 'FourDOR':
                hwf_local = [400, 400, hwf[2]]

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                rgbs, disps, estimated_depth = render_path(poses[i_test], hwf_local, args.chunk, render_kwargs_test,
                                          gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

            images_eval = images[i_test].permute(0, 3, 1, 2).to(device)
            if args.quantitative_evaluation:
                import lpips
                from piqa import SSIM
                # psnr = PSNR().cuda()
                ssim = SSIM().cuda()
                # lpips = LPIPS().cuda()
                lpips = lpips.LPIPS(net='vgg').to(device)
                rgbs = torch.from_numpy(rgbs).clip(0, 1).permute(0, 3, 1, 2).to(device)

                psnr_metric = mse2psnr(img2mse(rgbs, images_eval)).item()
                ssim_metric = ssim(rgbs, images_eval).item()
                lpips_metric = lpips(rgbs, images_eval).mean().item()

                print("Max rgbs Image:", torch.max(rgbs).item())
                print("PSNR;SSIM;LPIPS")
                print(psnr_metric, ";", ssim_metric, ";", lpips_metric)

                if args.wandb:
                    wandb.log({'psnr_test': psnr_metric})
                    wandb.log({'ssim_test': ssim_metric})
                    wandb.log({'lpips_test': lpips_metric})

                if args.use_depth:
                    depths_eval = depth_maps[i_test]
                    # print("depth_eval:", depths_eval.shape)
                    estimated_depth = torch.from_numpy(np.asarray(estimated_depth)).to(device)
                    print("estimated_depth map: ", estimated_depth.shape, " device: ", estimated_depth.device)

                    print("Far: ", far, "depth_maps max:", depth_maps.max(), "estimated_depth max:", estimated_depth.max())
                    # abs_depth_diff = torch.abs(depths_eval / depths_eval.max() - estimated_depth / estimated_depth.max())
                    abs_depth_diff = torch.abs(depths_eval - estimated_depth) / far
                    save_depth_differences(abs_depth_diff, testsavedir)

                    if args.wandb:
                        psnr_metric_depth = mse2psnr(img2mse(estimated_depth, depths_eval)).item()
                        wandb.log({'psnr_depth_test': psnr_metric_depth})



        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
