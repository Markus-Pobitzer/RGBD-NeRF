import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, load_depth=False, far=6.0):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depths = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        depths = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            if load_depth:
                dname = os.path.join(basedir, frame['file_path'] + '_depth.png')
                depths.append(imageio.imread(dname))
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

        if load_depth:
            d = np.array(depths)
            shape = d.shape

            if len(shape) > 3:
                # Depth value stored several times
                d = d[:, :, :, 0]  # keep only 1 channel (depth alone)

            print("Shape", shape)
            depths = (far * d.reshape((shape[0], shape[1], shape[2], 1)) / np.max(d)).astype(np.float32)  # Works pretty well
            # depths = (far * d.reshape((shape[0], shape[1], shape[2], 1)) / 255.).astype(np.float32)


            # Discussion at: https://github.com/bmild/nerf/issues/77
            # depths = (far * (255. - d[:, :, :, 0].reshape((shape[0], shape[1], shape[2], 1))) / 255.).astype(np.float32) # keep only 1 channel (depth alone)
            # depths = (d[:, :, :, 0].reshape((shape[0], shape[1], shape[2], 1)) / 255.).astype(np.float32) # keep only 1 channel (depth alone)
            # depths = (d[:, :, :, 0].reshape((shape[0], shape[1], shape[2], 1))).astype(np.float32) # keep only 1 channel (depth alone)
            all_depths.append(depths)
            # cv2.imwrite('Depth_test.png', depths[0].reshape(shape[1], shape[2]))

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    if load_depth:
        depths = np.concatenate(all_depths, 0)
        depths[depths == 0] = far
    else:
        depths = None

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        if load_depth:
            depths_half_res = np.zeros((depths.shape[0], H, W, 1))
            for i, depth in enumerate(depths):
                depths_half_res[i] = (cv2.resize(depth, (W, H))).reshape(W, H, 1)
            depths = depths_half_res
            print("Experimantal: Rescaling depth maps to (H, W).")
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, depths, render_poses, [H, W, focal], i_split
