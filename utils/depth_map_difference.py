import numpy as np
import os
import cv2
import imageio

def save_depth_differences(abs_depth_diff, testsavedir, far=None, granularity=4):
    """
        Input:
            abs_depth_diff: List of Torch tensors. An entry in the List corresponds to the absolute difference of two
            depth maps.
            testsavedir: Path to where the output images are saved
            granularity: How many interpolation steps between 0 (blue) and 1 (red) in the ouput image
        Saves the absolute difference of two depth maps in a blue-red scaled image. Where blue corresponds to 0 and red
        to the max depth difference in the abs_depth_diff input.
    """
    for depth_index in range(len(abs_depth_diff)):
        depth_diff = abs_depth_diff[depth_index]

        # from https://stackoverflow.com/questions/64071648/converting-grey-image-to-blue-and-red-image
        img = depth_diff
        img = np.expand_dims(img, -1)

        print("Min: ", img.min(), "Max:", img.max())

        # Just do define min and max errors
        img[0][0][0] = 0.
        img[0][1][0] = 1.
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

def visualize_difference(depths_eval, estimated_depth):
    abs_depth_diff = np.abs(depths_eval - estimated_depth)
    save_depth_differences(abs_depth_diff, "")

if __name__ == '__main__':
    test = imageio.imread("../../data/ship_rgbd_4/train/r_1_depth_0002.png")
    a = test[400]


    img_numb = "000_depth.png"
    basedir = "../../TestOut"

    gt = "../../data/nerf_llff_data/fern_depth/images_8/image" + img_numb
    gt_img = imageio.imread(gt) * 1.
    gt_img /= 255.
    # gt_img /= np.max(gt_img)

    path_rgb_d = os.path.join(basedir, "DS-RGB-D-Fern-6/" + img_numb)
    depth_rgb_d = imageio.imread(path_rgb_d) * 1.
    depth_rgb_d /= 255.
    # depth_rgb_d /= np.max(depth_rgb_d)

    path_rgb = os.path.join(basedir, "DS-RGB-Fern-6/" + img_numb)
    depth_rgb = imageio.imread(path_rgb) * 1.
    depth_rgb /= 255.
    # depth_rgb /= np.max(depth_rgb)

    visualize_difference(gt_img, np.asarray([depth_rgb, depth_rgb_d], dtype=float))