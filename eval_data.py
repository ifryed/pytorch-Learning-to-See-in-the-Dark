import glob
import os
import re
import sys

import cv2
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
from pyssim.ssim.ssimlib import SSIM


def checkQuality(folder_path: str, meaure_fun: callable, dist_type: str):
    folder_path = './' + folder_path
    gt_arr = glob.glob(folder_path + '/*gt.png')

    mse_scores = []
    names = []
    for gt_img_path in gt_arr:
        print('\r\tChecking: %s' % gt_img_path, end='')
        output_per_gt = gt_img_path.replace('gt', 'out')

        names.append(re.match(folder_path + "[^\d]*(\d.*)_gt.png", gt_img_path).group(1))

        dist = meaure_fun(gt_img_path, output_per_gt)
        mse_scores.append(dist)

    mse_scores = np.asarray(mse_scores)
    names = np.asarray(names)
    sorted_idx = mse_scores.argsort()
    mse_scores = mse_scores[sorted_idx]
    names = names[sorted_idx]
    print(mse_scores)
    print("Mean %s:\t%.3f" % (dist_type, mse_scores.mean()))
    k = min(len(mse_scores), 10)
    print("%d Lowest:" % k)
    for i in range(k):
        print("\t%d) Image: %s\tScore:%.3f" % (i + 1, names[i], mse_scores[i]))

    print("%d Highest:" % k)
    for i in range(k):
        idx = i + 1
        print("\t%d) Image: %s\tScore:%.3f" % (idx, names[-idx], mse_scores[-idx]))


def checkMSE(folder_path: str):
    checkQuality(folder_path,
                 lambda gt, out:
                 np.power(np.asarray(Image.open(gt)) - np.asarray(Image.open(out)), 2).mean(),
                 'MSE')


def checkPSNR(folder_path: str):
    dist_func = lambda gt, out: 10 * np.log10(
        (2 ** 8 - 1 if np.asarray(Image.open(gt)).dtype == np.uint8 else 1.0) ** 2 /
        np.power(np.asarray(Image.open(gt)) - np.asarray(Image.open(out)), 2).mean()
    )
    checkQuality(folder_path, dist_func, 'MY PSNR')

    dist_func = lambda gt, out: compare_psnr(
        np.asarray(Image.open(gt)), np.asarray(Image.open(out))
    )
    checkQuality(folder_path, dist_func, 'PSNR')


def checkSSIMM(folder_path: str):
    print("Checking SSIM\n")
    dist_func = lambda gt, out: compare_ssim(
        cv2.cvtColor(np.asarray(Image.open(gt)), cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(np.asarray(Image.open(out)), cv2.COLOR_RGB2GRAY))
    checkQuality(folder_path, dist_func, 'SSIM')


def checkMySSIMM(folder_path: str):
    print("Checking My SSIM\n")
    dist_func = lambda gt, out: mySSIM(gt, out)

    dist_func = lambda gt, out: pySSIM(gt, out)
    checkQuality(folder_path, dist_func, 'My SSIM')


def get_gaussian_kernel(gaussian_kernel_width=11, gaussian_kernel_sigma=1.5):
    """Generate a gaussian kernel."""
    # 1D Gaussian kernel definition
    gaussian_kernel_1d = np.ndarray(gaussian_kernel_width)
    norm_mu = int(gaussian_kernel_width / 2)

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        gaussian_kernel_1d[i] = (np.exp(-((i - norm_mu) ** 2) /
                                        (2 * (gaussian_kernel_sigma ** 2))))
    return gaussian_kernel_1d / np.sum(gaussian_kernel_1d)


def pySSIM(gt_path, out_path):
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(
        gaussian_kernel_width, gaussian_kernel_sigma)
    gt = np.asarray(Image.open(gt_path))
    size = gt.shape[:2]

    ssim = SSIM(gt_path, gaussian_kernel_1d, size=size)
    ssim_value = ssim.ssim_value(out_path)

    return ssim_value


def Covariance(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar) * (y - ybar)) / (len(x) - 1)


def mySSIM(img1, img2):
    img1 = cv2.cvtColor(np.asarray(Image.open(img1)), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.asarray(Image.open(img2)), cv2.COLOR_RGB2GRAY)
    L = 2 ** 8 - 1
    img1 = img1.squeeze()
    img2 = img2.squeeze()

    u1 = img1.mean()
    u2 = img2.mean()

    sig1 = np.var(img1)
    sig2 = np.var(img2)

    sig12 = Covariance(img1, img2)

    k1 = 0.01
    k2 = 0.03
    c1 = np.power(k1 * L, 2)
    c2 = np.power(k2 * L, 2)

    ret_ssim = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / \
               (
                       (np.power(u1, 2) + np.power(u2, 2) + c1) * (np.power(sig1, 2) + np.power(sig2, 2) + c2)
               )
    return ret_ssim


if __name__ == '__main__':
    checkPSNR(sys.argv[1])
    # checkSSIMM(sys.argv[1])
    checkMySSIMM(sys.argv[1])
