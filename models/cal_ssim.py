import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToTensor

def cal_ssim(img1, img2, data_range=1.0, multichannel=True):
    """
    计算两幅图像的 SSIM（结构相似性）值
    :param img1: 输入图像1（Tensor 或 Numpy 数组）
    :param img2: 输入图像2（Tensor 或 Numpy 数组）
    :param data_range: 像素值范围（例如 255 或 1.0）
    :param multichannel: 是否为多通道图像（如 RGB）
    :return: SSIM 值
    """
    # 确保输入为 Numpy 数组
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose((1, 2, 0))
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose((1, 2, 0))

    # 计算 SSIM
    return ssim(img1, img2, data_range=data_range, multichannel=multichannel)
