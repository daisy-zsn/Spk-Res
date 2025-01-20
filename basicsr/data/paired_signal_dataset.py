import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2


class Dataset_Reconstruction(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_Reconstruction, self).__init__()
        self.opt = opt

        # if self.opt['phase'] == 'train':
        #     self.sigma_type  = opt['sigma_type']
        #     self.sigma_range = opt['sigma_range']
        #     assert self.sigma_type in ['constant', 'random', 'choice']
        # else:
        #     self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']
        self.paths = [os.path.join(self.gt_folder, file) for file in os.listdir(self.gt_folder)]

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        path = self.paths[index]
        gt_path = os.path.join(path,'gt.npy')
        lq_path = os.path.join(path,'lq.npy')

        img_gt = np.load(gt_path)
        img_lq = np.load(lq_path)

        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

        # augmentation for training
        if self.opt['phase'] == 'train':
            if self.geometric_augs:
                gt_size = self.opt['gt_size']
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale)
                # flip, rotation
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            else:
                img_gt, img_lq = img2tensor([img_gt, img_lq],bgr2rgb=False,float32=True)
        else:
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False,float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

