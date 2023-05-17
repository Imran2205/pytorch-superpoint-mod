"""

"""
import os

import numpy as np
# import tensorflow as tf
import cv2
from pathlib import Path

import torch
import torch.utils.data as data

# from .base_dataset import BaseDataset
# from .utils import pipeline
from utils.tools import dict_update

from models.homographies import sample_homography
from settings import DATA_PATH

from imageio import imread


def load_as_float(path):
    return imread(path).astype(np.float32)/255


class TestDataset(data.Dataset):
    default_config = {
        'dataset': 'test',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'dataset_path': '',
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, transform=None, **config):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.files = self._init_dataset(**self.config)
        sequence_set = []
        for (img, img_warped) in zip(self.files['image_paths'], self.files['warped_image_paths']):
            sample = {'image': img, 'warped_image': img_warped, 'homography': None}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform
        if config['preprocessing']['resize']:
            self.sizer = np.array(config['preprocessing']['resize'])
        pass

    def __getitem__(self, index):
        """

        :param index:
        :return:
            image:
                tensor (1,H,W)
            warped_image:
                tensor (1,H,W)
        """
        def _read_image(path):
            input_image = cv2.imread(path)
            return input_image

        def _preprocess(image):
            s = max(self.sizer /image.shape[:2])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[:int(self.sizer[0]/s), :int(self.sizer[1]/s)]
            image = cv2.resize(image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            if self.transform is not None:
                image = self.transform(image)
            return image

        sample = self.samples[index]
        image_original = _read_image(sample['image'])
        image = _preprocess(image_original)
        warped_image = _preprocess(_read_image(sample['warped_image']))
        to_numpy = False
        if to_numpy:
            image, warped_image = np.array(image), np.array(warped_image)

        sample = {'image': image, 'warped_image': warped_image, 'homography': None}
        return sample

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        base_path = config['dataset_path']
        images_path = [os.path.join(base_path, x) for x in os.listdir(base_path) if not x.startswith('.') and x.endswith('.png')]
        images_path.sort()
        image_paths = images_path[0:-1]
        warped_image_paths = images_path[1:]

        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': []}
        return files


