import logging
import os
import sys
sys.path.append("./../")
from torch.utils.data import Dataset
from utils.img_io import read_img
import numpy as np
import albumentations as A
import random
import cv2
from .Anomaly_simulating.Spectral_anomaly_simulating import Spectral_simulator
from .Anomaly_simulating.Spatial_anomaly_simulating import Spatial_simulator

"""
Simulating deviating spectral and spatial anomalies
"""

class Anomaly_Training_Dataset(Dataset): 
    def __init__(self, spectral_img_dirs, spatial_img_dir, spatial_mask_dir, object_bank_img_dir, object_bank_label_dir, transforms):
        self.spectral_img_dirs = spectral_img_dirs
        self.spatial_img_dir = spatial_img_dir
        self.spatial_mask_dir = spatial_mask_dir
        self.spatial_image_files = os.listdir(spatial_img_dir)
        self.transform = transforms
        self.spectral_img_files = []
        for i in range(len(self.spectral_img_dirs)):
            file_names = os.listdir(self.spectral_img_dirs[i])
            for j in range(len(file_names)):
                file_names[j] = os.path.join(self.spectral_img_dirs[i], file_names[j])
            self.spectral_img_files += file_names

        self.shift_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.0, p=1, interpolation=cv2.INTER_NEAREST), \
                                          A.IAAPiecewiseAffine (scale=(0.03, 0.05), order=0, p=1.0)])
        
        self.shuffle_transform = A.ChannelShuffle(p=1.0)
        self.spectral_simulator = Spectral_simulator(anomaly_ratio_range=(0.0064, 0.0225), \
                                                     normal_ratio_range=(0.0225, 0.5), transform=transforms)
        self.spatial_simulator = Spatial_simulator(224, 0.02, 0.06, 0.5, object_bank_img_dir,\
                                                    object_bank_label_dir, transform=transforms)

        self.dataset_length = 500

        logging.info(f'Creating dataset with {self.dataset_length} examples') 

    def __len__(self): 
        return 500

    def __getitem__(self, i):
        """
        return values:
        transformed_img: input image after the data argumentation
        transformed_mask: mask with 0-(backround+large normal object), 1-anomaly, 2-ignored
        detailed_mask: mask with 0-backround, 1-anomaly, 2-ignored, 3-large normal object
        """
        if random.random() > 0.5:
            i = random.choice(np.arange(len(self.spectral_img_files)))
            img_path = self.spectral_img_files[i]
            img = read_img(img_path=img_path)[:,:,:] # 1024, 1024, 4
            img = img/img.max()
            transformed_img, transformed_mask, detailed_mask = self.spectral_simulator.simulate(img)
            return transformed_img, transformed_mask, detailed_mask
        else:
            i = random.choice(np.arange(len(self.spatial_image_files)))
            image_file = self.spatial_image_files[i]

            img_path = os.path.join(self.spatial_img_dir, image_file)
            label_path = os.path.join(self.spatial_mask_dir, image_file.replace('.png', '_instance_id_RGB.png'))
           
            img = read_img(img_path=img_path)
            mask = read_img(label_path).astype(np.int16)

            transformed_img, transformed_mask, detailed_mask = self.spatial_simulator.simulate(img, mask)
            return transformed_img, transformed_mask, detailed_mask