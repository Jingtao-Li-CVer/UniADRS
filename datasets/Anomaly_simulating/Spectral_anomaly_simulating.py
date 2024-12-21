import numpy as np
import albumentations as A
import cv2
from utils.img_io import write_img
import time


class Spectral_simulator:
    def __init__(self, normal_ratio_range=(0.0015, 0.5), anomaly_ratio_range=(0.00024, 0.0015), transform=None):
        """
        normal_ratio_range: the ratio range of the pre-set normal object area
        anomaly_ratio_range: the ratio range of the pre-set anomaly object area
        transform: data transforms for the simulated anomaly samples, with albumentations format
        """
        self.shuffle_transform = A.ChannelShuffle(p=1.0)
        self.shift_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.0, p=1, interpolation=cv2.INTER_NEAREST), A.IAAPiecewiseAffine (scale=(0.03, 0.05), order=0, p=1.0)])
        self.normal_ratio_range = normal_ratio_range
        self.anomaly_ratio_range = anomaly_ratio_range
        self.transform = transform

    def get_pn_select_transform(self, img_shape):
        """
        img_shape: The spatial shape of the given spectral image. 
                   It is used to select the generating region randomly for the normal and anomaly objects.
        """
        max_height_width = int(np.sqrt(img_shape[0]*img_shape[1]*self.anomaly_ratio_range[1]))
        min_height_width = int(np.sqrt(img_shape[0]*img_shape[1]*self.anomaly_ratio_range[0]))
        max_height_width2 = int(np.sqrt(img_shape[0]*img_shape[1]*self.normal_ratio_range[1]))
        min_height_width2 = int(np.sqrt(img_shape[0]*img_shape[1]*self.normal_ratio_range[0]))
        p_location_select_transform = A.CoarseDropout(max_holes=5 , min_holes=2, max_height=max_height_width, \
                                                           max_width=max_height_width, min_height=min_height_width, min_width=min_height_width, \
                                                            fill_value=1, mask_fill_value=1, p=1.0)
        n_location_select_transform = A.CoarseDropout(max_holes=5 , min_holes=2, max_height=max_height_width2, \
                                                           max_width=max_height_width2, min_height=min_height_width2, min_width=min_height_width2, \
                                                            fill_value=0, mask_fill_value=3, p=1.0)
        return p_location_select_transform, n_location_select_transform

    def simulate(self, img, ):
        mask =  np.zeros((img.shape[0], img.shape[1]))
        p_location_select_transform, n_location_select_transform = self.get_pn_select_transform((img.shape[0], img.shape[1]))

        # select the region randomly for negative objects (large normal objects), and substitue with the shuffled spectra
        sample_negative = n_location_select_transform(image=img, mask = mask)
        img_negative = self.shuffle_transform(image=img)['image']
        locs = np.where(sample_negative['mask'] == 3)
        sample_negative['image'][locs[0], locs[1], :] = img_negative[locs[0], locs[1], :] 

        # select the region randomly for positive objects (anomaly objects), and substitue with the shuffled spectra
        sample = p_location_select_transform(image=sample_negative['image'], mask = sample_negative['mask'])
        img2 = self.shuffle_transform(image=img)['image']
        locs = np.where(sample['mask'] == 1)

        sample['image'][locs[0], locs[1], :] = img2[locs[0], locs[1], :] 

        # apply shift transform and some other user-defined transforms
        sample = self.shift_transform(image=sample['image'], mask = sample['mask']) 

        sample = self.transform(image=sample['image'], mask = sample['mask'])
        detailed_mask = sample['mask'].clone()
        sample['mask'][sample['mask'] == 3] = 0

        select_bands_index = np.random.randint(0, img.shape[2], size=(270))

        return sample['image'][select_bands_index, :, :].float(), sample['mask'].float(), detailed_mask.float()