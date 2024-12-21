import logging
import os
import sys
sys.path.append("./../")
from torch.utils.data import Dataset
from utils.img_io import read_img
import numpy as np
import cv2

"""
Dataset class for inferring the folder image
"""

class Test_Folder_Image(Dataset):
    def __init__(self ,img_dir, mask_dir, test_patch_size, test_size, test_pad_size, normalize, replace_org, replace_dst, transforms):
        """
        Args:
            img_dir (str): Paths of input images.
            mask_dir (str): Paths of corresponding masks.
            test_patch_size (int): Inferring patch size. The original image is inferred in cropped patches
            test_size (int): Inferring size for each input patch. The cropped patches will be resized to the test size
            test_pad_size (int): For the overlapped inferring, the test pad size decides the overlap surrounding.
            normalize (bool): Whether to normalize the input patch.
            replace_org (str): The difference of the filename in the image name compared to the groundtruth
            replace_dst (str): Replace the replace_org with the replace_dst to get the groundtruth name
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.test_patch_size = test_patch_size
        self.test_pad_size = test_pad_size
        self.replace_org = replace_org
        self.replace_dst = replace_dst
        self.normalize = normalize
        self.test_size = test_size
        self.transform = transforms

        self.img_file_names = os.listdir(self.img_dir)

        self.dataset_length = len(self.img_file_names)

        logging.info(f'Creating dataset with {self.dataset_length} examples')

    def __len__(self):
        return self.dataset_length
 
    def __getitem__(self, i):

        img_path = os.path.join(self.img_dir, self.img_file_names[i])
        label_path = os.path.join(self.mask_dir, self.img_file_names[i])

        (_, img_file_name) = os.path.split(img_path)
        img_file_name = img_file_name.split('.')[-2]
        img = read_img(img_path=img_path).astype(np.float32)
        label_path = label_path.replace(self.replace_org, self.replace_dst)
        mask = read_img(img_path=label_path).astype(np.float32)
        mask[mask >= 1] = 1

        if self.normalize:
            img = img/img.max((0,1))
        
        try:
            sample = self.transform(image=img, mask = mask)
        except cv2.error:
            print(1)

        return sample['image'].float(), sample['mask'].float(), img_file_name, self.test_patch_size, self.test_pad_size, self.test_size
