import logging
import os
import sys
sys.path.append("./../")
from torch.utils.data import Dataset
from utils.img_io import read_img
import numpy as np

"""
Dataset class for inferring the single image
"""


class Test_Single_Image(Dataset):
    def __init__(self, img_paths, mask_paths, test_patch_sizes, test_sizes, test_pad_sizes, normalize, transforms):
        """
        Args:
            img_paths (list): Paths of input images.
            mask_paths (list): Paths of corresponding masks.
            test_patch_sizes (int): Inferring patch size. The original image is inferred in cropped patches
            test_sizes (int): Inferring size for each input patch. The cropped patches will be resized to the test size
            test_pad_sizes (int): For the overlapped inferring, the test pad size decides the overlap surrounding.
            normalize (bool): Whether to normalize the input patch.
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.test_patch_sizes = test_patch_sizes
        self.test_pad_sizes = test_pad_sizes
        self.normalize = normalize
        self.test_sizes = test_sizes
        self.transform = transforms

        self.dataset_length = len(self.img_paths)

        logging.info(f'Creating dataset with {self.dataset_length} examples')

    def __len__(self):
        return self.dataset_length
 
    def __getitem__(self, i):
        img_path = self.img_paths[i]
        label_path = self.mask_paths[i]
        test_patch_size = self.test_patch_sizes[i]
        test_pad_size = self.test_pad_sizes[i]

        (_, img_file_name) = os.path.split(img_path)
        img_file_name = img_file_name.split('.')[-2] if '.' in img_file_name else img_file_name
        img = read_img(img_path=img_path).astype(np.float32)

        if self.normalize[i]:
            img = img/img.max((0,1))

        mask = read_img(label_path) 
        mask[mask >= 1] = 1
        mask  = 1 - mask   

        sample = self.transform(image=img, mask = mask)

        return sample['image'].float(), sample['mask'].float(), img_file_name, test_patch_size, test_pad_size, self.test_sizes[i]