import os
import numpy as np
import albumentations as A
from skimage.transform import resize
import random
import sys
sys.path.append("./../")
from utils.img_io import read_img

class Spatial_simulator:
    def __init__(self, cropped_size, min_valid_ratio, max_anomaly_ratio, \
                 max_normal_ratio, object_bank_img_dir, object_bank_label_dir, transform):
        self.cropped_size = cropped_size
        self.min_valid_ratio = min_valid_ratio
        self.max_anomaly_ratio = max_anomaly_ratio
        self.max_normal_ratio = max_normal_ratio
        self.object_bank_img_dir = object_bank_img_dir
        self.object_bank_label_dir = object_bank_label_dir
        self.object_bank_file_names = os.listdir(self.object_bank_img_dir)
        self.transform = transform

    def resize_contras_img(self, subset_img, subset_mask, new_ratio, area_ratio, paste_shape):
        new_height = subset_img.shape[0] * np.sqrt(new_ratio/area_ratio)
        new_width = subset_img.shape[1] * np.sqrt(new_ratio/area_ratio)

        if new_height > paste_shape[0]:
            alpha = 0.9*(paste_shape[0]/new_height)
            new_height2 = new_height*alpha
            new_width = (new_height * new_width)/new_height2
            new_height = new_height2

        if new_width > paste_shape[1]:
            alpha = 0.9*(paste_shape[1]/new_width)
            new_width2 = new_width*alpha
            new_height = (new_height * new_width)/new_width2
            new_width = new_width2

        new_height = int(new_height)
        new_width = int(new_width)

        try:
            trans_img = resize(subset_img.astype(np.float32), (new_height, new_width, 3))
        except ValueError as e:
            print(e)
            exit(0) 
        trans_mask = resize(subset_mask.astype(np.bool8), (new_height, new_width), order=0).astype(np.int8)
        return trans_img, trans_mask


    def copy_paste(self, img, mask, trans_img, trans_mask, mask_id):
        object_locs = np.where(trans_mask == 1)
        try:
            min_x = np.min(object_locs[1])
            min_y = np.min(object_locs[0])
        except ValueError as e:
            print(1)

        target_min_x = int(random.uniform(0, img.shape[1] - object_locs[1].max() + min_x))
        target_min_y = int(random.uniform(0, img.shape[0] - object_locs[0].max() + min_y))

        if trans_img.shape[0] > img.shape[0] or trans_img.shape[1] > img.shape[1]:
            return img, mask

        else:
            img[object_locs[0] - (min_y - target_min_y), object_locs[1] - (min_x - target_min_x), 0:3] = \
            trans_img[object_locs[0], object_locs[1], 0:3]

            mask[object_locs[0] - (min_y - target_min_y), object_locs[1] - (min_x - target_min_x)] = mask_id
            return img, mask


    def make_constractive_samples(self, label, img, min_valid_ratio = 0.02, max_anomaly_ratio = 0.06, max_normal_ratio = 0.5):
        initial_mask = np.zeros((label.shape[0], label.shape[1]))
        non_zero_locs = np.where(np.any(label != 0, axis=-1))
        initial_mask[non_zero_locs[0], non_zero_locs[1]] = 2
        anomaly_object_num = 2
        normal_object_num = 2

        for i in range(anomaly_object_num + normal_object_num):
            file = random.sample(self.object_bank_file_names, k=1)
            object_img = read_img(os.path.join(self.object_bank_img_dir, file[0]))[:,:,0:3]
            object_mask = read_img(os.path.join(self.object_bank_label_dir, file[0]))/255
            if object_img.shape[0] > 100 or object_img.shape[1] > 100:
                object_img = resize(object_img, (100,100,3))
                object_mask = resize(object_mask.astype(np.bool8), (100,100)).astype(np.int8)
            area_ratio = np.sum(object_mask)/(label.shape[0] * label.shape[1])
            if i >= anomaly_object_num and area_ratio > min_valid_ratio and area_ratio < max_anomaly_ratio:
                try:
                    if object_mask.max() < 1:
                        print(1)
                    img, initial_mask = self.copy_paste(img, initial_mask, object_img, object_mask, 1)
                    continue
                except IndexError as e:
                    print(e)
                    exit(0)
            if i < anomaly_object_num and area_ratio > max_anomaly_ratio:
                try:
                    if object_mask.max() < 1:
                        print(1)
                    img, initial_mask = self.copy_paste(img, initial_mask, object_img, object_mask, 3)
                    continue
                except IndexError as e:
                    print(e)
                    exit(0)

            if i >= anomaly_object_num:
                new_ratio = random.uniform(min_valid_ratio, max_anomaly_ratio)
            else:
                new_ratio = random.uniform(max_anomaly_ratio, max_normal_ratio)
            trans_img, trans_mask = self.resize_contras_img(object_img, object_mask, new_ratio, area_ratio, img.shape)
            
            mask_id = 1 if i >= anomaly_object_num else 3
            img, initial_mask = self.copy_paste(img, initial_mask, trans_img, trans_mask, mask_id)

        initial_mask = initial_mask.astype(np.int8)

        return img, initial_mask
    
    def simulate(self, img, mask):
        cropped_height = min(self.cropped_size, img.shape[0])
        cropped_width = min(self.cropped_size, img.shape[1])
        trans_op = A.Compose([
            A.RandomCrop(height=cropped_height, width=cropped_width, p=1.0),
            A.GaussNoise(per_channel=True, p=0.7),
            A.Resize(height=self.cropped_size,width=self.cropped_size,p=1.0)
        ])

        trans = trans_op(image=img, mask=mask)
        img = trans['image']
        mask = trans['mask']

        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img = np.concatenate((img,img,img), 2)

        cons_img, cons_mask, = self.make_constractive_samples(mask, img, self.min_valid_ratio, self.max_anomaly_ratio, self.max_normal_ratio)
        img = cons_img
        mask = cons_mask

        if random.random() > 0.5:
            img = img[:,:, random.choice([0,1,2])]
        img = img/img.max()

        sample = self.transform(image=img, mask=mask)
        detailed_mask = sample['mask'].clone()
        sample['mask'][sample['mask'] == 3] = 0

        return sample['image'].float(),sample['mask'].float(), detailed_mask.float()