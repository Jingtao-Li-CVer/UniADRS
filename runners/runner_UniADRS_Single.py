from runners.base_runner import BaseRunner
import logging
import numpy as np
import torch
from typing import Dict, List
from numpy import ndarray as NDArray
from utils.savefig import save_heatmap
import time
from utils.average_meter import AverageMeter
from utils.overlap_infer_v2 import overlap_infer
from metrics.auroc import compute_auroc
from utils.img_io import read_img, write_img
import os
from tqdm import tqdm 
import time

"""
Train the model, and infer the single image.
The training process is identity with the runner_UniADRS_Folder.py
"""

class runner_UniADRS_Single(BaseRunner):
    def _train(self, epoch: int) -> None:
        self.model.train()
        train_iter_loss = AverageMeter()
        epoch_start_time = time.time()
        self.train_loader_size = self.dataloaders['train'].__len__()

        if self.first_epoch:
            self.model.loss = self.criterions['P_AUC']
            self.opt_lagrange_multiplier = torch.optim.Adam([self.model.loss.lambdas, self.model.loss.biases], lr=0.001, weight_decay=0.0005)
            self.first_epoch = False


        for batch_idx, (img, mask, detailed_mask) in enumerate(self.dataloaders["train"]): 

            if batch_idx > 100:
                break
            self.optimizer.zero_grad()
            self.opt_lagrange_multiplier.zero_grad()

            if torch.any(torch.isnan(img)) or torch.any(torch.isnan(mask)):
                continue

            img = img.to(self.cfg.params.device)
            mask = mask.to(self.cfg.params.device)

            detailed_mask = detailed_mask.to(self.cfg.params.device)

            _, loss = self.model(img, mask.unsqueeze(1), detailed_mask, False)

            if loss > 0:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                self.optimizer.step()
                self.opt_lagrange_multiplier.step()

            train_iter_loss.update(loss.item())

            if batch_idx % self.cfg.params.print_intervals == 0: 
                spend_time = time.time() - epoch_start_time
                logging.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(epoch, batch_idx, self.train_loader_size, 
                batch_idx / self.train_loader_size * 100, self.optimizer.param_groups[-1]['lr'], train_iter_loss.avg, spend_time / (batch_idx + 1) * self.train_loader_size // 60 - spend_time // 60))

                train_iter_loss.reset()



    def _test(self, epoch: int, save_anomaly_map = True) -> None:

        self.model.eval()
        save_dir = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, 'test_images') or not hasattr(self, 'test_img_gts'):
            self.test_images = []
            self.test_img_gts = []
            self.test_file_names = []
            self.test_patch_sizes = []
            self.test_pad_sizes = []
            self.test_sizes = []

            for test_image, test_img_gt, test_file_name, test_patch_size, test_pad_size, test_size in (self.dataloaders["test"]):
                self.test_images.append(test_image)
                self.test_img_gts.append(test_img_gt)
                self.test_file_names.append(test_file_name[0])
                self.test_patch_sizes.append(test_patch_size[0])
                self.test_pad_sizes.append(test_pad_size[0])
                self.test_sizes.append(test_size)
            
        for i in tqdm(range(len(self.test_images))):

            mb_img = self.test_images[i]
            mb_gt = self.test_img_gts[i]
            file_name = self.test_file_names[i]
            file_name = str(i) + '_' + file_name

            artifacts: Dict[str, List[NDArray]] = {
                "img": [],
                "gt": [],
                "amap": [],
            }

            logging.info("test model on the image: " + file_name)
            start = time.time()

            with torch.no_grad():
                if mb_img.shape[1] > self.cfg.params.training_channels:
                    slice_number = int(mb_img.shape[1]/self.cfg.params.training_channels + 1)
                else:
                    slice_number = 1
                predicted_test_overlap = 0
                for j in range(slice_number):
                    logging.info("test on the " + str(j) + " slice")
                    if j == slice_number - 1:
                        img_t = mb_img[:,-self.cfg.params.training_channels:,:,:]
                    else:
                        img_t = mb_img[:,j:j+self.cfg.params.training_channels,:,:]
                    cfg_test = dict(title_size=[self.test_patch_sizes[i], self.test_patch_sizes[i]],
                    pad_size=[self.test_pad_sizes[i], self.test_pad_sizes[i]], batch_size=1,  
                    padding_mode='mirror', num_classes=1, device=self.cfg.params.device, test_size=self.test_sizes[i])  
                    data_overlap_input = img_t.squeeze(0)
                    predicted_test_overlap = (predicted_test_overlap + overlap_infer(cfg_test, model=self.model, img=data_overlap_input)['score_map'])/2.0

                artifacts["amap"].extend(predicted_test_overlap.permute((2,0,1)).detach().cpu().numpy())

            end = time.time()
            logging.info("used time: " + str(end - start))
            
            artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).detach().cpu().numpy()) 
            artifacts["gt"].extend((1-mb_gt).detach().cpu().numpy())
            artifacts["amap"] = np.array(artifacts["amap"])

            artifacts["amap"] = (artifacts["amap"] - artifacts["amap"].min())/(artifacts["amap"].max() - artifacts["amap"].min())
            
            save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
            if not os.path.exists(save_path):
                os.makedirs(save_path) 

            anomaly_map = artifacts["amap"][0, :, :] 
            save_ratio = 20

            if save_anomaly_map:
                write_img((anomaly_map), os.path.join(save_path, file_name + '_anomaly_map.tif'))
                save_heatmap(anomaly_map, save_path, save_height=anomaly_map.shape[0]/save_ratio, save_width=anomaly_map.shape[1]/save_ratio ,dpi=150, file_name=file_name) 
            
            try: 
                auroc = compute_auroc(epoch, np.array(artifacts["amap"]), np.array(artifacts["gt"]), self.working_dir)
            except IndexError:
                logging.info('Error happened when computing AUC')
                auroc = 0.0
                pass

            if auroc > 0.93:
                self.save(epoch)


    def save(self, epoch):
        save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'checkpoint-model.pth')
        torch.save(self.model.state_dict(), filename)


    def load(self, save_dir):
        filename = os.path.join(save_dir, 'checkpoint-model.pth')
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.cfg.params.device)), strict=False)