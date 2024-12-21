from runners.base_runner import BaseRunner
import logging
import numpy as np
import torch
from typing import Dict, List
from numpy import ndarray as NDArray
import time
from utils.overlap_infer_v2 import overlap_infer
from metrics.auroc import compute_auroc
import os
from utils.savefig import save_heatmap
from utils.img_io import read_img, write_img
from tqdm import tqdm 
import time
from utils.average_meter import AverageMeter

"""
Train the model, and infer the folder image.
The training process is identity with the runner_UniADRS_Single.py
"""

class runner_UniADRS_Folder(BaseRunner):
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


    def _test(self, epoch: int, save_anomaly_map = False) -> None:

        self.model.eval()
        save_dir = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for test_dataloader_names in [['test_HSI']]: # 'test_sar', 'test_low_light', 'test_visible_light', 'test_thermal', test_HSI'
            start = time.time()
            artifacts: Dict[str, List[NDArray]] = {
                "img": [],
                "gt": [],
                "amap": [],
            }

            for i in range(len(test_dataloader_names)):
                test_dataloader_name = test_dataloader_names[i]

                for mb_img, mb_gt, file_name, test_patch_size, test_pad_size, test_size in tqdm(self.dataloaders[test_dataloader_name]):
                

                    save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
            
                    with torch.no_grad():
                        if mb_img.shape[1] > self.cfg.params.training_channels:
                            slice_number = int(mb_img.shape[1]/self.cfg.params.training_channels + 1)
                        else:
                            slice_number = 1
                        predicted_test_overlap = 0
                        for j in range(slice_number):
                            # logging.info("test on the " + str(j) + " slice")
                            if j == slice_number - 1:
                                img_t = mb_img[:,-self.cfg.params.training_channels:,:,:]
                            else:
                                img_t = mb_img[:,j:j+self.cfg.params.training_channels,:,:]
                            cfg_test = dict(title_size=[test_patch_size, test_patch_size],
                            pad_size=[test_pad_size, test_pad_size], batch_size=1,  
                            padding_mode='mirror', num_classes=1, device=self.cfg.params.device, test_size=test_size)  
                            data_overlap_input = img_t.squeeze(0)
                            predicted_test_overlap = (predicted_test_overlap + overlap_infer(cfg_test, model=self.model, img=data_overlap_input)['score_map'])/2.0

                        artifacts["amap"].extend(predicted_test_overlap.permute((2,0,1)).detach().cpu().numpy())
                
                    artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).detach().cpu().numpy()) 
                    artifacts["gt"].extend((mb_gt).detach().cpu().numpy())

                    if not os.path.exists(save_path):
                        os.makedirs(save_path) 

                    if save_anomaly_map:
                        anomaly_map = predicted_test_overlap[:, :,0].detach().cpu().numpy()
                        save_ratio = 20
                        file_name = file_name[0]
                        write_img(anomaly_map, os.path.join(save_path, file_name + '_anomaly_map.tif'))
                        save_heatmap(anomaly_map, save_path, save_height=anomaly_map.shape[0]/save_ratio, save_width=anomaly_map.shape[1]/save_ratio ,dpi=150, file_name=file_name) 
                
            end = time.time()
            logging.info("used time: " + str(end - start))
            
            try:
                auroc = compute_auroc(epoch, np.array(artifacts["amap"]), np.array(artifacts["gt"]), self.working_dir)
            except IndexError:
                logging.info('Error happened when computing AUC')
                auroc = 0.0
                pass


    def save(self, epoch):
        save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'checkpoint-model.pth')
        torch.save(self.model.state_dict(), filename)


    def load(self, save_dir):
        filename = os.path.join(save_dir, 'checkpoint-model.pth')
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.cfg.params.device)), strict=False)