# -*- coding: utf-8 -*-

import logging
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .initializer import initialize

"""
Network architecture
"""

def FloatTensor(*args):
    if 1:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


class UniADRS(nn.Module):
    def __init__(self, n_channels=270, descriptor_dim=64, num_classes=1, stem_identity=True, device='cuda:2'):
        super(UniADRS, self).__init__() 

        self.n_channels = n_channels
        self.device = device

        self.spectral_stem = make_layers([descriptor_dim, descriptor_dim], n_channels)
        
        self.spatial_stem = self.spectral_stem if stem_identity else self.spatial_stem_
       
        self.normality_extractor = Normality_extractor(descriptor_dim=64, channels=[128, 256, 512, 512, 1024], fusion_mode='LLLLL', device=device)
        
        self.features_DN_combine = DecoderCell(
                                        in_channel=descriptor_dim,
                                        out_channel=num_classes,
                                        mode='C',
                                        device = device)

        initialize(self, 'xavier-uniform')

    def forward(self, *input):
        if len(input) == 1:
            x = input[0]
            tar = None
            test_mode = True
        if len(input) == 2:
            x = input[0]
            tar = input[1]
            test_mode = False
        if len(input) == 3:
            x = input[0]
            tar = input[1]
            test_mode = input[2]
        if len(input) == 4:
            x = input[0]
            tar = input[1]
            full_mask = input[2]
            test_mode = input[3]
            full_mask = full_mask.unsqueeze(1)


        ratio = int(self.n_channels/x.shape[1])
        x = x.repeat( (1,ratio,1,1))
        if x.shape[1] < self.n_channels:
            x = torch.cat([x, x[:,0:(self.n_channels -x.shape[1]),:,:]], 1)

        pixel_descriptors = self.spectral_stem(x) if ratio <= 30 else self.spatial_stem(x)
        normality_descripotrs = self.normality_extractor(pixel_descriptors)
        features_DN = self.features_DN_combine(pixel_descriptors, normality_descripotrs)
        _, features_T, pred = features_DN

        if test_mode:
            return [pred], 0.0
        
        feature_level_loss = 0.1*self.feature_level_loss(full_mask, features_T)

        tar = tar.ravel()
        labeled_locs = tar != 2
        score_labeled = (pred.squeeze(1).ravel())[labeled_locs]
        gt_labeled = tar[labeled_locs]
        weight = torch.ones(score_labeled.shape,).to(self.device)
        weight[gt_labeled == 0] *=  0.5
        pixel_level_loss = self.loss(score_labeled, gt_labeled.int())

        loss =  pixel_level_loss + feature_level_loss

        return [pred], loss


    def feature_level_loss(self, gt, output):
        gt = gt.squeeze(0).squeeze(0)
        gt[gt == 2] = 4
        gt[gt == 1] = 2
        gt[gt == 3] = 1
        output = output.squeeze(0).squeeze(0)
        locs_bkg_normal = torch.where(gt < 2)
        locs_anomaly = torch.where(gt == 2)

        normal_features = output[:, locs_bkg_normal[0], locs_bkg_normal[1]].permute((1,0))
        anomaly_features = output[:, locs_anomaly[0], locs_anomaly[1]].permute((1,0))

        loss = torch.tensor(0.0).to(self.device)
        if len(normal_features) > 0:
            loss_normal, c_normal, r_normal = self.compact_loss(normal_features)
            loss += loss_normal

        if len(anomaly_features) > 0:
            loss_anomaly, c_anomaly, r_anomaly = self.compact_loss(anomaly_features)
            loss += loss_anomaly

        if torch.isnan(loss):
            exit(0)

        if c_anomaly != None:
            loss += -1*F.pairwise_distance(c_normal, c_anomaly, p=2)
        else:
            loss = 0.0
        return loss


    def compact_loss(self, normal_features):
        normal_c = normal_features.mean(0)
        normal_dists = torch.sum((normal_features - normal_c.repeat(normal_features.shape[0], 1))**2, dim=1)
        normal_radius = torch.mean(torch.sqrt((normal_dists) + 1e-8))
        normal_scores = normal_dists - normal_radius**2  
        normal_compact_loss = normal_radius ** 2 + (1 / 0.1) * torch.mean(torch.max(torch.zeros_like(normal_scores), normal_scores))
        return normal_compact_loss, normal_c, normal_radius

def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
        

class Normality_extractor(nn.Module):
    def __init__(self, descriptor_dim=64, channels=[ 128, 256, 512, 512, 1024], fusion_mode='LLLLL', device='cuda:0'):
        super(Normality_extractor, self,).__init__()
        self.normality_encoder_1 = make_layers(['M', channels[0], channels[0]], descriptor_dim).to(device)
        self.normality_encoder_2 = make_layers(['M', channels[1], channels[1], channels[1]], channels[0]).to(device)
        self.normality_encoder_3 = make_layers(['M', channels[2], channels[2], channels[2]], channels[1]).to(device)
        self.normality_encoder_4 = make_layers(['m', channels[3], channels[3], channels[2], 'm'] , channels[2]).to(device)
        self.normality_encoder_5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=12, dilation=12),
            nn.Conv2d(channels[4], channels[4], 3, 1, 1)
        ).to(device)

        self.normality_decoders = []
        channels.insert(0, descriptor_dim)
        channels.reverse()
        for i in range(5):
            self.normality_decoders.append(
                DecoderCell(
                            in_channel=channels[i],
                            out_channel=channels[i + 1], 
                            mode=fusion_mode[i],
                            device = device).to(device))


    def forward(self, *input):

        x = input[0]
        encoder_features_1 = self.normality_encoder_1(x)
        encoder_features_2 = self.normality_encoder_2(encoder_features_1)
        encoder_features_3 = self.normality_encoder_3(encoder_features_2)
        encoder_features_4 = self.normality_encoder_4(encoder_features_3)
        encoder_features_5 = self.normality_encoder_5(encoder_features_4)
        encoder_features = [encoder_features_1, encoder_features_2, encoder_features_3, encoder_features_4, encoder_features_5]

        dec = None
        for i in range(5): 
            dec, _, _ = self.normality_decoders[i](encoder_features[4 - i], dec)

        return  dec


class DecoderCell(nn.Module):
    def __init__(self, in_channel, out_channel, mode, device):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.InstanceNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'L':
            self.attn = LAM(in_channel)
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.InstanceNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        elif mode == 'G':
            self.attn = GAM(in_channel)
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.InstanceNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        elif mode == 'C':
            self.attn = None
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)
        else:
            assert 0

        self.device = device

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0] 
            dec = input[0]
        else:
            en = input[0]
            dec = input[1]

        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.interpolate(dec.cpu(), scale_factor=2, mode='nearest').to(self.device)
        elif dec.size()[2] != en.size()[2]:
            assert 0
        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        feature_ranking_logits = F.sigmoid(fmap)
        fmap = F.relu(fmap)

        if not self.mode == 'C':
            fmap_att = self.attn(fmap)  # F_att
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            dec_out = self.bn_feature(x)
            dec_out = F.relu(dec_out)
            _y = torch.sigmoid(self.conv3(dec_out))
        else:
            dec_out = self.conv2(fmap)
            _y = torch.sigmoid(dec_out)

        return dec_out, feature_ranking_logits, _y


class GAM(nn.Module):
    def __init__(self, in_channel):
        super(GAM, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(in_channel, 1, batch_first=True) 
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel*2, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        shape = x.shape
        qk = self.conv1(x)
        q, k = qk[:,0:self.in_channel, :, :], qk[:,self.in_channel:2*self.in_channel, :, :]

        q = q.flatten(2).permute((0,2,1))
        k = k.flatten(2).permute((0,2,1)) 
        x2 = x.flatten(2).permute((0,2,1))

        attn_output, _ = self.multihead_attn(q, k, x2)
        attn_output = attn_output.permute((0,2,1)) 
        attn_output = attn_output.reshape((attn_output.shape[0], attn_output.shape[1], shape[2], shape[3])) 
        return attn_output 


class LAM(nn.Module):
    def __init__(self, in_channel):
        super(LAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size() # torch.Size([1, 1024, 28, 28])
        kernel = self.conv1(x) # torch.Size([1, 128, 28, 28])
        kernel = self.conv2(kernel) # torch.Size([1, 49, 28, 28])
        kernel = F.softmax(kernel, 1) 
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7) # torch.Size([1, 1, 784, 49])
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1) # 
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x


if __name__ == '__main__':
    # vgg = torchvision.models.vgg16(pretrained=True)

    device = torch.device("cuda:2") 
    batch_size = 1
    noise = torch.randn((batch_size, 3, 224, 224)).to(device)
    target = torch.randn((batch_size, 1, 224, 224)).to(device)
 
    model = UniADRS(n_channels=270).to(device)  
    # model.encoder.seq.load_state_dict(vgg.features.state_dict(), strict=False) 
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    # print('Time: {}'.format(time.clock()))
    _, loss = model(noise, target)
    loss.backward() 
