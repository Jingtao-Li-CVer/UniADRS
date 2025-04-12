## Learning a Cross-Modality Anomaly Detector for Remote Sensing Imagery (TIP2024)

<p align="center">
  <img src=./figs/Figure1.jpg width="600"> 
</p>

This is a PyTorch implementation of the [UniADRS model](https://ieeexplore.ieee.org/abstract/document/10747828): 
```
@article{li2024learning,
  title={Learning a Cross-modality Anomaly Detector for Remote Sensing Imagery},
  author={Li, Jingtao and Wang, Xinyu and Zhao, Hengwei and Zhong, Yanfei},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```

### Outline
1. In this paper, a **cross-modality and cross-scene anomaly detector** for remote sensing imagery has been proposed.
2. Different from traditional deep detectors, we convert the learning target from **varying** background distribution to **consistent** deviation relationship between anomalies and background for **zero-shot detection**.
3. We **theoretically prove** that meeting the large margin condition in training samples can guarantee the correct deviation rank for unseen anomaly and background.
4. We have built **an anomaly detection dataset with five modalities** including hyperspectral, visible light, synthetic aperture radar
(SAR), infrared and low-light. 


### Introduction

Current anomaly detectors aim to learn the certain background distribution, the trained model cannot be transferred to unseen images. Inspired by the fact that the deviation metric for score ranking isconsistent and independent from the image distribution, this study exploits the learning target conversion from the varying background distribution to the consistent deviation metric. We theoretically prove that the large-margin condition in labeled samples ensures the transferring ability of learned deviation metric. To satisfy this condition, two large margin losses for pixel-level and feature-level deviation ranking are proposed respectively. Since the real anomalies are difficult to acquire, anomaly simulation strategies are designed to compute the model loss. With the large-margin learning for deviation metric, the trained model achieves cross-modality detection ability in five modalities—hyperspectral, visible light, synthetic aperture radar (SAR), infrared and low-light—in zero-shot manner.

<p align="center">
  <img src=./figs/Figure2.jpg width="600"> 
</p>

### Preparation

1. Install required packages according to the requirements.txt.
2. Download the required datasets for anomaly simulating (data.rar) and testing (data2.rar) from the following link and put all subforders in the `data` folder.
    (https://www.wjx.cn/vm/ruVKZ9e.aspx#) Link would be shown after filling the form.

### Model Training and Testing

1. UniADRS can be trained on simulated deviating samples and infer the unseen modalities and scenes directly.
2. Starting the training and testing process using the following command.

```
python run.py ./configs/training&infer_img_folder.yaml
```


### Testing on Single Image

1. Our code also supports the common need of inferring single image.
2. Firstly, assigning the trained checkpoint path for ckpt_dir key, and filling the parameters in datasets/test key in training&infer_sinlge_image.yaml file. Secondly, uncommenting the following codes in base_runner.py.

```
    # Infer only
    # self._test(0, save_anomaly_map=True)
    # exit(0)
```
3. You can also run the training and testing (single image) process using the following command.

```
python run.py ./configs/training&infer_sinlge_image.yaml
```


### Qualitative result  

 &emsp;The following are the zero-shot detection results of UniADRS on five modalities.

<p align="center">
  <img src=./figs/Figure4.jpg width="600"> 
</p>

