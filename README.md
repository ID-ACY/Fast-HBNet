# Fast-HBNet
The code of "Fast-HBNet: Hybrid Branch Network for Fast Lane Detection" will be released here later.


# LaneNet on CULane Dataset

## Installation

This code has been tested on Ubuntu 16.04, python 3.7, cuda 9.0, cudnn 7.0, opencv(C++)3.4.8 with GPU RTX-2080Super. To run this code you should have a environment with pytorch==1.7.0, and the lower version pytorch maybe not work. Other environment dependencies can be installed by 'pip'.

## Performance
| Category  | F1-measure          |
| --------- | ------------------- |
| Normal    | 82.9                |
| Crowded   | 61.1                |
| HLight    | 53.4                |
| Shadow    | 56.2                |
| No line   | 37.7                |
| Arrow     | 72.2                |
| Curve     | 59.3                |
| Crossroad | 5828 （FP measure） |
| Night     | 53.4                |
| FPS       | 44                  |
| Total     | 61.8                |

## Data preparation

### CULane

The dataset is available in [CULane](https://xingangpan.github.io/projects/CULane.html). Please download and unzip the files in one folder, which later is represented as `CULane_path`.  Then modify the path of `CULane_path` in `config.py`.
```
CULane_path
├── driver_100_30frame
├── driver_161_90frame
├── driver_182_30frame
├── driver_193_90frame
├── driver_23_30frame
├── driver_37_30frame
├── laneseg_label_w16
├── laneseg_label_w16_test
└── list
```

## Demo Test

For single image visual, run:

```Bash
python visual.py -i visualization/02445.jpg 
                    -w path/to/pretrainedModel
                    -b 1.5
                    [--visualize / -v]
```

The predicted lane result will be saved in "visualization/". Our pretrained model can be downloaded [here](https://drive.google.com/drive/folders/1hJCMzft-BkNj6jylihGX2Yumbnyxr1Dy?usp=sharing).  Please download and put it under "experiment/exp0/".

'demo result image'

![result image](https://github.com/ZT-GroupR/Fast-HBNet/blob/master/LaneNetonCULane/visualization/02445_result.jpg)


## Train 

1. Start training:

   ```python
   python train_CULane.py -e ./experiments/exp0 [-r]
   ```
  **Note**


  - [-r] is a optional parameters. If you train with "-r", your train process will start with our pretrained model.

   
## Test 

1. For speed test, run:
   ```python
   python test_fps.py -e ./experiments/exp0
   ```

2. For quickly calculate F1-score, you should unzip `/experiments/exp0/corrd_output.zip` to `/experiments/exp0/`, and make some operations as the following:

   a). Modify "root" as absolute project path in `utils/lane_evaluation/CULane/Run.sh`.
   ```bash
   cd utils/lane_evaluation/CULane
   mkdir build 
   cd build
   cmake ..
   make
   ```
   b). Then you should run the following command to calculate quickly. The predicted lane of each road scenes will be saved into "experiments/exp0" directory.
   ``` shell
   python test_CULane.py -e ./experiments/exp0 [-R]
   ```
  **Note**


  - [-R] is a optional parameters. If you test without "-R", your can test LaneNet on CULane dataset quickly. If you test with "-R", the predicted result will be regenerated, and it will take several hours due to the post processing operations are added.

    c). The total F1 can be calculated by `utils/lane_evaluation/CULane/calTotal.m` after finishing step b).



## Reference

[1]. Neven, Davy, et al. "[Towards end-to-end lane detection: an instance segmentation approach.](https://arxiv.org/pdf/1802.05591.pdf)" *2018 IEEE Intelligent Vehicles Symposium (IV)*. IEEE, 2018.

[2]. De Brabandere, Bert, Davy Neven, and Luc Van Gool. "[Semantic instance segmentation with a discriminative loss function.](https://arxiv.org/pdf/1708.02551.pdf)" CVPR2017.

[3]. Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, Xiaoou Tang. "[Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://arxiv.org/abs/1712.06080)" AAAI2018.

[4]. [https://github.com/harryhan618/LaneNet](https://github.com/harryhan618/LaneNet)

[5]. [https://github.com/XingangPan/SCNN](https://github.com/XingangPan/SCNN)

[6]. [https://github.com/MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)





