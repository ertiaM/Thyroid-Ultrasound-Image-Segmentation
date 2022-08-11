# Thyroid Nodule Segmentation in Ultrasound Image
It is an image segmentation in Thyroid nodule ultrasound images, whose Implementation is by Fully Convolutional Network. The main code is modified from the PyTorch version of [FCN](https://github.com/L1aoXingyu/Deep-Learning-Project-Template). 

### Requirements
- pytorch
- torchvision
- ignite
- yacs
- tensorboardX (tensorflow for the tensorboardx)

### Dataset
The original dataset [DDTI](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) used in this experiment is an open access database of thyroid ultrasound images, and is public and available on Kaggle. After the preliminary enhancements are deployed and the masks are generated, the dataset is used for the segementation. And the new [dataset](https://www.kaggle.com/datasets/eiraoi/thyroidultrasound) is uploaded and is available on Kaggle, too. 

### Configeration
Before the program being executed, the major configuration files lie in the the folder `configs`. The settings that required for configuration are the basically the file paths of `dataset root`, `backbone weight` and `output directory`.  
    The `backbone weight` that used in this experiment is the PyTorch pretrained model of vgg16. 
```bash
https://download.pytorch.org/models/vgg16-397923af.pth
```  
Moreover, for the other configurations such as `learning rate` and `weight decay`, you need to modify the parameters in the file `default.py` of the folder `configs`.

### Training
Basically, to execute the training process, the `train_fcn.py` file lying in the `tools` folder should be run with the choosing configuration file.  
In this experiment, the result is coming from after `FCN32s`, `FCN16s` and `FCN8s`. Hence, configuration file should be set first as `train_fcn32s.yml`, `train_fcn16s.yml` and `train_fcn8s.yml` in order. You can modify the configuration in the following two ways.  

#### 1. External Modification

```bash
python3 tools/train_fcn.py --config_file='configs/train_fcn32s.yml'
```

#### 2. Internal Modification
You can change configuration parameter inside the file `train_fcn.py`. 

```bash
parser.add_argument(
        "--config_file", default="../configs/train_fcn8s.yml", help="path to config file", type=str
)
``` 
 
### Result
The result is recorded by `tensorboard`, which is packed with the pretrained model in [ertiaM](https://pan.baidu.com/s/1wd__23_YazZT53ko6aMwUg?pwd=1989)
The result from DDTI is shown below, where 80% are randomly selected for training, and the rest is for validation and testing. 

|Model| Epoch | Mean IU |
|-|-|-|
| FCN32s| 13 | 72.9|
| FCN16s| 8 | 75.8| 
| FCN8s | 7 |  76.1 |  
