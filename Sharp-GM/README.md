# SAPA+GM


This is the official implementation for method [SAPA+Gradient-Match](https://openreview.net/pdf?id=bxITGFPVWh). The implemetation is based on the implementation of Witch's Brew.


### Dependencies:
* PyTorch => 1.6.*
* torchvision > 0.5.*
- efficientnet_pytorch [```pip install --upgrade efficientnet-pytorch``` only if EfficientNet is used]
* python-lmdb [only if datasets are supposed to be written to an LMDB]


## USAGE:

To run the attack with Cifar10 and ResNet18, an example as follows:

```shell
python -u brew_poison.py --net ResNet18 --dataset CIFAR10 --eps 16 --budget 0.002 --pbatch 128 --name Your_Name --save numpy --savename Your_Save_Name --optimization Your_Optimization --target_criterion worstsharp --sharpsigma 0.05
```

## Some parameters for training uncertainty
- ```--optimization```: customized optimization
- ```--mixing_method```: augmentations such as Mixup, CutOut
- ```--ensemble```: ensemble
- ```--targets```: number of targets, support multiple targets
- ...