# Sharp Agent: a sharpness-aware nehancement for Sleeper Agent

This code is the official PyTroch implementation of method [SAPA+Sleeper Agent](https://openreview.net/pdf?id=bxITGFPVWh). The implementation is based on the official code of Sleeper Agent.


## Dependencies

- PyTorch => 1.6.*
- torchvision > 0.5.*
- higher [best to directly clone https://github.com/facebookresearch/higher and use ```pip install .```]
- python-lmdb [only if datasets are supposed to be written to an LMDB]




## USAGE

The wrapper for the Sleeper Agent can be found in sleeper_agent.py. The default values are set for attacking ResNet-18 on CIFAR-10.

There are a buch of optional arguments in the ```forest/options.py```. Here are some of them:

- ```--patch_size```, ```--eps```, and ```--budget``` : determine the power of backdoor attack.
- ```--dataset``` : which dataset to poison.
- ```--net``` : which model to attack on.
- ```--retrain_scenario``` : enable the retraining during poison crafting.
- ```--poison_selection_strategy``` : enables the data selection (choose ```max_gradient```)
- ```--ensemble``` : number of models used to craft poisons.
- ```--sources``` : Number of sources to be triggered in inference time.

### Cifar10

To craft poisons on Cifar10 and ResNet18 you can use the following sample command:

```shell
python -u sleeper_agent.py --net ResNet18 --dataset CIFAR10 --patch_size 8 --budget 0.01 --eps 16  --pbatch 128 --epochs 80 --sources 1000 --pbatch 128 --source_gradient_batch 300 --save numpy --optimization standard2 --name Your Name --source_criterion sharpness --sharpsigma 0.01
```