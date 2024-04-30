# CS5296 Group 14 Project

Welcome to our Distributed Machine Learning Project's code repository. This repository contains code for executing distributed deep learning tasks using PyTorch.

## Prerequisites

Before running the code, please ensure that you have the following requirements installed:

- **PyTorch 2.0+**
- **NumPy**

## Setup

1. You will need to download the CIFAR-10 dataset. The dataset can be downloaded from the following link: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

2. Once downloaded, extract the dataset into the `./datas` folder located in the project's root directory.

## Training

The training results will be saved in the `outputs` directory, with filenames structured as `{mode}_{model}_Nodes_{size}_Epoches_{epoch}.part{rank}`. Each component of the filename will be explained in detail below.

To train a model using this program, run the following command:

```bash
python train.py --model your_model --nodes node_num --epoch your_epoch --mode your_mode
```
or you can run the train.sh file to train a VGG11 model with 4 nodes for 20 epochs using Distributed Data Parallel (DDP)

### Argument Descriptions

- `--model`: Specifies the deep learning model to run. Available options are `ResNet`, `RepVGG`, `VGG11`, `VGG13`, `VGG16`, `VGG19`.
- `--nodes`: Specifies the number of distributed nodes to use during execution. Each node is set to use 4 threads by default. Please note that due to CPU performance limitations, this setting might not be effective.
- `--epoch`: Specifies the number of training epochs.
- `--mode`: Specifies the mode of operation. Currently, only two modes are supported:
  - `DDP` (Distributed Data Parallel)
  - `allReduce`

## Contributing

We welcome contributions to our project. If you have suggestions or improvements, please make a pull request or open an issue in this repository.

Thank you for being a part of our project!
