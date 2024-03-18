# PyTorch Pycm Template Project
---
> To easily create a PyTorch deep learning project, consider using this template.    
> It's based on the [pytorch-template](https://github.com/victoresque/pytorch-template), so some parts aren't explained in detail.    
> Unlike the original, it utilizes the pycm library for metrics computation.    
> Consequently, the example files have been adjusted to incorporate the pycm library.     
> Moreover, the template supports pre-split npz data examples for training, validation, and testing sets.    

<br>

> ### **INDEX**
> - [PyTorch Pycm Template Project](#pytorch-pycm-template-project)
>    * [Requirements](#requirements)
>    * [Features](#features)
>    * [Folder Structure](#folder-structure)
>    * [Usage](#usage)
>        + [Config file](#config-file)
>        + [Resuming from checkpoints](#resuming-from-checkpoints)
>        + [Using Multiple GPU](#using-multiple-gpu)
>        + [Testing with checkpoints](#testing-with-checkpoints)
>    * [Customization](#customization)
>    * [Automated Experimentation Process](#automated-experimentation-process)
>    * [Contribution](#contribution)
>    * [License](#license)
>    * [Acknowledgements](#acknowledgements)
> 
> <small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

<br>

## Requirements
---
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (2.0 recommended)
* pycm >= 4.0
* numpy
* pandas
* tensorboard >= 2.16 (Optional for visualization during training and testing)
* scikit-learn (Optional for ROC curve, 1.1 recommended)
* tqdm (Optional for `test.py`)
* torchinfo (Optional for Model Information)
* torchviz (Optional for Model Visualization)
    * If you failed to execute PosixPath, execute `conda install python-graphviz`.
<br>

## Features
---
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseTester` handles checkpoint resuming, test process logging, and more.
  * `BaseRawDataLoader` is the [`BaseDataLoader`](https://github.com/victoresque/pytorch-template/blob/master/base/base_data_loader.py) class of the [pytorch-template](https://github.com/victoresque/pytorch-template). It equally handles batch creation, data shuffling, and validation data segmentation.
  * `BaseSplitDatasetLoader` loads pre-split npz data based on the mode (training, validation, testing).
  * `BaseModel` provides basic model summary.
  * `MetricTracker` and `ConfusionTracker` provides basic metric and confusion matrix tracker.  
<br>

## Folder Structure
---
Please check [here](docs/folder_structure.md) for explanations.

## Usage
---
This repository's code supports a total of 4 essential options and 2 additional options.   
* Essential options:
    1. `-c`, `--config`: Set the path to the config file defining necessary components for the experiment environment such as data loaders and models.
    2. `-r`, `--resume`: Resume from a checkpoint. Set the path to the saved model checkpoint before the interruption.
    3. `-d`, `--device`: Specify the GPU number to use during training or testing.
    4. `-t`, `--test`: Enable the test mode.
* Additional options:
    1. `-lr`, `--learning_rate`: Set or modify the learning rate.
    2. `-bs`, `--batch_size`:Set or modify the batch size.

When writing the main Python code, you can input and configure it in the following structure.  
 ```python
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    
    import argparse
    import collections
    from parse_config import ConfigParser
    
    def main(config):
        ...
    
    args = argparse.ArgumentParser(description='PyTorch pycm Template')
    args.add_argument('-c', '--config', default=None,  type=str,  help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None,  type=str,  help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None,  type=str,  help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--test',   default=False, type=bool, help='Whether to enable test mode (default: False)')
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    
    main(config)
 ```
<br>

### Config file
---
Config files are in `.json` format.    
Please check [here](docs/config.md) for examples and detailed explanations.

Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```
<br>

### Resuming from checkpoints
---
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint_model.pth
  ```
<br>

### Using Multiple GPU
---
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 0,1 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```
<br>

### Testing with checkpoints
---
You can resume from a previously saved checkpoint by:

  ```
  python test.py --resume path/to/checkpoint_best_model.pth
  ```
<br>

## Customization
---
Please check [here](docs/customization.md) for explanations.   
This description includes instructions for initial project setup, CLI options, data loaders, models, loss functions, metrics, logging, and tests.   
<br>

## Automated Experimentation Process
---
You can set up repetitive training and testing using a bash script. After modifying only the necessary information such as the config file name and the device to be used in `auto_process.sh`, you can execute it with the following command.
```bash
bash auto_process.sh
```
In the case of testing, it searches for the most recently modified best_model within the folder of the date when the training was executed and performs the test.

<br>

## Contribution
---
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.
<br>

## License
---
This project is licensed under the MIT License. See  LICENSE for more details
<br>

## Acknowledgements
---
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
This project is inspired by the project [pytorch-template](https://github.com/victoresque/pytorch-template) by [victoresque](https://github.com/victoresque)
<br>
