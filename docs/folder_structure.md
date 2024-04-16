# Folder Structure

<!--
The following commands in the terminal let you know the tree structure of the storage folder:
tree -I "__pycache__" -I "saved" -I "__init__.py"
-->


  ```
  pytorch-pycm-template/
  │
  ├── train_{dataloader type}.py - main script to start training
  ├── test_{dataloader type}.py  - evaluation of trained model
  │
  ├── create_single_config.py    - Example of creating a configuration file
  ├── config                     - Example of a configuration file
  │   └── {dataloader type}
  │       └── {optimizer}-lr_{lr}-{lr_scheduler}
  │           └── ....json
  │
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_metric.py
  │   ├── base_hook.py
  │   ├── base_model.py
  │   ├── base_raw_data_loader.py
  │   ├── base_split_data_loader.py
  │   ├── base_tester.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── DA - Please specify the folder for data augmentation management.
  │   │   └── ...
  │   ├── data_augmentation.py
  │   ├── sampling - Please specify the folder for data sampling management.
  │   │   └── ...
  │   ├── data_sampling.py
  │   ├── mnist_data_loaders.py
  │   ├── npz_data_loaders.py
  │   └── transforms.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, optims, losses, and metrics
  ├── model.py
  │   ├── models - Please specify the folder for model management.
  │   │   └── ...
  │   ├── optims - Please specify the folder for optimizer management.
  │   │   └── ...
  │   ├── model.py
  │   ├── optim.py
  │   ├── metric.py
  │   ├── metric_curve_plot.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   ├── log/ - default logdir for tensorboard and logging output
  │   └── output/ - Optional. To save 1. model visualization image  2. performance plot at last epoch 3. metrics result per epoch
  │
  ├── runner/ - trainers and testers
  │   ├── tester.py
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
<br>