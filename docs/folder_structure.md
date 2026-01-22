# Folder Structure

<!-- 
The following commands in the terminal let you know the tree structure of the storage folder:
tree -I "__pycache__" -I "saved" -I "__init__.py"
-->

  ```
  pytorch-pycm-template/
  │
  ├── .flake8 - Python code style checker configuration
  ├── .gitignore - Git version control exclusion settings
  ├── .github/ - GitHub Actions workflows and configurations
  │
  ├── README.md - Project documentation
  ├── requirements.txt - Python package dependencies
  │
  ├── parse_config.py - class to handle config file and cli options
  ├── parse_save_path.py - module to parse and manage save paths
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
├── utils/ - Utility functions
│   ├── util.py - General-purpose utility functions
│   ├── config_util.py - Configuration file handling and settings management utilities
│   ├── model_util.py - Model-related utility functions (e.g., loading, saving, architecture helpers)
│   ├── plot_util.py - Plotting and chart generation utilities
│   └── output_visualization_util.py - Result visualization and output rendering utilities
  │
├── base/ - Abstract base classes for core components
│   ├── base_model.py - Base class for PyTorch model implementations
│   ├── base_raw_data_loader.py - Base class for data loaders with unsplit datasets
│   ├── base_split_data_loader.py - Base class for data loaders with pre-split train/test datasets
│   ├── base_sampler.py - Base class for sampling strategies (e.g., oversampling, undersampling)
│   ├── base_trainer.py - Base class for model training procedures
│   ├── base_tester.py - Base class for model evaluation and testing
│   ├── base_metric.py - Base class for accumulating values and computing running averages
│   ├── base_metric_ftns.py - Base class for performance metrics (e.g., TPR, specificity)
│   ├── base_confusion.py - Base class for tracking confusion matrices using PyCM (includes fixed-specificity variant)
│   ├── base_tracker.py - Base class for managing and saving tracked information via dictionaries
│   ├── base_hook.py - Base class for registering PyTorch model hooks
│   ├── base_explainer.py - Base class for XAI (explainable AI) methods with utilities like finding last convolutional layers
│   └── base_resultvisualization.py - Base class for visualizing training and evaluation results
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── DA/ - Data augmentation techniques folder
  │   │   └── ...
  │   ├── data_augmentation.py - Definitions and implementations for augmentation methods in DA folder
  │   ├── sampler/ - Data sampling strategies folder
  │   │   └── ...
  │   ├── data_sampling.py - Definitions and implementations for samplers in sampler folder
  │   ├── mnist_data_loaders.py - Example data loader for MNIST dataset
  │   ├── excel_img_loaders.py - Example data loader for image datasets managed via Excel files
  │   ├── npz_loaders.py - Example data loader for NPZ format datasets (recommended as base template)
  │   └── transforms.py - Custom transformation functions (e.g., padding options for rectangular to square image conversion)
  │
├── model/ - Model-related components and configurations
│   ├── models/ - Custom model architectures
│   │   └── ...
│   ├── optims/ - Custom optimizer implementations
│   │   └── ...
│   ├── loss_functions/ - Custom loss function implementations
│   │   └── ...
│   ├── lr_schedulers/ - Custom learning rate scheduler implementations
│   │   └── ...
│   ├── model.py - Model definitions (imports custom models from models/ and standard models)
│   ├── optim.py - Optimizer definitions (imports custom optimizers from optims/ and standard optimizers)
│   ├── loss.py - Loss function definitions (imports custom losses from loss_functions/ and standard losses)
│   ├── lr_scheduler.py - Learning rate scheduler definitions (imports custom schedulers from lr_schedulers/ and standard schedulers)
│   ├── metric.py - Scalar evaluation metrics (e.g., accuracy, precision, recall)
│   └── plottable_metrics.py - Plot-based evaluation metrics (e.g., ROC curve, PR curve, confusion matrix)
  │
├── runner/ - Training and testing execution modules
│   ├── trainer.py - Standard model training
│   ├── trainer_fixedSpec.py - Model training with fixed-specificity tracking
│   ├── tester.py - Standard model evaluation and testing
│   ├── tester_fixedSpec.py - Model testing with fixed-specificity performance tracking
│   ├── tester_only_excel_output.py - Lightweight testing that outputs only prediction probabilities to Excel
│   └── explainer.py - XAI (explainable AI) visualization using torchcam library for saved models
  │
├── runfile/ - Executable scripts that instantiate and run template components
│   ├── train_mnist.py - Example training script using MNIST dataset
│   ├── train_npz.py - Example training script using NPZ dataset
│   ├── test_mnist.py - Example testing script for trained models using MNIST dataset
│   ├── test_npz.py - Example testing script for trained models using NPZ dataset
│   ├── test_npz_fixedSpec.py - Example testing script with fixed-specificity evaluation using NPZ dataset
│   ├── test_npz_only_excel_output.py - Lightweight testing script that outputs only probability predictions to Excel using NPZ dataset
│   └── explainable_torchcam.py - Example XAI visualization script using torchcam library
  │
  ├── raytuner_example.ipynb - Example notebook demonstrating hyperparameter tuning with Ray Tune
  │
  ├── config/ - Example of a configuration file
  │   └── {dataloader type}
  │       └── {optimizer}-lr_{lr}-{lr_scheduler}
  │           └── ....json
  │
  └── saved/ - Example of a outputs
      ├── models/ - trained models are saved here
      ├── log/ - default logdir for tensorboard and logging output
      └── output/ - Optional. To save 1. model visualization image  2. performance plot at last epoch 3. metrics result per epoch
  ```
<br>
```
