# Customization

> This description includes instructions for initial project setup, CLI options, data loaders, models, loss functions, metrics, logging, and tests.   

> ### **INDEX**
> - [Customization](#customization)
>    + [Project initialization](#project-initialization)
>    + [Custom CLI options](#custom-cli-options)
>    + [Data Loader](#data-loader)
>    + [Data Augmentation](#data-augmentation)
>    + [Data Sampling](#data-sampling)
>    + [Learning Rate Scheduler](#learning-rate-scheduler)
>    + [Trainer](#trainer)
>    + [Tester](#tester)
>    + [Model](#model)
>    + [Loss](#loss)
>    + [Optimizer](#optimizer)
>    + [Metrics](#metrics)
>    + [Additional logging](#additional-logging)
>    + [Testing](#testing)
>    + [Checkpoints](#checkpoints)
>    + [Tensorboard Visualization](#tensorboard-visualization)
> <small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>



<br>

### Project initialization

Use the `new_project.py` script to make your new project directory with template files.   
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file.     
<br>

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.     
<br>


### Data Loader

1. **Writing your own data loader**
    - **Inherit ```BaseRawDataLoader```**   
        `BaseRawDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.     
        Please refer to `data_loader/mnist_data_loaders.py` for an MNIST data loading example.     
    
        `BaseRawDataLoader` handles:
        * Generating next batch
        * Data shuffling
        * Generating validation data loader by calling `BaseRawDataLoader.split_validation()`
            * It will return a data loader for validation of size specified in your config file.
            * The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).
                * **Note**: the `split_validation()` method will modify the original data loader
                * **Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`     

    - **Inherit ```BaseSplitDatasetLoader```**
        `BaseSplitDatasetLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.     
        Please refer to `data_loader/npz_loaders.py` for an npz data loading example.     
    
        `BaseSplitDatasetLoader` handles:
        * Generating next batch
        * Data shuffling
<br>

2. **DataLoader Usage**     
`DataLoader` is an iterator, to iterate through batches:
      ```python
      for batch_idx, (x_batch, y_batch) in data_loader:
          pass
      ```
<br>

### Data Augmentation

* **Writing your own data augmentation**

1. **Inherit augmentation class or implement callable**

    Data augmentation techniques are located in `data_loader/DA/` folder.     
    Please refer to `data_loader/DA/cutmix.py` for a CutMix example.
    
    Data augmentation classes handle:
    * Random transformation of input batch data
    * Probability-based augmentation application
    * Preservation of batch shape and data type
    
    When implementing custom augmentation:
    * Either inherit from `BaseHook` class
    * Accept batch data as input (typically tensor or numpy array)
    * Accept optional hyperparameters through `__init__`
    * Return augmented batch maintaining original shape

2. **Configure augmentation in config file**

    To enable data augmentation, add the following configuration:
    
    ```json
    "data_augmentation": {
        "type": "Cutmix",
        "args": {
            "beta": 0.1,
            "prob": 0.5
        },
        "hook_args": {
            "layer_idx": 0,
            "pre": true
        }
    }
    ```
    
    Configuration parameters:
    * `type`: Name of the augmentation class (must be defined in `data_loader/data_augmentation.py`)
    * `args`: Hyperparameters specific to the augmentation technique
        * Example for CutMix: `beta` controls the cut area, `prob` is application probability
    * `hook_args`: Specifies where and how augmentation is applied
        * `layer_idx`: Index of the model layer where hook will be attached (0-based)
        * `pre`: If `true`, augmentation is applied to input values; if `false`, applied to output values

3. **Hook mechanism for selective layer augmentation**

    The trainer applies data augmentation via PyTorch forward hooks on the specified layer:
    
    * When `hook_args` is configured, a forward hook is registered on the layer at `layer_idx`
    * During forward pass, augmentation is triggered at that specific layer
    * This allows selective augmentation without modifying model code
    * **Note**: Make sure `layer_idx` corresponds to an existing layer in your model architecture
    * **Note**: If augmentation is applied to outputs (`pre: false`), ensure the augmented data format matches expected layer input

<br>
   
### Data Sampling

* **Handling imbalanced datasets with data sampling**

1. **Understanding data sampling strategies**

    Data sampling is located in `data_loader/DA/data_sampling.py`.     
    Data sampling handles:
    * **Downsampling**: Randomly remove majority class samples to balance dataset
    * **Upsampling**: Duplicate minority class samples to increase their representation
    * Per-batch sampling during training
    * Helpful for addressing class imbalance issues

    Choose upsampling or downsampling based on your dataset:
    * **Downsampling (`"down"`)**: Recommended when majority class is very large; speeds up training by reducing data volume
    * **Upsampling (`"up"`)**: Recommended when minority class is very small; preserves all information by repeating samples

2. **Configure sampling in config file**

    To enable data sampling, add the following configuration:
    
    ```json
    "data_sampling": {
        "type": "down",
        "name": "random_downsampling"
    }
    ```
    
    Configuration parameters:
    * `type`: Sampling strategy
        * `"down"`: Downsample majority classes
        * `"up"`: Upsample minority classes
    * `name`: Name of the specific sampling technique implemented in `data_loader/DA/data_sampling.py`

3. **How sampling is applied during training**

    * Sampling is performed for each batch during the training phase
    * The trainer automatically applies the configured sampling strategy
    * Sampling ratios are typically determined by class distribution
    * **Note**: Sampling is applied after data loading but before passing to the model
    * **Note**: If `data_sampling` is not configured, no sampling will be performed and original class distribution is maintained
    * **Note**: For validation and testing, sampling is typically disabled to preserve original data distribution

<br>

### Learning Rate Scheduler

* **Using learning rate scheduler to control training dynamics**

1. **Configure PyTorch built-in schedulers**

    PyTorch provides various learning rate scheduling strategies. To use them, add to your config file:
    
    ```json
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 10,
            "gamma": 0.1
        }
    }
    ```
    
    Common PyTorch scheduler types:
    * `StepLR`: Multiply learning rate by `gamma` every `step_size` epochs
    * `MultiStepLR`: Multiply learning rate by `gamma` at specified milestone epochs
    * `ExponentialLR`: Multiply learning rate by `gamma` after each epoch
    * `CosineAnnealingLR`: Decrease learning rate following cosine curve
    * `ReduceLROnPlateau`: Decrease learning rate when validation metric plateaus
    
    For detailed information on scheduler parameters and types, refer to [PyTorch Optimizer documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

2. **Implementing custom learning rate scheduler**

    Custom schedulers can be implemented in `model/lr_schedulers.py` as a PyTorch `_LRScheduler` subclass:
    
    ```python
    from torch.optim.lr_scheduler import _LRScheduler
    
    class CustomScheduler(_LRScheduler):
        def __init__(self, optimizer, **kwargs):
            # Initialize custom scheduler parameters
            super().__init__(optimizer)
        
        def get_last_lr(self):
            # Return current learning rate
            return [group['lr'] for group in self.optimizer.param_groups]
        
        def step(self, epoch=None):
            # Update learning rate based on custom logic
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.compute_lr(epoch)
        
        def compute_lr(self, epoch):
            # Implement custom learning rate computation
            return base_lr * (1 - epoch / total_epochs)
    ```
    
    Then register it in config by setting `"type"` to your custom scheduler name.

3. **Using fixed learning rate (optional)**

    If you prefer a fixed learning rate without scheduling, simply remove the `lr_scheduler` section from your config file:
    
    ```json
    // Remove or comment out the lr_scheduler section
    // "lr_scheduler": { ... }
    ```
    
    The trainer will automatically detect the absence of `lr_scheduler` and maintain a constant learning rate throughout training.

<br>


### Trainer

* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving & Checkpoint resuming
        * This requires device information for `map_location`.
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If `monitor` is set in the config to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If `early_stop` is set in the config, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.
      * If `accumulation_steps` is set in the config, gradient accumulation will be performed. For example, if the `batch size` is 4 and `accumulation_steps` is 4, it means employing a technique where the effect is similar to having a `batch size` of 16.
      * If `data_augmentation` is set in the config, the specified technique will be applied. However, this will be performed by setting a hook on the designated layer using `hook_args`. In this case, `pre` determines whether to apply data augmentation to the input values.
      * If the `data_sampling` is set in the config, sampling will be performed for each batch. If the `type` is set to "down," the data will decrease; if it's set to "up," the data will increase.
    * outputs saving
        * This requires device information for `classes` (class name list).
      * If `save_performance_plot` is configured in the config, the performance of the model up to the last epoch is saved in both `plot` and `json` formats.
      * If `tensorboard` is configured in the config, the model performance is visualized on TensorBoard.
          * `tensorboard_projector`: enable save projector at first epoch
          * `tensorboard_pred_plot`: enable save prediction example plot per epoch (5 data)

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`.    
    Please refer to `trainer/trainer.py` for training.
    * **Iteration-based training**   
      `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

3. **Make a runfile for terminal**

   Please refer to `train_{dataloader type}.py` for training.
    1. To ensure experiments are conducted in the same environment, the random seed is fixed.
         ```python
            SEED = 123
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(SEED)
         ```
    2. Objects such as data loaders and models are created based on the contents of the config file.
    3. Option 1: Visualize the model graph
       - Using the `torchviz` library, the model can be visualized as a graph.
         ```python
            from torchviz import make_dot
            import model.model as module_arch
            ...
            
            if config['arch']['visualization']:
                graph_path = config.output_dir / config['arch']['type']
                modelviz = config.init_obj('arch', module_arch)
                make_dot(modelviz(next(iter(train_data_loader))[0]), params=dict(list(modelviz.named_parameters())), show_attrs=True, show_saved=True).render(graph_path, format='png')
         ```
    4. Option 2: Print model information
       - Using functions from `BaseModel`, basic output can be printed with the following command.
         ```python
            import model.model as module_arch
            ...
            model = config.init_obj('arch', module_arch)
            logger.info(model)
         ```
       - Additionally, for more detailed output, the `torchinfo` library can be used.
         ```python
            from torchinfo import summary
            ...
            model = config.init_obj('arch', module_arch)
            input_size = next(iter(train_data_loader))[0].shape
            model_info = str(summary(model, input_size=input_size, verbose=0))
            logger.info('{}\n'.format(model_info))
         ```
<br>

### Tester

* **Writing your own tester**

1. **Inherit ```BaseTester```**

    `BaseTester` handles:
    * Test process logging
    * Checkpoint resuming
        * This requires device information for `map_location`.
    * outputs saving
        * This requires device information for `classes` (class name list).
        * Basic settings follow the `trainer` in the config.
          * Only the `tensorboard_projector` setting follows the configuration in the `tester`.

2. **Implementing abstract methods**

    You need to implement `_test()` for your test process.        
    Please refer to `trainer/tester.py` for training.

3. **Make a runfile for terminal**

   Please refer to `test_{dataloader type}.py` for testing.
<br>

### Model

* **Writing your own model**

1. **Inherit `BaseModel`**   
   Please refer to `model/model.py` for a LeNet example.  

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.
    * `conv_output_size`: Calculate the output size of convolution layer.
    * `pooling_output_size`: Calculate the output size of pooling layer.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`
<br>

### Loss

Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.     
<br>

### Optimizer

Custom optimizer can be implemented in 'model/optim.py'. Use them by changing the name given in "optimizer" in config file, to corresponding name.     
<br>

### Metrics

Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": {"ACC":1, "ACC_class":null},
  "curve_metrics": {"ROC":null},
  ```

Additionally, if curve plots such as ROC are needed, please refer to '[model/plottable_metrics.py](../model/plottable_metrics.py)'. Here, ROC plots are implemented using `scikit-learn` to calculate macro and micro values. If simply class-wise ROC plots are needed, they can be easily implemented using [pycm](https://www.pycm.io/doc/#ROC-curve).
<br>

### Additional logging

If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```     
<br>

### Testing

You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.  
In the current example file, the test option is enabled by default. If you've set the test option's default value to false in your modified file, you need to explicitly enable it in the command.
<br>

### Checkpoints

You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.module.state_dict() if isinstance(self.model, DP) or isinstance(self.model, DDP) else self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```     
<br>

### Tensorboard Visualization

This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/ --port 1111 --bind_all` at the project root, then server will open at `http://localhost:1111`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at logger/visualization.py will track current steps.     
<br>
