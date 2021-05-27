### Training from scratch
To start the training, run the following command:
```
CUDA_VISIBLE_DEVICES=0 ipython experiments/cremi/train_model.py -- <yourExperimentName> --DATA_HOMEDIR <path/to/the/cremi/data/you/downloaded> --inherit main_config.yml
```

The experiment data are saved by default in the `experiments/cremi/runs` folder.

### Keep training existing model
To start a new training and loading a previous model, run for example the following command:

`CUDA_VISIBLE_DEVICES=0 ipython experiments/cremi/train_model.py -- name_new_experiment --DATA_HOMEDIR <path/to/the/cremi/data/you/downloaded> --inherit main_config.yml  --update0 new_experiment_config.yml --config.model.model_kwargs.loadfrom PATH_TO_OLD_CHECKPOINT.pytorch --config.trainer.optimizer.Adam.lr 6e-5`

### Inference

`CUDA_VISIBLE_DEVICES=0 ipython experiments/cremi/infer.py -- test_infer --DATA_HOMEDIR <path/to/the/cremi/data/you/downloaded> 
--inherit main_config.yml --update0 infer_config.yml --config.model.model_kwargs.loadfrom RUNS__HOME/model_name/checkpoint.pytorch --config.name_experiment your_infer_experiment_name --config.loaders.infer.loader_config.batch_size 1 --config.export_path <directory-where-to-save-affinities>`

More specific infer-parameters are found in `configs/infer_config.yml`. Some examples:

- Predict only a small part of the data using the parameter `loaders.infer.volume_config.data_slice`
- You may need to adjust the parameter `model.slicing_config.window_size` according to the memory available on your GPU. 

### Post-processing
After predicting the affinities, have a look at the example script `postprocess_affinities.py` showing how to convert the affinities into an instance segmentation using GASP or Mutex Watershed.

