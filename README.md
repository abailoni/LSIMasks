# LSIMasks
Proposal-free instance segmentation from Latent Single-Instance Masks

### Installation (on linux)
If you plan to use the code to train your model, then you will need to install some extra packages:

- Clone the repository: `git clone https://github.com/abailoni/LSIMasks.git`
- `cd LSIMasks`
- `chmod +x ./install_dependencies.sh`
- To install the dependencies, you will need [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Install the dependencies and the package by running `./install_dependencies.sh`. While the script is running, you will need to confirm twice.
- The script will create a new conda environment called `LSIMasks` including all you need

<!--
### Training your model  
- Download the training data from [here](https://heibox.uni-heidelberg.de/d/e182f3807b0c4761a999/)
- Start the training script with:
`CUDA_VISIBLE_DEVICES=0 ipython experiments/cremi/train_model.py -- yourExperimentName --DATA_HOMEDIR path/to/the/cremi/data/you/downloaded --inherit v1_main.yml  --config.loaders.general.loader_config.batch_size 1 --config.trainer.optimizer.Adam.lr 1e-4 `
- You find the main configuration file in `experiments/cremi/configs/v1_main.yml`   
-->

### Starting the training from scratch:
To start the training, run the following command:
```
CUDA_VISIBLE_DEVICES=0 ipython experiments/cremi/train_model.py -- <yourExperimentName> --DATA_HOMEDIR <path/to/the/cremi/data/you/downloaded> --inherit v1_main.yml
```
(the one just given is a single command: for readability it was split into multiple lines)


### Visualizing the training results in tensorboard
Go to the experiment folder (by default placed in the `experiments/cremi/runs` folder) and then start tensorboard:

`tensorboard --logdir=./ --bind_all`

For this, you will need to install tensorflow, with `pip install --upgrade tensorflow`




### Visualize your predictions in tensorboard
Coming soon
