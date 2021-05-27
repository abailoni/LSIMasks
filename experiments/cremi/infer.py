from LSIMasks.utils.various import change_paths_config_file

from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin, FirelightLogger, AffinityInferenceMixin
from speedrun.log_anywhere import register_logger, log_image, log_scalar
from speedrun.py_utils import create_instance

from copy import deepcopy

import os
import torch
import numpy as np

from segmfriends.utils.config_utils import recursive_dict_update
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils.various import writeHDF5

import sys

from segmfriends.datasets.cremi import get_cremi_loader

torch.backends.cudnn.benchmark = True

class BaseCremiExperiment(BaseExperiment, AffinityInferenceMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)


        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        # register_logger(FirelightLogger, "image")
        register_logger(self, 'scalars')


        # offsets = self.get_boundary_offsets()
        # self.set('global/offsets', offsets)
        # self.set('loaders/general/volume_config/segmentation/affinity_config/offsets', offsets)

        if "model_class" in self.get('model'):
            self.model_class = self.get('model/model_class')
        else:
            self.model_class = list(self.get('model').keys())[0]

        if self.get("loaders/general/master_config/downscale_and_crop") is not None:
            master_conf = self.get("loaders/general/master_config")

            ds_config = self.get("loaders/general/master_config/downscale_and_crop")
            nb_tensors = len(ds_config)
            nb_inputs = self.get("model/model_kwargs/number_multiscale_inputs")
            nb_targets = nb_tensors - nb_inputs
            if "affinity_config" in master_conf:
                affs_config = deepcopy(master_conf.get("affinity_config", {}))
                if affs_config.get("use_dynamic_offsets", False):
                    raise NotImplementedError
                    nb_targets = 1
                else:
                    affs_config.pop("global", None)
                    nb_targets = len(affs_config)
            self.set("trainer/num_targets", nb_targets)
        else:
            self.set("trainer/num_targets", 1)

        self.set_devices()


    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config

        if "model_kwargs" in model_config:
            assert "model_class" in model_config
            model_class = model_config["model_class"]
            model_kwargs = model_config["model_kwargs"]
            model_path = model_kwargs.pop('loadfrom', None)
            stacked_models_path = model_kwargs.pop('load_stacked_models_from', None)
            model_config = {model_class: model_kwargs}
            model = create_instance(model_config, self.MODEL_LOCATIONS)
        else:
            model_path = model_config[next(iter(model_config.keys()))].pop('loadfrom', None)
            stacked_models_path = model_config[next(iter(model_config.keys()))].pop('load_stacked_models_from', None)
            model = create_instance(model_config, self.MODEL_LOCATIONS)

        if model_path is not None:
            print(f"loading model from {model_path}")
            loaded_model = torch.load(model_path)["_model"]
            if self.get("legacy_experiment", False):
                state_dict = loaded_model.models[0].state_dict()
            else:
                state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict)

        # if stacked_models_path is not None:
        #     for mdl in stacked_models_path:
        #         stck_mdl_path = stacked_models_path[mdl]
        #         print("loading stacked model {} from {}".format(mdl, stck_mdl_path))
        #         mdl_state_dict = torch.load(stck_mdl_path)["_model"].models[mdl].state_dict()
        #         model.models[mdl].load_state_dict(mdl_state_dict)


        return model

    def set_devices(self):
        # --------- In case of multiple GPUs: ------------
        # n_gpus = torch.cuda.device_count()
        # gpu_list = range(n_gpus)
        # self.set("gpu_list", gpu_list)
        # self.trainer.cuda(gpu_list)

        # --------- For one GPU only: ------------
        self.set("gpu_list", [0])
        self.trainer.cuda([0])

    def inferno_build_criterion(self):
        print("Building criterion")
        loss_kwargs = self.get("trainer/criterion/kwargs", {})
        loss_name = self.get("trainer/criterion/loss_name", "LSIMasks.losses.latent_mask_loss.LatentMaskLoss")
        loss_config = {loss_name: loss_kwargs}
        loss_config[loss_name]['model'] = self.model
        if "model_kwargs" in self.get('model'):
            model_kwargs = self.get('model/model_kwargs')
        else:
            model_kwargs = self.get('model/{}'.format(self.model_class))
        loss_config[loss_name]['model_kwargs'] = model_kwargs
        loss_config[loss_name]['devices'] = tuple(self.get("gpu_list"))

        loss = create_instance(loss_config, self.CRITERION_LOCATIONS)
        self._trainer.build_criterion(loss)
        self._trainer.build_validation_criterion(loss)

    def build_train_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)

    def build_val_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)

    def build_infer_loader(self):
        kwargs = deepcopy(self.get('loaders/infer'))
        loader = get_cremi_loader(kwargs)
        return loader

    def save_infer_output(self, output):
        print("Saving....")
        assert self.get("export_path") is not None
        dir_path = os.path.join(self.get("export_path"),
                                self.get("name_experiment", default="generic_experiment"))
        check_dir_and_create(dir_path)
        filename = os.path.join(dir_path, "predictions_sample_{}.h5".format(self.get("loaders/infer/name")))
        print("Writing to ", self.get("inner_path_output", 'data'))
        writeHDF5(output.astype(np.float16), filename, self.get("inner_path_output", 'data'))
        print("Saved to ", filename)

        # Dump configuration to export folder:
        self.dump_configuration(os.path.join(dir_path, "prediction_config.yml"))

if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    # Read path of the dataset:
    assert "--DATA_HOMEDIR" in sys.argv, "Script parameter DATA_HOMEDIR is missing"
    idx = sys.argv.index('--DATA_HOMEDIR')
    DATA_HOMEDIR = sys.argv.pop(idx+1)
    DATA_HOMEDIR = DATA_HOMEDIR if DATA_HOMEDIR.endswith('/') else DATA_HOMEDIR + '/'
    sys.argv.pop(idx)

    # Update HCI_HOME paths:
    for i, key in enumerate(sys.argv):
        if "DATA_HOMEDIR" in sys.argv[i]:
            sys.argv[i] = sys.argv[i].replace("DATA_HOMEDIR/", DATA_HOMEDIR)


    # Update RUNS paths:
    for i, key in enumerate(sys.argv):
        if "RUNS__HOME" in sys.argv[i]:
            sys.argv[i] = sys.argv[i].replace("RUNS__HOME", experiments_path)


    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]), dataset_main_dir=DATA_HOMEDIR)
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]), dataset_main_dir=DATA_HOMEDIR)
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = change_paths_config_file(os.path.join(config_path, sys.argv[ind]), dataset_main_dir=DATA_HOMEDIR)
            i += 1
        else:
            break
    cls = BaseCremiExperiment
    cls().infer()
