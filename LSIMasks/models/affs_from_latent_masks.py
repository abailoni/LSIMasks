import torch

from segmfriends.utils import parse_data_slice
import numpy as np

from .latent_mask_model import LatentMaskModel
from ..losses.latent_mask_loss import extract_patches_torch

class AffinitiesFromLatentMasks(LatentMaskModel):
    """
    Subclass of LatentMaskModel.
    To be used only during inference, once LatentMaskModel has been trained.

    It predicts affinities from latent instance masks, given certain offsets.
    """
    def __init__(self, offsets, num_IoU_workers=1,
                 pre_crop_pred=None,
                 patch_size_per_offset=None,
                 slicing_config=None,
                 affinity_mode="classic",
                 temperature_parameter=1.,
                 patch_threshold=0.5,
                 T_norm_type=None,
                 *super_args, **super_kwargs):
        super(AffinitiesFromLatentMasks, self).__init__(*super_args, **super_kwargs)

        self.ptch_kwargs = [kwargs for i, kwargs in
                            enumerate(self.collected_patchNet_kwargs) if
                            i in self.trained_patchNets]

        assert 'window_size' in slicing_config and slicing_config is not None
        slicing_config['stride'] = slicing_config['window_size']
        self.slicing_config = slicing_config

        assert all(isinstance(off, (tuple, list)) for off in offsets)
        self.offsets = offsets

        # TODO: Assert
        # assert affinity_mode in ["classic", "probabilistic", "probNoThresh", "fullPatches"]
        self.temperature_parameter = temperature_parameter
        self.T_norm_type = T_norm_type
        self.affinity_mode = affinity_mode
        self.patch_threshold = patch_threshold
        if patch_size_per_offset is None:
            patch_size_per_offset = [None for _ in range(len(offsets))]
        else:
            assert len(patch_size_per_offset) == len(offsets)
        self.patch_size_per_offset = patch_size_per_offset
        self.num_IoU_workers = num_IoU_workers
        if pre_crop_pred is not None:
            assert isinstance(pre_crop_pred, str)
            pre_crop_pred = (slice(None), slice(None)) + parse_data_slice(pre_crop_pred)
        self.pre_crop_pred = pre_crop_pred

    def forward(self, *inputs):
        if self.affinity_mode == "classic":
            return self.forward_affinities(*inputs)
        else:
            raise ValueError

    def forward_affinities(self, *inputs):
        with torch.no_grad():
            all_predictions = super(AffinitiesFromLatentMasks, self).forward(*inputs)

        def make_sliding_windows(volume_shape, window_size, stride, downsampling_ratio=None):
            from inferno.io.volumetric import volumetric_utils as vu
            assert isinstance(volume_shape, tuple)
            ndim = len(volume_shape)
            if downsampling_ratio is None:
                downsampling_ratio = [1] * ndim
            elif isinstance(downsampling_ratio, int):
                downsampling_ratio = [downsampling_ratio] * ndim
            elif isinstance(downsampling_ratio, (list, tuple)):
                # assert_(len(downsampling_ratio) == ndim, exception_type=ShapeError)
                downsampling_ratio = list(downsampling_ratio)
            else:
                raise NotImplementedError

            return list(vu.slidingwindowslices(shape=list(volume_shape),
                                               ds=downsampling_ratio,
                                               window_size=window_size,
                                               strides=stride,
                                               shuffle=False,
                                               add_overhanging=True))

        del inputs
        # torch.cuda.empty_cache()

        total_nb_patchnets = 0
        for _, off_specs in enumerate(self.offsets):
            new_max = np.array(off_specs[1]).max()
            total_nb_patchnets = new_max if new_max > total_nb_patchnets else total_nb_patchnets
        all_predictions = all_predictions[:total_nb_patchnets+1]
        patch_nets = range(total_nb_patchnets+1)
        # TODO: generalize to multiscale outputs
        first_shape = all_predictions[0].shape
        for pred in all_predictions[1:]:
            assert first_shape == pred.shape

        if self.pre_crop_pred is not None:
            all_predictions = [pred[self.pre_crop_pred] for pred in all_predictions]

        # TODO: generalize
        assert all(shp % wdw_shp == 0 for wdw_shp, shp in zip(self.slicing_config["window_size"], all_predictions[0].shape[2:])), \
            "The slicing window size {} should be an exact multiple of the prediction shape {}".format(self.slicing_config["window_size"],
                                                                                                       all_predictions[0].shape[2:])

        # Initialize stuff:
        device = all_predictions[0].get_device()
        sliding_windows = make_sliding_windows(all_predictions[0].shape[2:], **self.slicing_config)

        # Get padding of each patch: it will be useful later to crop/pad the predictions
        patch_padding = {}
        for nb_patch_net in patch_nets:
            kwargs = self.ptch_kwargs[nb_patch_net]
            patch_padding[nb_patch_net] = [int(shp/2)*dws for shp, dws in zip(kwargs["patch_size"], kwargs["patch_dws_fact"])]

        # Create array with output affinities:
        out_affinities_shape = (len(self.offsets),) + all_predictions[0].shape[2:]
        out_affinities = torch.zeros(out_affinities_shape).cuda(device)
        mask_affinities = torch.zeros(out_affinities_shape).cuda(device)

        # Predict all patches in a sliding window style:
        for i, current_slice in enumerate(sliding_windows):
            for pred, nb_patch_net in zip(all_predictions, patch_nets):
                assert pred.shape[0] == 1, "Only batch == 1 is supported atm"

                kwargs = self.ptch_kwargs[nb_patch_net]

                # Collect options from config:
                patch_shape = kwargs.get("patch_size")
                assert all(i % 2 == 1 for i in patch_shape), "Patch should be odd"
                patch_dws_fact = kwargs.get("patch_dws_fact", [1, 1, 1])

                full_slice = (slice(None), slice(None),) + current_slice
                # Now this is simply doing a reshape...
                emb_vectors, _, nb_patches = extract_patches_torch(pred[full_slice], shape=(1, 1, 1), stride=(1,1,1))
                patches = self.models[-1].patch_models[nb_patch_net](emb_vectors[:, :, 0, 0, 0])

                patches = patches.view(*nb_patches, *patch_shape)

                # Make sure to have me-masks:
                if patch_dws_fact[1] <= 6:
                    patches = 1. - patches

                # Compute affinities:
                for nb_off, off_specs in enumerate(self.offsets):
                    if nb_patch_net in off_specs[1]:
                        assert all(off%dws == 0 for off, dws in zip(off_specs[0], patch_dws_fact))
                        aff_coord = [int(shp/2)+int(off/dws)  for off, dws, shp in zip(off_specs[0], patch_dws_fact, patches.shape[-3:])]

                        # Get the requested affinity:
                        current_affinities = patches[:,:,:,aff_coord[0],aff_coord[1],aff_coord[2]]

                        out_affinities[nb_off][current_slice] = out_affinities[nb_off][current_slice] + current_affinities
                        mask_affinities[nb_off][current_slice] = mask_affinities[nb_off][current_slice] + 1

        # Normalize:
        valid_affs = mask_affinities > 0.
        out_affinities[valid_affs] = out_affinities[valid_affs] / mask_affinities[valid_affs]

        return out_affinities.unsqueeze(0)
