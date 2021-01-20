import warnings
import gc
import numpy as np
from copy import deepcopy

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.parallel.data_parallel import data_parallel

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from inferno.extensions.containers.graph import Identity
from speedrun.log_anywhere import log_image, log_embedding, log_scalar
from segmfriends.utils.various import parse_data_slice
from segmfriends.transform.volume import DownSampleAndCropTensorsInBatch
from segmfriends.utils.various import parse_data_slice

from ..utils.various import auto_crop_tensor_to_shape
from .sparse_affinitiees_loss import MultiLevelSparseAffinityLoss


class LatentMaskLoss(nn.Module):
    def __init__(self, model, apply_checkerboard=False, loss_type="Dice",
                 ignore_label=0,
                 train_glia_mask=False,
                 boundary_label=None,
                 glia_label=None,
                 train_patches_on_glia=False,
                 fix_bug_multiscale_patches=False,
                 defected_label=None,
                 IoU_loss_kwargs=None,
                 sparse_affs_loss_kwargs=None,
                 indx_trained_patchNets=None,
                 model_kwargs=None, devices=(0, 1)):
        super(LatentMaskLoss, self).__init__()
        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

        self.apply_checkerboard = apply_checkerboard
        self.fix_bug_multiscale_patches = fix_bug_multiscale_patches
        self.ignore_label = ignore_label
        self.boundary_label = boundary_label
        self.glia_label = glia_label
        self.defected_label = defected_label
        self.train_glia_mask = train_glia_mask
        self.train_patches_on_glia = train_patches_on_glia
        self.indx_trained_patchNets = indx_trained_patchNets
        self.add_IoU_loss = False
        if IoU_loss_kwargs is not None:
            raise NotImplementedError()
            # self.add_IoU_loss = True
            # from .compute_IoU import IoULoss
            # self.IoU_loss = IoULoss(model, model_kwargs=model_kwargs, devices=devices, **IoU_loss_kwargs)

        self.devices = devices
        # TODO: get rid of kwargs
        self.model_kwargs = model_kwargs
        self.MSE_loss = nn.MSELoss()
        self.smoothL1_loss = nn.SmoothL1Loss()
        # TODO: use nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        self.soresen_loss = SorensenDiceLoss()

        self.model = model

        self.train_sparse_loss = False
        self.sparse_multilevelDiceLoss = None
        if sparse_affs_loss_kwargs is not None:
            self.train_sparse_loss = True
            self.sparse_multilevelDiceLoss = MultiLevelSparseAffinityLoss(model, model_kwargs=model_kwargs,
                                                                          devices=devices,
                                                                          **sparse_affs_loss_kwargs)

    def forward(self, all_predictions, target):
        mdl = self.model

        nb_inputs = mdl.number_multiscale_inputs

        # Plot some patches with the raw:
        if self.model.return_input:
            raw_inputs = all_predictions[-nb_inputs:]
            all_predictions = all_predictions[:-nb_inputs]

        loss = 0

        if self.train_sparse_loss:
            raise NotImplementedError
            loss = loss + self.sparse_multilevelDiceLoss(all_predictions, target)
            # Delete affinities from targets:
            target = [tar[:, :2].int() for tar in target]

        # ----------------------------
        # Loss on patches:
        # ----------------------------
        for mask_dec_indx in range(len(all_predictions)):
            # ----------------------------
            # Initializations:
            # ----------------------------
            mask_dec = self.model.mask_decoders[mask_dec_indx]
            pred = all_predictions[mask_dec_indx]

            gt_segm = target[mask_dec.target_index]

            # Collect options from config:
            mask_shape = mask_dec.mask_shape
            mask_dws_fact = mask_dec.mask_dws_fact
            sample_strides = mask_dec.sample_strides
            pred_dws_fact = mask_dec.pred_dws_fact
            crop_slice_prediction = mask_dec.crop_slice_prediction
            limit_nb_decoded_masks_to = mask_dec.limit_nb_decoded_masks_to

            if crop_slice_prediction is not None:
                precrop_pred_slice = (slice(None), slice(None)) + parse_data_slice(crop_slice_prediction)
                pred = pred[precrop_pred_slice]

            max_random_crop = mask_dec.max_random_crop

            real_shape_mask = tuple(pt * fc for pt, fc in zip(mask_shape, mask_dws_fact))

            full_target_shape = gt_segm.shape[-3:]
            assert all([i <= j for i, j in zip(real_shape_mask, full_target_shape)]), "Real-sized patch is too large!"

            # ----------------------------
            # Deduce crop size of the prediction and select target patches accordingly:
            # ----------------------------
            # TODO: explain better what is going on here
            crop_slice_targets, crop_slice_prediction = get_slicing_crops(pred.shape[2:], full_target_shape,
                                                                          pred_dws_fact, real_shape_mask)
            gt_segm = gt_segm[crop_slice_targets]
            pred = pred[crop_slice_prediction]
            full_target_shape = gt_segm.shape[-3:]

            # # # ----------------------------
            # # # Plot some random patches with associated raw patch:
            # # # ----------------------------
            # if self.model.return_input and mask_dec_indx<5:
            #     # raw = raw_inputs[kwargs["nb_target"]][crop_slice_targets]
            #     # FIXME: raw is not correct for deeper ones
            #     raw = raw_inputs[0][crop_slice_targets]
            #     raw_to_plot, gt_labels_to_plot, gt_masks_to_plot, pred_emb_to_plot = [], [], [], []
            #     for n in range(40):
            #         # Select a random pixel and define sliding-window crop slices:
            #         selected_coord = [np.random.randint(shp) for shp in pred.shape[2:]]
            #         # selected_coord[0] = 4 # For plots, get always 4
            #         full_patch_slice = (slice(None), slice(0,1)) + tuple(
            #             slice(selected_coord[i], selected_coord[i] + real_shape_mask[i]) for i in range(len(selected_coord)))
            #         emb_slice = (slice(None), slice(0,1)) + tuple(slice(selected_coord[i] + int(real_shape_mask[i] / 2),
            #                                                             selected_coord[i] + int(
            #                                                                 real_shape_mask[i] / 2) + 1) for i in
            #                                                       range(len(selected_coord)))
            #         pred_center_coord = [int(selected_coord[i] / pred_dws_fact[i]) for i in range(len(selected_coord))]
            #         emb_slice_pred = (slice(None), slice(None)) + tuple(
            #             slice(pred_center_coord[i], pred_center_coord[i] + 1)
            #             for i in range(len(selected_coord)))
            #
            #         # Collect data for current sliding window:
            #         center_label = gt_segm[emb_slice]
            #         center_label_repeated = center_label.repeat(1, 1, *real_shape_mask)
            #         gt_patch_labels = gt_segm[full_patch_slice]
            #         gt_masks_to_plot.append(gt_patch_labels != center_label_repeated)
            #         gt_labels_to_plot.append(gt_patch_labels)
            #         # ignore_mask_patch = (gt_patch_labels == 0)
            #         pred_emb_to_plot.append(pred[emb_slice_pred])
            #
            #         raw_to_plot.append(raw[full_patch_slice])
            #
            #     # Highlight center pixel:
            #     raw_to_plot = torch.cat(raw_to_plot, dim=0)
            #     center_pixel_coord = (slice(None), 0) + tuple(int(shp / 2) for shp in real_shape_mask)
            #     raw_to_plot[center_pixel_coord] = raw_to_plot.min() - 1.
            #
            #     gt_labels_to_plot = torch.cat(gt_labels_to_plot, dim=0)
            #     gt_masks_to_plot = torch.cat(gt_masks_to_plot, dim=0)
            #     pred_emb_to_plot = torch.cat(pred_emb_to_plot, dim=0)
            #
            #     # Decode embeddings:
            #     ptch_num = kwargs["patchNet_number"]
            #     pred_patch_to_plot = data_parallel(self.model.patch_models[ptch_num], pred_emb_to_plot[:, :, 0, 0, 0], self.devices)
            #
            #     # Downscale and rescale targets:
            #     down_sc_slice = (slice(None), slice(None)) + tuple(
            #         slice(int(dws_fact / 2), None, dws_fact) for dws_fact in mask_dws_fact)
            #     gt_masks_to_plot = torch.nn.functional.interpolate(gt_masks_to_plot[down_sc_slice].float(), scale_factor=tuple(mask_dws_fact))
            #     pred_patch_to_plot = torch.nn.functional.interpolate(pred_patch_to_plot,
            #                                                          scale_factor=tuple(mask_dws_fact))
            #
            #     gt_masks_to_plot = 1. - gt_masks_to_plot
            #     if mask_dws_fact[1] <= 6:
            #         pred_patch_to_plot = 1. - pred_patch_to_plot
            #
            #     log_image("raw_patch_l{}".format(mask_dec_indx), raw_to_plot)
            #     log_image("gt_label_patch_l{}".format(mask_dec_indx), gt_labels_to_plot)
            #     log_image("gt_mask_patch_l{}".format(mask_dec_indx), gt_masks_to_plot)
            #     log_image("pred_patch_l{}".format(mask_dec_indx), pred_patch_to_plot)

            # # ----------------------------
            # # Patch-Loss:
            # # ----------------------------

            # If multiple strides were given, process all of them:
            sample_strides = sample_strides if isinstance(sample_strides[0], list) else [sample_strides]
            if limit_nb_decoded_masks_to is not None:
                limit_nb_decoded_masks_to = limit_nb_decoded_masks_to if isinstance(limit_nb_decoded_masks_to[0],
                                                                                    list) else [
                    limit_nb_decoded_masks_to]
            else:
                limit_nb_decoded_masks_to = [None for _ in sample_strides]

            for nb_stride, smpl_stride, max_nb_masks in zip(range(len(sample_strides)), sample_strides,
                                                            limit_nb_decoded_masks_to):

                # ----------------------------
                # Get some random prediction embeddings:
                # ----------------------------
                prediction_strides = get_prediction_strides(pred_dws_fact, smpl_stride)
                selected_embeddings, crop_slice_pred, nb_selected_masks = extract_patches_torch(pred, (1, 1, 1),
                                                                                                stride=prediction_strides,
                                                                                                max_random_crop=max_random_crop)

                # ----------------------------
                # Collect gt_segm patches and corresponding center labels:
                # ----------------------------
                crop_slice_targets = tuple(slice(sl.start, None) for sl in crop_slice_pred)
                gt_patches, _, _ = extract_patches_torch(gt_segm, real_shape_mask, stride=smpl_stride,
                                                         apply_specific_crop_slice=crop_slice_targets,
                                                         limit_patches_nb_to=nb_selected_masks)
                gt_patches = gt_patches[:, [0]]

                # Make sure to crop some additional border and get the centers correctly:
                # TODO: this can be now easily done by cropping the gt_patches...
                crop_slice_center_labels = (slice(None), slice(None)) + tuple(
                    slice(slc.start + int(sh / 2), slc.stop) for slc, sh in
                    zip(crop_slice_targets[2:], real_shape_mask))
                target_at_patch_center, _, _ = extract_patches_torch(gt_segm, (1, 1, 1), stride=smpl_stride,
                                                                     apply_specific_crop_slice=crop_slice_center_labels,
                                                                     limit_patches_nb_to=nb_selected_masks)
                # Get GT and other masks separately:
                label_at_patch_center = target_at_patch_center[:, [0]]
                mask_at_patch_center = target_at_patch_center[:, [1]]

                # ----------------------------
                # Ignore patches on the boundary or involving ignore-label:
                # ----------------------------
                # Ignore pixels involving ignore-labels:
                ignore_masks = (gt_patches == self.ignore_label)
                valid_patches = (label_at_patch_center != self.ignore_label)

                patch_is_on_boundary = None
                if self.boundary_label is not None:
                    patch_is_on_boundary = (mask_at_patch_center == self.boundary_label).repeat(1, 1, *real_shape_mask)

                # Delete non-valid patches from batch:
                valid_batch_indices = np.argwhere(valid_patches[:, 0, 0, 0, 0].cpu().detach().numpy())[:, 0]
                if max_nb_masks is not None:
                    limit = max_nb_masks[0]
                    if max_nb_masks[1] == 'number':
                        if valid_batch_indices.shape[0] > limit:
                            valid_batch_indices = np.random.choice(valid_batch_indices, limit, replace=False)
                    elif max_nb_masks[1] == 'factor':
                        assert limit <= 1. and limit >= 0.
                        valid_batch_indices = np.random.choice(valid_batch_indices,
                                                               int(limit * valid_batch_indices.shape[0]), replace=False)

                if valid_batch_indices.shape[0] == 0:
                    # Avoid problems if all patches are invalid and
                    # torch complaining that autograd cannot be performed:
                    loss += selected_embeddings.sum() * 0.
                    print("ZERO valid patches at level {}".format(mask_dec_indx))
                    continue

                # ----------------------------
                # Compute the actual (inverted) MeMasks targets: (0 is me, 1 are the others)
                # best targets for Dice loss (usually more me than others)
                # ----------------------------
                center_labels_repeated = label_at_patch_center.repeat(1, 1, *real_shape_mask)
                target_me_masks = gt_patches != center_labels_repeated

                if patch_is_on_boundary is not None:
                    # If on boundary, we make (inverted) me_masks completely 1 (split from everything)
                    target_me_masks = target_me_masks | patch_is_on_boundary

                # Downscaling patches:
                down_sc_slice = (slice(None), slice(None)) + tuple(
                    slice(int(dws_fact / 2), None, dws_fact) for dws_fact in mask_dws_fact)

                # Final targets:
                target_me_masks = target_me_masks[valid_batch_indices].float()[down_sc_slice]
                ignore_masks = ignore_masks[valid_batch_indices][down_sc_slice].byte()

                # Invert MeMasks:
                # best targets for Dice loss are: meMask == 0; others == 1
                # TODO: generalize
                if mask_dws_fact[1] > 6:
                    target_me_masks = 1. - target_me_masks

                assert valid_batch_indices.max() < selected_embeddings.shape[
                    0], "Something went wrong, more target patches were collected than those predicted: {} targets vs {} pred...".format(
                    valid_batch_indices.max(), selected_embeddings.shape[0])
                selected_embeddings = selected_embeddings[valid_batch_indices]
                selected_embeddings = selected_embeddings[:, :, 0, 0, 0]

                # ----------------------------
                # Decode the actual predicted using the decoder models:
                # ----------------------------
                decoded_masks = data_parallel(mask_dec, selected_embeddings, self.devices)
                # print(expanded_patches.shape)
                assert decoded_masks.shape[1] == 1, "MaskDecoder should output only single-channel masks!"

                # Some logs:
                if nb_stride == 0:
                    log_image("ptc_trg_l{}".format(mask_dec_indx), target_me_masks)
                    log_image("ptc_pred_l{}".format(mask_dec_indx), decoded_masks)
                    # log_image("ptc_ign_l{}".format(nb_patch_net), patch_ignore_masks)
                    log_scalar("avg_targets_l{}".format(mask_dec_indx), target_me_masks.float().mean())

                # ----------------------------
                # Apply ignore mask and compute loss:
                # ----------------------------
                valid_masks = 1. - ignore_masks.float()
                decoded_masks = decoded_masks * valid_masks
                target_me_masks = target_me_masks * valid_masks
                with warnings.catch_warnings(record=True) as w:
                    reconstruction_loss = data_parallel(self.loss, (decoded_masks, target_me_masks.float()),
                                                        self.devices).mean()

                loss = loss + reconstruction_loss
                if nb_stride == 0:
                    log_scalar("loss_l{}".format(mask_dec_indx), reconstruction_loss)
                    log_scalar("nb_patches_l{}".format(mask_dec_indx), decoded_masks.shape[0])

        gc.collect()
        return loss


def get_slicing_crops(pred_shape, target_shape, pred_ds_factor, real_patch_shape):
    """
    In few words, this function tries to deduce how the target and predicted tensors should be cropped, so that
    if we extract patches from both of them, these patches are consistent.

    Let's see some examples:

    1) If the target and prediction tensors have the same shape:
            then the prediction tensor should be partially cropped, because the embedding in the top left corner
            will need some extra context from the target tensor in order to be trained properly.

    2) However, in some cases the prediction tensor will be much smaller than the target (for example because some crops
            were performed inside the UNet model), so we will further crop the target so that they match.

    :return: Two tuples containing the crop slices to be applied to the target and prediction tensors, respectively.
    """
    # Compute new left crops:
    # (we do not care about the right crops, because anyway the extra patches are
    # ignored with the option `limit_patches_to`)
    upscaled_pred_shape = [sh * fctr for sh, fctr in zip(pred_shape, pred_ds_factor)]

    shape_diff = [orig - trg for orig, trg in zip(target_shape, upscaled_pred_shape)]
    assert all([diff >= 0 for diff in shape_diff]), "Prediction should be smaller or equal to the targets!"
    assert all([diff % 2 == 0 for diff in shape_diff])
    padding = [int(diff / 2) for diff in shape_diff]

    crop_slice_targets = [slice(None), slice(None)]
    crop_slice_prediction = [slice(None), slice(None)]
    import math
    for dim, pad in enumerate(padding):
        # Consider the patch-padding:
        real_pad = pad - int(real_patch_shape[dim] / 2)
        if real_pad > 0:
            # We should crop targets
            crop_slice_targets.append(slice(real_pad, -real_pad))
            crop_slice_prediction.append(slice(None))
        elif real_pad < 0:
            # We should crop prediction:
            # (use floor to round up, since pad is negative)
            crop_slice_prediction.append(
                slice(-math.floor(real_pad / pred_ds_factor[dim]), math.floor(real_pad / pred_ds_factor[dim])))
            crop_slice_targets.append(slice(None))
        else:
            # No need to crop:
            crop_slice_targets.append(slice(None))
            crop_slice_prediction.append(slice(None))

    return tuple(crop_slice_targets), tuple(crop_slice_prediction)


def get_prediction_strides(pred_ds_factor, strides):
    # Compute updated strides:
    assert all(strd % pred_fctr == 0 for strd, pred_fctr in
               zip(strides, pred_ds_factor)), "Stride {} should be divisible by downscaling factor {}".format(strides,
                                                                                                              pred_ds_factor)
    pred_strides = tuple(int(strd / pred_fctr) for strd, pred_fctr in zip(strides, pred_ds_factor))

    return pred_strides


def extract_patches_torch(tensor, shape, stride,
                          precrop_tensor=None,
                          max_random_crop=None,
                          apply_specific_crop_slice=None,
                          limit_patches_nb_to=None,
                          reshape_to_batch_dim=True):
    """
    :param tensor: PyTorch tensor from which to extract patches
    :param shape: Shape of the extracted patches
    :param stride: Stride of the extracted patches

    :param precrop_tensor: How much to precrop the tensor (list of length tensor.dim()-2)
                            Example: [(0,0), (2,4), (1,4)]

    :param max_random_crop: How much to randomly crop the tensor (to create some variability in the strided sampling).
                            Same format as `precrop_tensor`

    :param apply_specific_crop_slice:
                            This is the second argument that is returned by the function: it represents the
                            actual crop that was performed (including a possible random crop that was applied).
                            If you apply this function to multiple tensors and you want to get consistent results,
                            then the second time call the function passing the output_crop from the first call using
                            this argument.
                            If this is passed, then `precrop_tensor` and `max_random_crop` should be None.
    :param limit_patches_nb_to:
                            This is the third argument that is returned by the function and it represents how many
                            patches were extracted along each dimension of the original tensor.
                            Use this argument to make sure that if you apply the function to multiple tensors,
                            you get the same number of patches for all of them.

    :param reshape_to_batch_dim:

    :return: See description of `apply_specific_crop_slice` and `limit_patches_nb_to`
    """
    assert tensor.dim() == 4 or tensor.dim() == 5
    dim = tensor.dim() - 2
    assert len(shape) == dim and len(stride) == dim

    if apply_specific_crop_slice is not None:
        assert max_random_crop is None and precrop_tensor is None

    if precrop_tensor is not None:
        assert len(precrop_tensor) == dim
        assert all([isinstance(sl, (tuple, list)) for sl in precrop_tensor]) and all(
            [len(sl) == 2 for sl in precrop_tensor])
    else:
        precrop_tensor = [(0, 0) for _ in range(dim)]

    max_random_crop = [0 for _ in range(dim)] if max_random_crop is None else deepcopy(max_random_crop)
    assert len(max_random_crop) == dim
    if isinstance(max_random_crop, tuple):
        max_random_crop = list(max_random_crop)
    for d in range(dim):
        max = tensor.shape[2 + d] - precrop_tensor[d][0] - precrop_tensor[d][1] - shape[d]
        if max_random_crop[d] > max:
            max_random_crop[d] = max

    if limit_patches_nb_to is not None:
        assert len(limit_patches_nb_to) == dim

    # Pick a random crop:
    if apply_specific_crop_slice is None:
        rnd_crop = [np.random.randint(max_offs + 1) for max_offs in max_random_crop]
        apply_specific_crop_slice = (slice(None), slice(None)) + tuple(
            slice(precrop[0] + off, full_shp - precrop[1]) for off, precrop, full_shp in
            zip(rnd_crop, precrop_tensor, tensor.shape[2:]))

    # Unfold it:
    tensor = tensor[apply_specific_crop_slice]
    N, C = tensor.shape[:2]
    for d in range(dim):
        tensor = tensor.unfold(d + 2, size=shape[d], step=stride[d])

    # Reshape:
    nb_patches = tensor.shape[2:2 + len(shape)]

    # Along each dimension, we make sure to keep only a specific number of patches (not more):
    # This assures compatibility with other patches already extracted from other tensors.
    if limit_patches_nb_to is not None:
        actual_limits = tuple(lim if lim < nb else nb for nb, lim in zip(nb_patches, limit_patches_nb_to))
        valid_patch_slice = (slice(None), slice(None)) + tuple(slice(None, lim) for lim in actual_limits)
        tensor = tensor[valid_patch_slice]
        nb_patches = actual_limits

    # Reshape
    if reshape_to_batch_dim:
        tensor = tensor.contiguous().view(N, C, -1, *shape)
        tensor = tensor.permute(0, 2, 1, *range(3, 3 + dim)).contiguous().view(-1, C, *shape)

    return tensor, apply_specific_crop_slice, nb_patches
