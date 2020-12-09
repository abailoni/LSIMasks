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
                 model_kwargs=None, devices=(0,1)):
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

        # TODO: hack to adapt to stacked model:
        self.downscale_and_crop_targets = {}
        if hasattr(self.model, "collected_patchNet_kwargs"):
            self.model_kwargs["patchNet_kwargs"] = [kwargs for i, kwargs in enumerate(self.model.collected_patchNet_kwargs) if i in self.model.trained_patchNets]

            # FIXME: generalize to the non-stacked model (there I also have global in the keys...)
            for nb, kwargs in enumerate(self.model_kwargs["patchNet_kwargs"]):
                if "downscale_and_crop_target" in kwargs:
                    # FIXME: adapt to new segmfriend function
                    raise NotImplementedError
                    self.downscale_and_crop_targets[nb] = DownsampleAndCrop3D(**kwargs["downscale_and_crop_target"])



    def forward(self, all_predictions, target):
        mdl_kwargs = self.model_kwargs
        ptch_kwargs = mdl_kwargs["patchNet_kwargs"]

        nb_inputs = mdl_kwargs.get("number_multiscale_inputs")

        # print([(pred.shape[-3], pred.shape[-2], pred.shape[-1]) for pred in all_predictions])
        # print([(targ.shape[-3], targ.shape[-2], targ.shape[-1]) for targ in target])

        # Plot some patches with the raw:
        if self.model.return_input:
            raw_inputs = all_predictions[-nb_inputs:]
            all_predictions = all_predictions[:-nb_inputs]

        loss = 0

        # # ----------------------------
        # # Predict glia mask:
        # # ----------------------------
        if self.train_glia_mask:
            assert self.glia_label is not None

            frg_kwargs = self.model.foreground_prediction_kwargs
            if frg_kwargs is None:
                # Legacy:
                nb_glia_preds = 1
                nb_glia_targets = [0]
            else:
                nb_glia_preds = len(frg_kwargs)
                nb_glia_targets = [frg_kwargs[dpth]["nb_target"] for dpth in frg_kwargs]

            all_glia_preds = all_predictions[-nb_glia_preds:]
            all_predictions = all_predictions[:-nb_glia_preds]
            loss_glia = 0
            for counter, glia_pred, nb_tar in zip(range(len(all_glia_preds)), all_glia_preds, nb_glia_targets):
                glia_target = (target[nb_tar][:, [1]] == self.glia_label).float()
                valid_mask = (target[nb_tar][:, [0]] != self.ignore_label).float()

                glia_target = auto_crop_tensor_to_shape(glia_target, glia_pred.shape)
                valid_mask = auto_crop_tensor_to_shape(valid_mask, glia_pred.shape)
                glia_pred = glia_pred * valid_mask
                glia_target = glia_target * valid_mask
                with warnings.catch_warnings(record=True) as w:
                    loss_glia_cur = data_parallel(self.loss, (glia_pred, glia_target), self.devices).mean()
                loss_glia = loss_glia + loss_glia_cur
                log_image("glia_target_d{}".format(counter), glia_target)
                log_image("glia_pred_d{}".format(counter), glia_pred)
            loss = loss + loss_glia
            log_scalar("loss_glia", loss_glia)
        else:
            glia_pred = all_predictions.pop(-1)

        if self.train_sparse_loss:
            loss = loss + self.sparse_multilevelDiceLoss(all_predictions, target)
            # Delete affinities from targets:
            target = [tar[:, :2].int() for tar in target]

        # IoU loss:
        if self.add_IoU_loss:
            assert self.boundary_label is None, "Not implemented"
            assert self.indx_trained_patchNets is None
            loss = loss + self.IoU_loss(all_predictions, target)

        if self.indx_trained_patchNets is None:
            nb_preds = len(all_predictions)
            assert len(ptch_kwargs) == nb_preds
            indx_trained_patchNets = zip(range(nb_preds), range(nb_preds))
        else:
            indx_trained_patchNets = self.indx_trained_patchNets

        # ----------------------------
        # Loss on patches:
        # ----------------------------
        for nb_patch_net, nb_pr in indx_trained_patchNets:
            # ----------------------------
            # Initializations:
            # ----------------------------
            pred = all_predictions[nb_pr]
            kwargs = ptch_kwargs[nb_patch_net]
            if isinstance(target, (list, tuple)):
                assert "nb_target" in kwargs, "Multiple targets passed. Target should be specified"
                gt_segm = target[kwargs["nb_target"]]
            else:
                gt_segm = target

            # Collect options from config:
            patch_shape_input = kwargs.get("patch_size")
            assert all(i % 2 ==  1 for i in patch_shape_input), "Patch should be odd"
            patch_dws_fact = kwargs.get("patch_dws_fact", [1,1,1])
            stride = tuple(kwargs.get("patch_stride", [1,1,1]))
            pred_dws_fact = kwargs.get("pred_dws_fact", [1,1,1])
            # print(nb_patch_net, patch_dws_fact, pred_dws_fact)
            precrop_pred = kwargs.get("precrop_pred", None)
            limit_nb_patches = kwargs.get("limit_nb_patches", None)
            from segmfriends.utils.various import parse_data_slice
            if precrop_pred is not None:
                precrop_pred_slice = (slice(None), slice(None)) + parse_data_slice(precrop_pred)
                pred = pred[precrop_pred_slice]

            central_shape = tuple(kwargs.get("central_shape", [1,3,3]))
            max_random_crop = tuple(kwargs.get("max_random_crop", [0,5,5]))
            if self.fix_bug_multiscale_patches:
                real_patch_shape = tuple(pt * fc - fc + 1 for pt, fc in zip(patch_shape_input, patch_dws_fact))
            else:
                real_patch_shape = tuple(pt*fc for pt, fc in zip(patch_shape_input, patch_dws_fact))

            full_target_shape = gt_segm.shape[-3:]
            assert all([i <= j for i, j in zip(real_patch_shape, full_target_shape)]), "Real-sized patch is too large!"

            # ----------------------------
            # Deduce crop size of the prediction and select target patches accordingly:
            # ----------------------------
            # print(pred.shape, full_target_shape, pred_dws_fact, real_patch_shape)
            crop_slice_targets, crop_slice_prediction = get_slicing_crops(pred.shape[2:], full_target_shape, pred_dws_fact, real_patch_shape)
            # print(crop_slice_prediction, crop_slice_targets, nb_patch_net)
            gt_segm = gt_segm[crop_slice_targets]
            pred = pred[crop_slice_prediction]
            full_target_shape = gt_segm.shape[-3:]

            # # ----------------------------
            # # Plot some random patches with associated raw patch:
            # # ----------------------------
            if self.model.return_input and nb_patch_net<5:
                # raw = raw_inputs[kwargs["nb_target"]][crop_slice_targets]
                # FIXME: raw is not correct for deeper ones
                raw = raw_inputs[0][crop_slice_targets]
                raw_to_plot, gt_labels_to_plot, gt_masks_to_plot, pred_emb_to_plot = [], [], [], []
                for n in range(40):
                    # Select a random pixel and define sliding-window crop slices:
                    selected_coord = [np.random.randint(shp) for shp in pred.shape[2:]]
                    # selected_coord[0] = 4 # For plots, get always 4
                    full_patch_slice = (slice(None), slice(0,1)) + tuple(
                        slice(selected_coord[i], selected_coord[i] + real_patch_shape[i]) for i in range(len(selected_coord)))
                    emb_slice = (slice(None), slice(0,1)) + tuple(slice(selected_coord[i] + int(real_patch_shape[i] / 2),
                                                                        selected_coord[i] + int(
                                                                            real_patch_shape[i] / 2) + 1) for i in
                                                                  range(len(selected_coord)))
                    pred_center_coord = [int(selected_coord[i] / pred_dws_fact[i]) for i in range(len(selected_coord))]
                    emb_slice_pred = (slice(None), slice(None)) + tuple(
                        slice(pred_center_coord[i], pred_center_coord[i] + 1)
                        for i in range(len(selected_coord)))

                    # Collect data for current sliding window:
                    center_label = gt_segm[emb_slice]
                    center_label_repeated = center_label.repeat(1, 1, *real_patch_shape)
                    gt_patch_labels = gt_segm[full_patch_slice]
                    gt_masks_to_plot.append(gt_patch_labels != center_label_repeated)
                    gt_labels_to_plot.append(gt_patch_labels)
                    # ignore_mask_patch = (gt_patch_labels == 0)
                    pred_emb_to_plot.append(pred[emb_slice_pred])

                    raw_to_plot.append(raw[full_patch_slice])

                # Highlight center pixel:
                raw_to_plot = torch.cat(raw_to_plot, dim=0)
                center_pixel_coord = (slice(None), 0) + tuple(int(shp / 2) for shp in real_patch_shape)
                raw_to_plot[center_pixel_coord] = raw_to_plot.min() - 1.

                gt_labels_to_plot = torch.cat(gt_labels_to_plot, dim=0)
                gt_masks_to_plot = torch.cat(gt_masks_to_plot, dim=0)
                pred_emb_to_plot = torch.cat(pred_emb_to_plot, dim=0)

                # Decode embeddings:
                ptch_num = kwargs["patchNet_number"]
                pred_patch_to_plot = data_parallel(self.model.patch_models[ptch_num], pred_emb_to_plot[:, :, 0, 0, 0], self.devices)

                # Downscale and rescale targets:
                down_sc_slice = (slice(None), slice(None)) + tuple(
                    slice(int(dws_fact / 2), None, dws_fact) for dws_fact in patch_dws_fact)
                gt_masks_to_plot = torch.nn.functional.interpolate(gt_masks_to_plot[down_sc_slice].float(), scale_factor=tuple(patch_dws_fact))
                pred_patch_to_plot = torch.nn.functional.interpolate(pred_patch_to_plot,
                                                                     scale_factor=tuple(patch_dws_fact))

                gt_masks_to_plot = 1. - gt_masks_to_plot
                if patch_dws_fact[1] <= 6:
                    pred_patch_to_plot = 1. - pred_patch_to_plot

                log_image("raw_patch_l{}".format(nb_patch_net), raw_to_plot)
                log_image("gt_label_patch_l{}".format(nb_patch_net), gt_labels_to_plot)
                log_image("gt_mask_patch_l{}".format(nb_patch_net), gt_masks_to_plot)
                log_image("pred_patch_l{}".format(nb_patch_net), pred_patch_to_plot)


            # # ----------------------------
            # # Patch-Loss:
            # # ----------------------------
            if kwargs.get("skip_standard_patch_loss", False):
                continue

            # If multiple strides were given, process all of them:
            all_strides = stride if isinstance(stride[0], list) else [stride]
            if limit_nb_patches is not None:
                all_limit_nb_patches = limit_nb_patches if isinstance(limit_nb_patches[0], list) else [limit_nb_patches]
            else:
                all_limit_nb_patches = [None for _ in all_strides]

            for nb_stride, stride, limit_nb_patches in zip(range(len(all_strides)), all_strides, all_limit_nb_patches):

                # ----------------------------
                # Get some random prediction embeddings:
                # ----------------------------
                pred_strides = get_prediction_strides(pred_dws_fact, stride)
                pred_patches, crop_slice_pred, nb_patches = extract_patches_torch(pred, (1, 1, 1), stride=pred_strides,
                                                                                  max_random_crop=max_random_crop)


                # Try to get some raw patches:
                # TODO: the factor is simply the level in the UNet
                # get_slicing_crops(pred.shape[2:], full_target_shape, [1,1,1], real_patch_shape)

                # ----------------------------
                # Collect gt_segm patches and corresponding center labels:
                # ----------------------------
                crop_slice_targets = tuple(slice(sl.start, None) for sl in crop_slice_pred)
                gt_patches, _, _ = extract_patches_torch(gt_segm, real_patch_shape, stride=stride,
                                                         crop_slice=crop_slice_targets, limit_patches_to=nb_patches)
                gt_patches = gt_patches[:, [0]]

                # Make sure to crop some additional border and get the centers correctly:
                # TODO: this can be now easily done by cropping the gt_patches...
                crop_slice_center_labels = (slice(None), slice(None)) + tuple(slice(slc.start+int(sh/2), slc.stop) for slc, sh in zip(crop_slice_targets[2:], real_patch_shape))
                target_at_patch_center, _, _ = extract_patches_torch(gt_segm, (1, 1, 1), stride=stride,
                                                                     crop_slice=crop_slice_center_labels,
                                                                     limit_patches_to=nb_patches)
                # Get GT and other masks separately:
                label_at_patch_center = target_at_patch_center[:,[0]]
                mask_at_patch_center = target_at_patch_center[:,[1]]

                # ----------------------------
                # Ignore patches on the boundary or involving ignore-label:
                # ----------------------------
                # Ignore pixels involving ignore-labels:
                ignore_masks = (gt_patches == self.ignore_label)
                valid_patches = (label_at_patch_center != self.ignore_label)

                assert self.boundary_label is not None, "Old boundary method is deprecated"
                # # Exclude a patch from training if the central region contains more than one gt label
                # # (i.e. it is really close to a boundary):
                # central_crop = (slice(None), slice(None)) + convert_central_shape_to_crop_slice(gt_patches.shape[-3:], central_shape)
                # mean_central_crop_labels = gt_patches[central_crop].mean(dim=-1, keepdim=True) \
                #     .mean(dim=-2, keepdim=True) \
                #     .mean(dim=-3, keepdim=True)
                #
                # valid_patches = valid_patches & (mean_central_crop_labels == center_labels)
                # is_on_boundary_mask = None
                patch_is_on_boundary = (mask_at_patch_center == self.boundary_label).repeat(1, 1, *real_patch_shape)

                # Ignore patches that represent a glia:
                if not self.train_patches_on_glia:
                    assert self.glia_label is not None
                    # print("Glia: ", (mask_at_patch_center != self.glia_label).min())
                    valid_patches = valid_patches & (mask_at_patch_center != self.glia_label)

                # Delete redundant patches from batch:
                valid_batch_indices = np.argwhere(valid_patches[:, 0, 0, 0, 0].cpu().detach().numpy())[:, 0]
                if limit_nb_patches is not None:
                    limit = limit_nb_patches[0]
                    if limit_nb_patches[1] == 'number':
                        if valid_batch_indices.shape[0] > limit:
                            valid_batch_indices = np.random.choice(valid_batch_indices, limit, replace=False)
                    elif limit_nb_patches[1] == 'factor':
                        assert limit <= 1. and limit >= 0.
                        valid_batch_indices = np.random.choice(valid_batch_indices, int(limit*valid_batch_indices.shape[0]), replace=False)
                if valid_batch_indices.shape[0] == 0:
                    print("ZERO valid patches at level {}!".format(nb_patch_net))
                    # Avoid problems if all patches are invalid and torch complains that autograd cannot be performed:
                    loss += pred_patches.sum() * 0.
                    continue

                # ----------------------------
                # Compute the actual (inverted) MeMasks targets: (0 is me, 1 are the others)
                # best targets for Dice loss (usually more me than others)
                # ----------------------------
                center_labels_repeated = label_at_patch_center.repeat(1, 1, *real_patch_shape)
                me_masks = gt_patches != center_labels_repeated

                if patch_is_on_boundary is not None:
                    # If on boundary, we make (inverted) me_masks completely 1 (split from everything)
                    me_masks = me_masks | patch_is_on_boundary

                # Downscale MeMasks using MaxPooling (preserve narrow processes):
                # moreover, during the maxPool, better shrink me mask than expanding (avoid merge predictions)
                if all(fctr == 1 for fctr in patch_dws_fact):
                    maxpool = Identity()
                else:
                    maxpool = nn.MaxPool3d(kernel_size=patch_dws_fact,
                                           stride=patch_dws_fact,
                                           padding=0)

                # Downscaling patch:
                down_sc_slice = (slice(None), slice(None)) + tuple(slice(int(dws_fact/2), None, dws_fact) for dws_fact in patch_dws_fact)

                # Final targets:
                patch_targets = me_masks[valid_batch_indices].float()[down_sc_slice]
                patch_ignore_masks = ignore_masks[valid_batch_indices][down_sc_slice].byte()


                # Invert MeMasks:
                # best targets for Dice loss are: meMask == 0; others == 1
                # FIXME: generalize
                if patch_dws_fact[1] > 6:
                    patch_targets = 1. - patch_targets

                assert valid_batch_indices.max() < pred_patches.shape[0], "Something went wrong, more target patches were collected than predicted: {} targets vs {} pred...".format(valid_batch_indices.max(), pred_patches.shape[0])
                pred_embed = pred_patches[valid_batch_indices]
                pred_embed = pred_embed[:, :, 0, 0, 0]

                # ----------------------------
                # Expand embeddings to patches using PatchNet models:
                # ----------------------------
                if "model_number" in kwargs:
                    # FIXME: update this crap
                    # In this case we are training a stacked model:
                    mdl_num = kwargs["model_number"]
                    ptch_num = kwargs["patchNet_number"]
                    expanded_patches = data_parallel(self.model.models[mdl_num].patch_models[ptch_num], pred_embed, self.devices)
                else:
                    expanded_patches = data_parallel(self.model.patch_models[nb_patch_net], pred_embed, self.devices)
                # print(expanded_patches.shape)
                assert expanded_patches.shape[1] == 1, "PatchNet should output only a one-channel mask!"

                # Some logs:
                if nb_stride == 0:
                    log_image("ptc_trg_l{}".format(nb_patch_net), patch_targets)
                    log_image("ptc_pred_l{}".format(nb_patch_net), expanded_patches)
                    # log_image("ptc_ign_l{}".format(nb_patch_net), patch_ignore_masks)
                    log_scalar("avg_targets_l{}".format(nb_patch_net), patch_targets.float().mean())

                # Train only checkerboard pattern:
                if self.apply_checkerboard:
                    checkerboard = np.zeros(patch_shape_input)
                    # Verticals:
                    center_coord = [int(sh/2) for sh in patch_shape_input]
                    checkerboard[:,center_coord[1],:] = 1
                    checkerboard[:,:,center_coord[2]] = 1
                    # Two diagonals:
                    indices = np.indices(patch_shape_input)
                    checkerboard[indices[1] == indices[2]] = 1
                    checkerboard[indices[1] == (patch_shape_input[2] - indices[2] - 1)] = 1
                    # Reduce z-context:
                    z_mask = np.zeros_like(checkerboard)
                    z_mask[center_coord[0]] = 1
                    for z in range(patch_shape_input[0]):
                        offs = abs(center_coord[0]-z)
                        if offs != 0:
                            z_mask[z,offs:-offs, offs:-offs] = 1
                    checkerboard[np.logical_not(z_mask)] = 0
                    # Expand channels and wrap:
                    checkerboard = torch.from_numpy(checkerboard).cuda(patch_ignore_masks.get_device()).float()
                    checkerboard = checkerboard.unsqueeze(0).unsqueeze(0)
                    checkerboard = checkerboard.repeat(*patch_ignore_masks.shape[:2], 1, 1, 1)

                # ----------------------------
                # Apply ignore mask and compute loss:
                # ----------------------------
                patch_valid_masks = 1. - patch_ignore_masks.float()
                if self.apply_checkerboard:
                    patch_valid_masks = patch_valid_masks * checkerboard
                expanded_patches = expanded_patches * patch_valid_masks
                patch_targets = patch_targets * patch_valid_masks
                with warnings.catch_warnings(record=True) as w:
                    loss_unet = data_parallel(self.loss, (expanded_patches, patch_targets.float()), self.devices).mean()

                loss = loss + loss_unet
                if nb_stride == 0:
                    log_scalar("loss_l{}".format(nb_patch_net), loss_unet)
                    log_scalar("nb_patches_l{}".format(nb_patch_net), expanded_patches.shape[0])

        # print("Loss done, memory {}", torch.cuda.memory_allocated(0)/1000000)
        # TODO: use Callback from Roman to run it every N iterations
        gc.collect()
        return loss




def get_slicing_crops(pred_shape, target_shape, pred_ds_factor, real_patch_shape):
    # Compute new left crops:
    # (we do not care about the right crops, because anyway the extra patches are
    # ignored with the option `limit_patches_to`)
    upscaled_pred_shape = [sh*fctr for sh, fctr in zip(pred_shape, pred_ds_factor)]

    shape_diff = [orig - trg for orig, trg in zip(target_shape, upscaled_pred_shape)]
    assert all([diff >= 0 for diff in shape_diff]), "Prediction should be smaller or equal to the targets!"
    assert all([diff % 2 == 0 for diff in shape_diff])
    padding = [int(diff/2) for diff in shape_diff]

    crop_slice_targets = [slice(None), slice(None)]
    crop_slice_prediction = [slice(None), slice(None)]
    import math
    for dim, pad in enumerate(padding):
        # Consider the patch-padding:
        real_pad = pad - int(real_patch_shape[dim]/2)
        if real_pad > 0:
            # We should crop targets
            crop_slice_targets.append(slice(real_pad, -real_pad))
            crop_slice_prediction.append(slice(None))
        elif real_pad < 0:
            # We should crop prediction:
            # (use floor to round up, since pad is negative)
            crop_slice_prediction.append(slice(-math.floor(real_pad/pred_ds_factor[dim]), math.floor(real_pad/pred_ds_factor[dim])))
            crop_slice_targets.append(slice(None))
        else:
            # No need to crop:
            crop_slice_targets.append(slice(None))
            crop_slice_prediction.append(slice(None))

    return tuple(crop_slice_targets), tuple(crop_slice_prediction)


def get_prediction_strides(pred_ds_factor, strides, max_crops=None):
    # Compute updated strides:
    assert all(strd % pred_fctr == 0 for strd, pred_fctr in
               zip(strides, pred_ds_factor)), "Stride {} should be divisible by downscaling factor {}".format(strides,
                                                                                                            pred_ds_factor)
    pred_strides = tuple(int(strd / pred_fctr) for strd, pred_fctr in zip(strides, pred_ds_factor))

    return pred_strides


def extract_patches_torch(tensor, shape, stride, precrop_target=None, max_random_crop=None,
                          # downscale_fctr=None,
                          crop_slice=None,
                          limit_patches_to=None,
                          reshape_to_batch_dim=True):
    assert tensor.dim() == 4 or tensor.dim() == 5
    dim = tensor.dim() - 2
    assert len(shape) == dim and len(stride) == dim
    if crop_slice is not None:
        assert max_random_crop is None and precrop_target is None
    if precrop_target is not None:
        assert len(precrop_target) == dim
        assert all([isinstance(sl, (tuple, list)) for sl in precrop_target]) and all([len(sl) == 2 for sl in precrop_target])
    else:
        precrop_target = [(0, 0) for _ in range(dim)]

    max_random_crop = [0 for _ in range(dim)] if max_random_crop is None else deepcopy(max_random_crop)
    assert len(max_random_crop) == dim
    if isinstance(max_random_crop, tuple):
        max_random_crop = list(max_random_crop)
    for d in range(dim):
        max = tensor.shape[2 + d] - precrop_target[d][0] - precrop_target[d][1] - shape[d]
        if max_random_crop[d] > max:
            max_random_crop[d] = max

    # if downscale_fct is not None:
    #     assert len(downscale_fct) == dim

    if limit_patches_to is not None:
        assert len(limit_patches_to) == dim

    # Pick a random crop:
    if crop_slice is None:
        rnd_crop = [np.random.randint(max_offs+1) for max_offs in max_random_crop]
        crop_slice = (slice(None), slice(None)) + tuple(slice(precrop[0]+off, full_shp-precrop[1]) for off, precrop, full_shp in zip(rnd_crop, precrop_target, tensor.shape[2:]))

    # Unfold it:
    tensor = tensor[crop_slice]
    N, C = tensor.shape[:2]
    for d in range(dim):
        tensor = tensor.unfold(d+2, size=shape[d],step=stride[d])
    # Reshape:
    nb_patches = tensor.shape[2:2+len(shape)]
    # Along each dimension, we make sure to keep only a specific number of patches (not more):
    # this assures compatibility with other patches already extracted
    if limit_patches_to is not None:
        actual_limits  = tuple( lim if lim<nb else nb for nb, lim in zip(nb_patches, limit_patches_to))
        valid_patch_slice = (slice(None), slice(None)) + tuple(slice(None,lim) for lim in actual_limits)
        tensor = tensor[valid_patch_slice]
        nb_patches = actual_limits
    # Reshape
    if reshape_to_batch_dim:
        tensor = tensor.contiguous().view(N,C,-1,*shape)
        tensor = tensor.permute(0,2,1,*range(3,3+dim)).contiguous().view(-1,C,*shape)
    # else:
    #     tensor = tensor.permute(0, 1, *range(3, 3 + dim), 2).contiguous()

    # if downscale_fct is not None:
    #     # TODO: use MaxPool instead?
    #     for d, dw in enumerate(downscale_fct):
    #         slc = tuple(slice(None) for _ in range(2+d)) + (slice(None,None,dw),)
    #         tensor = tensor[slc]

    return tensor, crop_slice, nb_patches


