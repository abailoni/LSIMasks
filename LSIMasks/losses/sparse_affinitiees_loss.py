import warnings
import gc
import torch.nn as nn
from torch.nn.parallel.data_parallel import data_parallel

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from speedrun.log_anywhere import log_image, log_embedding, log_scalar
from segmfriends.utils.various import parse_data_slice

from ..utils.various import auto_crop_tensor_to_shape


class MultiLevelSparseAffinityLoss(nn.Module):
    """
    Perform deep supervision by applying loss at several depth levels of a U-Net like architecture.
    """
    def __init__(self, model, loss_type="Dice", model_kwargs=None, devices=(0,1),
                 predictions_specs=None,
                 train_glia_mask=False,
                 target_has_label_segm=False,
                 target_has_various_masks=False,
                 precrop_pred=None):
        super(MultiLevelSparseAffinityLoss, self).__init__()
        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

        self.devices = devices
        self.model_kwargs = model_kwargs
        self.MSE_loss = nn.MSELoss()
        self.smoothL1_loss = nn.SmoothL1Loss()
        # TODO: use nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        self.soresen_loss = SorensenDiceLoss()

        self.model = model
        assert predictions_specs is not None, "A dictionary should be passed"
        self.predictions_specs = predictions_specs
        self.precrop_pred = precrop_pred
        self.target_has_label_segm = target_has_label_segm
        self.target_has_various_masks = target_has_various_masks
        self.train_glia_mask = train_glia_mask

    def forward(self, predictions, all_targets):
        predictions = [predictions] if not isinstance(predictions, (list, tuple)) else predictions
        all_targets = [all_targets] if not isinstance(all_targets, (list, tuple)) else all_targets

        loss = 0

        # # ----------------------------
        # # Predict glia mask:
        # # ----------------------------
        if self.train_glia_mask:
            assert not self.target_has_various_masks, "To be implemented"
            frg_kwargs = self.model.models[-1].foreground_prediction_kwargs
            if frg_kwargs is None:
                # Legacy:
                nb_glia_preds = 1
                nb_glia_targets = [0]
            else:
                nb_glia_preds = len(frg_kwargs)
                nb_glia_targets = [frg_kwargs[dpth]["nb_target"] for dpth in frg_kwargs]

            all_glia_preds = predictions[-nb_glia_preds:]
            predictions = predictions[:-nb_glia_preds]

            loss_glia = 0
            for counter, glia_pred, nb_tar in zip(range(len(all_glia_preds)), all_glia_preds, nb_glia_targets):
                glia_target = all_targets[nb_tar][:,[-1]]
                all_targets[nb_tar] = all_targets[nb_tar][:, :-1]
                assert self.target_has_label_segm
                gt_segm = all_targets[nb_tar][:,[0]]

                glia_target = auto_crop_tensor_to_shape(glia_target, glia_pred.shape)
                gt_segm = auto_crop_tensor_to_shape(gt_segm, glia_pred.shape)
                # TODO: generalize ignore label:
                valid_mask = (gt_segm != 0).float()
                glia_pred = glia_pred * valid_mask
                glia_target = glia_target * valid_mask
                with warnings.catch_warnings(record=True) as w:
                    loss_glia_new = data_parallel(self.loss, (glia_pred, glia_target), self.devices).mean()
                loss_glia = loss_glia + loss_glia_new
                log_image("glia_target_d{}".format(counter), glia_target)
                log_image("glia_pred_d{}".format(counter), glia_pred)
            loss = loss + loss_glia
            log_scalar("loss_glia", loss_glia)

        for counter, nb_pred in enumerate(self.predictions_specs):
            assert len(predictions) > nb_pred
            pred = predictions[nb_pred]
            # TODO: add precrop_pred?
            # if self.precrop_pred is not None:
            #     from segmfriends.utils.various import parse_data_slice
            #     crop_slc = (slice(None), slice(None)) + parse_data_slice(self.precrop_pred)
            #     predictions = predictions[crop_slc]
            pred_specs = self.predictions_specs[nb_pred]
            target = all_targets[pred_specs.get("target", 0)]

            target_dws_fact = pred_specs.get("target_dws_fact", None)
            if target_dws_fact is not None:
                assert isinstance(target_dws_fact, list) and len(target_dws_fact) == 3
                target = target[(slice(None), slice(None)) + tuple(slice(None,None,dws) for dws in target_dws_fact)]

            target = auto_crop_tensor_to_shape(target, pred.shape,
                                            ignore_channel_and_batch_dims=True)

            if self.target_has_label_segm:
                if self.target_has_various_masks:
                    target = target[:, 2:]
                else:
                    target = target[:,1:]
            assert target.shape[1] % 2 == 0, "Target should include both affinities and masks"

            # Get ignore-mask and affinities:
            nb_channels = int(target.shape[1] / 2)

            affs_channels = pred_specs.get("affs_channels", None)
            if affs_channels is not None:
                if isinstance(affs_channels, str):
                    affs_slice = parse_data_slice(affs_channels)[0]
                elif isinstance(affs_channels, list):
                    # TODO: make as a tuple???
                    affs_slice = affs_channels
                else:
                    raise ValueError("The passed affinities channels are not compatible")
            else:
                affs_slice = slice(None)

            gt_affs = target[:,:nb_channels][:, affs_slice]

            assert gt_affs.shape[1] == pred.shape[1], "Prediction has a wrong number of offset channels"

            valid_pixels = target[:,nb_channels:][:, affs_slice]

            # Invert affinities for Dice loss: (1 boundary, 0 otherwise)
            gt_affs = 1. - gt_affs

            pred = pred*valid_pixels
            gt_affs = gt_affs*valid_pixels

            with warnings.catch_warnings(record=True) as w:
                loss_new = data_parallel(self.loss, (pred, gt_affs), self.devices).mean()
            loss = loss + loss_new
            log_scalar("loss_sparse_d{}".format(counter), loss_new)

        # TODO: use Callback from Roman to run it every N iterations
        gc.collect()
        return loss




