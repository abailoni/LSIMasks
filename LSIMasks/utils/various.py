

def auto_crop_tensor_to_shape(to_be_cropped, target_tensor_shape, return_slice=False,
                              ignore_channel_and_batch_dims=True):
    initial_shape = to_be_cropped.shape
    diff = [int_sh - trg_sh for int_sh, trg_sh in zip(initial_shape, target_tensor_shape)]
    if ignore_channel_and_batch_dims:
        assert all([d >= 0 for d in diff[2:]]), "Target shape should be smaller!"
    else:
        assert all([d >= 0 for d in diff]), "Target shape should be smaller!"
    left_crops = [int(d / 2) for d in diff]
    right_crops = [shp - int(d / 2) if d % 2 == 0 else shp - (int(d / 2) + 1) for d, shp in zip(diff, initial_shape)]
    if ignore_channel_and_batch_dims:
        crop_slice = (slice(None), slice(None)) + tuple(slice(lft, rgt) for rgt, lft in zip(right_crops[2:], left_crops[2:]))
    else:
        crop_slice = tuple(slice(lft, rgt) for rgt, lft in zip(right_crops, left_crops))
    if return_slice:
        return crop_slice
    else:
        return to_be_cropped[crop_slice]
