import numpy as np
from GASP.segmentation import GaspFromAffinities


IMAGE_SHAPE = (10, 200, 200)

# TODO: use the same offsets that you defined in the infer_config.yml file:
offsets = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [0, -4, 0],
    [0, 0, -4],
    [0, -4, -4],
    [0, 4, -4],
    [-2, 0, 0],
    [0, -8, -8],
    [0, 8, -8],
    [0, -12, 0],
    [0, 0, -12],
]
# TODO: Load the affinities that were saved by the `infer.py` script.
# Here we generate some random ones:
random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE).astype('float32')

run_GASP_kwargs = {'linkage_criteria': 'average',
                   'add_cannot_link_constraints': False}

gasp_instance = GaspFromAffinities(offsets,
                                   run_GASP_kwargs=run_GASP_kwargs)
final_instance_segmentation, runtime = gasp_instance(random_affinities)
print("Clustering took {} s".format(runtime))

# In case some ground-truth annotations are available, here we compute the ARAND score:
from segmfriends.utils import cremi_score
# TODO: load corresponding GT labels
# Here we generate some random ones:
gt = np.random.uniform(size=IMAGE_SHAPE).astype('uint32')
assert gt.shape == final_instance_segmentation.shape
scores = cremi_score(gt, final_instance_segmentation, return_all_scores=True)
print(scores)
