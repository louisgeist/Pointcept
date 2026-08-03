"""
SensatUrban split config for preprocessing.

Official labeled train pool (paper split): 12 Birmingham blocks + 25 Cambridge blocks = 37 zones.

Custom train/val/test partition (within that pool):
- Birmingham: 6 train / 2 val / 4 test
- Cambridge: 13 train / 4 val / 8 test

Within each city, block names are shuffled with a fixed RNG seed, then assigned in order to
train, val, and test. Test set size is 12/37 zones (~32.4%).
"""

import random

SEED = 0

BIRMINGHAM_OFFICIAL_TRAIN = [
    "birmingham_block_0",
    "birmingham_block_1",
    "birmingham_block_3",
    "birmingham_block_4",
    "birmingham_block_5",
    "birmingham_block_6",
    "birmingham_block_7",
    "birmingham_block_9",
    "birmingham_block_10",
    "birmingham_block_11",
    "birmingham_block_12",
    "birmingham_block_13",
]

CAMBRIDGE_OFFICIAL_TRAIN = [
    "cambridge_block_0",
    "cambridge_block_1",
    "cambridge_block_2",
    "cambridge_block_3",
    "cambridge_block_4",
    "cambridge_block_6",
    "cambridge_block_7",
    "cambridge_block_8",
    "cambridge_block_9",
    "cambridge_block_10",
    "cambridge_block_12",
    "cambridge_block_13",
    "cambridge_block_14",
    "cambridge_block_17",
    "cambridge_block_18",
    "cambridge_block_19",
    "cambridge_block_20",
    "cambridge_block_21",
    "cambridge_block_23",
    "cambridge_block_25",
    "cambridge_block_26",
    "cambridge_block_28",
    "cambridge_block_32",
    "cambridge_block_33",
    "cambridge_block_34",
]

OFFICIAL_TEST = [
    "birmingham_block_2",
    "birmingham_block_8",
    "cambridge_block_15",
    "cambridge_block_16",
    "cambridge_block_22",
    "cambridge_block_24",
    "cambridge_block_27",
]

N_TRAIN_BHM, N_VAL_BHM, N_TEST_BHM = 6, 2, 4
N_TRAIN_CAM, N_VAL_CAM, N_TEST_CAM = 13, 4, 8

assert len(BIRMINGHAM_OFFICIAL_TRAIN) == N_TRAIN_BHM + N_VAL_BHM + N_TEST_BHM
assert len(CAMBRIDGE_OFFICIAL_TRAIN) == N_TRAIN_CAM + N_VAL_CAM + N_TEST_CAM

rng = random.Random(SEED)


def split_three(
    zones: list[str],
    n_train: int,
    n_val: int,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(zones)
    rng.shuffle(shuffled)
    end_val = n_train + n_val
    return (
        shuffled[:n_train],
        shuffled[n_train:end_val],
        shuffled[end_val:],
    )


bhm_train, bhm_val, bhm_test = split_three(
    BIRMINGHAM_OFFICIAL_TRAIN, N_TRAIN_BHM, N_VAL_BHM
)
cam_train, cam_val, cam_test = split_three(
    CAMBRIDGE_OFFICIAL_TRAIN, N_TRAIN_CAM, N_VAL_CAM
)

SPLITS = {
    "train": sorted(bhm_train + cam_train),
    "val": sorted(bhm_val + cam_val),
    "test": sorted(bhm_test + cam_test),
}

print(SPLITS)