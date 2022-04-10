from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = ''
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''

__C.GPU_ID = ''
__C.CUDA = True
__C.LOAD_PRE_TRAIN = False
__C.WRONG_CAPTION = 0
__C.RANDOM_SEED = 0
__C.NUM_BATCH_FOR_TEST = 0
__C.R_PRECISION_FILE = ''
__C.R_PRECISION_FILE_HIDDEN = ''
__C.R_PRECISION_DIR = ''
__C.WORKERS = 6
__C.PRE_WORKERS = 6
__C.IMAGE_SIZE = 0
__C.BATCH_SIZE = 32

__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = False

__C.CHECKPOINT_DIR = ''
__C.CHECKPOINT_NAME = ''

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64

__C.PRE_TREE = edict()
__C.PRE_TREE.BRANCH_NUM = 3
__C.PRE_TREE.BASE_SIZE = 64
# Test options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = False
__C.TEST.GENERATED_TEST_IMAGES = ''
__C.TEST.GENERATED_HIDDEN_TEST_IMAGES = ''
__C.TEST.BATCH_SIZE = 32

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 0.
__C.TRAIN.COEFF.UNCOND_LOSS =  0.
__C.TRAIN.COEFF.COLOR_LOSS = 0.

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

__C.PRE_TRAIN = edict()
__C.PRE_TRAIN.BATCH_SIZE = 32
__C.PRE_TRAIN.MAX_EPOCH = 600
__C.PRE_TRAIN.SNAPSHOT_INTERVAL = 2000
__C.PRE_TRAIN.DISCRIMINATOR_LR = 2e-4
__C.PRE_TRAIN.GENERATOR_LR = 2e-4
__C.PRE_TRAIN.ENCODER_LR = 2e-4
__C.PRE_TRAIN.RNN_GRAD_CLIP = 0.25
__C.PRE_TRAIN.FLAG = True
__C.PRE_TRAIN.NET_E = ''
__C.PRE_TRAIN.NET_G = ''
__C.PRE_TRAIN.B_NET_D = True

__C.PRE_TRAIN.SMOOTH = edict()
__C.PRE_TRAIN.SMOOTH.GAMMA1 = 5.0
__C.PRE_TRAIN.SMOOTH.GAMMA3 = 10.0
__C.PRE_TRAIN.SMOOTH.GAMMA2 = 5.0
__C.PRE_TRAIN.SMOOTH.LAMBDA = 1.0

# Modal options

__C.RNN = edict()
__C.RNN.EMBEDDING_DIM = 0
__C.RNN.VOCAB_SIZE = 0
__C.RNN.WORD_EMBEDDING_DIM = 0
__C.RNN.H_DIM = 0
__C.RNN.TYPE = ''

__C.CNN = edict()
__C.CNN.EMBEDDING_DIM = 0
__C.CNN.H_DIM = 0

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False
__C.GAN.B_CONDITION = False
__C.GAN.EMBEDDING_DIM = 0

__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering theoptions in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.unsafe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
