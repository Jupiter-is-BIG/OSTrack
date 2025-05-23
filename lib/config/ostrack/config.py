from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_ce"  # 'vit_base_patch16_224_ce'
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.CE_LOC = [3, 6, 9]
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = [0.7, 0.7, 0.7]
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.GNN_LAYERS_PER_STAGE= 2 # THIS IS THE NEW PARAMETER
cfg.MODEL.BACKBONE.GNN_TYPE = "GAT" # THIS IS THE NEW PARAMETER
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'CTR_POINT'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CORNER"  # CORNER, CENTER
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 6 # UPDATED
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 128
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 2 # UPDATED
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.DROP_PATH_RATE = 0.1
cfg.TRAIN.CE_START_EPOCH = 20
cfg.TRAIN.CE_WARM_EPOCH = 80
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1


# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # causal, joint
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["GOT10K_vottrain"]  # , "LASOT", "COCO17", "TRACKINGNET"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]  # , 1, 1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 288
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.CENTER_JITTER = 3
cfg.DATA.SEARCH.SCALE_JITTER = 0.25
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.TEST
cfg.DATA.TEST = edict()
cfg.DATA.TEST.DATASETS_NAME = ["VOT20"]
cfg.DATA.TEST.DATASETS_RATIO = [1]
cfg.DATA.TEST.SAMPLE_PER_EPOCH = 5000

# DATA.SAMPLER
cfg.DATA.SAMPLER = edict()
cfg.DATA.SAMPLER.JOINT = edict()
cfg.DATA.SAMPLER.JOINT.TEMPLATE = edict()
cfg.DATA.SAMPLER.JOINT.TEMPLATE.SHIFT = [64, 64]
cfg.DATA.SAMPLER.JOINT.TEMPLATE.SCALE = [0.05, 0.05]
cfg.DATA.SAMPLER.JOINT.SEARCH = edict()
cfg.DATA.SAMPLER.JOINT.SEARCH.SHIFT = [64, 64]
cfg.DATA.SAMPLER.JOINT.SEARCH.SCALE = [0.05, 0.05]
cfg.DATA.SAMPLER.CAUSAL = edict()
cfg.DATA.SAMPLER.CAUSAL.TEMPLATE = edict()
cfg.DATA.SAMPLER.CAUSAL.TEMPLATE.SHIFT = [64, 64]
cfg.DATA.SAMPLER.CAUSAL.TEMPLATE.SCALE = [0.05, 0.05]
cfg.DATA.SAMPLER.CAUSAL.SEARCH = edict()
cfg.DATA.SAMPLER.CAUSAL.SEARCH.SHIFT = [64, 64]
cfg.DATA.SAMPLER.CAUSAL.SEARCH.SCALE = [0.05, 0.05]

# tracking
cfg.TRACK = edict()
cfg.TRACK.WINDOW_INFLUENCE = 0.175
cfg.TRACK.PENALTY_K = 0.04
cfg.TRACK.LR = 0.4
cfg.TRACK.UPDATE_TEMPLATE = True
cfg.TRACK.TEMPLATE_FACTOR = 2.0
cfg.TRACK.TEMPLATE_SIZE = 128

cfg.TEST = edict()
cfg.TEST.EPOCH = 6 # UPDATED
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                # raise ValueError("{} not exist in config.py".format(k))
                print(f"[WARNING] '{k}' not found in config.py — skipping")
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)