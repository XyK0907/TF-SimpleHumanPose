import tensorflow as tf
import argparse
import numpy as np
# #set random seed for the numpy pseudo random number generator in order to obtain comparable results
# np.random.seed(2223)

from model import Model
from config import cfg
from tfflat.base import Trainer
from tfflat.utils import mem_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument("--cnt_val_it",type=int,dest="cnt_val_itr",default=-1)
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
args = parse_args()

cfg.set_args(args.gpu_ids, args.cnt_val_itr,args.continue_train)
trainer = Trainer(Model(), cfg)
trainer.train()



