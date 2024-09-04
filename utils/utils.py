import os
import os.path as osp
import errno
import logging
import json
import sys
import numpy as np
from piq import ssim, psnr

def normalize(noise, max_num=255, min_num=0):
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * (max_num - min_num) + min_num
    # noise = noise 
    return noise

def mkdir_if_missing(directory):
	if not osp.exists(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

def compute_batch_recovery_metric(x1, x2):
    _psnr = psnr(x1, x2)
    _ssim = ssim(x1, x2)
    _mse = np.mean((x1.cpu().numpy() - x2.cpu().numpy()) ** 2)
    return _psnr, _mse, _ssim


def setup_logger(name, save_dir, distributed_rank, save_name='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
