# encoding: utf-8

import numpy as np
import torch
from ignite.metrics import Metric
import os
import time
from torchvision import utils as v_utils
import os.path as osp
from .re_ranking import re_ranking

def mkdir_if_missing(directory):
	if not osp.exists(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP


def extract_features_for_reid(logger, args, data_loader, reid_model, pixelFade, state='test'):
    save_imgs_dir = os.path.join(args.output, 'imgs')
    mkdir_if_missing(save_imgs_dir)
    feats_O, feats_P, pids, camids = [], [], [], []
    for idx, batch in enumerate(data_loader):
        logger.info("-----------{}-----------".format(idx))
        
        if state=='train':
            img_O, pid = batch
        else:
            img_O, pid, camid = batch
             
        with torch.no_grad():
            feat_O = reid_model(img_O.cuda())
            
        img_P, feat_P = pixelFade.protect_image(reid_model, img_O.cuda())
        
        feats_O.append(feat_O.cpu())
        feats_P.append(feat_P.cpu())
        pids.extend(np.asarray(pid))
        if state!='train':
            camids.extend(np.asarray(camid))
        
        # Save protected images for visualization
        if idx<10:
            all_picture = torch.cat([img_O.cpu(), img_P.cpu()], dim=0)
            v_utils.save_image(all_picture, os.path.join(save_imgs_dir, '{}_{}.png'.format(state, str(idx))), normalize=True, padding=0, nrow=args.batch_size)

        # Save protected images for recovery attack
        if args.save_image_npy:
            save_npy_dir = os.path.join(args.output, 'img_npy')
            mkdir_if_missing(save_npy_dir)
            np.save(os.path.join(save_npy_dir, 'clean-{}-{}-images'.format(state,idx)), img_O.cpu().numpy())
            np.save(os.path.join(save_npy_dir, 'encry-{}-{}-images'.format(state,idx)), img_P.cpu().numpy())
            np.save(os.path.join(save_npy_dir, '{}-{}-ids'.format(state,idx)), np.array(pid))
            if state!='train':
                np.save(os.path.join(save_npy_dir, '{}-{}-camids'.format(state,idx)), np.array(camid))
    
    feats_o = torch.cat(feats_O, dim=0)
    feats_p = torch.cat(feats_P, dim=0)
    return feats_o, feats_p, pids, camids

def compute_reid_metric(num_query, datas, feat_norm='on', gallery_mode='clean', query_mode='clean'):
    feats_o, feats_p, pids, camids = datas
    
    if gallery_mode == 'encry':
        gallery_feats = feats_p
    else:
        gallery_feats = feats_o
    
    if query_mode == 'encry':
        query_feats = feats_p
    else:
        query_feats = feats_o
    
    if feat_norm == 'on':
        print("The test feature is normalized")
        gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)
        query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2)
    # query
    qf = query_feats[:num_query]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    # gallery
    gf = gallery_feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP, mINP


class r1_mAP_mINP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='on'):
        super(r1_mAP_mINP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, mINP


class r1_mAP_mINP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='on'):
        super(r1_mAP_mINP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, mINP
    
