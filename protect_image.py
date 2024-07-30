
import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from configs import cfg
from data import make_data_loader_for_encry
from modeling import build_model
from utils.utils import setup_logger
from datetime import datetime
from pixelfade import PixelFade
from utils.reid_metric import extract_features_for_reid, compute_reid_metric

def main():
    parser = argparse.ArgumentParser(description="PixelFade protect Re-ID image")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--dataset', type=str, default='market1501')
    parser.add_argument('--output', help='output directory', type=str, default='./log/market1501_protect')
    parser.add_argument('--reid_ckpt_path', type=str, default='', required=True)
    parser.add_argument('--save_image_npy', action='store_true', default=False)

    # parameter of PixelFade
    parser.add_argument('--num_iter', help='Number of iterations.', type=int, default=200)
    parser.add_argument('--batch_size', help='batch size', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--beta_temp', type=float, default=0.04)
    parser.add_argument('--momentum_w', type=float, default=0.6)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--fade_num_iter', help='Number of iterations of fade stage.', type=int, default=5)
    parser.add_argument('--warmup_iter', help='Number of iterations of warmup stage.', type=int, default=5)


    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = args.output
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PixelFade protect ReID images", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True

    # prepare data
    data_loader, num_query, num_classes = make_data_loader_for_encry(cfg, args)
    logger.info("{} query num {}".format(cfg.DATASETS.NAMES, num_query))

    # build pretrained reid model
    model = build_model(cfg, num_classes).cuda()
    logger.info("Loading pretrained ReID model from {}".format(args.reid_ckpt_path))
    model.load_param(args.reid_ckpt_path)
    model.eval()

    # prepare pixelfade
    pixelfade = PixelFade(args, logger, (cfg.INPUT.IMG_SIZE[0], cfg.INPUT.IMG_SIZE[1]))

    running_time_start = datetime.now()

    # protect reid images
    if args.save_image_npy: # if you want to test recovery attack, set it as True to prepare original-protected pairs
        datas_train = extract_features_for_reid(logger, 
                                                args, 
                                                data_loader['train'], 
                                                model, 
                                                pixelfade,
                                                state='train')
    test_datas = extract_features_for_reid(logger, 
                                      args, 
                                      data_loader['eval'], 
                                      model, 
                                      pixelfade,
                                      state='test')

    # test reid metrics
    cmc, mAP, mINP = compute_reid_metric(num_query, test_datas, feat_norm='on', gallery_mode='encry', query_mode='encry')
    logger.info('**** encry2encry Results ****')
    logger.info('rank1:{:.1%}'.format(cmc[0]))
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('mINP: {:.1%}'.format(mINP))

    cmc, mAP, mINP = compute_reid_metric(num_query, test_datas, feat_norm='on', gallery_mode='clean', query_mode='clean')
    logger.info('**** clean2clean Results ****')
    logger.info('rank1:{:.1%}'.format(cmc[0]))
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('mINP: {:.1%}'.format(mINP))

    cmc, mAP, mINP = compute_reid_metric(num_query, test_datas, feat_norm='on', gallery_mode='clean', query_mode='encry')
    logger.info('**** encry2clean Results ****')
    logger.info('rank1:{:.1%}'.format(cmc[0]))
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('mINP: {:.1%}'.format(mINP))

    cmc, mAP, mINP = compute_reid_metric(num_query, test_datas, feat_norm='on', gallery_mode='encry', query_mode='clean')
    logger.info('**** clean2encry Results ****')
    logger.info('rank1:{:.1%}'.format(cmc[0]))
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('mINP: {:.1%}'.format(mINP))

    running_time = datetime.now() - running_time_start
    logger.info("running time: {}".format(running_time))
if __name__ == '__main__':
    main()
