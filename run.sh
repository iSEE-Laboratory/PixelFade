# protect image
CUDA_VISIBLE_DEVICES=0 python protect_image.py \
    --config_file 'configs/AGW_baseline_market1501.yml' \
    --reid_ckpt_path 'ckpt/market1501_resnet50_nl_model_120.pth' \
    --output 'log/market1501_protect' --save_image_npy

CUDA_VISIBLE_DEVICES=0 python protect_image.py \
    --config_file 'configs/AGW_baseline_cuhk03.yml' \
    --reid_ckpt_path 'ckpt/cuhk03_resnet50_nl_model_120.pth' \
    --beta_temp 0.02 \
    --output 'log/cuhk03_protect' --save_image_npy

CUDA_VISIBLE_DEVICES=0 python protect_image.py \
    --config_file 'configs/AGW_baseline_msmt17.yml' \
    --reid_ckpt_path 'ckpt/msmt17_resnet50_nl_model_120.pth' \
    --output 'log/msmt17_protect' --save_image_npy


# recovery attack
CUDA_VISIBLE_DEVICES=4 python recovery_attack.py --npy_path 'log/market1501_protect' 

CUDA_VISIBLE_DEVICES=5 python recovery_attack.py --npy_path 'log/cuhk03_protect' 
