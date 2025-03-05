arch="resnet50"  # 模型结构
d="/mnt/yixiaojie/imagenet"  # 数据集的data路径
CUDA_VISIBLE_DEVICES=$1 python scripts/get_calib_data.py --arch $arch -d $d --evaluate --test-batch 512