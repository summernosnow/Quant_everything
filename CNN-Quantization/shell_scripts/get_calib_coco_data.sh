d="/mnt/yixiaojie/coco"  # 数据集的data路径
outdir="/root/yixiaojie/CNN-Quantization/calib_data"
CUDA_VISIBLE_DEVICES=$1 python scripts/get_calib_coco_data.py --d $d --outdir $outdir --calib_images 256