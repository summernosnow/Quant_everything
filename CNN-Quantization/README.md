## Install
* Install [PyTorch](http://pytorch.org/)
* pip install -e .

## 代码执行流程

### calibration dataset
```
d="/mnt/yixiaojie/coco"  # 数据集的data路径
outdir="/root/yixiaojie/CNN-Quantization/calib_data"
CUDA_VISIBLE_DEVICES=$1 python scripts/get_calib_coco_data.py --d $d --outdir $outdir --calib_images 256
```

### PTQ
```
config="/root/yixiaojie/CNN-Quantization/configs/configyolov10-1.yaml"
arch="yolov10x"
outdir="quant_output_random/yolov10x1_w8a8"

# --------- conduct ptq --------
CUDA_VISIBLE_DEVICES=0 python scripts/ptq.py --outdir $outdir --pretrained --arch $arch --config $config
```

### mixed_precision_quantization
```
BASE_PATH="./quant_output_random/yolov10x1_w8a8"
arch="yolov10x"
sensitivity_type="weight"
min_bit=8
required_map=0.448

# --------- conduct mixed precision quantization --------
CUDA_VISIBLE_DEVICES=0 python scripts/mixed_precision_quantization.py \
    --base_path $BASE_PATH \
    --pretrained \
    --arch $arch \
    --sensitivity_type $sensitivity_type \
    --min_bit $min_bit \
    --required_map $required_map
