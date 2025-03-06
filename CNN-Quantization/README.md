## Install
* Install [PyTorch](http://pytorch.org/)
* pip install -e .

## 代码执行流程
- 在推理过程中手动提取calib data: 
```
# --------- conduct quantized inference --------
arch="vgg"  # 模型结构
d="/mnt/public/yuanzhihang/imagenet"  # 数据集的data路径
CUDA_VISIBLE_DEVICES=$1 python scripts/get_calib_data.py --arch $arch -d $d --evaluate
```

### PTQ
```
config="/home/fangtongcheng/base_code_test/Quant_Base_Model/pytorch-classification/configs/config.yaml"  # 量化配置config
arch="vgg"  # 模型结构
outdir="quant_output_random/vgg_w4a8_mse"  # PTQ输出存储路径

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=$1 python scripts/ptq.py --outdir $outdir --pretrained --arch $arch --config $config
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
