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






### quant_inference
```
BASE_PATH="./quant_output_random/vgg_w4a8"  # PTQ结果存储的路径
arch="vgg"
d="/mnt/public/yuanzhihang/imagenet"  # 数据集的data路径
# 如果这个不是None，则perform混合精度推理
config_weight_mp="/home/fangtongcheng/base_code_test/Quant_Base_Model/pytorch-classification/quant_output_random/vgg_w4a8/vgg_w4/weight_4.00.yaml"  
# --keep_fp
# 开启这个会把最敏感的层置为fp16，实施fp16和int的混合精度推理
# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=$1 python scripts/quant_inference.py --base_path $BASE_PATH --pretrained --arch $arch --skip_quant_act -d $d --config_weight_mp $config_weight_mp
