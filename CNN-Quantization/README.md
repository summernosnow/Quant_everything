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

### 获取模型的权重参数量比例
  - 调用：./scripts/weight_ration.ipynb
  - 这个脚本可以用来计算模型每一层的参数量，因为我们最终在计算模型的平均位宽时，需要用每一层的参数量和每一层的权重进行加权除以总参数量

### 获取模型的量化敏感度
  - 获取每一层的量化敏感度（分别对应2/4/8 bit）后，可以尝试使用整数规划进行混合精度量化，也可以尝试直接根据敏感度手工将模型的某些层设置成FP16
  - 具体结果可以参考 quant_output_random/vgg_w4a8/sensitivity.yaml，其中数值代表sqnr，量化噪声比，数值越大，认为该层对量化越不敏感
```
BASE_PATH="./quant_output_random/vgg_w4a8"  # 之前放PTQ结果的路径
arch="vgg"
sensitivity_type="weight"  # 测试权重还是激活值敏感度，我们这里关注权重敏感度

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=$1 python scripts/get_sensitivity.py --base_path $BASE_PATH --pretrained --arch $arch --sensitivity_type $sensitivity_type
```

### Perform 整数规划
- 会导出一份混合精度的config，给出对应每一层的位宽
- 具体可以参考：quant_output_random/vgg_w4a8/weight_4.00.yaml
```
python scripts/integer_programming.py --mixed_precision_type weight\
                                      --sensitivity /home/fangtongcheng/base_code_test/Quant_Base_Model/pytorch-classification/quant_output_random/vgg_w4a8/sensitivity.yaml\
                                      --para_size_config cnn_weight_ratio/vgg_weight_ratio.yaml\
                                      --mixed_precision_config vgg_w4\
                                      --target_bitwidth 4  # 我们希望做到的混合精度平均位宽
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
```

### 以上为量化模拟，当我们希望真正节省显存时，请运行save_int_ckpt.sh
```
BASE_PATH="./quant_output_random/mobilenet_w8a8"  # 原本PTQ结果的路径
arch="mobilenet"
d="/mnt/public/yuanzhihang/imagenet"
save_path_for_int_ckpt="quant_output_random/mobilenet_w8a8/mobilenet_w8_mp_int8.pt"  # int版本权重的存储路径
# config_weight_mp="/home/fangtongcheng/base_code_test/Quant_Base_Model/pytorch-classification/quant_output_random/mobilenet_w4a8_mse/weight_mp.yaml"
# --keep_fp
# --save_int 开启后才能存储一份int权重

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=$1 python scripts/quant_hardware/quant_inference_hardware.py --base_path $BASE_PATH --pretrained --arch $arch --skip_quant_act -d $d \
                                                          --save_int \
                                                          --keep_fp \
                                                          --int_ckpt_path $save_path_for_int_ckpt
```

### 在此基础上，运行quant_inference_hardware.sh，可以在推理过程中节省GPU上的显存
```
BASE_PATH="./quant_output_random/mobilenet_w8a8"  # 原本PTQ结果的路径
arch="mobilenet"
d="/mnt/public/yuanzhihang/imagenet"
int_ckpt_path="quant_output_random/mobilenet_w8a8/mobilenet_w8_mp_int8.pt"  # int版本权重的存储路径
# config_weight_mp="/home/fangtongcheng/base_code_test/Quant_Base_Model/pytorch-classification/quant_output_random/mobilenet_w4a8_mse/weight_mp.yaml"

# --------- conduct quantized inference --------
# while(true); do
CUDA_VISIBLE_DEVICES=$1 python scripts/quant_hardware/quant_inference_hardware.py --base_path $BASE_PATH --pretrained --arch $arch --skip_quant_act -d $d \
                                                                          --keep_fp \
                                                                          --from_int \
                                                                          --ckpt $int_ckpt_path
```