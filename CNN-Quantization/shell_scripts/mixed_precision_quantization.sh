BASE_PATH="./quant_output_random/yolov10x1_w8a8"
# BASE_PATH="./quant_output_random/resnet50_w8a8"
arch="yolov10x"
# arch="resnet50"
sensitivity_type="weight"
min_bit=8
required_map=0.448
data="/mnt/yixiaojie/imagenet"
required_acc=0.448   

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=0 python scripts/mixed_precision_quantization.py \
    --data $data \
    --base_path $BASE_PATH \
    --pretrained \
    --arch $arch \
    --sensitivity_type $sensitivity_type \
    --min_bit $min_bit \
    --required_map $required_map
    #--save_int