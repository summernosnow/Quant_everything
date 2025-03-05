BASE_PATH="./quant_output_random/yolov10s_w8a8"
arch="yolov10"
sensitivity_type="weight"

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=0 python scripts/get_sensitivity.py --base_path $BASE_PATH --pretrained --arch $arch --sensitivity_type $sensitivity_type