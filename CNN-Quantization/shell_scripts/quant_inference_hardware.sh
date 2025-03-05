BASE_PATH="./quant_output_random/yolov10s_w8a8"
arch="yolov10"
d="/mnt/yixiaojie/coco"
int_ckpt_path="quant_output_random/yolov10s_w8a8/yolov10s_w8_mp_int8.pt"
config_weight_mp="/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_layers.yaml"

# --------- conduct quantized inference --------
# while(true); do
CUDA_VISIBLE_DEVICES=0 python scripts/quant_hardware/quant_inference_hardware.py --base_path $BASE_PATH --pretrained --arch $arch --skip_quant_act -d $d \
																		  --keep_fp \
																		  --from_int \
																		  --ckpt $int_ckpt_path \
																		#   --config_weight_mp $config_weight_mp


