BASE_PATH="/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8"
arch="yolov10"
d="/mnt/yixiaojie/coco"
save_path_for_int_ckpt="/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/yolov10s_w8_mp_int8.pt"  # save path for int ckpt
config_weight_mp="/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_layers1.yaml"
# if config_weight_mp is not None, perform mixed precision quantization

# --keep_fp: keep some layers at fp32

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=0 python scripts/quant_hardware/quant_inference_hardware.py --base_path $BASE_PATH --pretrained --arch $arch --skip_quant_act -d $d \
														  --save_int \
														  --int_ckpt_path $save_path_for_int_ckpt \
														  --config_weight_mp $config_weight_mp


