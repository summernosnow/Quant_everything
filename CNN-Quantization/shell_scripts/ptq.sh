# config="/root/yixiaojie/CNN-Quantization/configs/configresnet50.yaml"
# arch="resnet50"
# outdir="quant_output_random/resnet50_w8a8"

config="/root/yixiaojie/CNN-Quantization/configs/configyolov10-1.yaml"
arch="yolov10x"
outdir="quant_output_random/yolov10x1_w8a8"

# --------- conduct quantized inference --------
CUDA_VISIBLE_DEVICES=0 python scripts/ptq.py --outdir $outdir --pretrained --arch $arch --config $config