python scripts/integer_programming.py --mixed_precision_type weight\
                                      --sensitivity /root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/sensitivity_start.yaml\
                                      --para_size_config cnn_weight_ratio/yolov10_weight_ratio.yaml\
                                      --mixed_precision_config /root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/yolov10_w8a8\
                                      --target_bitwidth 8  # 我们希望做到的混合精度平均位宽