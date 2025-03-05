import yaml

# 读取 YAML 文件
def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# 读取两个 YAML 文件
quantized_layers = load_yaml("/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/quantized_layers.yaml")
sensitivity_start = load_yaml("/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/sensitivity_start.yaml")

# 计算差集 (quantized_layers - sensitivity_start)
diff_keys = set(sensitivity_start.keys()) - set(quantized_layers.keys())

# 将差集保存到新的 YAML 文件
diff_yaml_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/no_quant_layers.yaml"
with open(diff_yaml_path, 'w') as f:
    yaml.dump(list(diff_keys), f, default_flow_style=False)

print(f"差集已保存到 {diff_yaml_path}")
