import yaml


def load_sensitivity_data(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data


threshold_8bit = 0


def select_quantized_layers(sensitivity_data):
    quantization_config = {}
    
    for layer, sensitivities in sensitivity_data.items():
        sensitivity_8bit = sensitivities[2]
        
        if sensitivity_8bit < threshold_8bit:
            quantization_config[layer] = 'no-quantization'  
        else:
            quantization_config[layer] = 8 
    
    return quantization_config



def save_quantization_config(quantization_config, output_yaml):
    with open(output_yaml, 'w') as file:
        for layer, quantization in quantization_config.items():
            if quantization == 8:  
                file.write(f"{layer}: 8\n")


yaml_file = '/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/sensitivity.yaml'
sensitivity_data = load_sensitivity_data(yaml_file)


quantization_config = select_quantized_layers(sensitivity_data)


output_yaml = '/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_layers.yaml'
save_quantization_config(quantization_config, output_yaml)


print(f"Quantization configuration saved to {output_yaml}.")
