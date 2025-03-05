import torch

def print_weights(pth_file_path, output_txt_file):
    # Load the .pth file
    model = torch.load(pth_file_path, map_location=torch.device('cpu'))
    
    # Open a text file to save the weights
    with open(output_txt_file, 'w') as f:
        for name, param in model.items():
            f.write(f"Layer: {name}\n")
            f.write(f"Shape: {param.shape}\n")
            f.write("Values:\n")
            f.write(f"{param}\n\n")

# Example usage
pth_file_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/yolov10s_w8_mp_int8.pt"  # Replace with your .pth file path
output_txt_file = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/yolov10s_w8_mp_int8.txt"

print(f"Weights have been saved to {output_txt_file}")
