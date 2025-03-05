import torch
import os

def save_quant_params_to_txt(checkpoint_path, output_txt_path):
    """
    读取保存的量化参数字典并写入到 TXT 文件
    """
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    with open(output_txt_path, 'w') as f:
        # 直接打印整个 checkpoint 结构
        f.write("=== Raw Checkpoint Data ===\n")
        f.write(f"Type: {type(checkpoint)}\n")
        f.write(f"Content: {checkpoint}\n\n")

        # 检查是否为空字典
        if isinstance(checkpoint, dict) and len(checkpoint) > 0:
            f.write("=== Parsed Quantization Parameters ===\n")
            f.write(f"Checkpoint Type: {type(checkpoint)}\n")
            
            # 获取所有键
            keys = list(checkpoint.keys())
            f.write(f"Keys: {keys}\n\n")

            # 遍历量化参数字典
            for key, value in checkpoint.items():
                f.write(f"Key: {key}\n")

                # 处理张量类型
                if isinstance(value, torch.Tensor):
                    f.write(f"  Shape: {value.shape}, Type: {value.dtype}\n")
                    f.write(f"  Values (first 10): {value.flatten().tolist()[:10]}\n\n")
                
                # 处理字典类型（嵌套字典）
                elif isinstance(value, dict):
                    f.write(f"  Nested Dictionary with {len(value)} keys\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            f.write(f"    {sub_key}: Shape: {sub_value.shape}, Type: {sub_value.dtype}\n")
                            f.write(f"      Values (first 10): {sub_value.flatten().tolist()[:10]}\n")
                        else:
                            f.write(f"    {sub_key}: Type: {type(sub_value)}, Value: {sub_value}\n")
                    f.write("\n")
                
                # 处理其他数据类型（如标量、列表、字符串等）
                else:
                    f.write(f"  Type: {type(value)}, Value: {value}\n\n")
        else:
            f.write("Checkpoint is an empty dictionary!\n")

    print(f"Quantization parameters saved to: {output_txt_path}")

# 示例使用：
checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s7_w8a8/ckpt.pth"
output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s7_w8a8/ckpt.txt"
save_quant_params_to_txt(checkpoint_path, output_txt_path)

