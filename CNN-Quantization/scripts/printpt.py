# import torch

# def save_checkpoint_details_to_txt(checkpoint_path, output_txt_path):
#     """
#     将 PyTorch .pt/.pth 文件的详细信息保存到 .txt 文件
#     :param checkpoint_path: checkpoint 文件路径
#     :param output_txt_path: 输出 .txt 文件路径
#     """
#     try:
#         # 加载 checkpoint 文件
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
#         # 打开输出文件
#         with open(output_txt_path, 'w') as f:
#             # 检查 checkpoint 的类型
#             if isinstance(checkpoint, dict):
#                 f.write("\n=== Checkpoint Details ===\n")
#                 f.write(f"Type: {type(checkpoint)}\n")
                
#                 # 获取所有键
#                 keys = list(checkpoint.keys())
#                 f.write(f"Keys: {keys}\n")
                
#                 # 打印每个键的内容
#                 for key in keys:
#                     value = checkpoint[key]
#                     if isinstance(value, dict):
#                         f.write(f"\nKey: {key} -> Nested Dictionary with {len(value)} keys\n")
#                         for sub_key, sub_value in value.items():
#                             if hasattr(sub_value, 'shape'):
#                                 f.write(f"  Sub-Key: {sub_key}, Shape: {sub_value.shape}, Type: {sub_value.dtype}\n")
#                             else:
#                                 f.write(f"  Sub-Key: {sub_key}, Type: {type(sub_value)}\n")
#                     elif hasattr(value, 'shape'):  # 处理张量类型
#                         f.write(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}\n")
#                         f.write(f"  Values: {value.tolist()}\n")

#                     else:  # 处理其他类型
#                         f.write(f"Key: {key}, Type: {type(value)}, Value: {value}\n")
#             else:
#                 f.write("\nThe checkpoint is not a dictionary!\n")
#                 f.write(f"Type: {type(checkpoint)}\n")
        
#         print(f"Checkpoint details saved to {output_txt_path}")
    
#     except Exception as e:
#         print(f"Error loading checkpoint: {e}")

# # 使用示例
# # 替换为实际路径
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_model.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_model_pt.txt"
# # checkpoint_path = "/root/CNN-Quantization-master/quant_output_random/vgg_w4a8_mse/ckpt.pth"
# # output_txt_path = "/root/CNN-Quantization-master/quant_output_random/vgg_w4a8_mse/ckpt.txt"
# save_checkpoint_details_to_txt(checkpoint_path, output_txt_path)
# import torch

# def save_checkpoint_details_to_txt(checkpoint_path, output_txt_path):
#     """
#     将 PyTorch .pt/.pth 文件的详细信息保存到 .txt 文件，并打印权重的具体值
#     :param checkpoint_path: checkpoint 文件路径
#     :param output_txt_path: 输出 .txt 文件路径
#     """
#     def extract_weights(module, f, prefix=""):
#         """递归提取模块的权重并写入文件"""
#         for name, child in module.named_children():
#             child_prefix = f"{prefix}{name}."
#             extract_weights(child, f, child_prefix)
        
#         # 打印当前模块的参数
#         for param_name, param in module.named_parameters(recurse=False):
#             f.write(f"{prefix}{param_name}: Shape: {param.shape}, Type: {param.dtype}\n")
#             f.write(f"  Values: {param.tolist()}\n")

#     try:
#         # 加载 checkpoint 文件
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')

#         # 打开输出文件
#         with open(output_txt_path, 'w') as f:
#             # 检查 checkpoint 的类型
#             if isinstance(checkpoint, dict):
#                 f.write("\n=== Checkpoint Details ===\n")
#                 f.write(f"Type: {type(checkpoint)}\n")

#                 # 获取所有键
#                 keys = list(checkpoint.keys())
#                 f.write(f"Keys: {keys}\n")

#                 # 打印每个键的内容
#                 for key in keys:
#                     value = checkpoint[key]
#                     if key == "model" and hasattr(value, 'model'):
#                         f.write(f"\nKey: {key} -> Model Details\n")
#                         extract_weights(value.model, f)
#                     elif isinstance(value, dict):
#                         f.write(f"\nKey: {key} -> Nested Dictionary with {len(value)} keys\n")
#                         for sub_key, sub_value in value.items():
#                             if hasattr(sub_value, 'shape'):
#                                 f.write(f"  Sub-Key: {sub_key}, Shape: {sub_value.shape}, Type: {sub_value.dtype}\n")
#                                 f.write(f"    Values: {sub_value.tolist()}\n")
#                             else:
#                                 f.write(f"  Sub-Key: {sub_key}, Type: {type(sub_value)}, Value: {sub_value}\n")
#                     elif hasattr(value, 'shape'):  # 处理张量类型
#                         f.write(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}\n")
#                         f.write(f"  Values: {value.tolist()}\n")
#                     else:  # 处理其他类型
#                         f.write(f"Key: {key}, Type: {type(value)}, Value: {value}\n")
#             else:
#                 f.write("\nThe checkpoint is not a dictionary!\n")
#                 f.write(f"Type: {type(checkpoint)}\n")

#         print(f"Checkpoint details saved to {output_txt_path}")

#     except Exception as e:
#         print(f"Error loading checkpoint: {e}")

# # 使用示例
# # 替换为实际路径
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model_all.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model_all.txt"
# save_checkpoint_details_to_txt(checkpoint_path, output_txt_path)

import torch

def save_checkpoint_details_to_txt(checkpoint_path, output_txt_path):
    """
    将 PyTorch .pt/.pth 文件的详细信息保存到 .txt 文件，并打印权重的前 10 个具体值
    :param checkpoint_path: checkpoint 文件路径
    :param output_txt_path: 输出 .txt 文件路径
    """
    def extract_weights(module, f, prefix=""):
        """递归提取模块的权重并写入文件"""
        for name, child in module.named_children():
            child_prefix = f"{prefix}{name}."
            extract_weights(child, f, child_prefix)
        
        # 打印当前模块的参数
        for param_name, param in module.named_parameters(recurse=False):
            f.write(f"{prefix}{param_name}: Shape: {param.shape}, Type: {param.dtype}\n")
            f.write(f"  Values (first 10): {param.flatten().tolist()[:10]}\n")

    try:
        # 加载 checkpoint 文件
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 打开输出文件
        with open(output_txt_path, 'w') as f:
            # 检查 checkpoint 的类型
            if isinstance(checkpoint, dict):
                f.write("\n=== Checkpoint Details ===\n")
                f.write(f"Type: {type(checkpoint)}\n")

                # 获取所有键
                keys = list(checkpoint.keys())
                f.write(f"Keys: {keys}\n")

                # 打印每个键的内容
                for key in keys:
                    value = checkpoint[key]
                    if key == "model" and hasattr(value, 'model'):
                        f.write(f"\nKey: {key} -> Model Details\n")
                        extract_weights(value.model, f)
                    elif isinstance(value, dict):
                        f.write(f"\nKey: {key} -> Nested Dictionary with {len(value)} keys\n")
                        for sub_key, sub_value in value.items():
                            if hasattr(sub_value, 'shape'):
                                f.write(f"  Sub-Key: {sub_key}, Shape: {sub_value.shape}, Type: {sub_value.dtype}\n")
                                f.write(f"    Values (first 10): {sub_value.flatten().tolist()[:10]}\n")
                            else:
                                f.write(f"  Sub-Key: {sub_key}, Type: {type(sub_value)}, Value: {sub_value}\n")
                    elif hasattr(value, 'shape'):  # 处理张量类型
                        f.write(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}\n")
                        f.write(f"  Values (first 10): {value.flatten().tolist()[:10]}\n")
                    else:  # 处理其他类型
                        f.write(f"Key: {key}, Type: {type(value)}, Value: {value}\n")
            else:
                f.write("\nThe checkpoint is not a dictionary!\n")
                f.write(f"Type: {type(checkpoint)}\n")

        print(f"Checkpoint details saved to {output_txt_path}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

# 使用示例
# 替换为实际路径
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_model.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s_w8a8/quantized_model.txt"
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/model/yolov10s.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/model/yolov10s.txt"
checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10x1_w8a8/ckpt.pth"
output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10x1_w8a8/ckpt.txt"
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model13.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model13.txt"
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model12.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model12.txt"
# checkpoint_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/quantized_model_all.pt"
# output_txt_path = "/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s6_w8a8/quantized_model_all.txt"
save_checkpoint_details_to_txt(checkpoint_path, output_txt_path)
