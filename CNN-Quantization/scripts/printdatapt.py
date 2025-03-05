import torch

def save_dataset_details_to_txt(file_path, output_txt_path):
    """
    打印并保存保存在 .pt 文件中的数据详细信息，包括每个值到 txt 文件
    :param file_path: .pt 文件路径
    :param output_txt_path: 保存打印内容的 txt 文件路径
    """
    try:
        # 加载 .pt 文件
        data = torch.load(file_path, map_location='cpu')
        
        # 创建保存内容的字符串列表
        output_lines = []
        output_lines.append("\n=== Dataset Details ===")
        output_lines.append(f"Type of loaded data: {type(data)}")
        
        # 检查数据类型并生成详细内容
        if isinstance(data, torch.Tensor):
            output_lines.append(f"Data is a tensor with Shape: {data.shape}, Type: {data.dtype}")
            output_lines.append("Values:")
            output_lines.append(data.__str__())  # 打印整个张量值
        elif isinstance(data, list):
            output_lines.append(f"Data is a list with {len(data)} elements.")
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    output_lines.append(f"Element {i}: Shape: {item.shape}, Type: {item.dtype}")
                    output_lines.append("Values:")
                    output_lines.append(item.__str__())  # 打印张量值
                else:
                    output_lines.append(f"Element {i}: {item}")
        elif isinstance(data, dict):
            output_lines.append(f"Data is a dictionary with {len(data)} keys.")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    output_lines.append(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}")
                    output_lines.append("Values:")
                    output_lines.append(value.__str__())  # 打印张量值
                else:
                    output_lines.append(f"Key: {key}, Value: {value}")
        else:
            output_lines.append(f"Data is of type {type(data)}. Content: {data}")

        # 将内容保存到 txt 文件
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(output_lines))
        
        print(f"Details saved to {output_txt_path}")

    except Exception as e:
        print(f"Error loading the .pt file: {e}")

# 使用示例
# file_path = "/root/CNN-Quantization-master/calib_data/calib_data_bs512_random.pt"  # 替换为你的实际路径
# output_txt_path = "/root/CNN-Quantization-master/calib_data/calib_data_bs512_random.txt"  # 输出 txt 文件的路径
file_path = "/root/yixiaojie/CNN-Quantization/calib_data/calib_coco_val2017_256_random.pt"  # 替换为你的实际路径
output_txt_path = "/root/yixiaojie/CNN-Quantization/calib_data/calib_coco_val2017_256_random.txt"  # 输出 txt 文件的路径
save_dataset_details_to_txt(file_path, output_txt_path)

