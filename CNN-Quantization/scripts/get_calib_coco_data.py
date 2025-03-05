import torch
import random
from pycocotools.coco import COCO
import os
import json
import argparse
import cv2
import numpy as np

def preprocess_images(image_paths, input_size=640):
    """根据输入路径批量预处理图像，返回形状为 [batch_size, 3, input_size, input_size] 的数组"""

    # 创建输入张量
    input_tensor = np.zeros((len(image_paths), 3, input_size, input_size), np.float32)
    original_shapes = []

    for index, image_path in enumerate(image_paths):
        # 读取和调整图像大小#####修改
        image_ori = cv2.imread(image_path)
        ratio = min((640 / image_ori.shape[0]),(640 / image_ori.shape[1]))
        in_w = int(image_ori.shape[0] * ratio)
        in_h = int(image_ori.shape[1] * ratio)
        im_temp = cv2.resize(image_ori, (in_h, in_w), interpolation=cv2.INTER_CUBIC)
        top, bottom, left, right = 0,(640-im_temp.shape[0]),0,(640-im_temp.shape[1])
        original_shape = image_ori.shape[:2]
        original_shapes.append(original_shape)
        image = cv2.copyMakeBorder(im_temp, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])#cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        # 读取和调整图像大小#####修改
        # 将图像转换为 (C, H, W) 并存储
        input_tensor[index] = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # 直接归一化到 [0, 1]

    return input_tensor, original_shapes

def main():
    parser = argparse.ArgumentParser(description="Generate calibration data from COCO dataset")
    parser.add_argument("--d", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save calibration data")
    parser.add_argument("--calib_images", type=int, default=200, help="Number of calibration images to select")
    args = parser.parse_args()

    # 设置 COCO 数据集的路径
    data_dir = args.d
    ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')
    image_dir = os.path.join(data_dir, 'images/val2017')

    # 加载 COCO 数据集
    coco = COCO(ann_file)

    # 获取所有的图片ID
    image_ids = coco.getImgIds()

    # 随机选择指定数量的图片
    random_image_ids = random.sample(image_ids, args.calib_images)

    # 存储选中的图片路径
    image_paths = []

    # 加载图片路径
    for img_id in random_image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        image_paths.append(img_path)

    calib_images, original_shapes = preprocess_images(image_paths, input_size=640)

    # 确保输出目录存在
    os.makedirs(args.outdir, exist_ok=True)

    # 保存图片路径到 .pt 文件
    calib_data_path = os.path.join(args.outdir, f"calib_coco_val2017_{args.calib_images}_random.pt")
    torch.save(calib_images, calib_data_path)

    # 保存图片 ID 到 JSON 文件
    id_file_path = os.path.join(args.outdir, f"calib_coco_val2017_{args.calib_images}_random_ids.json")
    with open(id_file_path, 'w') as id_file:
        json.dump(random_image_ids, id_file)

    print(f"{args.calib_images} 张图片路径已保存到 {calib_data_path}")
    print(f"对应的图片 ID 已保存到 {id_file_path}")

if __name__ == "__main__":
    main()
