import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch.cuda import amp
from contextlib import nullcontext
import json
import sys
import math
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params, prepare_coco_text_and_image
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer, lp_loss

from tqdm.auto import tqdm
import sys
import copy

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
import models.imagenet as customized_models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


logger = logging.getLogger(__name__)


# def LossFunction(pred, tgt, grad=None):
#         """
#         Compute the quant error: MES and the SQNR
#         """

#         # MSE Loss
#         # mse = lp_loss(pred, tgt, p=2, reduction='all')
#         pred = torch.tensor(calculatemap(pred))
#         tgt = torch.tensor(calculatemap(tgt))
#         mse = pred - tgt
        
#         # SQNR
#         # err = pred - tgt
#         # tgt = torch.sum(tgt**2)  #or tgt = torch.norm(tgt)**2
#         # err = torch.sum(err**2)  #or err = torch.norm(err)**2
#         # 二者相除
#         # divided = tgt / err
#         # 直接计算信噪比
#         # sqnr = 10*torch.log10(divided)
#         # err = torch.abs(pred - tgt)
#         err = pred - tgt
#         tgt_norm = torch.norm(tgt)
#         if tgt_norm == 0:
#             sqnr = torch.tensor(0.0)  
#         else:
#             sqnr = (err / tgt_norm) * 100

#         return mse, sqnr
        
def pad_to_match_shape(tensor1, tensor2):
    """
    对两个张量进行形状对齐，使用零填充到相同形状。
    """
    if tensor1.shape != tensor2.shape:
        max_shape = [max(s1, s2) for s1, s2 in zip(tensor1.shape, tensor2.shape)]
        # 创建填充后的张量，初始化为零
        padded_tensor1 = torch.zeros(*max_shape, dtype=tensor1.dtype).to(tensor1.device)
        padded_tensor2 = torch.zeros(*max_shape, dtype=tensor2.dtype).to(tensor2.device)

        # 根据不同的维度情况填充
        for i in range(len(tensor1.shape)):
            padded_tensor1[tuple(slice(0, min(tensor1.shape[i], max_shape[i])) for i in range(len(tensor1.shape)))] = tensor1
            padded_tensor2[tuple(slice(0, min(tensor2.shape[i], max_shape[i])) for i in range(len(tensor2.shape)))] = tensor2
        
        return padded_tensor1, padded_tensor2
    else:
        return tensor1, tensor2  # 如果形状相同，不进行对齐

def compute_sqnr(pred, tgt):
    """
    计算 SQNR (Signal-to-Quantization-Noise Ratio)
    """
    noise = pred - tgt
    signal_power = torch.sum(tgt**2).item()
    noise_power = torch.sum(noise**2).item()
    sqnr = 10 * torch.log10(torch.tensor(signal_power / noise_power)) if noise_power != 0 else float('inf')
    return sqnr

def get_pred_key(pred, i, key):
    """
    根据输入的 key，自动解析并访问多层次的属性。
    """
    if "." in key:
        key_parts = key.split(".")  # 拆分多层属性
        current_attr = getattr(pred[i], key_parts[0])  # 获取第一级属性
        for part in key_parts[1:]:  # 遍历后续的属性
            current_attr = getattr(current_attr, part)  # 获取下一级属性
        return current_attr
    else:
        return getattr(pred[i], key)  # 直接访问单层属性

def LossFunction(pred, tgt, output_format=["boxes.data", "boxes.conf", "boxes.cls"]):
    """
    计算模型输出的敏感度。
    :param pred: 模型的预测输出
    :param tgt: 模型的目标输出
    :param output_format: 人为定义的输出格式，例如 ["boxes.data", "boxes.conf", "boxes.cls"]
    """
    if isinstance(pred, torch.Tensor) and isinstance(tgt, torch.Tensor):
        # 如果是单一 tensor 输出，直接计算
        sqnr_total = compute_sqnr(pred, tgt)
    elif isinstance(pred, list) and isinstance(tgt, list):
        # 多输出处理
        assert len(pred) == len(tgt), "Pred and Tgt lists must have the same length!"
        
        # 创建用于保存拼接后的长张量的字典
        combined_pred = {key: [] for key in output_format}
        combined_tgt = {key: [] for key in output_format}
        
        # 遍历每个样本的输出
        for i in range(len(pred)):
            for key in output_format:
                # 根据人工定义输出格式来读取预测和目标
                pred_key = get_pred_key(pred, i, key)  # 读取预测
                tgt_key = get_pred_key(tgt, i, key)    # 读取目标
                
                # 对齐形状：仅在形状不一致时进行对齐
                pred_key, tgt_key = pad_to_match_shape(pred_key, tgt_key)
                
                # 将对齐后的张量加入对应列表
                combined_pred[key].append(pred_key)
                combined_tgt[key].append(tgt_key)

        # 拼接每个维度的张量并计算 SQNR
        sqnr_total = 0
        for key in output_format:
            pred_tensor = torch.cat(combined_pred[key], dim=0)
            tgt_tensor = torch.cat(combined_tgt[key], dim=0)
            sqnr_total += compute_sqnr(pred_tensor, tgt_tensor)
    else:
        raise ValueError("Unsupported output type. Must be Tensor or List.")

    return sqnr_total,sqnr_total


# def set_quant_state(model=None, weight_quant: bool = False, act_quant: bool = False):
#     for m in model.modules():
#         if isinstance(m, (QuantLayer, BaseQuantBlock)):
#             m.set_quant_state(weight_quant, act_quant)


def layer_set_quant(model=None, quantized_model: QuantModel=None, fp_model=None, weight_quant=True, act_quant=False, input_data=None, config_sqnr={}, cur_bit=0, prefix=""):
    '''
    compute the error of the output of the model with a certain layer quantized
    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {)}")
        if isinstance(module, QuantLayer):
            # if not 'ff' in full_name and not 'attn2' in full_name:
                module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                torch.cuda.empty_cache()
                logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")

                # mse_mean = 0
                # sqnr_mean = 0
                # for idx, input_data in enumerate(input_list):
                with torch.no_grad():
                    output_quant = quantized_model(input_data.cuda())
                    output_fp = fp_model(input_data.cuda())
                mse, sqnr = LossFunction(output_quant, output_fp)
                logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse*1e5), float(sqnr)))
                if full_name not in config_sqnr.keys():
                    config_sqnr[full_name] = []
                config_sqnr[full_name].append(float(sqnr))

                quantized_model.set_quant_state(False, False)
        else:
            layer_set_quant(model=module, quantized_model=quantized_model, fp_model=fp_model, weight_quant=weight_quant, act_quant=act_quant, input_data=input_data, config_sqnr=config_sqnr, cur_bit=cur_bit, prefix=full_name+".")


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="path for generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # parser.add_argument(
    #     "--model_id", type=str, required=True,
    #     default="stabilityai/sdxl-turbo",
    #     help="the model type: sdxl or sdxl-turbo"
    # )
    # parser.add_argument(
    #     "--unet_input_path", type=str, required=True,
    #     help="the input of the unet"
    # )
    # parser.add_argument(
    #     "--unet_output_path", type=str, required=True,
    #     help="the output of the unet with the weight of fp32"
    # )
    parser.add_argument(
        "--is_fp16", action="store_true", 
        help="if to use fp16 weight to inference"
    )
    parser.add_argument(
        "--sensitivity_type", type=str, required=True,
        help="weight or act"
    )
    parser.add_argument(
        "--sensitivity_path", type=str,
        help="weight or act"
    )
    # quantization configs
    parser.add_argument(
        "--config",
        type=str,
        # default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # default="/root/qdiffusion/q-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--skip_quant_act",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_weight",
        action='store_true',
    )
    parser.add_argument(
        "--template_config",
        type=str,
        help="a template to init a sensitivity config",
    )
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('-d', '--data', default='path to dataset', type=str)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        # choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=42)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    outpath = opt.base_path
    log_path = os.path.join(outpath, "run_sensitivity.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # load the config from the log path
    if opt.config is None:
        opt.config = os.path.join(opt.base_path,'config.yaml')
    if opt.ckpt is None:
        opt.ckpt = os.path.join(opt.base_path,'ckpt.pth')
    config = OmegaConf.load(f"{opt.config}")

    use_cuda = torch.cuda.is_available()
    # create model
    if opt.pretrained:
        if opt.arch == "resnet18":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.resnet18(weights="IMAGENET1K_V1")
            if use_cuda:
                model.cuda()
        elif opt.arch == "alexnet":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.alexnet(weights="IMAGENET1K_V1")
            if use_cuda:
                model.cuda()
        elif opt.arch == "vgg":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.vgg16(weights="IMAGENET1K_V1")
            if use_cuda:
                model.cuda()
        elif opt.arch == "mobilenet":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.mobilenet_v2(weights="IMAGENET1K_V1")
            if use_cuda:
                model.cuda()
        elif opt.arch == "yolov10":
            print("=> using pre-trained model '{}'".format(opt.arch))
            from ultralytics import YOLO
            model = YOLO('/root/CNN-Quantization-master/model/yolov10s.pt')
            if use_cuda:
                model.cuda()

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is False else True
    # use_act_quant = False if aq_params is False else True
    use_weight_quant = not opt.skip_quant_weight
    use_act_quant = not opt.skip_quant_act

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        aq_params['mixed_precision'] = config.mixed_precision

    fp_model = copy.deepcopy(model)
    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        # act_quant_mode="qdiff",\
        # sm_abit=config.quant.softmax.n_bits,\
    )
    qnn.cuda()
    # qnn.eval()
    logger.info(qnn)

    qnn.set_quant_state(False, False)

    # set the init flag True, otherwise will recalculate params
    qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    # TODO: load quant params
    load_quant_params(qnn, opt.ckpt)
    qnn.cuda()

    # use_weight_mp = opt.config_weight_mp is not None
    # use_act_mp = opt.config_act_mp is not None

    global best_acc
    
    ######################################################################
    # compute the sensitivity
    logger.info("quant_error_unet_output!")

    # use min-max based quantized model
    input_data= torch.load(config.calib_data.path, map_location='cpu')

    # disable the quant mode
    qnn.set_quant_state(False, False)

    # output_fp32_list = []
    logger.info("Verify the correctness")
    with torch.no_grad():
        output_quant = qnn(input_data.cuda())
        output_fp = fp_model(input_data.cuda())
    # output_quant_map = calculatemap(output_quant)
    # output_fp_map = calculatemap(output_fp)
    # print(f"output_quant_map: {output_quant_map}, output_fp_map: {output_fp_map}")
    mse, sqnr = LossFunction(output_quant, output_fp)
    logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse*1e5), float(sqnr)))


    ##################################################################################
    if opt.sensitivity_type == 'weight':
        config_sqnr = {}
        # with open(opt.template_config, 'r') as file:
        #     config_sqnr = yaml.safe_load(file)
        
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            layer_set_quant(model=qnn, quantized_model=qnn, fp_model=fp_model, weight_quant=True, act_quant=False, input_data=input_data, config_sqnr=config_sqnr, cur_bit=bit_width)
    
    elif opt.sensitivity_type == 'act':
        config_sqnr = {}
        # with open(opt.template_config, 'r') as file:
        #     config_sqnr = yaml.safe_load(file)
        
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            layer_set_quant(model=qnn, quantized_model=qnn, fp_model=fp_model, weight_quant=False, act_quant=True, input_data=input_data, config_sqnr=config_sqnr, cur_bit=bit_width)
    save_path_config = opt.base_path+'/sensitivity.yaml'
    with open(save_path_config, 'w') as file:
        yaml.dump(config_sqnr, file)

coco_img_dir = "/mnt/yixiaojie/coco/images/val2017"
coco_ann_file = "/mnt/yixiaojie/coco/annotations/instances_val2017.json"
json_file_path ="/root/yixiaojie/CNN-Quantization/calib_data/calib_coco_val2017_256_random_ids_process.json"

def coco80_to_coco91_class():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76,
            77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def calculatemap(model_outputs):
    """
    直接使用模型的输出计算mAP

    :param model_outputs: 模型输出的列表，每个元素包含[boxes, scores, class_ids]。
    :param json_file_path: 保存图片ID的JSON文件路径。
    :param coco_ann_file: COCO标注文件路径。
    :param coco_img_dir: COCO图片目录路径。
    """
    # 加载 COCO 数据集
    coco = COCO(coco_ann_file)
    with open(json_file_path, 'r') as f:
        img_ids = json.load(f)  # 读取256张图片的ID列表

    # 类别映射
    class_map = coco80_to_coco91_class()

    # 生成COCO格式的结果
    coco_results = []
    for img_id, output in zip(img_ids, model_outputs):
        boxes, scores, class_ids = [], [], []
        if output.boxes:  # 检查是否有检测框
        # 提取边界框、置信度和类别ID
            boxes = output.boxes.data # 
            scores = output.boxes.conf
            class_ids = output.boxes.cls
            print(f"  Boxes: {boxes}")
            print(f"  Scores: {scores}")
            print(f"  Class IDs: {class_ids}")
        for box, score, class_id in zip(boxes, scores, class_ids):
            result = {
                'image_id': img_id,
                'category_id': int(class_map[int(class_id.item())]),
                'bbox': [float(box[0].item()), float(box[1].item()), float((box[2] - box[0]).item()), float((box[3] - box[1]).item())],
                'score': float(score)
            }
            coco_results.append(result)

    # 计算mAP
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回第一个mAP值
    return coco_eval.stats[0]

if __name__ == "__main__":
    main()



                                            