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
# from qdiff.models.quant_block import BaseQuantBlock
# from qdiff.models.quant_layer import QuantLayer
# from qdiff.models.quant_model import QuantModel
# from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
# from qdiff.utils import get_model, load_quant_params, prepare_coco_text_and_image
# from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer, lp_loss

from qdiff.qcnn_hardware.models.quant_block import BaseQuantBlock
from qdiff.qcnn_hardware.models.quant_layer import QuantLayer
from qdiff.qcnn_hardware.models.quant_model import QuantModel
from qdiff.qcnn_hardware.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.qcnn_hardware.utils import get_model, load_quant_params, prepare_coco_text_and_image

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
import os
os.environ['YOLO_VERBOSE'] = str(False)

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

def batch_inference(model, input_data, batch_size=32):
    """
    处理模型输出为自定义对象的分批推理
    :param model: 要推理的模型
    :param input_data: 输入数据
    :param batch_size: 每批大小
    :return: 拼接后的输出
    """
    outputs = []
    num_batches = (len(input_data) + batch_size - 1) // batch_size  # 向上取整
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(input_data))
        batch = input_data[start:end].cuda()

        with torch.no_grad():
            output = model(batch)
            outputs.append(output)

    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    else:
        return outputs  

def get_pred_key(pred, i, key):
    """
    根据输入的 key，自动解析并访问多层次的属性。
    """
    if "." in key:
        key_parts = key.split(".")  
        current_attr = getattr(pred[i], key_parts[0])  
        for part in key_parts[1:]:  
            current_attr = getattr(current_attr, part)  
        return current_attr
    else:
        return getattr(pred[i], key)  




def calculate_sensitivity(pred, tgt, output_format=["boxes.data", "boxes.conf", "boxes.cls"]):
    """
    计算模型输出的敏感度。
    :param pred: 模型的预测输出
    :param tgt: 模型的目标输出
    :param output_format: 人为定义的输出格式，例如 ["boxes.data", "boxes.conf", "boxes.cls"]
    """
    if isinstance(pred, torch.Tensor) and isinstance(tgt, torch.Tensor):
        sqnr_total = compute_sqnr(pred, tgt)
    elif isinstance(pred, list) and isinstance(tgt, list):
        assert len(pred) == len(tgt), "Pred and Tgt lists must have the same length!"
        
        combined_pred = {key: [] for key in output_format}
        combined_tgt = {key: [] for key in output_format}
        
        for batch_pred, batch_tgt in zip(pred, tgt):
            assert len(batch_pred) == len(batch_tgt), "Each batch must have the same length!"

            for i in range(len(batch_pred)):  # 遍历每个 batch 内的元素
                for key in output_format:
                    pred_key = get_pred_key(batch_pred, i, key)
                    tgt_key = get_pred_key(batch_tgt, i, key)

                    pred_key, tgt_key = pad_to_match_shape(pred_key, tgt_key)

                    combined_pred[key].append(pred_key)
                    combined_tgt[key].append(tgt_key)

        # for i in range(len(pred)):
        #     for key in output_format:
                
        #         pred_key = get_pred_key(pred, i, key)  
        #         tgt_key = get_pred_key(tgt, i, key)    
                
        #         pred_key, tgt_key = pad_to_match_shape(pred_key, tgt_key)
                
        #         combined_pred[key].append(pred_key)
        #         combined_tgt[key].append(tgt_key)

        sqnr_total = 0
        for key in output_format:
            pred_tensor = torch.cat(combined_pred[key], dim=0)
            tgt_tensor = torch.cat(combined_tgt[key], dim=0)
            sqnr_total += compute_sqnr(pred_tensor, tgt_tensor)
    else:
        raise ValueError("Unsupported output type. Must be Tensor or List.")

    return sqnr_total

def layer_set_quant(model=None, quantized_model: QuantModel=None, fp_model=None, weight_quant=True, act_quant=True, input_data=None, config_sqnr={}, cur_bit=0, prefix="", quantized_layers={}, batch_size=32):
    '''
    Compute the sensitivity of the model output to quantization by quantizing each layer separately.
    '''

    # First, quantize all layers recorded in quantized_layers
    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        if isinstance(module, QuantLayer):
            # Skip layers that are already in quantized_layers
            if full_name in quantized_layers.keys():
                module.set_quant_state(weight_quant=True, act_quant=True)
                # logger.info(f"{full_name} already quantized, skipping.")
                continue

    # Then, iterate through the model to quantize other layers
    for name, module in model.named_children():
        full_name = prefix + name if prefix else name

        if isinstance(module, QuantLayer):
            # Skip layers that are already in quantized_layers
            if full_name in quantized_layers.keys():
                # logger.info(f"{full_name} already quantized, skipping.")
                continue

            # Temporarily quantize the current layer
            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
            torch.cuda.empty_cache()
            # logger.info(f"Quantizing {full_name}: weight_quant={weight_quant}, act_quant={act_quant}")

            # with torch.no_grad():
            #     output_quant = quantized_model(input_data.cuda())
            #     output_fp = fp_model(input_data.cuda())
            output_quant = batch_inference(quantized_model, input_data, batch_size)
            output_fp = batch_inference(fp_model, input_data, batch_size)



            # Calculate SQNR for the current quantized layer
            sqnr = calculate_sensitivity(output_quant, output_fp)
            # logger.info('SQNR for {}: {:.5f}dB \n'.format(full_name, float(sqnr)))

            # Store SQNR result
            if full_name not in config_sqnr.keys():
                config_sqnr[full_name] = []
            config_sqnr[full_name].append(float(sqnr))

            # Revert the quantization state of the layer
            module.set_quant_state(weight_quant=False, act_quant=False)

        else:
            # Recursively process child modules
            layer_set_quant(model=module, 
                            quantized_model=quantized_model, 
                            fp_model=fp_model, 
                            weight_quant=weight_quant, 
                            act_quant=act_quant, 
                            input_data=input_data, 
                            config_sqnr=config_sqnr, 
                            cur_bit=cur_bit, 
                            prefix=full_name + ".", 
                            quantized_layers=quantized_layers,
                            batch_size=batch_size)

logger = logging.getLogger(__name__)
# Suppress YOLO model outputs
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
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
        "--save_int",
        action='store_true'
    )
    parser.add_argument('--min_bit', type=int, default=2, help='Minimum bit-width for quantization')
    parser.add_argument('--required_map', type=float, default=0.5, help='Required mAP threshold for quantization')
    parser.add_argument('--required_acc', type=float, default=0.5, help='Required acc threshold for quantization')


    parser.add_argument('-d', '--data', default='path to dataset', type=str)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        # choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    # Miscs
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    opt = parser.parse_args()

    seed_everything(opt.seed)

    outpath = opt.base_path
    import os
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
        elif opt.arch == "resnet50":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.resnet50(weights="IMAGENET1K_V1")
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
        elif opt.arch == "yolov10s":
            print("=> using pre-trained model '{}'".format(opt.arch))
            from ultralytics import YOLO
            model = YOLO('/root/yixiaojie/CNN-Quantization/model/yolov10s.pt', verbose=False)
            if use_cuda:
                model.cuda()
        elif opt.arch == "yolov10x":
            print("=> using pre-trained model '{}'".format(opt.arch))
            from ultralytics import YOLO
            model = YOLO('/root/yixiaojie/CNN-Quantization/model/yolov10x.pt', verbose=False)
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

    # logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
    calib_data = torch.load(config.calib_data.path, map_location='cpu')
    calib_data = torch.tensor(calib_data).cuda()



    # prepare data for init the model
    calib_batch_size = config.calib_data.batch_size  # DEBUG: actually for weight quant, only bs=1 is enough
    _ = model(calib_data[:calib_batch_size].cuda())
    # fp_model = copy.deepcopy(model)
    fp_model = model
    tmp_model = model
    save_qnn= QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        save_int=opt.save_int,
    )
    save_qnn.cuda()
    save_qnn.set_quant_init_done('weight')

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        save_int=opt.save_int,
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


    global best_acc
    
    ######################################################################
    # compute the sensitivity
    logger.info("quant_error_unet_output!")

    # use min-max based quantized model
    input_data = torch.load(config.calib_data.path, map_location='cpu')
    input_data = torch.tensor(input_data).cuda()

    # disable the quant mode
    qnn.set_quant_state(False, False)

    # output_fp32_list = []
    logger.info("Verify the correctness")
    # with torch.no_grad():
    #     with suppress_stdout(): 
    #         output_quant = qnn(input_data.cuda())
    #         # output_quant = copy.deepcopy(qnn)(input_data.cuda())
    #         output_fp = fp_model(input_data.cuda())
    with torch.no_grad():
        output_quant = batch_inference(qnn, input_data, batch_size=calib_batch_size)
        output_fp = batch_inference(fp_model, input_data, batch_size=calib_batch_size)
    # print(output_quant)
    # print(output_fp)
    sqnr = calculate_sensitivity(output_quant, output_fp)
    # current_map = 0.458
    # current_acc = 0.458
    # _, current_acc = test(val_loader, qnn, criterion, start_epoch, use_cuda)

    # model_file_name1 = os.path.join(opt.base_path, "quantized_model1.pt")
    # qnn.model.save(model_file_name1)
    val_results = qnn.model.val(data='coco.yaml', batch=128)
    # 获取验证阶段的AP值
    current_map = val_results.box.map  # mAP@[IoU=0.50:0.95]
    # val_results = fp_model.val(data='coco.yaml', batch=256)
    # # # 获取验证阶段的AP值
    # current_map = val_results.box.map  # mAP@[IoU=0.50:0.95]
    target_map = current_map - 0.01
    logger.info('SQNR:{:.5f}dB \n'.format(float(sqnr)))


    quantized_layers = {}

    if opt.sensitivity_type == 'weight':
        config_sqnr = {}
        # model_qnn=copy.deepcopy(qnn)

        for bit_width in [8]:
            logger.info(f"\nThe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            layer_set_quant(model=qnn, quantized_model=qnn, fp_model=fp_model, weight_quant=True, act_quant=True, input_data=input_data, config_sqnr=config_sqnr, cur_bit=bit_width, quantized_layers=quantized_layers, batch_size=calib_batch_size)

        layer_names = list(config_sqnr.keys())
        logger.info(f"Extracted layer names: {layer_names}")

    elif opt.sensitivity_type == 'act':
        config_sqnr = {}

        for bit_width in [8]:
            logger.info(f"\nThe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            layer_set_quant(model=qnn, quantized_model=qnn, fp_model=fp_model, weight_quant=False, act_quant=True, input_data=input_data, config_sqnr=config_sqnr, cur_bit=bit_width, quantized_layers=quantized_layers, batch_size=calib_batch_size)

        layer_names = list(config_sqnr.keys())
        logger.info(f"Extracted layer names: {layer_names}")

    import yaml

    sensitivity_file_name = os.path.join(opt.base_path, "sensitivity_start.yaml")
    with open(sensitivity_file_name, "w") as sensitivity_file:
        yaml.dump(config_sqnr, sensitivity_file)

    # config_sqnr = yaml.safe_load(open('/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10x_w8a8/sensitivity_start.yaml', 'r'))


    while True: 
        qnn.set_quant_state(False, False)
        # Print current status
        quantized_count = len(quantized_layers)
        logger.info(f"\n{'='*40}\nExecution Round: {len(quantized_layers)+1}\nQuantized Layers: {quantized_count}\nCurrent mAP: {current_map:.5f}\n{'='*40}\n")

        #Save sensitivity information before removing already quantized layers
        import yaml
        import os
        # 覆盖保存敏感度信息到 YAML 文件
        sensitivity_file_name = os.path.join(opt.base_path, "sensitivity_info.yaml")
        with open(sensitivity_file_name, "w") as sensitivity_file:
            yaml.dump(config_sqnr, sensitivity_file)
        logger.info(f"Saved sensitivity information to {sensitivity_file_name}")


        for quantized_layer in list(quantized_layers.keys()):
            config_sqnr.pop(quantized_layer, None)

        if not config_sqnr:
            logger.info("No layers left to quantize.")
            break

        lowest_sensitivity_layer = min(config_sqnr, key=lambda x: config_sqnr[x][-1])
        logger.info(f"Quantizing layer: {lowest_sensitivity_layer}")

        for bit in [b for b in [8, 4, 2] if b >= opt.min_bit]:
            # 使用量化配置文件执行实际量化
            bit_config = quantized_layers.copy()  # 当前的量化配置文件
            bit_config[lowest_sensitivity_layer] = bit
            # if "model.model.model.23.dfl.conv" in bit_config:
            #     del bit_config["model.model.model.23.dfl.conv"]  # 移除特定层
            # logger.info(f"Quantizing layer {lowest_sensitivity_layer} to {bit}-bit")
            qnn.set_quant_state(True, True)
            qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')
            # Create a list of layers to exclude from quantization
            module_name_list = [key for key in config_sqnr.keys() if key != lowest_sensitivity_layer]
            qnn.set_layer_quant(model=qnn, module_name_list=module_name_list, quant_level='per_channel', weight_quant=False, act_quant=False)

        
            # if opt.save_int:
            if True:

                val_results = qnn.model.val(data='/root/yixiaojie/CNN-Quantization/coco.yaml', batch=16)
                average_precision = val_results.box.map  # mAP@[IoU=0.50:0.95]
                print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = {average_precision:.3f}")
                logger.info(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = {average_precision:.3f}")

            current_map = average_precision
            logger.info(f"Current mAP after {bit}-bit quantization: {current_map}")

            if current_map < target_map:
                logger.warning(f"mAP requirement not met for {bit}-bit quantization. Stopping quantization for this layer.")
                break

        if current_map >= target_map:
            # Save the updated model
            save_qnn.set_quant_state(True, False)
            save_qnn.load_bitwidth_config(model=save_qnn, bit_config=bit_config, bit_type='weight')
            module_name_list = [key for key in config_sqnr.keys() if key != lowest_sensitivity_layer]
            save_qnn.set_layer_quant(model=save_qnn, module_name_list=module_name_list, quant_level='per_channel', weight_quant=False, act_quant=False)

            weight_set_quant(model=save_qnn)

            model_file_name = os.path.join(opt.base_path, "quantized_model.pt")
            save_qnn.model.save(model_file_name)

            logger.info(f"Saved updated model to {model_file_name}")

            quantized_layers[lowest_sensitivity_layer] = bit
            logger.info(f"Updated quantized_layers with: {quantized_layers}")
            logger.info(f"Layer {lowest_sensitivity_layer} fully quantized to {bit}-bit.")


            #Save quantized layers information to a YAML file
            quantized_layers_file_name = os.path.join(opt.base_path, "quantized_layers.yaml")  
            with open(quantized_layers_file_name, "w") as yaml_file:
                yaml.dump(quantized_layers, yaml_file)
            logger.info(f"Saved quantized layers information to {quantized_layers_file_name}")
        

            # config_sqnr = {}
            # qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')
            # qnn.set_quant_state(False, False)
            # layer_set_quant(
            #         model=qnn,
            #         quantized_model=qnn,
            #         fp_model=fp_model,
            #         weight_quant=True,
            #         act_quant=True,
            #         input_data=input_data,
            #         config_sqnr=config_sqnr,
            #         cur_bit=8,
            #         prefix="",
            #         quantized_layers=quantized_layers,
            #         batch_size=calib_batch_size        
            #     )
        else:
            logger.error("Current mAP does not meet the required threshold. Exiting loop.")
            break


    save_path_config = opt.base_path + '/quantized_layers.yaml'
    with open(save_path_config, 'w') as file:
        yaml.dump(quantized_layers, file)



def weight_set_quant(model=None, prefix=""):
    '''
    Compute the sensitivity of the model output to quantization by quantizing each layer separately.

    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name

        if isinstance(module, QuantLayer) and module.weight_quant:
            weight_int = module.weight_quantizer(module.weight)
            module.weight = nn.Parameter(weight_int)
        else:
            weight_set_quant(model=module, 
                            prefix=full_name + ".")


@torch.no_grad()
def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            with torch.no_grad():
                inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        # torch.save(inputs, "/home/fangtongcheng/base_code_test/Quant_Base_Model/pytorch-classification/calib_data/calib_data_bs200.pt")
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()



                                            




