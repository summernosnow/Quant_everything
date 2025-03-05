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

from qdiff.qcnn_hardware.models.quant_block import BaseQuantBlock
from qdiff.qcnn_hardware.models.quant_layer import QuantLayer
from qdiff.qcnn_hardware.models.quant_model import QuantModel
from qdiff.qcnn_hardware.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.qcnn_hardware.utils import get_model, load_quant_params, prepare_coco_text_and_image

from tqdm.auto import tqdm
import sys

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


def make_memory_friendly(bytes):

    MBs = bytes / (1024*1024)

    B = bytes % 1024
    bytes = bytes // 1024
    kB = bytes % 1024
    bytes = bytes // 1024
    MB = bytes % 1024
    GB = bytes // 1024

    return f"{GB} G {MB} M {B} {kB} K {B} Bytes ({MBs} MBs)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="how many batches to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to config which constructs model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
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
        "--keep_fp",
        action='store_true',
    )
    parser.add_argument(
        "--config_weight_mp",
        type=str,
        help="path for weight configs",
    )
    parser.add_argument(
        "--config_act_mp",
        type=str,
        help="path for act configs",
    )
    parser.add_argument(
        "--act_protect",
        type=str,
        help="the path for extremely sensitive acts",
    )
    parser.add_argument(
        "--save_int",
        action='store_true'
    )
    parser.add_argument(
        "--from_int",
        action='store_true'
    )
    parser.add_argument(
        "--int_ckpt_path",
        type=str,
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

    # opt.outdir = os.path.join(opt.base_path,'generated_images')
    # os.makedirs(opt.outdir, exist_ok=True)
    # outpath = opt.outdir
    log_path = os.path.join(opt.base_path, "run.log")
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

    # if opt.cfg is None:
    #     opt.cfg = config.calib_data.scale_value

    use_cuda = torch.cuda.is_available()
    # create model
    if opt.pretrained:
        if opt.arch == "resnet18":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.resnet18(weights="IMAGENET1K_V1")
        elif opt.arch == "alexnet":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.alexnet(weights="IMAGENET1K_V1")
        elif opt.arch == "vgg":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.vgg16(weights="IMAGENET1K_V1")
        elif opt.arch == "mobilenet":
            print("=> using pre-trained model '{}'".format(opt.arch))
            # model = models.__dict__[args.arch](pretrained=True)
            model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        elif opt.arch == "yolov10":
            print("=> using pre-trained model '{}'".format(opt.arch))
            from ultralytics import YOLO
            model = YOLO('/root/CNN-Quantization-master/model/yolov10s.pt')
            if use_cuda:
                model.cuda()

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

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        save_int=opt.save_int,
        from_int=opt.from_int,
    )
    qnn.cuda()
    # qnn.eval()
    logger.info(qnn)

    dtype = torch.float32 if not opt.fp16 else torch.float16
    
    qnn.set_quant_state(False, False)

    # set the init flag True, otherwise will recalculate params
    qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')
    load_quant_params(qnn, opt.ckpt, dtype=dtype, from_int=opt.from_int)
    qnn.cuda()

    model_memory = torch.cuda.memory_allocated()
    print("Static (weights) memory usage:", make_memory_friendly(model_memory))

    use_weight_mp = opt.config_weight_mp is not None
    use_act_mp = opt.config_act_mp is not None

    global best_acc

    # if not os.path.isdir(opt.checkpoint):
    #     mkdir_p(opt.checkpoint)

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(opt.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])


    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=opt.test_batch, shuffle=False,
    #     num_workers=opt.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    start_epoch = opt.start_epoch  # start from epoch 0 or last checkpoint epoch
    title = 'ImageNet-' + opt.arch
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('\nEvaluation only')


    # inference with the quantized model with 
    if use_weight_mp:
        with open(opt.config_weight_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("---------------- load the bitwidth config for weight! -------------------")
        logger.info(f"------------------ config: {opt.config_weight_mp} ---------------------")

        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

        if use_weight_mp and not use_act_mp:
            logger.info("-------- Inference with weight-only quantized and mixed precision, using fp16(optional) int4 and int8! -----------")
            if opt.keep_fp:
                if opt.arch == "resnet18":
                    qnn.set_layer_quant(model=qnn, module_name_list=["model.conv1"],  quant_level='per_layer', weight_quant=False, act_quant=False)
                elif opt.arch == "alexnet":
                    qnn.set_layer_quant(model=qnn, module_name_list=["model.features.0"],  quant_level='per_layer', weight_quant=False, act_quant=False)
                elif opt.arch == "vgg":
                    qnn.set_layer_quant(model=qnn, module_name_list=["model.features.0"],  quant_level='per_layer', weight_quant=False, act_quant=False)
                elif opt.arch == "mobilenet":
                    qnn.set_layer_quant(model=qnn, module_name_list=["model.features.1.conv.0.0"],  quant_level='per_layer', weight_quant=False, act_quant=False)
                elif opt.arch == "yolov10":
                    qnn.set_layer_quant(model=qnn, module_name_list=["model.model.model.0.conv"], quant_level='per_layer', weight_quant=False, act_quant=False)
            
        if opt.save_int:
            # for name, module in qnn.named_modules():
            #     if isinstance(module, QuantLayer) and module.weight_quant:
            #         module.weight = nn.Parameter(module.weight_quantizer(module.weight, save_int=True).to(torch.uint8), requires_grad=False)
            layer_set_quant(model=qnn)
            # torch.save(model.state_dict(), opt.int_ckpt_path)

            qnn.model.save(opt.int_ckpt_path)
        test_map = test(model) 
        print(' Test Map:  %.8f' % (test_map))
            # test_loss, test_acc = test(val_loader, qnn, criterion, start_epoch, use_cuda, opt.save_int)
            # print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

    else:
        logger.info("-------- Inference with weight-only quantization under int4/8 and fp16(optional) --------")
        # if keep_fp, then perform mixed precision quantization
        # gen_image(prompts, qnn, pipe, num_timesteps, opt)
        if opt.keep_fp:
            if opt.arch == "resnet18":
                qnn.set_layer_quant(model=qnn, module_name_list=["model.conv1"],  quant_level='per_layer', weight_quant=False, act_quant=False)
            elif opt.arch == "alexnet":
                qnn.set_layer_quant(model=qnn, module_name_list=["model.features.0"],  quant_level='per_layer', weight_quant=False, act_quant=False)
            elif opt.arch == "vgg":
                qnn.set_layer_quant(model=qnn, module_name_list=["model.features.0"],  quant_level='per_layer', weight_quant=False, act_quant=False)
            elif opt.arch == "mobilenet":
                qnn.set_layer_quant(model=qnn, module_name_list=["model.features.1.conv.0.0"],  quant_level='per_layer', weight_quant=False, act_quant=False)
            elif opt.arch == "yolov10":
                qnn.set_layer_quant(model=qnn, module_name_list=["model.model.model.0.conv"], quant_level='per_layer', weight_quant=False, act_quant=False)
        # test_loss, test_acc = test(val_loader, qnn, criterion, start_epoch, use_cuda, opt.save_int)
        # print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        # test_map = test(qnn)
        # print(' Test Map:  %.8f' % (test_map))

    if opt.save_int:
        torch.save(model.state_dict(), opt.int_ckpt_path)
    
    if(opt.from_int):
        test_map = test(model) 
        print(' Test Map:  %.8f' % (test_map))
    return None

    
        



@torch.no_grad()
def test(val_loader, model, criterion, epoch, use_cuda, save_int):
    # if save int, one epoch inference is enough

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
        if save_int:
            assert batch_idx==0, "when we save ckpt in int format, just run inference once"
            break
    bar.finish()
    return (losses.avg, top1.avg)

coco_img_dir = "/mnt/yixiaojie/coco/images/val2017"
coco_ann_file = "/mnt/yixiaojie/coco/annotations/instances_val2017.json"
# json_file_path ="/root/yixiaojie/CNN-Quantization/calib_data/calib_coco_val2017_256_random_ids.json"

def coco80_to_coco91_class():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76,
            77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def preprocess_images(image_paths, input_size=640):
    """根据输入路径批量预处理图像，返回形状为 [batch_size, 3, input_size, input_size] 的数组"""

    # 创建输入张量
    input_tensor = np.zeros((len(image_paths), 3, input_size, input_size), np.float32)
    original_shapes = []

    for index, image_path in enumerate(image_paths):
        # 读取和调整图像大小
        image = cv2.imread(image_path)
        original_shape = image.shape[:2]
        original_shapes.append(original_shape)
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

        # 将图像转换为 (C, H, W) 并存储
        input_tensor[index] = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # 直接归一化到 [0, 1]

    return input_tensor, original_shapes


def test(model, batch_size=1):

    coco = COCO(coco_ann_file)  
    img_ids = coco.getImgIds()  
    coco_results = []
    class_map = coco80_to_coco91_class()  
    processed_img_ids = []  
    print(model)

    # 获取所有图像的路径并进行批量处理
    for i in range(0, 1, batch_size):
        # 获取当前批次的图像ID
        batch_ids = img_ids[i:i + batch_size]
        processed_img_ids.extend(batch_ids)

        # 获取当前批次的图像路径
        image_paths = [os.path.join(coco_img_dir, coco.loadImgs(img_id)[0]['file_name']) for img_id in batch_ids]

        # 批量加载和预处理图像
        # batch_input, original_shapes = preprocess_images(image_paths)

        # 推理
        outputs = model(image_paths)
        print(f"Outputs: {outputs}")
        print(outputs[0].boxes)

        
        # 为每张图像和每个框生成结果
        for img_id, output in zip(batch_ids, outputs):
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
    print(f"Processed {len(processed_img_ids)} images.")
    coco_dt = coco.loadRes(coco_results)  # 加载检测结果
    coco_eval = COCOeval(coco, coco_dt, 'bbox')  # 创建评估对象
    coco_eval.params.imgIds = processed_img_ids  # 设置评估的图像ID
    coco_eval.evaluate()  # 评估
    coco_eval.accumulate()  # 累积结果
    coco_eval.summarize()  # 输出结果

    # 返回第一个mAP值
    return coco_eval.stats[0]


def layer_set_quant(model=None, prefix=""):
    '''
    Compute the sensitivity of the model output to quantization by quantizing each layer separately.

    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name

        if isinstance(module, QuantLayer) and module.weight_quant:
            # Skip layers that are already in quantized_layers

            # Temporarily quantize the current layer
            _, weight_int = module.weight_quantizer(module.weight, save_int=True)
            module.weight = nn.Parameter(weight_int.to(torch.uint8), requires_grad=False)
            print(f"weight_int: {module.weight}")
            module.weight = nn.Parameter(module.weight.data.to(torch.float32), requires_grad=False)
            print(f"Converted weight (FP32): {module.weight}")


        else:
            # Recursively process child modules
            layer_set_quant(model=module, 
                            prefix=full_name + ".")





if __name__ == "__main__":
    main()


