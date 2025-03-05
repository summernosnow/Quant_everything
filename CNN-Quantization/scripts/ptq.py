from __future__ import print_function

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

# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import argparse, os, datetime, gc, yaml
import logging
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
import torch
import shutil
import sys
from qdiff.optimization.model_recon import model_reconstruction
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params, get_quant_calib_data
from qdiff.models.quant_block_forward_func import convert_model_split, convert_transformer_storable, set_shortcut_split

logger = logging.getLogger(__name__)


import os
# os.environ['YOLO_VERBOSE'] = str(False)

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

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/sdxl.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='use pre-trained model'
    )
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet18',
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INFO: add bakup file and bakup cfg into logpath for debug
    if os.path.exists(os.path.join(outpath,'config.yaml')):
        os.remove(os.path.join(outpath,'config.yaml'))
    shutil.copy(opt.config, os.path.join(outpath,'config.yaml'))
    if os.path.exists(os.path.join(outpath,'qdiff')): # if exist, overwrite
        shutil.rmtree(os.path.join(outpath,'qdiff'))
    shutil.copytree('/root/yixiaojie/CNN-Quantization/qdiff', os.path.join(outpath,'qdiff'))

    log_path = os.path.join(outpath, "run.log")
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

    config = OmegaConf.load(f"{opt.config}")
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
            from ultralytics import YOLOv10
            # model = YOLOv10('/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model_all.pt')
            model = YOLOv10('/root/yixiaojie/CNN-Quantization/quant_output_random/yolov10s1_w8a8/quantized_model_all.pt')
            # model = YOLOv10('/root/yixiaojie/CNN-Quantization/model/yolov10s.pt')
            if use_cuda:
                model.cuda()
        elif opt.arch == "yolov10x":
            print("=> using pre-trained model '{}'".format(opt.arch))
            from ultralytics import YOLO
            from ultralytics import YOLOv10
            model = YOLOv10('/root/yixiaojie/CNN-Quantization/model/yolov10x.pt')
            if use_cuda:
                model.cuda()
                
    # model = get_model(config.model, fp16=False, return_pipe=False)
    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is None else True
    use_act_quant = False if aq_params is None else True

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        aq_params['mixed_precision'] = config.mixed_precision

    # logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
    calib_data = torch.load(config.calib_data.path, map_location='cpu')
    calib_data = torch.tensor(calib_data).cuda()

    # prepare data for init the model
    print(type(calib_data))
    calib_batch_size = config.calib_data.batch_size  # DEBUG: actually for weight quant, only bs=1 is enough
    _ = model(calib_data[:calib_batch_size].cuda())

    # model.val(data='coco.yaml', batch=256)
    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
    )
    qnn.cuda()
    # qnn.model.eval()
    logger.info(qnn)

    if not config.quant.grad_checkpoint:
        logger.info('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)

    # # logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
    # calib_data = torch.load(config.calib_data.path, map_location='cpu')


    # # prepare data for init the model
    # calib_batch_size = config.calib_data.batch_size  # DEBUG: actually for weight quant, only bs=1 is enough


    # ----------------------- get the quant params (training-free), using the calibration data -------------------------------------
    with torch.no_grad():
        _ = qnn(calib_data[:calib_batch_size].cuda())
        qnn.set_module_name_for_quantizer(module=qnn.model)  # add the module name as attribute for each quantizer

        # --- the weight quantization -----
        qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
        pred= qnn(calib_data[:calib_batch_size].cuda())
        

        logger.info("weight initialization done!")
        qnn.set_quant_init_done('weight')
        torch.cuda.empty_cache()

        # --- the activation quantization -----
        # by default, use the running_mean of calibration data to determine activation quant params
        qnn.set_quant_state(True, True) # quantize activation with fixed quantized weight
        logger.info('Running stat for activation quantization')
        inds = np.arange(calib_data.shape[0])
        np.random.shuffle(inds)
        rounds = int(calib_data.size(0) / calib_batch_size)

        for i in trange(rounds):
            _ = qnn(calib_data[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda())
        # qnn.set_quant_init_done('activation')
        qnn.set_quant_init_done('activation')
        logger.info("activation initialization done!")
        torch.cuda.empty_cache()

    # save the quant params
    logger.info("Saving calibrated quantized CNN model")
    quant_params_dict = qnn.get_quant_params_dict()
    print(quant_params_dict)
    torch.save(quant_params_dict, os.path.join(outpath, "ckpt.pth"))

if __name__ == "__main__":
    main()                                        
