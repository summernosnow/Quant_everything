#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import sys
import math

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


arch = "yolov10"

if arch == "resnet18":
    print("=> using pre-trained model '{}'".format(arch))
    # model = models.__dict__[args.arch](pretrained=True)
    model = models.resnet18(weights="IMAGENET1K_V1")

elif arch == "alexnet":
    print("=> using pre-trained model '{}'".format(arch))
    # model = models.__dict__[args.arch](pretrained=True)
    model = models.alexnet(weights="IMAGENET1K_V1")

elif arch == "vgg":
    print("=> using pre-trained model '{}'".format(arch))
    # model = models.__dict__[args.arch](pretrained=True)
    model = models.vgg16(weights="IMAGENET1K_V1")
elif arch == "mobilenet":
    print("=> using pre-trained model '{}'".format(arch))
    # model = models.__dict__[args.arch](pretrained=True)
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
elif arch == "yolov10":
    print("=> using pre-trained model '{}'".format(arch))
    from ultralytics import YOLO
    from ultralytics import YOLOv10
    model = YOLOv10('/root/yixiaojie/CNN-Quantization/model/yolov10s.pt')




# In[ ]:


import torch
import yaml

def print_layer_info(model):
    for name, module in model.named_modules():
        print(f"Layer: {name} | Type: {type(module).__name__}")
        if hasattr(module, 'weight'):
            print(f"Weight elements: {module.weight.numel()}")
        print("------------------------")


def count_parameters(model):
    total_params = 0
    param_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            print(name, module)
            parameter = module.weight.data
            params = parameter.numel()
            param_dict["model."+name] = params
            total_params += params

    # Find the layer with the minimum parameters
    min_params = min(param_dict.values())
    param_dict = {name: params / min_params for name, params in param_dict.items()}

    return param_dict

calib_data = torch.load("/root/yixiaojie/CNN-Quantization/calib_data/calib_coco_val2017_256_random_process.pt", map_location='cpu')
# prepare data for init the model
calib_batch_size = 32  # DEBUG: actually for weight quant, only bs=1 is enough
_ = model(calib_data[:calib_batch_size].cuda())

param_dict = count_parameters(model)

# Dump to a YAML file
with open("cnn_weight_ratio/yolov10_weight_ratio.yaml", "w") as f:
    yaml.dump(param_dict, f)

