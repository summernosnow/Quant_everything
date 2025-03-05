import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time # DEBUG_ONLY

from qdiff.qcnn_hardware.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough

logger = logging.getLogger(__name__)


class QuantLayer(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff', save_int=False, from_int=False):
        super(QuantLayer, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        self.save_int = save_int
        self.from_int = from_int

        if isinstance(org_module, nn.Conv2d):
            self.in_channels = org_module.in_channels  
            self.out_channels = org_module.out_channels
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride  
            self.padding = org_module.padding 
            self.dilation = org_module.dilation  
            self.groups = org_module.groups
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.in_features = org_module.in_features
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        
        if not from_int:
            self.weight = org_module.weight
        else:
            self.weight = self.register_buffer('weight',torch.tensor([]))
        # self.org_weight = org_module.weight
        if org_module.bias is not None:
            if not from_int:
                self.bias = org_module.bias
            else:
                self.bias = self.register_buffer('bias',torch.tensor([]))
            # self.org_bias = org_module.bias
        else:
            self.bias = None
            # self.org_bias = None

        # set use_quant as False, use set_quant_state to set
        self.weight_quant = False
        self.act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant

        # initialize quantizer
        if self.weight_quant_params is not None:
            self.weight_quantizer = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_params is not None:
            self.act_quantizer = ActQuantizer(self.act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):

        t_start = time.time()
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            # logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        if not self.disable_act_quant and self.act_quant:
            if self.act_quant_mode == 'qdiff':
                input = self.act_quantizer(input)

        if self.weight_quant:
            if self.save_int is False:
                # self.weight = self.weight_quantizer(self.weight, from_int=self.from_int)
                # print(1)
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight, from_int=self.from_int))
                # print(1)
                # self.weight_save = self.weight_quantizer(self.weight, from_int=self.from_int)
            else:
                self.weight, weight_int = self.weight_quantizer(self.weight, save_int=self.save_int)
                # self.weight_save = self.weight_quantizer(self.weight, from_int=self.from_int)
                # print("****************************************************************")
                # print()
            bias = self.bias

        else:
            self.weight = self.weight  # debug only, what dose the self.org_weight do?
            # self.weight_save = self.weight_quantizer(self.weight, from_int=self.from_int)
            bias = self.bias

        out = self.fwd_func(input, self.weight, bias, **self.fwd_kwargs) 
        out = self.activation_function(out)
        
        if self.weight_quant:
            if self.save_int:
                # print("MAX:",)
                self.weight = nn.Parameter(weight_int.to(torch.uint8), requires_grad=False)
                # if weight_int.max() >127:
                    # print(f"weight_int: {self.weight}")

        torch.cuda.empty_cache()  

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):  
        self.weight_quant = weight_quant
        self.act_quant = act_quant

    def get_quant_state(self):
        return self.weight_quant, self.act_quant

    def set_split(self):
        self.weight_quantizer_0 = WeightQuantizer(self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = ActQuantizer(self.act_quant_params)