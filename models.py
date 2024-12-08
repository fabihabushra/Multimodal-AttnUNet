import math
import numpy as np
# PyTorch
import torch
from torch import NoneType, cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import segmentation_models_pytorch as smp 
from dual_cascaded_fusion_att_unet import FusionAttU_Net
import attn_unet

def reset_function_generic(m):
    if hasattr(m,'reset_parameters'): 
        m.reset_parameters()

def get_pretrained_model(parentdir, model_type, model_to_load, encoder_depth, encoder_weights, q_order, max_shift, U_init_features,in_channels,out_channels,train_on_gpu,multi_gpu, decoder_attention = None, activation = None, unfolding = 1):
    if model_to_load == 'attn_UNet':
        model = attn_unet.UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)    
    elif model_to_load == 'FusionAttU_Net':
        # Initialize the FusionAttU_Net with appropriate parameters
        model = FusionAttU_Net(img_ch=1, output_ch=out_channels) 

    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)
    return model 