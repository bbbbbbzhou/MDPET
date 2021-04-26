import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import SVR_DP, Unet, Discriminator
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):
    if name == 'svr_dp':
        network = SVR_DP(inshape=opts.cropsize_train,
                         nb_dpnet_features=[opts.enc1_1, opts.dec1_1, opts.dec1_2],
                         int_steps=opts.int_steps,
                         int_downsize=opts.int_downsize,
                         use_lstm=True)

    elif name == 'unet':
        network = Unet(inshape=opts.cropsize_train,
                       nb_features=[opts.enc2, opts.dec2])

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)


def get_discriminator(name, opts):
    if name == 'patchGAN':
        network = Discriminator(in_channels=1, norm_layer='IN')
    else:
        raise NotImplementedError

    return set_gpu(network, opts.gpu_ids)