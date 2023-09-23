import numpy as np
import torch.nn as nn
from detect import model_init

print('load pytorch model ... ')
checkpoint_path = 'weights/yolov2_final.pth'
B, C = 5, 20
model = model_init(checkpoint_path, B, C)

print('convert to darknet ... ')
with open('weights/yolov2-tiny-final.weights', 'wb') as f:
    np.asarray([0, 1, 0, 12800], dtype=np.int32).tofile(f)

    for module in model.features.features:
        if isinstance(module[0], nn.Conv2d):
            conv_layer = module[0]
            if isinstance(module[1], nn.BatchNorm2d):
                bn_layer = module[1]
                # bn bias
                num_b = bn_layer.bias.numel()
                bn_b = bn_layer.bias.data.view(num_b).numpy()
                bn_b.tofile(f)
                # bn weights
                num_w = bn_layer.weight.numel()
                bn_w = bn_layer.weight.data.view(num_w).numpy()
                bn_w.tofile(f)
                # bn running mean
                num_rm = bn_layer.running_mean.numel()
                bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
                bn_rm.tofile(f)
                # bn running var
                num_rv = bn_layer.running_var.numel()
                bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
                bn_rv.tofile(f)
            else:
                # conv bias
                num_b = conv_layer.bias.numel()
                conv_b = conv_layer.bias.data.view(num_b).numpy()
                conv_b.tofile(f)
            # conv weights
            num_w = conv_layer.weight.numel()
            conv_w = conv_layer.weight.data.view(num_w).numpy()
            conv_w.tofile(f)

    # addition module
    addition_module = model.additional
    conv_layer = addition_module[0]
    if isinstance(addition_module[1], nn.BatchNorm2d):
        bn_layer = addition_module[1]
        # bn bias
        num_b = bn_layer.bias.numel()
        bn_b = bn_layer.bias.data.view(num_b).numpy()
        bn_b.tofile(f)
        # bn weights
        num_w = bn_layer.weight.numel()
        bn_w = bn_layer.weight.data.view(num_w).numpy()
        bn_w.tofile(f)
        # bn running mean
        num_rm = bn_layer.running_mean.numel()
        bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
        bn_rm.tofile(f)
        # bn running var
        num_rv = bn_layer.running_var.numel()
        bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
        bn_rv.tofile(f)
    else:
        # conv bias
        num_b = conv_layer.bias.numel()
        conv_b = conv_layer.bias.data.view(num_b).numpy()
        conv_b.tofile(f)
    # conv weights
    num_w = conv_layer.weight.numel()
    conv_w = conv_layer.weight.data.view(num_w).numpy()
    conv_w.tofile(f)

    # region_conv
    region_conv_module = model.region_conv
    # fc bias
    num_b = region_conv_module.bias.numel()
    region_conv_b = region_conv_module.bias.data.view(num_b).numpy()
    region_conv_b.tofile(f)
    # fc weights
    num_w = region_conv_module.weight.numel()
    region_conv_w = region_conv_module.weight.data.view(num_w).numpy()
    region_conv_w.tofile(f)

print('done!')