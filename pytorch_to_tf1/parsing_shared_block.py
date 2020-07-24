from .aliases import Module, Sequential, Conv2d, ReLU
import tensorflow as tf
from tensorpack.models import FixedUnPooling, GlobalAvgPooling
from .bifpn import MyFixedUnPooling
import numpy as np
import torch


class Interpolate(Module):
    def __init__(self, scale_factor, input_size, name='interp', data_format='NCHW'):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        if data_format == 'NHWC':
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channels_first'
        self.input_size = input_size
        self.name = name

    def forward(self, x):
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0,2,3,1])
            shv = [self.input_size[0], self.input_size[2], self.input_size[3], self.input_size[1]]
            new_shape = tf.convert_to_tensor([shv[1]*self.scale_factor, shv[2]*self.scale_factor])
            x = tf.image.resize_nearest_neighbor(x, new_shape)
            x = tf.transpose(x, [0,3,1,2])
            return x
            # return FixedUnPooling(
            #     self.name, x, self.scale_factor, unpool_mat=np.ones((self.scale_factor, self.scale_factor), dtype='float32'),
            #     data_format='channels_first'
            # )
        else:
            shv = self.input_size
            new_shape = tf.convert_to_tensor([shv[1]*self.scale_factor, shv[2]*self.scale_factor])
            x = tf.image.resize_nearest_neighbor(x, new_shape)
            return x

class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size, input_size, data_format='NHWC'):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        if data_format == 'NHWC':
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channels_first'
    def forward(self, x):
        x = GlobalAvgPooling('gap', x, data_format=self.data_format)
        s = self.input_size
        if self.data_format == 'channels_first':
            return tf.reshape(x, [s[0], s[1], 1, 1])
        else:
            return tf.reshape(x, [s[0], 1, 1, s[1]])
    
class ParsingSharedBlock(Module):
    def __init__(self, in_channels=64, data_format='NHWC'):
        super(ParsingSharedBlock, self).__init__()
        
        self.aspp1 = Sequential(
            Conv2d(64, 64, kernel_size=1, stride=1, data_format=data_format),
            ReLU(inplace=True), 
        )
        self.aspp2 = Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding=6, dilation=6, data_format=data_format),
            ReLU(inplace=True),
        )
        self.aspp3 = Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding=12, dilation=12, data_format=data_format),
            ReLU(inplace=True)
        )
        self.aspp4 = Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding=18, dilation=18, data_format=data_format),
            ReLU(inplace=True)
        )
        self.aspp5 = Sequential(
            AdaptiveAvgPool2d(output_size=(1, 1), input_size=[100,64,32,32], data_format=data_format),
            Conv2d(64, 64, kernel_size=1, stride=1, data_format=data_format),
            ReLU(inplace=True),
            Interpolate(scale_factor=32, input_size=[100,64,1,1], name='aspp5_32x', data_format=data_format)
        )
        self.aspp_agg = Sequential(
            Conv2d(320, 64, kernel_size=1, stride=1, data_format=data_format),
            ReLU(inplace=True)
        )
        self.conv_after_aspp_nl = Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, data_format=data_format),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, data_format=data_format),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, data_format=data_format),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, data_format=data_format),
            ReLU(inplace=True),
        )

    def forward(self, features):
        x = features
        x = tf.concat(
            [self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x), self.aspp5(x)], 
            axis=1
        )
        x = self.aspp_agg(x)
        x = self.conv_after_aspp_nl(x)
        return x


def get_shared_block():
    model_dict = torch.load(
        '/root/s0_bv2_bifpn_f64_s3x.pth',
        map_location=torch.device('cpu'),    
    )
    prefix = 'roi_heads.shared_block.'
    state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
    state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}
    m = ParsingSharedBlock(in_channels=64, data_format='NCHW')
    m.load_state_dict(state_dict)
    return m