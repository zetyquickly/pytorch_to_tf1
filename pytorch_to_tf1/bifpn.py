
import tensorflow as tf
import numpy as np
import torch

from .aliases import Module, Conv3x3BnReLU, FastNormalizedFusion, Conv2d, EfficientNetFeatures
from tensorpack.models import FixedUnPooling, MaxPooling

def MyFixedUnPooling(name, x, shape, unpool_mat, data_format):
    if data_format == 'channels_first':
        x = tf.transpose(x, [0,2,3,1])
        shv = tf.shape(x)
        new_shape = tf.convert_to_tensor([shv[1]*2, shv[2]*2])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        # x = FixedUnPooling(name, x, shape, None, data_format='channels_last')
        x = tf.transpose(x, [0,3,1,2])
    else:
        x = FixedUnPooling(name, x, shape, unpool_mat, data_format)
    return x  

class BiFPN(Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, out_channels, top_block=None, data_format='NHWC'):
        super().__init__()

        self.data_format = data_format

        self.bottom_up = bottom_up
        self.top_block = top_block

        self.l5 = Conv2d(1152, out_channels, kernel_size=1, data_format=data_format)
        self.l4 = Conv2d(288, out_channels, kernel_size=1, data_format=data_format)
        self.l3 = Conv2d(120, out_channels, kernel_size=1, data_format=data_format)
        self.l2 = Conv2d(72, out_channels, kernel_size=1, data_format=data_format)

        self.p4_tr = Conv3x3BnReLU(out_channels, data_format=data_format)
        self.p3_tr = Conv3x3BnReLU(out_channels, data_format=data_format)

        self.fuse_p4_tr = FastNormalizedFusion(in_nodes=2)
        self.fuse_p3_tr = FastNormalizedFusion(in_nodes=2)

        self.down_p2 = Conv3x3BnReLU(out_channels, stride=2, data_format=data_format)
        self.down_p3 = Conv3x3BnReLU(out_channels, stride=2, data_format=data_format)
        self.down_p4 = Conv3x3BnReLU(out_channels, stride=2, data_format=data_format)

        self.fuse_p5_out = FastNormalizedFusion(in_nodes=2)
        self.fuse_p4_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p3_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p2_out = FastNormalizedFusion(in_nodes=2)

        self.p5_out = Conv3x3BnReLU(out_channels, data_format=data_format)
        self.p4_out = Conv3x3BnReLU(out_channels, data_format=data_format)
        self.p3_out = Conv3x3BnReLU(out_channels, data_format=data_format)
        self.p2_out = Conv3x3BnReLU(out_channels, data_format=data_format)

        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = 32
        self._out_feature_strides = {}
        for k, name in enumerate(self._out_features):
            self._out_feature_strides[name] = 2 ** (k + 2)

    # @property
    # def size_divisibility(self):
    #     return self._size_divisibility

    def forward(self, x):
        p5, p4, p3, p2 = self.bottom_up(x)  # top->down

        p5 = self.l5(p5)
        p4 = self.l4(p4)
        p3 = self.l3(p3)
        p2 = self.l2(p2)

        if self.data_format == 'NHWC':
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'

        p5_up = MyFixedUnPooling(
            'p5_up', p5, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
            data_format=data_format
        )
        p4_tr = self.p4_tr(self.fuse_p4_tr(p4, p5_up))
        p4_tr_up = MyFixedUnPooling(
            'p4_tr_up', p4_tr, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
            data_format=data_format
        )
        p3_tr = self.p3_tr(self.fuse_p3_tr(p3, p4_tr_up))
        p3_tr_up = MyFixedUnPooling(
            'p3_tr_up', p3_tr, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
            data_format=data_format
        )
        p2_out = self.p2_out(self.fuse_p2_out(p2, p3_tr_up))
        p3_out = self.p3_out(self.fuse_p3_out(p3, p3_tr, self.down_p2(p2_out)))
        p4_out = self.p4_out(self.fuse_p4_out(p4, p4_tr, self.down_p3(p3_out)))
        p5_out = self.p5_out(self.fuse_p5_out(p5, self.down_p4(p4_out)))
        p6 = MaxPooling('maxpool_p5_out', p5_out, pool_size=1, strides=2, data_format=data_format, padding='VALID')
        return (p2_out, p3_out, p4_out, p5_out, p6)

def get_bifpn():
    model_dict = torch.load(
        '/root/s0_bv2_bifpn_f64_s3x.pth',
        map_location=torch.device('cpu'),    
    )
    prefix = 'backbone.'
    state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
    state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}
    m = EfficientNetFeatures(data_format='NCHW')
    m2 = BiFPN(m, 64, data_format='NCHW')
    m2.load_state_dict(state_dict)
    return m2


if __name__ == "__main__":
    model_dict = torch.load(
        '/root/s0_bv2_bifpn_f64_s3x.pth',
        map_location=torch.device('cpu'),    
    )
    prefix = 'backbone.'
    state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
    state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}
    print(state_dict.keys())

    torch_to_tf_input_permutation = [0,2,3,1]
    torch_x = torch.rand((1, 3, 224, 224))
    tf_x = torch_x.cpu().numpy()
    tf_x = tf_x.transpose(torch_to_tf_input_permutation)
    m = EfficientNetFeatures()
    m2 = BiFPN(m, 64)
    m2.load_state_dict(state_dict)

    print(m2(tf_x))