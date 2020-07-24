import torch
import numpy as np
import tensorflow as tf

model_dict = torch.load(
    '/root/s0_bv2_bifpn_f64_s3x.pth',
    map_location=torch.device('cpu'),    
)

def test(torch_x, pt_class, tf_class, state_dict, *args, **kwargs):
    pt_layer = pt_class(*args, **kwargs)
    tf_layer = tf_class(*args, **kwargs)
    pt_layer.load_state_dict(state_dict)
    tf_layer.load_state_dict(state_dict)
    pt_layer.eval()
    
    data_format = kwargs.get('data_format', 'NHWC')
    if data_format == 'NHWC':
        torch_to_tf_input_permutation = [0,2,3,1] # NCHW -> NHWC
    else:
        torch_to_tf_input_permutation = [0,1,2,3] # NCHW -> NCHW

    if isinstance(torch_x, list):
        tf_x = [item.cpu().numpy().transpose(torch_to_tf_input_permutation) for item in torch_x]
    else:
        tf_x = torch_x.cpu().numpy()
        tf_x = tf_x.transpose(torch_to_tf_input_permutation)

    sess = tf.InteractiveSession()
    if isinstance(torch_x, list):
        res = np.sum(
            (
                pt_layer(*torch_x).cpu().detach().numpy().transpose(torch_to_tf_input_permutation) - 
                tf_layer(*tf_x).eval()
            ) ** 2
        )
    else:
        res = np.sum(
            (
                pt_layer(torch_x).cpu().detach().numpy().transpose(torch_to_tf_input_permutation) - 
                tf_layer(tf_x).eval()
            ) ** 2
        )
    return res

#############

# ### DepthwiseSeparableConv2d ###
# from bifpn_torch import DepthwiseSeparableConv2d as PTDSConv2d
# from pytorch_to_tf1.aliases import DepthwiseSeparableConv2d as TFDSConv2d

# in_channels = 64
# stride = 1
# kernel_size = 3

# torch_x = torch.rand((10, in_channels, 50, 60))

# prefix = 'backbone.p4_tr.0.'
# state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
# state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}

# print(
#     test(
#         torch_x,
#         PTDSConv2d,
#         TFDSConv2d,
#         state_dict,
#         # args
#         in_channels,
#         in_channels,
#         kernel_size=kernel_size,
#         bias=False,
#         padding=1,
#         stride=stride,
#         data_format='NCHW'
#     )
# )

# ### Conv3x3BnReLU ###
# from bifpn_torch import Conv3x3BnReLU as PTMod
# from pytorch_to_tf1.aliases import Conv3x3BnReLU as TFMod, Module, Sequential

# in_channels = 64

# torch_x = torch.rand((10, in_channels, 5, 5))

# prefix = 'backbone.p4_tr.'
# state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
# state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}

# print(
#     test(
#         torch_x,
#         PTMod,
#         TFMod,
#         state_dict,
#         # args + kwargs
#         in_channels,
#         data_format='NCHW'
#     )
# )

# ### FastNormalizedFusion ###
# from pytorch_to_tf1.aliases import FastNormalizedFusion as TFMod
# from bifpn_torch import FastNormalizedFusion as PTMod

# prefix = 'backbone.fuse_p5_out.'
# state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
# state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}

# in_nodes = 2
# torch_x = [torch.rand((1, 10, 50, 50)) for _ in range(in_nodes)]

# print(
#     test(
#         torch_x,
#         PTMod,
#         TFMod,
#         state_dict,
#         # args + kwargs
#         in_nodes,
#     )
# )

# #### 
# # from pytorch_to_tf1.aliases import EfficientNetFeatures as TFMod
# # import timm

# # prefix = 'backbone.bottom_up.'
# # state_dict_keys = filter(lambda x: prefix in x , model_dict['model'].keys())
# # state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}

# # pt_model = timm.create_model('spnasnet_100', pretrained=True, features_only=True, out_indices=(1, 2, 3, 4), )
# # tf_model = TFMod()

# # pt_model.load_state_dict(state_dict)
# # pt_model.eval()
# # tf_model.load_state_dict(state_dict)

# # torch_to_tf_input_permutation = [0,2,3,1]
# # torch_x = torch.rand((1, 3, 224, 224))
# # tf_x = torch_x.cpu().numpy()
# # tf_x = tf_x.transpose(torch_to_tf_input_permutation)

# # sess = tf.InteractiveSession()
# # for tf_y, pt_y in zip(tf_model(tf_x), pt_model(torch_x)):
# #     res = np.sum(
# #         (
# #             pt_y.cpu().detach().numpy().transpose(torch_to_tf_input_permutation) - 
# #             tf_y.eval()
# #         ) ** 2
# #     )
# #     print(res)

# ##### BiFPN
# from bifpn_torch import BiFPN as PTMod
# from pytorch_to_tf1.bifpn import BiFPN as TFMod

# prefix = 'backbone.'
# state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
# state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}
# # print(state_dict.keys())

# # torch_to_tf_input_permutation = [0,2,3,1]
# torch_to_tf_input_permutation = [0,1,2,3]
# torch_x = torch.rand((1, 3, 224, 224))

# tf_x = torch_x.cpu().numpy()
# tf_x = tf.convert_to_tensor(tf_x.transpose(torch_to_tf_input_permutation))
# from pytorch_to_tf1.aliases import EfficientNetFeatures
# eff = EfficientNetFeatures(data_format='NCHW')
# tf_model = TFMod(eff, 64, data_format='NCHW')
# tf_model.load_state_dict(state_dict)

# from detectron2.modeling.backbone.fpn import LastLevelMaxPool
# import timm
# spnas = timm.create_model('spnasnet_100', pretrained=True, features_only=True, out_indices=(1, 2, 3, 4), )
# pt_model = PTMod(spnas, 64, LastLevelMaxPool())
# pt_model.load_state_dict(state_dict)
# pt_model.eval()

# sess = tf.InteractiveSession()
# for idx, (tf_y, pt_y) in enumerate(zip(tf_model(tf_x), pt_model(torch_x).values())):
#     res = np.sum(
#         (
#             pt_y.cpu().detach().numpy().transpose(torch_to_tf_input_permutation) - 
#             tf_y.eval()
#         ) ** 2
#     )
#     print('p',idx+2, ' ', res)

# ### Interpolate()

# from tensorpack.models import FixedUnPooling
# from detectron2.layers.wrappers import interpolate

# torch_x = torch.rand((1, 3, 2, 2))
# tf_x = tf.convert_to_tensor(torch_x.cpu().numpy())

# sess = tf.InteractiveSession()
# tf_x_interp = FixedUnPooling(
#     'interp', tf_x, 32, unpool_mat=np.ones((32, 32), dtype='float32'),
#     data_format='channels_first'
# )
# torch_x_interp = interpolate(torch_x, size=None, scale_factor=32, mode="nearest", align_corners=None)

# res = np.sum(
#     (
#         torch_x_interp.cpu().detach().numpy() - 
#         tf_x_interp.eval()
#     ) ** 2
# )
# print(res)


# ### AveragePooling
# from torch.nn import AdaptiveAvgPool2d
# from tensorpack.models import GlobalAvgPooling

# m = AdaptiveAvgPool2d((1, 1))

# torch_x = torch.rand((1, 3, 2, 2))
# tf_x = tf.convert_to_tensor(torch_x.cpu().numpy())

# sess = tf.InteractiveSession()
# tf_x_pooled = GlobalAvgPooling('gap', tf_x, data_format='channels_first')
# torch_x_pooled = m(torch_x)
# print(tf_x_pooled)
# print(torch_x_pooled.shape)

# ### ParsingSharedBlock
# from pytorch_to_tf1.parsing_shared_block import ParsingSharedBlock as TFMod
# from densepose.modeling.shared_block import ParsingSharedBlock as PTMod
# from detectron2.config import get_cfg
# from densepose.modeling.config import add_efficientnet_config, add_roi_shared_config
# from densepose import add_densepose_config

# config_file = '/root/DensePose_ADASE/configs/s0_bv2_bifpn_f64_s3x.yaml'

# cfg = get_cfg()
# add_densepose_config(cfg)
# add_efficientnet_config(cfg)
# add_roi_shared_config(cfg)
# cfg.merge_from_file(config_file)
# cfg.freeze()

# prefix = 'roi_heads.shared_block.'
# state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
# state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}

# tf_mod = TFMod(data_format='NCHW')
# tf_mod.load_state_dict(state_dict)

# pt_mod = PTMod(cfg, 64)
# pt_mod.load_state_dict(state_dict)

# torch_x = torch.rand((100, 64, 32, 32))
# tf_x = tf.convert_to_tensor(torch_x.cpu().numpy())

# pt_y = pt_mod(torch_x)
# tf_y = tf_mod(tf_x)

# sess = tf.InteractiveSession()
# res = np.sum(
#     (
#         pt_y.cpu().detach().numpy() - 
#         tf_y.eval()
#     ) ** 2
# )
# print(res)

# ### ConvTranspose2d

# from pytorch_to_tf1.aliases import ConvTranspose2d as TFMod
# from torch.nn import ConvTranspose2d as PTMod

# prefix = 'roi_heads.densepose_predictor.v_lowres.'
# state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
# state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}

# torch_to_tf_input_permutation = [0,1,2,3] # NCHW -> NCHW
# # torch_to_tf_input_permutation = [0,2,3,1] # NCHW -> NHWC
# torch_x = torch.rand((1, 64, 50, 50))
# tf_x = torch_x.cpu().numpy()
# tf_x = tf_x.transpose(torch_to_tf_input_permutation)

# pt_layer = PTMod(64, 25, kernel_size=4, stride=2, padding=1)
# tf_layer = TFMod(64, 25, kernel_size=4, stride=2, padding=1, data_format='NCHW')
# tf_layer.load_state_dict(state_dict)
# pt_layer.load_state_dict(state_dict)

# sess = tf.InteractiveSession()
# res = np.sum(
#     (
#         pt_layer(torch_x).cpu().detach().numpy().transpose(torch_to_tf_input_permutation) - 
#         tf_layer(tf_x).eval()
#     ) ** 2
# )
# print(tf_layer(tf_x).eval().shape)
# print(res)

### DensePoseHead 
from pytorch_to_tf1.densepose_ending import DensePoseEnding as TFMod

tf_layer = TFMod(64, data_format='NCHW')

prefix = 'roi_heads.shared_block.'
state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}
tf_layer.shared_block.load_state_dict(state_dict)
prefix = 'roi_heads.densepose_predictor.'
state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}
tf_layer.densepose_predictor.load_state_dict(state_dict)

torch_x = torch.rand((100, 64, 32, 32))
tf_x = tf.convert_to_tensor(torch_x.cpu().numpy())
tf_y = tf_layer(tf_x)

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from densepose.modeling.config import add_efficientnet_config, add_roi_shared_config
from densepose import add_densepose_config

config_file = '/root/DensePose_ADASE/configs/s0_bv2_bifpn_f64_s3x.yaml'

cfg = get_cfg()
add_densepose_config(cfg)
add_efficientnet_config(cfg)
add_roi_shared_config(cfg)
cfg.merge_from_file(config_file)
cfg.freeze()


pt_model = build_model(cfg)
pt_model.load_state_dict(model_dict['model'])
pt_model.eval()
torch_y = pt_model.roi_heads.shared_block(torch_x)
torch_y = pt_model.roi_heads.densepose_predictor(torch_y)

sess = tf.InteractiveSession()
for torch_out, tf_out in zip(torch_y[0], tf_y):
    res = np.sum(
        (
            torch_out.detach().numpy() - 
            tf_out.eval()
        ) ** 2
    )
    print('####', res)