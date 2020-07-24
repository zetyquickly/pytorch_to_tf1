from tensorpack.predict import PredictConfig
from tensorpack.tfutils.export import ModelExporter
from pytorch_to_tf1.densepose_ending import DensePoseEndingTensorpack
import torch
import numpy as np
import tensorflow as tf

model_dict = torch.load(
    '/root/s0_bv2_bifpn_f64_s3x.pth',
    map_location=torch.device('cpu'),    
)

MODEL = DensePoseEndingTensorpack(64, data_format='NCHW')
prefix = 'roi_heads.shared_block.'
state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}
MODEL.shared_block.load_state_dict(state_dict)
prefix = 'roi_heads.densepose_predictor.'
state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
state_dict = { k[len(prefix): ] : model_dict['model'][k] for k in state_dict_keys}
MODEL.densepose_predictor.load_state_dict(state_dict)

predcfg = PredictConfig(
    model=MODEL,
    input_names=MODEL.get_inference_tensor_names()[0],
    output_names=MODEL.get_inference_tensor_names()[1])

output_pb = './out.pb'
ModelExporter(predcfg).export_compact(output_pb, optimize=False)