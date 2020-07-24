from .aliases import Module
from .parsing_shared_block import ParsingSharedBlock
from .densepose_predictor import DensePosePredictor

class DensePoseEnding(Module):
    def __init__(self, in_channels=64, data_format='NHWC'):
        super(DensePoseEnding, self).__init__()
        self.shared_block = ParsingSharedBlock(in_channels, data_format)
        self.densepose_predictor = DensePosePredictor(in_channels, data_format)
    
    def forward(self, x):
        x = self.shared_block(x)
        return self.densepose_predictor(x)

from tensorpack import ModelDesc
import tensorflow as tf

class DensePoseEndingTensorpack(ModelDesc):
    def __init__(self, in_channels=64, data_format='NHWC'):
        super(DensePoseEndingTensorpack, self).__init__()
        self.shared_block = ParsingSharedBlock(in_channels, data_format)
        self.densepose_predictor = DensePosePredictor(in_channels, data_format)

    def get_inference_tensor_names(self):
        out = []
        out.append('output/densepose_S')
        out.append('output/densepose_I')
        out.append('output/densepose_U')
        out.append('output/densepose_V')
        return ['features_dp'], out

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        x = inputs['features_dp']
        x = self.shared_block(x)
        ann_index, index_uv, u, v = self.densepose_predictor(x)
        tf.identity(ann_index, 'output/densepose_S')
        tf.identity(index_uv, 'output/densepose_I')
        tf.identity(u, 'output/densepose_U')
        tf.identity(v, 'output/densepose_V')

    def inputs(self):
        ret = [tf.TensorSpec((100, 64, 32, 32), tf.float32, 'features_dp')]
        return ret