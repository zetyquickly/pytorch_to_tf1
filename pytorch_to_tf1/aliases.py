import torch # == 1.4.0
import tensorflow as tf # == 1.13.0
import numpy as np


class Module():
    def __init__(self):
        self.property_setters = {}

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *x, **kwargs):
        return self.forward(*x, **kwargs)

    def to(self, *args):
        return self

    def eval(self, *args):
        return self

    def train(self, *args):
        return self

    def get_full_module_name(self, modules, full_name):
        properties = vars(self)
        for name in properties:
            module = properties[name]
            new_full_name = full_name + '.%s' % name if full_name != '' else name
            if isinstance(module, Module):
                module.get_full_module_name(modules, new_full_name)
            else:
                modules.append(new_full_name)
                
    def _load_from_state_dict(self, state_dict, full_name):
        # print(full_name)
        properties = vars(self)
        for name in properties:
            module = properties[name]
            new_full_name = full_name + '.%s' % name if full_name != '' else name
            if isinstance(module, Module):
                module._load_from_state_dict(state_dict, new_full_name)
            #else:
            #    if new_full_name in state_dict:
            #        print("set property %s:%s" % (new_full_name,state_dict[new_full_name].shape))
            #        self.property_setters[name](state_dict[new_full_name].numpy())
        for module in self.property_setters:
            new_full_name = full_name + '.%s' % module if full_name != '' else module
            if new_full_name in state_dict:
                print("set property %s:%s" % (new_full_name, state_dict[new_full_name].shape))
                self.property_setters[module](state_dict[new_full_name].numpy())

    def load_state_dict(self, state_dict, strict=True):
        print("load state_dict")
        # print(state_dict.keys())
        self._load_from_state_dict(state_dict, '')

class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_shape, stride=1, padding=0, dilation=1, groups=1, bias=True, data_format='NHWC', name=None):

        if isinstance(padding, int):
            padding = dilation * (kernel_size - 1) - padding
            padding = [[0,0],[padding,padding],[padding,padding],[0,0]]
        else:
            raise NotImplementedError("Non int values is not supported")
        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        if isinstance(stride, int):
            stride = (stride,stride)
        if isinstance(dilation, int):
            dilation = (dilation,dilation)

        assert groups == 1 or (groups == in_channels), "Not implemented not equals groups"

        self.bias = bias
        self.groups = groups
        self.dilation = [1,*dilation,1]

        self.padding = padding
        self.stride = [1,*stride,1]
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.name = name
        self.data_format = data_format
        if self.data_format == 'NCHW':
            order = [0,3,1,2]
            self.stride = [self.stride[i] for i in order]
            self.padding = [self.padding[i] for i in order]
            self.dilation = [self.dilation[i] for i in order]

        self.output_shape = output_shape
        self.property_setters = {'weight': self.set_torch_weight, 'bias': self.set_torch_bias}

    def set_torch_weight(self,weight):
        [I, O, H, W] = weight.shape
        new_filter_shape = [H, W, O, I]
        filter_permutation = [2, 3, 1, 0]
        tf_weight = weight.transpose(filter_permutation)

        #assertion
        new_shape = tf_weight.shape
        assert self.kernel_size[0] == new_shape[0], "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))
        assert self.kernel_size[1] == new_shape[1],  "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))
        assert self.out_channels == new_shape[2],  "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))

        self.kernel = tf_weight

    def set_torch_bias(self,bias):
        self.bias_value = bias

    def is_same_pad(self):
        return self.kernel_size[0] // 2 == self.padding[2][0] and self.stride[2] == 1

    def forward(self,x):
        
        padding = 'SAME'# if self.is_same_pad() else 'VALID'

        result = tf.nn.conv2d_transpose(
            x, filter=self.kernel, output_shape=self.output_shape,
            strides=self.stride, padding=padding,
            data_format=self.data_format
        )
        
        if self.bias:
            result = tf.nn.bias_add(result, self.bias_value, data_format=self.data_format, name=self.name)# result + self.bias_value
        return result


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, data_format='NHWC', name=None):

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        if isinstance(stride, int):
            stride = (stride,stride)

        if isinstance(padding, int):
            padding = [[0,0],[padding,padding],[padding,padding],[0,0]]
        else:
            raise NotImplementedError("Non int values is not supported")

        if isinstance(dilation, int):
            dilation = (dilation,dilation)

        assert groups == 1 or (groups == in_channels), "Not implemented not equals groups"


        self.bias = bias
        self.groups = groups
        self.dilation = [1,*dilation,1]


        self.padding = padding
        self.stride = [1,*stride,1]
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.name = name
        self.data_format = data_format
        if self.data_format == 'NCHW':
            order = [0,3,1,2]
            self.stride = [self.stride[i] for i in order]
            self.padding = [self.padding[i] for i in order]
            self.dilation = [self.dilation[i] for i in order]

        self.property_setters = {'weight': self.set_torch_weight, 'bias': self.set_torch_bias}

    def set_torch_weight(self,weight):
        [O, I, H, W] = weight.shape
        new_filter_shape = [H, W, I, O]
        filter_permutation = [2, 3, 1, 0]
        tf_weight = weight.transpose(filter_permutation)

        #assertion
        new_shape = tf_weight.shape
        assert self.kernel_size[0] == new_shape[0], "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))
        assert self.kernel_size[1] == new_shape[1],  "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))
        assert self.in_channels // self.groups == new_shape[2],  "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))
        assert self.out_channels == new_shape[3],  "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups,self.out_channels]))

        if self.groups > 1:
            print("ATTENTION!!! use depthwise")
            tf_weight = tf_weight.transpose([0,1,3,2])
        self.kernel = tf_weight

    def set_torch_bias(self,bias):
        self.bias_value = bias

    def is_same_pad(self):
        return self.kernel_size[0] // 2 == self.padding[2][0] and self.stride[2] == 1

    def forward(self,x):

        if not self.is_same_pad():
            x = tf.pad(x, self.padding)

        padding = 'SAME' if self.is_same_pad() else 'VALID'

        if self.groups == 1:
            result = tf.nn.conv2d(x, filter=self.kernel, strides=self.stride, padding=padding, dilations=self.dilation, data_format=self.data_format)
        else:
            result = tf.nn.depthwise_conv2d(x, filter=self.kernel, strides=self.stride, padding=padding, data_format=self.data_format)


        if self.bias:
            result = tf.nn.bias_add(result, self.bias_value, data_format=self.data_format, name=self.name)# result + self.bias_value
        return result


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.stride = stride or kernel_size
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        self.padding = padding

        assert dilation == 1, 'dilation not equals to 1 is not supported'

    def forward(self, input):

        if self.padding[1][0] > 0:
            input = tf.pad(input, self.padding)

        return tf.layers.max_pooling2d(input, pool_size=self.kernel_size, strides=self.stride)


class ReLU(Module):
    def __init__(self,*args, **kwargs):
        super().__init__()

    def forward(self, x):
        return tf.nn.relu(x)


class ReLU6(Module):
    def __init__(self,*args, **kwargs):
        super().__init__()

    def forward(self, x):
        return tf.nn.relu6(x)


class Sequential(Module):
    def __init__(self, *args):
        self.modules = list(args)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def _load_from_state_dict(self, state_dict, full_name):
        for i, module in enumerate(self.modules):
            new_full_name = full_name + '.%s' % str(i) if full_name != '' else str(i)
            module._load_from_state_dict(state_dict, new_full_name)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, type="custom", data_format='NHWC'):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.data_format = data_format
        assert self.affine == True, 'affine=False parameter is not supported'

        self.property_setters = {'weight': self.set_weight, 'bias': self.set_bias, 'running_mean': self.set_running_mean,
                                 'running_var': self.set_running_var, 'num_batches_tracked': lambda x: None}
        self.type = type

    def set_weight(self, weight):
        self.weight = weight

    def set_bias(self, bias):
        self.bias = bias

    def set_running_mean(self, mean):
        self.running_mean = mean

    def set_running_var(self, running_var):
        self.running_var = running_var

    @staticmethod
    def batch_normalization(x,
                            mean,
                            variance,
                            offset,
                            scale,
                            variance_epsilon,
                            name=None):
        inv = 1 / np.sqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale
        return x * inv + (offset - mean * inv) if offset is not None else -mean * inv

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.type == "vanila" and self.data_format == "NHWC":
            return tf.nn.batch_normalization(input, mean=self.running_mean, variance=self.running_var, scale=self.weight, offset=self.bias, variance_epsilon=self.eps)
        elif self.type == "custom" and self.data_format == "NHWC":
            return self.batch_normalization(input, mean=self.running_mean, variance=self.running_var, scale=self.weight, offset=self.bias, variance_epsilon=self.eps)
        elif self.type == "fused" or self.data_format == "NCHW":
            y, running_mean, running_var = tf.nn.fused_batch_norm(input, mean=self.running_mean, variance=self.running_var, scale=self.weight, offset=self.bias, epsilon=self.eps, is_training=False, data_format=self.data_format)
            return y
        else:
            raise NotImplementedError("BathNorm2d with type: {type},  data_format: {data_format}".format(type=self.type, data_format=self.data_format))

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, name=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.name = name
        self.property_setters = {'weight': self.set_torch_weight, 'bias': self.set_torch_bias}
        
    def set_torch_weight(self,weight):
        [O, I] = weight.shape
        new_filter_shape = [I, O]
        filter_permutation = [1, 0]
        tf_weight = weight.transpose(filter_permutation)

        #assertion
        new_shape = tf_weight.shape
        assert self.in_features == new_shape[0], "%s vs %s" % (self.in_features, new_shape[0])
        assert self.out_features == new_shape[1],  "%s vs %s" % (self.out_features, new_shape[1])
 
        self.kernel = tf_weight

    def set_torch_bias(self,bias):
        self.bias_value = bias
    
    def forward(self,x):
        x = tf.matmul(x, self.kernel)
        if self.bias:
            x = tf.add(x, self.bias_value, name=self.name)# result + self.bias_value
        return x


class Conv3x3BnReLU(Sequential):
    def __init__(self, in_channels, stride=1, data_format='NHWC'):
        conv = DepthwiseSeparableConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            bias=False,
            padding=1,
            stride=stride,
            data_format=data_format,
        )
        bn = BatchNorm2d(in_channels, momentum=0.03, type='fused', data_format=data_format)
        relu = ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class FastNormalizedFusion(Module):

    def __init__(self, in_nodes):
        self.in_nodes = in_nodes
        self.property_setters = {
            'weight': self.set_weight,
            'eps': self.set_eps,
        }

    def set_weight(self, weight):
        self.weight = np.maximum(weight, 0)
        self.weight_list = self.weight.tolist()

    def set_eps(self, eps):
        self.eps = eps
        self.divisor = np.sum(self.weight) + self.eps

    def forward(self, *x):
        if len(x) != self.in_nodes:
            raise RuntimeError(
                "Expected to have {} input nodes, but have {}.".format(self.in_nodes, len(x))
            )
        
        weighted_xs = [
            xi * wi for xi, wi in zip(x, self.weight_list)
        ]
        normalized_weighted_x = sum(weighted_xs) / self.divisor
        return normalized_weighted_x


class DepthwiseSeparableConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        name=None,
        data_format='NHWC',
    ):

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        if isinstance(stride, int):
            stride = (stride,stride)
        if isinstance(padding, int):
            padding = [[0,0],[padding,padding],[padding,padding],[0,0]]
        else:
            raise NotImplementedError("Non int values is not supported")
        if isinstance(dilation, int):
            if dilation == 1:
                dilation = (dilation, dilation)
            else:
                raise NotImplementedError("Dilation is not supported")

        self.bias = bias
        self.dilation = [1, *dilation, 1]
        self.padding = padding
        self.stride = [1, *stride, 1]
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.data_format = data_format
        if self.data_format == 'NCHW':
            order = [0,3,1,2]
            self.stride = [self.stride[i] for i in order]
            self.padding = [self.padding[i] for i in order]
            self.dilation = [self.dilation[i] for i in order]
        self.property_setters = {
            '0.weight': self.set_depthwise_filter,
            '1.weight': self.set_pointwise_filter,
            '1.bias': self.set_torch_bias,
        }

    def set_depthwise_filter(self, weight):
        # PyTorch weights := [O, I / groups, kH, kW], groups := I
        # O := (I / groups) * channel_multiplier
        # TF depthwise_filter := [kH, kW, I, channel_multiplier]
        self.channel_multiplier = weight.shape[0]
        filter_permutation = [2, 3, 0, 1]
        tf_weight = weight.transpose(filter_permutation)

        # assertions
        # new_shape = tf_weight.shape
        # assert self.kernel_size[0] == new_shape[0], "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups, self.out_channels]))
        # assert self.kernel_size[1] == new_shape[1], "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups, self.out_channels]))
        # assert self.out_channels == new_shape[3], "%s vs %s" % (str(new_shape), str([self.kernel_size[0],self.kernel_size[1],self.in_channels// self.groups, self.out_channels]))
        self.depthwise_filter = tf_weight

    def set_pointwise_filter(self, weight):
        # PyTorch weights := [O, I, kH, kW]
        # TF pointwise_filter := [1, 1, channel_multiplier * I, O]
        filter_permutation = [2, 3, 1, 0]
        tf_weight = weight.transpose(filter_permutation)

        self.pointwise_filter = tf_weight

    def set_torch_bias(self, bias):
        self.bias_value = bias

    def is_same_pad(self):
        return self.kernel_size[0] // 2 == self.padding[2][0] and self.stride[2] == 1

    def forward(self, x):
        if not self.is_same_pad():
            x = tf.pad(x, self.padding)
        padding = 'SAME' if self.is_same_pad() else 'VALID'
        # result = tf.nn.separable_conv2d(
        #     x,
        #     self.depthwise_filter,
        #     self.pointwise_filter,
        #     strides=self.stride,
        #     padding=padding,
        #     data_format='NCHW',
        # )

        x = tf.nn.depthwise_conv2d(
            x, filter=self.depthwise_filter, strides=self.stride, padding=padding, 
            data_format=self.data_format
        )
        result = tf.nn.conv2d(
            x, filter=self.pointwise_filter, strides=[1, 1, 1, 1], 
            padding="VALID", dilations=self.dilation, 
            data_format=self.data_format
        )
        
        if self.bias:
            result = tf.nn.bias_add(result, self.bias_value, data_format=self.data_format, name=self.name)# result + self.bias_value
        return result


class DepthwiseSeparableConv(Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3,
            stride=1, dilation=1, pad_type=1, noskip=False,
            pw_kernel_size=1, data_format='NHWC'
        ): 

        super(DepthwiseSeparableConv, self).__init__()
        self.mid_chs = None
        self.conv_dw = Conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, 
            dilation=dilation, padding=pad_type, groups=in_chs, bias=False, data_format=data_format
        )
        self.bn1 = BatchNorm2d(in_chs, data_format=data_format)
        self.act1 = ReLU(inplace=True)

        self.conv_pw = Conv2d(in_chs, out_chs, pw_kernel_size, bias=False, data_format=data_format)
        self.bn2 = BatchNorm2d(out_chs, data_format=data_format)

        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
    
    def forward(self, x):
        residual = x
        # print('r ', residual.shape)

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_pw(x)
        x = self.bn2(x)

        # print('x ', x.shape)
        if self.has_residual:
            x = tf.add(x, residual)
        return x


class InvertedResidual(Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3,
            stride=1, dilation=1, pad_type=1, noskip=False,
            exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
            data_format='NHWC',
        ):
        super(InvertedResidual, self).__init__()

        def make_divisible(v, divisor=8, min_value=None):
            min_value = min_value or divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
    
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.mid_chs = mid_chs

        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = Conv2d(in_chs, mid_chs, exp_kernel_size, bias=False, data_format=data_format)
        self.bn1 = BatchNorm2d(mid_chs, data_format=data_format)
        self.act1 = ReLU(inplace=True)

        # Depth-wise convolution
        self.conv_dw = Conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, groups=mid_chs, bias=False, data_format=data_format,
        )
        self.bn2 = BatchNorm2d(mid_chs, data_format=data_format)
        self.act2 = ReLU(inplace=True)

        # Point-wise linear projection
        self.conv_pwl = Conv2d(mid_chs, out_chs, pw_kernel_size, bias=False, data_format=data_format)
        self.bn3 = BatchNorm2d(out_chs, data_format=data_format)

    def forward(self, x, feature=False):
        residual = x
        # print('r ', residual.shape)

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        self.feature = x

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        # print('x ', x.shape)

        if self.has_residual:
            x = tf.add(x, residual)

        return x


            
class EfficientNetFeatures(Module):
    def __init__(self, data_format='NHWC'):
        super(EfficientNetFeatures, self).__init__()
        print(data_format)
        self.conv_stem = Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1, bias=False, data_format=data_format)
        self.bn1 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, data_format=data_format)
        self.act1 = ReLU(inplace=True)
        self.blocks = Sequential(
            # 0
            Sequential(
                DepthwiseSeparableConv(
                    32, 16, dw_kernel_size=3,
                    stride=1, dilation=1, pad_type=1, noskip=True,
                    pw_kernel_size=1, data_format=data_format
                )
            ),
            # 1
            Sequential(*(
                [
                    InvertedResidual(
                        16, 24, dw_kernel_size=3,
                        stride=2, dilation=1, pad_type=1, noskip=False,
                        exp_ratio=3.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    )
                ] + [
                    InvertedResidual(
                        24, 24, dw_kernel_size=3,
                        stride=1, dilation=1, pad_type=1, noskip=False,
                        exp_ratio=3.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) for _ in range(2)
                ]
            )),
            # 2
            Sequential(
                *([
                    InvertedResidual(
                        24, 40, dw_kernel_size=5,
                        stride=2, dilation=1, pad_type=2, noskip=False,
                        exp_ratio=6.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) 
                ] + [
                    InvertedResidual(
                        40, 40, dw_kernel_size=3,
                        stride=1, dilation=1, pad_type=1, noskip=False,
                        exp_ratio=3.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) for _ in range(3)
                ])
            ),
            # 3
            Sequential(
                *([
                    InvertedResidual(
                        40, 80, dw_kernel_size=5,
                        stride=2, dilation=1, pad_type=2, noskip=False,
                        exp_ratio=6.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) 
                ] + [
                    InvertedResidual(
                        80, 80, dw_kernel_size=3,
                        stride=1, dilation=1, pad_type=1, noskip=False,
                        exp_ratio=3.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) for _ in range(3)
                ])
            ),
            # 4
            Sequential(
                *([
                    InvertedResidual(
                        80, 96, dw_kernel_size=5,
                        stride=1, dilation=1, pad_type=2, noskip=False,
                        exp_ratio=6.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) 
                ] + [
                    InvertedResidual(
                        96, 96, dw_kernel_size=5,
                        stride=1, dilation=1, pad_type=2, noskip=False,
                        exp_ratio=3.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) for _ in range(3)
                ])
            ),
            # 5
            Sequential(
                *([
                    InvertedResidual(
                        96, 192, dw_kernel_size=5,
                        stride=2, dilation=1, pad_type=2, noskip=False,
                        exp_ratio=6.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    )
                ] + [
                    InvertedResidual(
                        192, 192, dw_kernel_size=5,
                        stride=1, dilation=1, pad_type=2, noskip=False,
                        exp_ratio=6.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                    ) for _ in range(3)
                ])
            ),
            # 6
            Sequential(
                InvertedResidual(
                    192, 320, dw_kernel_size=3,
                    stride=1, dilation=1, pad_type=1, noskip=True,
                    exp_ratio=6.0, exp_kernel_size=1, pw_kernel_size=1, data_format=data_format,
                )
            ),
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        features = []
        out_blocks = {(1,2), (2,3), (4,3), (6,0)}
        for i, stage in enumerate(self.blocks.modules):
            for j, b in enumerate(stage.modules):
                # print(i, j, b.has_residual, b.mid_chs)
                x = b(x)
                if (i, j) in out_blocks:
                    features.insert(0, b.feature)
        return features