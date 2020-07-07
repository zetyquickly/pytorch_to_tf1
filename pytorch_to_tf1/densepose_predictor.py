from .aliases import Module, ConvTranspose2d
import torch

class DensePosePredictor(Module):

    def __init__(self, input_channels, data_format='NCHW'):
        super(DensePosePredictor, self).__init__()
        NUM_ANN_INDICES = 15
        dim_in = input_channels
        dim_out_ann_index = NUM_ANN_INDICES
        dim_out_patches = 25
        kernel_size = 4
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, dim_out_ann_index, kernel_size, stride=2, padding=int(kernel_size / 2 - 1), data_format=data_format,
        )
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1), data_format=data_format,
        )
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1), data_format=data_format,
        )
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1), data_format=data_format,
        )

    def forward(self, head_outputs):
        ann_index_lowres = self.ann_index_lowres(head_outputs)
        index_uv_lowres = self.index_uv_lowres(head_outputs)
        u_lowres = self.u_lowres(head_outputs)
        v_lowres = self.v_lowres(head_outputs)

        return ann_index_lowres, index_uv_lowres, u_lowres, v_lowres

def get_densepose_predictor():
    model_dict = torch.load(
        '/root/s0_bv2_bifpn_f64_s3x.pth',
        map_location=torch.device('cpu'),    
    )
    prefix = 'roi_heads.densepose_predictor.'
    state_dict_keys = filter(lambda x: prefix in x, model_dict['model'].keys())
    state_dict = {k[len(prefix):]: model_dict['model'][k] for k in state_dict_keys}
    m = DensePosePredictor(input_channels=64, data_format='NCHW')
    m.load_state_dict(state_dict)
    return m