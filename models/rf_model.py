from typing import Tuple

import math
import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from einops import rearrange

from models.rf_eval import RFMetricsRecorder


class RotatingAutoEncoder(nn.Module):
    def __init__(self,
    img_resolution: Tuple[int, int],
    n_in_channels: int,
    enc_n_out_channels: str,
    enc_strides: str,
    enc_kernel_sizes: str,
    use_linear_layer: bool,
    d_linear: int,
    d_rotating: int,
    decoder_type: str,
    use_out_conv: bool,
    use_out_sigmoid: bool,
    norm_layer_type: str,
    rotating_mask_threshold: float,
    n_images_to_log: int,
    plot_resize_resolution: int,
    seed: int,
    ):
        super(RotatingAutoEncoder, self).__init__()

        # The below 2 options are saved for the evaluation
        self.img_resolution = img_resolution
        self.n_img_channels = n_in_channels
        self.seed = seed

        enc_n_out_channels = [int(i) for i in enc_n_out_channels.split(',')]
        self.enc_n_out_channels = enc_n_out_channels
        enc_strides = [int(i) for i in enc_strides.split(',')]
        enc_kernel_sizes = [int(i) for i in enc_kernel_sizes.split(',')]
        assert len(enc_n_out_channels) == len(enc_strides), f'Encoder output channels and strides lists need to have the same length, got {enc_n_out_channels} and {enc_strides}'
        assert len(enc_n_out_channels) == len(enc_kernel_sizes), f'Encoder output channels and strides lists need to have the same length, got {enc_n_out_channels} and {enc_kernel_sizes}'
        self.n_in_channels = n_in_channels
        self.d_rotating = d_rotating
        self.decoder_type = decoder_type
        self.use_out_conv = use_out_conv
        self.use_out_sigmoid = use_out_sigmoid
        self.norm_layer_type = norm_layer_type
        self.rotating_mask_threshold = rotating_mask_threshold
        self.n_images_to_log = n_images_to_log
        self.plot_resize_resolution = (plot_resize_resolution, plot_resize_resolution)
        self.use_linear_layer = use_linear_layer
        self.dec_in_resolution = (
            math.ceil(img_resolution[0]/math.prod(enc_strides)), 
            math.ceil(img_resolution[1]/math.prod(enc_strides))
            )
        
        self.encoder = RotatingEncoder(
            img_resolution=img_resolution,
            n_in_channels=self.n_in_channels,
            enc_n_out_channels=enc_n_out_channels,
            enc_strides=enc_strides,
            enc_kernel_sizes=enc_kernel_sizes,
            d_rotating=d_rotating,
            use_linear_layer=use_linear_layer,
            d_linear=d_linear,
            norm_layer_type=norm_layer_type,
        )
        self.decoder = RotatingConvDecoder(
            img_resolution=self.dec_in_resolution,
            decoder_type=self.decoder_type,
            n_in_channels=n_in_channels,
            enc_n_out_channels=enc_n_out_channels,
            enc_strides=enc_strides,
            enc_kernel_sizes=enc_kernel_sizes,
            d_rotating=d_rotating,
            use_linear_layer=use_linear_layer,
            d_linear=d_linear,
            enc_output_res=self.encoder.enc_output_res,
            norm_layer_type=norm_layer_type,
        )
            
        if self.use_out_conv:
            self.output_weight = nn.Parameter(torch.empty(n_in_channels))
            self.output_bias = nn.Parameter(torch.empty(1, n_in_channels, 1, 1))
            nn.init.constant_(self.output_weight, 1)
            nn.init.constant_(self.output_bias, 0)
    
    def get_metrics_recorder(self):
        return RFMetricsRecorder()

    def _prepare_input(self, input_images):
        bsz, c, h, w = input_images.size()
        input_images = rearrange(input_images, 'b c h w -> b 1 c h w')
        rotating_in = torch.zeros(
            bsz, self.d_rotating-1, c, h, w, device=input_images.device
        )
        inputs = torch.cat([input_images, rotating_in], dim=1)
        return inputs

    def _apply_module(self, z, module, channel_norm):
        m, phi = module(z)
        z = self._apply_activation_function(m, phi, channel_norm)
        return z

    def _apply_activation_function(self, m_bind, phi, channel_norm):
        m_bind = channel_norm(m_bind)
        m_out = torch.nn.functional.relu(m_bind)
        z_out = m_out[:, None] * torch.nn.functional.normalize(phi, dim=1)
        return z_out

    def _apply_conv_layers(self, model, z):
        layer_activation_maps = []
        for idx, _ in enumerate(model.conv_layers):
            z = self._apply_module(z, model.conv_layers[idx], model.conv_norm_layers[idx])
            layer_activation_maps.append(z)
        return z, layer_activation_maps

    def encode(self, x):
        self.batch_size = x.size()[0]
        z, enc_layer_activation_maps = self._apply_conv_layers(self.encoder, x)
        if self.encoder.linear_output_layer is not None:
            z = rearrange(z, "b r c h w -> b r (c h w)")
            z = self._apply_module(z, self.encoder.linear_output_layer, self.encoder.linear_output_norm)
        return z, enc_layer_activation_maps

    def decode(self, z):
        if self.decoder.linear_input_layer is not None:
            z = self._apply_module(
                z, self.decoder.linear_input_layer, self.decoder.linear_input_norm
            )
            z = rearrange(
                z, "b r (c h w) -> b r c h w", c=self.encoder.enc_n_out_channels[-1],
                h=self.dec_in_resolution[0], w=self.dec_in_resolution[1]
            )
        rotation_output, dec_layer_activation_maps = self._apply_conv_layers(self.decoder, z)
        reconstruction = torch.linalg.vector_norm(rotation_output, dim=1)
        # optionally run output model (shift-scale + sigmoid)
        if self.use_out_conv:
            reconstruction = self.apply_output_model(reconstruction)
        if self.use_out_sigmoid:
            reconstruction = torch.sigmoid(reconstruction)
        # crop if original image resolution not a multiple of encoder strides
        if self.img_resolution[0] % 8 !=0:
            rotation_output = CenterCrop(size=self.img_resolution[0])(rotation_output)
            reconstruction = CenterCrop(size=self.img_resolution[0])(reconstruction)
        return reconstruction, rotation_output, dec_layer_activation_maps

    def apply_output_model(self, z):
        reconstruction = (
            torch.einsum("b c h w, c -> b c h w", z, self.output_weight)
            + self.output_bias
        )
        return reconstruction

    def forward(self, input_images, step_number):
        rotation_input = self._prepare_input(input_images)
        z, _ = self.encode(rotation_input)
        reconstruction, rotation_output, _ = self.decode(z)
        outputs = {
            "reconstruction": reconstruction,
            "rotation_output": rotation_output,
            "rotation_magnitude": reconstruction,
            }
        return outputs


def init_conv_norms(conv_shapes, norm_layer_type):
    # conv_shape = (C, H, W) - if using BatchNorm, take only the first (C) dimension
    channel_norm = nn.ModuleList([None] * len(conv_shapes))
    for idx, conv_shape in enumerate(conv_shapes):
        if norm_layer_type == 'batch_norm':
            channel_norm[idx] = nn.BatchNorm2d(conv_shape[0], affine=True)
        elif norm_layer_type == 'layer_norm':
            channel_norm[idx] = nn.LayerNorm(conv_shape, elementwise_affine=True)
        elif norm_layer_type == 'none':
            channel_norm[idx] = nn.Identity()

    return channel_norm


def get_rotation_bias(n_out_channels, fan_in, d_rotating):
    rotation_bias = nn.Parameter(torch.empty((1, d_rotating, n_out_channels, 1, 1)))
    rotation_bias = init_rotation_bias(fan_in, rotation_bias)
    return rotation_bias


def init_rotation_bias(fan_in, bias):
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)
    return bias


def apply_layer(x, real_function, rotation_bias):
    bsz = x.size()[0]
    norm_x = torch.linalg.vector_norm(x, dim=1) # across d_rotating
    # layer compatible reshaping
    if isinstance(real_function, (nn.Conv2d, nn.ConvTranspose2d)):
        x = rearrange(x, 'b r c h w -> (b r) c h w')
    elif isinstance(real_function, nn.Linear):
        x = rearrange(x, 'b r d -> (b r) d')
    # run layer forward pass
    fw_of_x = real_function(x)
    # layer compatible reshaping back
    if isinstance(real_function, (nn.Conv2d, nn.ConvTranspose2d)):
        fw_of_x = rearrange(fw_of_x, '(b r) c h w -> b r c h w', b=bsz)
    elif isinstance(real_function, nn.Linear):
        fw_of_x = rearrange(fw_of_x, '(b r) d -> b r d', b=bsz)
    phi = fw_of_x + rotation_bias
    chi = real_function(norm_x)
    norm_phi = torch.linalg.vector_norm(phi, dim=1)
    # apply chi-binding
    m_bind = 0.5 * norm_phi + 0.5 * chi
    return m_bind, phi


def get_conv_layers_shapes(conv_layers, d_rotating, d_channels, d_height, d_width, eps=1e-8):
    z = torch.zeros(1, d_rotating, d_channels, d_height, d_width)
    shapes = []
    for module in conv_layers:
        z, phi = module(z)
        z = rearrange(z, 'b c h w -> b 1 c h w')
        z = z * (phi / (torch.linalg.vector_norm(phi, dim=1, keepdim=True) + eps)) 
        shapes.append(z.shape[2:])
    return shapes


class RotatingLinear(nn.Module):
    def __init__(self, n_in_channels: int, n_out_channels: int, d_rotating: int):
        super(RotatingLinear, self).__init__()
        self.fc = nn.Linear(n_in_channels, n_out_channels, bias=False)
        self.rotation_bias = self._get_bias(n_in_channels, n_out_channels, d_rotating)
        
    def _get_bias(self, n_in_channels, n_out_channels, d_rotating):
        fan_in = n_in_channels
        rotation_bias = nn.Parameter(torch.empty((1, d_rotating, n_out_channels)))
        rotation_bias = init_rotation_bias(fan_in, rotation_bias)
        return rotation_bias

    def forward(self, x):
        return apply_layer(x, self.fc, self.rotation_bias)


class RotatingConv2d(nn.Module):
    def __init__(self, n_in_channels: int, n_out_channels: int, d_rotating: int, kernel_size: int = 3, 
                stride: int = 1, padding: int = 0):
        super(RotatingConv2d, self).__init__()
        self.conv = nn.Conv2d(n_in_channels, n_out_channels, kernel_size, stride, padding, bias=False)
        fan_in = n_in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]
        self.rotation_bias = get_rotation_bias(n_out_channels, fan_in, d_rotating)
        
    def forward(self, x):
        return apply_layer(x, self.conv, self.rotation_bias)


class RotatingConvTranspose2d(nn.Module):
    def __init__(self, n_in_channels: int, n_out_channels: int, d_rotating: int,  kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0):
        super(RotatingConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            n_in_channels, n_out_channels, kernel_size, stride, padding, output_padding, bias=False
        )
        fan_in = n_out_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]
        self.rotation_bias = get_rotation_bias(n_out_channels, fan_in, d_rotating)
        
    def forward(self, x):
        return apply_layer(x, self.conv, self.rotation_bias)


class RotatingUpsample(nn.Module):
    def __init__(self, scale_factor: Tuple[float, float]):
        super(RotatingUpsample, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        bsz = x.size()[0]
        x = rearrange(x, 'b r c h w -> (b r) c h w')
        up_x = self.upsample(x)
        up_x = rearrange(up_x, '(b r) c h w -> b r c h w', b=bsz)
        return up_x


class RotatingConvUpsample2d(RotatingConv2d):
    def __init__(self, n_in_channels: int, n_out_channels: int, d_rotating: int, 
                kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0):
        super(RotatingConvUpsample2d, self).__init__(
            n_in_channels, n_out_channels, d_rotating, kernel_size, 1, padding
        )
        self.upsample = RotatingUpsample((stride, stride))

    def forward(self, x):
        x = self.upsample(x)
        return super().forward(x)
    

class RotatingEncoder(nn.Module):
    def __init__(
        self,
        img_resolution: Tuple[int, int], 
        n_in_channels: int,
        enc_n_out_channels: Tuple[int, ...],
        enc_strides: Tuple[int, ...],
        enc_kernel_sizes: Tuple[int, ...],
        d_rotating: int,
        use_linear_layer: bool,
        d_linear: int,
        norm_layer_type: str,
    ):
        super().__init__()

        enc_n_out_channels = [n_in_channels] + enc_n_out_channels
        self.enc_n_out_channels = enc_n_out_channels
        self.conv_layers = []

        for i in range(len(enc_n_out_channels) - 1):
            self.conv_layers.append(
                RotatingConv2d(
                    enc_n_out_channels[i],
                    enc_n_out_channels[i+1],
                    d_rotating,
                    kernel_size=enc_kernel_sizes[i],
                    stride=enc_strides[i],
                    padding=1,
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        conv_shapes = get_conv_layers_shapes(
            self.conv_layers, d_rotating, n_in_channels, img_resolution[0], 
            img_resolution[1]
        )
        self.enc_output_res = conv_shapes[-1][-2:]
        self.conv_norm_layers = init_conv_norms(conv_shapes, norm_layer_type)

        d_enc_output = self.enc_output_res[0] * self.enc_output_res[1] * enc_n_out_channels[-1]
        self.linear_output_layer = None
        if use_linear_layer:
            self.linear_output_layer = RotatingLinear(d_enc_output, d_linear, d_rotating)
            self.linear_output_norm = nn.LayerNorm(d_linear, elementwise_affine=True)


class RotatingConvDecoder(nn.Module):
    def __init__(
        self,
        img_resolution: Tuple[int, int],
        decoder_type: str,
        n_in_channels: int,
        enc_n_out_channels: Tuple[int, ...],
        enc_strides: Tuple[int, ...],
        enc_kernel_sizes: Tuple[int, ...],
        d_rotating: int,
        use_linear_layer: bool,
        d_linear: int,
        enc_output_res: Tuple[int, int],
        norm_layer_type: str,
    ):
        super().__init__()
        self.enc_output_res = enc_output_res
        dec_n_out_channels = enc_n_out_channels[::-1]
        dec_strides = enc_strides[::-1]
        dec_kernel_sizes = enc_kernel_sizes[::-1]
        dec_n_out_channels = dec_n_out_channels + [n_in_channels]

        if decoder_type == 'conv_transpose':
            conv_upsample_cls = RotatingConvTranspose2d
            for i in range(len(dec_kernel_sizes)):
                if dec_kernel_sizes[i] != 3:
                    print(f'Warning: resetting the kernel size from {dec_kernel_sizes[i]} to 3 since automatic padding calculation for random kernel sizes is not implemented for ConvTranspose.')
                    dec_kernel_sizes[i] = 3
        elif decoder_type == 'conv_upsample':
            conv_upsample_cls = RotatingConvUpsample2d

        self.linear_input_layer = None
        if use_linear_layer:
            d_linear_out = dec_n_out_channels[0] * self.enc_output_res[0] * self.enc_output_res[1]
            self.linear_input_layer = RotatingLinear(d_linear, d_linear_out, d_rotating)
            self.linear_input_norm = nn.LayerNorm(d_linear_out, elementwise_affine=True)

        self.conv_layers = []
        for i in range(len(dec_n_out_channels) - 1):
            if dec_strides[i] == 1:
                self.conv_layers.append(
                    RotatingConv2d(
                        dec_n_out_channels[i],
                        dec_n_out_channels[i+1],
                        d_rotating,
                        kernel_size=dec_kernel_sizes[i],
                        stride=dec_strides[i],
                        padding=1,
                    )
                )
            else:
                self.conv_layers.append(
                    conv_upsample_cls(
                        dec_n_out_channels[i],
                        dec_n_out_channels[i+1],
                        d_rotating,
                        kernel_size=dec_kernel_sizes[i],
                        stride=dec_strides[i],
                        padding=1,
                        output_padding=1,
                    )
                )
        self.conv_layers = nn.ModuleList(self.conv_layers)
        conv_shapes = get_conv_layers_shapes(
            self.conv_layers, d_rotating, dec_n_out_channels[0], self.enc_output_res[0], 
            self.enc_output_res[1]
        )
        self.conv_norm_layers = init_conv_norms(conv_shapes, norm_layer_type)