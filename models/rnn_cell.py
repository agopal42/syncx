from typing import Tuple

import torch.nn as nn

from models.complex_layers import ComplexConv2d, ComplexUpSample2d, Activation


class AutoEncoder3l(nn.Module):
    """ A 3-layer convolutional autoencoder with complex-weights. """
    def __init__(self,
        img_resolution: Tuple[int, int],
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        stride: int,
        activation_type: str,
        ):
        super().__init__()

        self.stride = stride
        self.img_resolution = img_resolution
        padding = "same" if stride == 1 else "valid"
        # Encoder layers
        self.enc_conv1 = ComplexConv2d(
            input_size, hidden_size, kernel_size, stride, padding=(kernel_size-stride)//2
            )
        self.enc_act1 = Activation(hidden_size, activation_type)
        self.enc_conv2 = ComplexConv2d(
            hidden_size, 2 * hidden_size, kernel_size, stride, padding=(kernel_size-stride)//2
        )
        self.enc_act2 = Activation(2 * hidden_size, activation_type)
        self.enc_conv3 = ComplexConv2d(
            2 * hidden_size, 2 * hidden_size, kernel_size, stride, padding=(kernel_size-stride)//2
        )
        self.enc_act3 = Activation(2 * hidden_size, activation_type)
        # Decoder layers
        self.dec_up1 = ComplexUpSample2d(size=img_resolution[0]//(stride**2))        
        self.dec_conv1 = ComplexConv2d(
            2 * hidden_size, 2 * hidden_size, kernel_size, stride=1, padding="same"
            )
        self.dec_act1 = Activation(2 * hidden_size, activation_type)
        self.dec_up2 = ComplexUpSample2d(size=img_resolution[0]//stride)
        self.dec_conv2 = ComplexConv2d(
            2 * hidden_size, hidden_size, kernel_size, stride=1, padding="same"
        )
        self.dec_act2 = Activation(hidden_size, activation_type)
        self.dec_up3 = ComplexUpSample2d(size=img_resolution[0])
        self.dec_conv3 = ComplexConv2d(
            hidden_size, hidden_size, kernel_size, stride=1, padding="same"
        )
        self.dec_act3 = Activation(hidden_size, activation_type)
        if img_resolution[0] in [64, 96]:
            self.dec_conv4 = ComplexConv2d(
                hidden_size, hidden_size, kernel_size=1, stride=1, padding="same"
            )
            self.dec_act4 = Activation(hidden_size, activation_type)

    def forward(self, x):
        enc_1 = self.enc_act1(self.enc_conv1(x))
        enc_2 = self.enc_act2(self.enc_conv2(enc_1))
        enc_3 = self.enc_act3(self.enc_conv3(enc_2))
        dec_1 = self.dec_up1(enc_3)
        dec_1 = self.dec_act1(self.dec_conv1(dec_1))
        dec_2 = self.dec_up2(dec_1)
        dec_2 = self.dec_act2(self.dec_conv2(dec_2))
        dec_3 = self.dec_up3(dec_2)
        dec_out = self.dec_act3(self.dec_conv3(dec_3))
        if self.img_resolution[0] in [64, 96]:
            dec_out = self.dec_act4(self.dec_conv4(dec_out))

        outputs = {
            "enc_1": enc_1,
            "enc_2": enc_2,
            "enc_3": enc_3,
            "dec_1": dec_1,
            "dec_2": dec_2,
            "dec_3": dec_3,
            "dec_out": dec_out,
        }
        return outputs