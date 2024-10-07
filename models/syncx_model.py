from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange

from models.syncx_eval import SynCxMetricsRecorder
from models.rnn_cell import AutoEncoder3l
from models.complex_layers import ComplexConv2d, Activation
from utils.utils_plot import process_layer_map


class SynchronousComplexNetwork(nn.Module):
    def __init__(self,
    img_resolution: Tuple[int, int],
    in_channels: int,
    phase_init_type: str,
    phase_init_min: float,
    phase_init_max: float,
    phase_init_conc: float,
    activation_type: str,
    cw_ksize: int,
    cw_stride: int,
    cw_hidden_dim: int,
    use_out_activation: bool,
    num_iters: int,
    phase_mask_threshold: float,
    n_images_to_log: int,
    seed: int,
    step_loss_type: str,
    eval_only_n_batches: int,
    features_to_cluster: str,
    ):
        super(SynchronousComplexNetwork, self).__init__()

        self.img_resolution = img_resolution
        self.phase_init_type = phase_init_type
        self.phase_init_min = phase_init_min
        self.phase_init_max = phase_init_max
        self.phase_init_conc = phase_init_conc
        self.activation_type = activation_type
        self.use_out_activation = use_out_activation
        self.num_iters = num_iters
        self.phase_mask_threshold = phase_mask_threshold
        self.n_images_to_log = n_images_to_log
        self.seed = seed
        self.step_loss_type = step_loss_type
        self.eval_only_n_batches = eval_only_n_batches
        self.features_to_cluster = features_to_cluster
        out_channels = in_channels

        # complex-weighted conv autoencoder.
        self.cw_rnn = AutoEncoder3l(
            img_resolution, in_channels, cw_hidden_dim, cw_ksize, cw_stride, 
            activation_type
        )
        
        # complex-weighted output layer.
        self.output_layer = ComplexConv2d(
            cw_hidden_dim, out_channels, kernel_size=1, stride=1, padding="same"
        )
        if use_out_activation:
            self.output_act = Activation(out_channels, activation_type)         
        
    def get_metrics_recorder(self):
        return SynCxMetricsRecorder()
        
    def _prepare_input(self, input_features: torch.Tensor, phase_init_type: str, low: float, high: float):
        # initialize phase map independently for all [H x W x C]
        if phase_init_type == "zero":
            phase = torch.zeros_like(input_features)
        elif phase_init_type == "uniform":
            # shifts range of uniform dist. from [0, 1] -> [low, high]
            phase = (high - low) * torch.rand_like(input_features) + low
        elif phase_init_type == "von_mises":
            dist = torch.distributions.von_mises.VonMises(
                loc=torch.tensor(0., device=input_features.device), 
                concentration=torch.tensor(self.phase_init_conc, device=input_features.device)
            )
            phase = dist.sample(input_features.shape)
        complex_input = input_features * torch.exp(phase * 1j)
        return complex_input 

    def forward(self, input_images, step_number):
        self.batch_size = input_images.size()[0]
        # Phase initialization for input_features
        z = self._prepare_input(
            input_images, self.phase_init_type, self.phase_init_min, self.phase_init_max
        )               
        # Run complex-weighted ConvRNN for num_iters steps, z: [b, n, c, h, w]
        phase_map_iters, magnitude_map_iters, z_iters = [], [], []
        # Initialize intermediate layers of UNet
        autoenc_layers_phase = {
            'enc_1': [],
            'enc_2': [],
            'enc_3': [],
            'dec_1': [],
            'dec_2': [],
            'dec_3': [],
            'dec_out': [],
        }
        autoenc_layers_magnitude = {
            'enc_1': [],
            'enc_2': [],
            'enc_3': [],
            'dec_1': [],
            'dec_2': [],
            'dec_3': [],
            'dec_out': [],
        }
        
        # Run complex-weighted autoencoder for N steps.
        for n_iter in range(self.num_iters):
            rnn_outputs = self.cw_rnn(z)
            z_out_rnn = rnn_outputs["dec_out"]
            # Run output layer
            z_out = self.output_layer(z_out_rnn)
            # output layer activation
            if self.use_out_activation:
                z_out = self.output_act(z_out)
            # store step-wise output magnitude & phases
            if not self.training and step_number == 0:
                z_viz = z_out if self.features_to_cluster == "complex_output" else z_out_rnn
                phase_map_iters.append(
                    process_layer_map(z_viz[:self.n_images_to_log], 'tsne')
                    )
                magnitude_map_iters.append(torch.linalg.vector_norm(
                    z_viz.abs(), dim=1, keepdim=True)
                    )
            z_iters.append(z_out)
            # use updated phase with init magnitudes as complex input at step=n+1
            z = input_images * torch.exp(z_out.angle() * 1j)

        # collect reconstruction from all RNN steps
        z_out_all = rearrange(z_iters, 'n b c h w -> b n c h w')
        # convert list of RNN output phases and magnitudes to tensors
        if phase_map_iters:
            phase_map_iters = rearrange(phase_map_iters, 'n b c h w -> b n c h w')
            magnitude_map_iters = rearrange(magnitude_map_iters, 'n b c h w -> b n c h w')

        # log the phase and magnitude of last iteration outputs
        last_iter_z_out = z_out_all[:, -1]
        recon_all = None
        if self.step_loss_type == "teacher":
            recon_all = z_out_all.abs()
        elif self.step_loss_type == "none":
            recon_all = last_iter_z_out.abs()
        reconstruction = recon_all

        rnn_outputs.update({
            "reconstruction": reconstruction,
            "complex_output": last_iter_z_out,
            "complex_magnitude": last_iter_z_out.abs(),
            "complex_phase": last_iter_z_out.angle(),
            "rnn_phase_map_iters": phase_map_iters,
            "rnn_magnitude_map_iters": magnitude_map_iters,
            "autoenc_layers_phase": autoenc_layers_phase,
            "autoenc_layers_magnitude": autoenc_layers_magnitude,
        })
        return rnn_outputs