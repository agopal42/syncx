from datetime import timedelta
import wandb
import os
import subprocess
import shutil
from functools import partial
from typing import Optional, Tuple

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat


def adjusted_rand_index(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor
    ) -> torch.Tensor:
    """
    Adapted from 
    https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (n_points <= n_true_groups and n_points <= n_pred_groups), ("adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32) 
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups)

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    _all_equal = lambda values: torch.all(torch.equal(values, values[..., :1]), dim=-1)
    both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def log_results(split, use_wandb, step, time_spent, metrics, extra_metrics):
    # Console print
    console_log = f"{split} \t \t" \
        f"Step: {step} \t" \
        f"Time: {timedelta(seconds=time_spent)} \t" \
        f"MSE Loss: {metrics['loss']:.4e} \t"
    if 'ARI-FULL' in metrics:
        console_log += f"ARI-FULL: {metrics['ARI-FULL']:.4e} \t"
    if 'ARI-FG' in metrics:
        console_log += f"ARI-FG: {metrics['ARI-FG']:.4e} \t"
    print(console_log)

    # wandb log
    if bool(use_wandb):
        wandb_log = {
            'step': step,
            split + '/time_spent': time_spent,
        }
        wandb_log.update({f"{split}/{k}": v for k, v in extra_metrics.items()})
        wandb_log.update({f"{split}/{k}": v for k, v in metrics.items()})
        wandb.log(wandb_log)


def print_model_size(model):
    line_len = 89
    line_len2 = 25
    print('-' * line_len)
    # Native pytorch
    try:
        print(model)
    except:
        print('Warning: could not print the Native PyTorch model info - probably some module is `None`.')

    # One-by-one layer
    print('-' * line_len)
    print("Model params:")
    total_params = 0
    module_name = ""
    module_n_params = 0
    for name, param in model.named_parameters():
        if name.find('.') != -1:
            if module_name == "":
                module_name = name[:name.index('.')]
            if module_name != name[:name.index('.')]:
                print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')
                module_name = name[:name.index('.')]
                module_n_params = 0
        else:
            if module_name == "":
                module_name = name
            if module_name != name:
                print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')
                module_name = name
                module_n_params = 0
        n_params = np.prod(param.size())
        module_n_params += n_params
        print(f"\t {name} {n_params:,}")
        total_params += n_params
    print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')

    # Total Number of params
    print('-' * line_len)
    print(f"Total number of params: {total_params:,}")
    print('-' * line_len)


def copy_git_src_files_to_logdir(log_dir):
    print(f'Copying *.py source files to {log_dir}:')
    srcdir = os.path.join(log_dir, 'src')
    os.makedirs(srcdir)
    src_git_files = subprocess.run(['git', 'ls-files'], stdout=subprocess.PIPE)
    src_git_files = src_git_files.stdout.decode('utf-8').split('\n')
    for file in src_git_files:
        if file.endswith(".py"):
            # handle nested dirs
            if '/' in file:
                sp = file.split('/')
                nested = os.path.join(*sp[:-1])
                nested = os.path.join(srcdir, nested)
                if not os.path.exists(nested):
                    os.makedirs(nested)
            else:
                nested = srcdir
            shutil.copy2(file, nested)
            print('Copied file {} to {}'.format(file, nested))
    print('-' * 89)


def build_grid(resolution: Tuple[int, int], l_min: int, l_max: int, pos_enc_type: str):
    ranges = [torch.linspace(l_min, l_max, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    if pos_enc_type == "softpos":
        grid = torch.cat([grid, 1.0 - grid], dim=-1)
    return grid


def sinusoidal_pe_2d(d_model: int, height: int, width: int):
    positional_encoding = torch.zeros(d_model, height, width)
    # each spatial dim use half of d_model
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(width)
    pos_w = rearrange(pos_w, 'w -> w 1')
    pos_h = torch.arange(height)
    pos_h = rearrange(pos_h, 'h -> h 1')
    sine_w_enc = torch.sin(pos_w * div_term)
    sine_w_enc = rearrange(sine_w_enc, 'w d -> d w')
    sine_w_enc = repeat(sine_w_enc, 'd w -> d h w', h=height)
    positional_encoding[0:d_model:2, :, :] = sine_w_enc
    cos_w_enc = torch.cos(pos_w * div_term)
    cos_w_enc = rearrange(cos_w_enc, 'w d -> d w')
    cos_w_enc = repeat(cos_w_enc, 'd w -> d h w', h=height)
    positional_encoding[1:d_model:2, :, :] = cos_w_enc
    sine_h_enc = torch.sin(pos_h * div_term)
    sine_h_enc = rearrange(sine_h_enc, 'h d -> d h')
    sine_h_enc = repeat(sine_h_enc, 'd h -> d h w', w=width)
    positional_encoding[d_model::2, :, :] = sine_h_enc
    cos_h_enc = torch.cos(pos_h * div_term)
    cos_h_enc = rearrange(cos_h_enc, 'h d -> d h')
    cos_h_enc = repeat(cos_h_enc, 'd h -> d h w', w=width)
    positional_encoding[d_model+1::2, :, :] = cos_h_enc
    return positional_encoding


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """ 
    Taken from huggingface transformers ->
    https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def stable_angle(x: torch.tensor, eps=1e-8):
    """ Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return y.angle()


def get_padding(input_size: Tuple[int, int], stride: int, kernel_size: int):
    """Function to return padding size such that input size [H, H] and output size [H/S, H/S]
    where S is the stride. """
    if stride > 1:
        padding = math.ceil(((input_size[0]//stride - 1)*stride + kernel_size - input_size[0])/2)
    else:
        padding = "same" 
    return padding