from pathlib import Path
import os
import random
import math
import string
from tqdm import tqdm
import time
from itertools import cycle
from datetime import datetime

import torch
import numpy as np
import wandb

from data import (EMORLHdF5Dataset,
    get_dataloader, 
    random_split, 
    DATASET_IMG_RESOLUTION,
    EMORL_DATASET_MAPPING
)
from utils.utils_general import (print_model_size, 
    get_constant_schedule_with_warmup, 
    log_results, 
    copy_git_src_files_to_logdir
)
from utils.utils_args import ArgumentParser
from models.syncx_model import SynchronousComplexNetwork
from models.rf_model import RotatingAutoEncoder

parser = ArgumentParser()

# Training
parser.add_argument('-max_iterations', type=int, default=100_000,
                    help='Maximum number of training steps.')
parser.add_argument('-max_patience', type=int, default=10,
                    help='Maximum patience for early-stopping.')
parser.add_argument('-learning_rate', type=float, default=0.0004,
                    help='Learning rate.')
parser.add_argument('-lr_schedule', type=str, choices=["constant", "cosine", "linear"],
                    default="constant", help="Type of schedule for learning rate warm-up and decay.")
parser.add_argument('-min_lr', type=float, default=0.0,
                    help='Minimum lr for cosine decay.')
parser.add_argument('-restart_iter', type=int, default=4000,
                    help='Number of steps for restart.')
parser.add_argument('-restart_factor', type=int, default=1,
                    help='Factor increases after a lr restart.')
parser.add_argument('-max_grad_norm', type=float, default=0.0,
                    help='Maximum norm for gradient clipping.')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Mini-batch size.')
parser.add_argument('-num_workers', type=int, default=4,
                    help='Number of DataLoader workers.')
# Model
parser.add_argument('-model', type=str, choices=["syncx", "rf"],
                    default="syncx", help='Model Name.')
parser.add_argument('-load_model_path', type=str,
                    default=None, help='Checkpoint path from where to load the model weights.')
parser.add_argument('-phase_init', type=str, choices=["zero", "uniform", "von_mises"],
                    default="zero", help='Type of phase map initialization at input layer')
parser.add_argument('-phase_init_min', type=float, default=-math.pi,
                    help='Minimum phase value used by uniform random initialization (SynCx).')
parser.add_argument('-phase_init_max', type=float, default=math.pi,
                    help='Maximum phase value used by uniform random initialization (SynCx).')
parser.add_argument('-phase_init_conc', type=float, default=1.,
                    help='Concentration parameter for phase initialization from von-Mises distribution (SynCx).')
parser.add_argument('-activation_type', type=str, choices=["modrelu", "relu"],
                    default="modrelu", help="Activation rule to be used in all layers of SynCx model.")
parser.add_argument('-cw_ksize', type=int, default=1,
                    help='Kernel size of conv. filter for complex-weighted RNN (SynCx).')
parser.add_argument('-cw_stride', type=int, default=1,
                    help='Stride length of conv. filter in complex weighted RNN (SynCx).')
parser.add_argument('-cw_hidden_size', type=int, default=64, 
                    help='Size of complex-weighted RNN (SynCx).')
parser.add_argument('-enc_n_out_channels', type=str, default='128,128,256,256,256',
                    help='Tuple in a string format denoting the number of output channels of the encoder layers. The decoder mirrors this.')
parser.add_argument('-enc_strides', type=str, default='2,1,2,1,2',
                    help='Tuple in a string format denoting the conv layer strides of the encoder. The decoder mirrors this.')
parser.add_argument('-enc_kernel_sizes', type=str, default='3,3,3,3,3',
                    help='Tuple in a string format denoting the conv layer strides of the encoder. The decoder mirrors this.')
parser.add_argument('-use_linear_layer', type=int, default=1, 
                    help='Whether to use linear layer mapping to/from the latent space.')
parser.add_argument('-d_linear', type=int, default=256,
                    help='Number of hidden units in the encoder linear mapping.')
parser.add_argument('-d_rotating', type=int, default=8,
                    help='Number of rotating features in RF baseline model.')
parser.add_argument('-decoder_type', type=str, choices=["conv_upsample", "conv_transpose"],
                    default="conv_upsample", help='Type of conv decoder architecture variation.')
parser.add_argument('-use_out_conv', type=int, default=0, 
                    help='Whether to use (1) 1x1 output convolution layer on Decoder or not (0).')
parser.add_argument('-use_out_sigmoid', type=int, default=0,
                    help='Whether to use (1) sigmoid activation on Decoder output or not (0).')
parser.add_argument('-norm_layer_type', type=str, default='batch_norm',
                    help='Norm layer type to use: batch_norm, layer_norm or none.')
parser.add_argument('-n_iters', type=int, default=2,
                    help='Number of iterations used in SlotAttention/SynCx.')
# Loss
parser.add_argument('-step_loss_type', type=str, default="teacher", choices=["teacher", "none"], 
                    help='Type of loss used by SynCx.')
# Data
parser.add_argument('-dataset_name', type=str,
                    choices=[
                        "tetrominoes", "multi_dsprites", "clevr",  # EMORL datasets
                    ],
                    default="multi_dsprites", help='Dataset name.')
parser.add_argument('-use_32x32_res', type=bool, default=1,
                    help='Whether to use 32x32 resolution (1) or default (tetr:35,35;dsp:64,64,clvr:96,96) (0).')
parser.add_argument('-use_64x64_res', type=bool, default=0,
                    help='Whether to use 32x32 resolution (1) or default (tetr:35,35;dsp:64,64,clvr:96,96) (0).')
parser.add_argument('-make_background_black', type=int, default=0,
                    help='Flag to make the background (pixels belonging to the first segmenatation mask) black.')
parser.add_argument('-n_objects_cutoff', type=int, default=0,
                    help='Number of objects by which to filter samples in the `getitem` function of the dataset loader.')
parser.add_argument('-cutoff_type', type=str, choices=["eq", "leq", "geq"],
                    default="eq", help='Cuttoff type: take only samples that equal, less-or-equal, greater-or-equal number of objects as specified by n_objects_cutoff.')
parser.add_argument('-use_grayscale', type=bool, default=0,
                    help='grayscale.')

# Eval/Logging
parser.add_argument('-root_dir', type=str, default="results",
                    help='Root directory to save logs, ckpts, load data etc.')
parser.add_argument('-log_interval', type=int, default=2_000,
                    help='Logging interval (in steps).')
parser.add_argument('-eval_interval', type=int, default=2_000,
                    help='Evaluation interval (in steps).')
parser.add_argument('-eval_only_n_batches', type=int, default=0,
                    help='Evaluate only on a part of the validation set (for debugging).')
parser.add_argument('-save_logs', type=int, default=1,
                    help='Whether to save model ckpts and logs (1) or not (0).')
parser.add_argument('-use_wandb', type=int, default=1,
                    help='Flag to log results on wandb (1) or not (0).')
parser.add_argument('-use_cuda', type=int, default=1,
                    help='Use GPU acceleration (1) or not (0).')
parser.add_argument('-gpu_id', type=int, default=0,
                    help='GPU device to run on.')
parser.add_argument('-seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('-n_images_to_log', type=int, default=8,
                    help='Set to 0 to disable image rendering and logging. Number of images to take from a batch to generate the image grid logs.')
parser.add_argument('-plot_resize_resolution', type=int, default=256,
                    help='Resolution to which to resize the images before plotting. This is introduced due to the fine-level details in matplotlib plots.')
parser.add_argument('-phase_mask_threshold', type=float, default=0.1,
                    help='Threshold on minimum magnitude to use when evaluating phases (CAE model); -1: no masking.')
parser.add_argument('-phase_mask_type', type=str, default="threshold", choices=["threshold", "bg_only", "none"], 
                    help='Masking strategy for phase values before clustering (SynCx).')
parser.add_argument('-use_eval_type', type=str, default="syncx", choices=["rf", "syncx"], 
                    help='Type of evaluation protocol to use.')
parser.add_argument('-features_to_cluster', type=str, default="dec_out",
                    choices=["enc_1", "enc_2", "enc_3",
                             "dec_1", "dec_2", "dec_3",
                             "dec_out", "complex_output"], 
                    help='Features to use for KMeans clustering.')

parser.add_profile([
    ################################################################
    # Base profiles
    ################################################################
    parser.Profile('rf', {
        'eval_only_n_batches': 0,
        'log_interval': 10_000,
        'eval_interval': 10_000,
        'model': 'rf',
        'enc_n_out_channels': '128,128,256,256,256',
        'enc_strides': '2,1,2,1,2',  # downsamples by 8x
        'enc_kernel_sizes': '3,3,3,3,3',
        'use_linear_layer': 1,
        'd_linear': 256,
        'd_rotating': 10,
        'decoder_type': 'conv_transpose',
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'norm_layer_type': 'batch_norm',
        'step_loss_type': 'none',
        'learning_rate': 0.001,
        'lr_schedule': 'linear',
        'restart_iter': 5000,
        'max_grad_norm': 0.1,
        'use_eval_type': 'rf',
        'phase_mask_threshold': 0.1
    }),

    parser.Profile('syncx', {
        'eval_only_n_batches': 0,
        'log_interval': 10_000,
        'eval_interval': 10_000,
        'model': 'syncx',
        'phase_init': 'von_mises',
        'phase_init_conc': 1.0,
        'cw_ksize': 3,
        'cw_stride': 2,
        'cw_hidden_size': 64,
        'activation_type': 'modrelu',
        'use_out_sigmoid': 0,
        'step_loss_type': 'teacher',
        'learning_rate': 0.0005,
        'lr_schedule': 'constant',
        'max_grad_norm': 1.0,
        'use_eval_type': 'syncx',
        'phase_mask_type': 'bg_only',
        'phase_mask_threshold': 0.0
    }),

    ################################################################
    # Tetrominoes profiles
    ################################################################

    parser.Profile('rf_tetrominoes', {
        'dataset_name': 'tetrominoes',
        'use_32x32_res': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'batch_size': 64,
        'max_iterations': 100_000,
    }, include='rf'),

    parser.Profile('syncx_tetrominoes', {
        'dataset_name': 'tetrominoes',
        'use_32x32_res': 0,
        'n_iters': 3,
        'batch_size': 64,
        'max_iterations': 40_000,
    }, include='syncx'),
    
    ################################################################
    # Multi-dsprites profiles
    ################################################################

    parser.Profile('rf_dsprites', {
        'dataset_name': 'multi_dsprites',
        'use_32x32_res': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'batch_size': 16,
        'max_iterations': 100_000,
    }, include='rf'),

    parser.Profile('syncx_dsprites', {
        'dataset_name': 'multi_dsprites',
        'use_32x32_res': 0,
        'n_iters': 3,
        'batch_size': 16,
        'max_iterations': 100_000,
    }, include='syncx'),

    ################################################################
    # CLEVR profiles
    ################################################################
    
    parser.Profile('rf_clevr', {
        'dataset_name': 'clevr',
        'use_32x32_res': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'batch_size': 64,
        'max_iterations': 30_000,

    }, include='rf'),

    parser.Profile('syncx_clevr', {
        'dataset_name': 'clevr',
        'use_32x32_res': 0,
        'n_iters': 4,
        'batch_size': 32,
        'max_iterations': 100_000,
    }, include='syncx'),

])

args = parser.parse_args()


torch.multiprocessing.set_sharing_strategy('file_system')

def build_datasets(args):
    """Function to build train, [validation] and test datasets loaders and initializers."""
    print(f"Loading and preprocessing data .....")

    return_masks = True
    return_factors = False
    print(f'Building {args.dataset_name} dataset.')
    if args.dataset_name in EMORL_DATASET_MAPPING:
        train_dataset = EMORLHdF5Dataset(
            dataset_name=args.dataset_name, 
            split='train', 
            return_masks=return_masks, 
            return_factors=return_factors, 
            make_background_black=args.make_background_black,
            use_32x32_res=bool(args.use_32x32_res),
            use_64x64_res=bool(args.use_64x64_res),
            n_objects_cutoff=args.n_objects_cutoff,
            cutoff_type=args.cutoff_type,
            use_grayscale=bool(args.use_grayscale),
        )
        # EMORL ds don't have an exclusive val split -> split train into 2 exclusive parts
        train_dataset, _ = random_split(
            train_dataset, [0.95, 0.05], torch.Generator().manual_seed(args.seed)
        )
        # Below is a 'hack' to make the train and val datasets different
        # This is especially needed for evaluating on samples containing different number of objects
        val_dataset = EMORLHdF5Dataset(
            dataset_name=args.dataset_name, 
            split='train', 
            return_masks=return_masks, 
            return_factors=return_factors, 
            make_background_black=args.make_background_black,
            use_32x32_res=bool(args.use_32x32_res),
            use_64x64_res=bool(args.use_64x64_res),
            n_objects_cutoff=0,
            cutoff_type='',
            use_grayscale=bool(args.use_grayscale),
        )
        _, val_dataset = random_split(
            val_dataset, [0.95, 0.05], torch.Generator().manual_seed(args.seed)
        )
        test_dataset = EMORLHdF5Dataset(
            dataset_name=args.dataset_name, 
            split='test', 
            return_masks=return_masks, 
            return_factors=return_factors, 
            make_background_black=args.make_background_black,
            use_32x32_res=bool(args.use_32x32_res),
            use_64x64_res=bool(args.use_64x64_res),
            n_objects_cutoff=0,
            cutoff_type='',
            use_grayscale=bool(args.use_grayscale),
        )
    else:
        raise ValueError(f'Unknown dataset {args.dataset_name}.')

    # Improve reproducibility in dataloader. (borrowed from Loewe)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    test_dataloader = get_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    n_channels = test_dataset.n_channels

    return train_dataloader, val_dataloader, test_dataloader, n_channels


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(args, n_channels):
    """Function to build model of requisite type and initialize optimizer."""

    if bool(args.use_32x32_res):
        img_resolution = (32, 32)
    elif bool(args.use_64x64_res):
        img_resolution = (64, 64)
    else:
        img_resolution = DATASET_IMG_RESOLUTION[args.dataset_name]

    if args.model == "syncx":
        model = SynchronousComplexNetwork(
            img_resolution=img_resolution,
            in_channels=n_channels,
            phase_init_type=args.phase_init,
            phase_init_min=args.phase_init_min,
            phase_init_max=args.phase_init_max,
            phase_init_conc=args.phase_init_conc,
            activation_type=args.activation_type,
            cw_ksize=args.cw_ksize,
            cw_stride=args.cw_stride,
            cw_hidden_dim=args.cw_hidden_size,
            use_out_activation=bool(args.use_out_sigmoid),
            num_iters=args.n_iters,
            phase_mask_threshold=args.phase_mask_threshold,
            n_images_to_log=args.n_images_to_log,
            seed=args.seed,
            step_loss_type=args.step_loss_type,
            eval_only_n_batches=args.eval_only_n_batches,
            features_to_cluster=args.features_to_cluster,
        )
    elif args.model == "rf":
        model = RotatingAutoEncoder(
            img_resolution=img_resolution,
            n_in_channels=n_channels,
            enc_n_out_channels=args.enc_n_out_channels,
            enc_strides=args.enc_strides,
            enc_kernel_sizes=args.enc_kernel_sizes,
            use_linear_layer=bool(args.use_linear_layer),
            d_linear=args.d_linear,
            d_rotating=args.d_rotating,
            decoder_type=args.decoder_type,
            use_out_conv=bool(args.use_out_conv),
            use_out_sigmoid=bool(args.use_out_sigmoid),
            norm_layer_type=args.norm_layer_type,
            rotating_mask_threshold=args.phase_mask_threshold,
            n_images_to_log=args.n_images_to_log,
            plot_resize_resolution=args.plot_resize_resolution,
            seed=args.seed,
        )
    else:
        raise ValueError("Invalid model architecture specified!!!!")
    
    if args.load_model_path is not None:
        full_state_dict = torch.load(args.load_model_path)
        model.load_state_dict(full_state_dict['model_state_dict'])

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print("Let's use", n_gpus, "GPUs!")
        model = MyDataParallel(model)
    
    if args.use_cuda:
        model = model.cuda()
    
    print_model_size(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1e-8
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_iter, T_mult=args.restart_factor,
        )
    elif args.lr_schedule == "linear":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, args.restart_iter
        )
    return model, optimizer, scheduler

def compute_loss(model_inputs, model_outputs, args):
    losses_dict = {}
    loss_total, loss_image_rec = 0, 0
    # iteration-level reconstruction targets
    if args.step_loss_type == "teacher":
        input_images_iters = model_inputs["images"][:, None].repeat(1, args.n_iters, 1, 1, 1)
        loss_image_rec = torch.nn.functional.mse_loss(
            model_outputs["reconstruction"], input_images_iters, reduction="mean"
        )
    elif args.step_loss_type == "none":
        loss_image_rec = torch.nn.functional.mse_loss(
            model_outputs["reconstruction"], model_inputs["images"], reduction="mean"
        )
    losses_dict['loss_rec'] = loss_image_rec
    loss_total += loss_image_rec
    losses_dict['loss'] = loss_total
    return loss_total, losses_dict


def train_step(model_inputs, model, optimizer, scheduler, args, step_number):
    optimizer.zero_grad()
    model_outputs = model(model_inputs["images"], step_number=step_number)
    loss, losses_dict = compute_loss(model_inputs, model_outputs, args)
    loss.backward()
    if args.max_grad_norm > 0.:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        model_outputs.update({"total_norm": total_norm})
    optimizer.step()
    if args.lr_schedule in ["cosine", "linear"]:
        scheduler.step()
    model_outputs.update(losses_dict)
    return model_outputs


def evaluation(split, step, args, eval_loader, model):
    model.eval()
    metrics_recorder = model.get_metrics_recorder()
    test_time = time.time()
    eval_iterator = iter(eval_loader)
    with torch.no_grad():
        for i_batch in tqdm(range(len(eval_loader))):
            eval_inputs = next(eval_iterator)
            if args.use_cuda:
                eval_inputs["images"] = eval_inputs["images"].cuda()
            # TODO(astanic): fix step_number to a dummy value to always use the same type of CL address?
            outputs = model(eval_inputs["images"], step_number=i_batch)
            _, losses_dict = compute_loss(eval_inputs, outputs, args)
            outputs.update(losses_dict)
            metrics_recorder.step(args, eval_inputs, outputs)
            if args.eval_only_n_batches > 0 and i_batch >= args.eval_only_n_batches:
                print(f'Stopping evaluation after {args.eval_only_n_batches} batches')
                break
    time_spent = time.time() - test_time
    extra_metrics = {}
    log_results(
        split=split, use_wandb=args.use_wandb, step=step, time_spent=time_spent, 
        metrics=metrics_recorder.log(), extra_metrics=extra_metrics
    )
    model.train()

    return metrics_recorder


def main(args):
    # set gpu-id to run process on
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["WANDB_SERVICE_WAIT"] = "300"
    # set randomness
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wandb init
    if bool(args.use_wandb):
        run = wandb.init(config=args, project="binding-by-synchrony", entity="agopal")
        
    # set logs and ckpts directories
    log_dir, ckpt_dir = "", ""
    if bool(args.save_logs):
        if bool(args.use_wandb):
            log_dir = Path(wandb.run.dir)
        else:
            run_name_id = str(''.join(random.choices(string.ascii_lowercase, k=5)))
            timestamp = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_%f")
            run_name_id = run_name_id + timestamp
            log_dir = Path(args.root_dir) / args.dataset_name / args.model / run_name_id

        ckpt_dir = Path(log_dir / "ckpts")
        # create logs and ckpt directories
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # copy the source files to logdir
        copy_git_src_files_to_logdir(log_dir)

    # LOAD DATA
    train_dataloader, val_dataloader, test_dataloader, n_channels = build_datasets(args)

    # BUILD MODEL
    model, optimizer, scheduler = build_model(args, n_channels)

    # TRAINING LOOP
    try:
        print("-" * 89)
        print(f"Starting training for max {args.max_iterations} steps.")
        print(f"Number of batches in epoch={len(train_dataloader)}, batch size={args.batch_size} -> dataset size ~ {len(train_dataloader)*args.batch_size}.")
        print(f"At any point you can hit Ctrl + C to break out of the training loop early, but still evaluate on the test set.")

        print(f"Training model .....")
        step_start_time = time.time()
        train_iterator = cycle(train_dataloader)
        early_stop_score_best = model.get_metrics_recorder().get_init_value_for_early_stop_score()
        for step in range(1, args.max_iterations + 1):  # Adding one such that we evaluate the last step too
            step_start_time = time.time()

            # TRAIN STEP
            train_inputs = next(train_iterator)
            if args.use_cuda == 1:
                train_inputs["images"] = train_inputs["images"].cuda()
            train_outputs = train_step(train_inputs, model, optimizer, scheduler, args, step_number=step)

            # LOGGING
            if step % args.log_interval == 0:
                metrics_recorder = model.get_metrics_recorder()
                metrics_recorder.step(args, train_inputs, train_outputs)
                step_time = time.time() - step_start_time
                extra_metrics = {
                    'lr': optimizer.param_groups[0]['lr'],
                }
                log_results(
                    split='train', use_wandb=args.use_wandb, step=step, time_spent=step_time,
                    metrics=metrics_recorder.log(), extra_metrics=extra_metrics,
                )

            # EVAL LOOP
            if step % args.eval_interval == 0:
                # Make sure to save the model in case evaluation NaNs out
                wandb_id = "-" + wandb.run.id if bool(args.use_wandb) else ""
                ckpt_fname = f"model-latest{wandb_id}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_outputs["loss"],
                }, ckpt_dir / ckpt_fname)

                eval_metrics_recorder = evaluation(
                    split='val', step=step, args=args, eval_loader=val_dataloader, 
                    model=model
                )                
                # CHECKPOINT MODEL
                ckpt_fname = ""
                early_stop_score_current = eval_metrics_recorder.get_current_early_stop_score()
                if eval_metrics_recorder.early_stop_score_improved(early_stop_score_current, early_stop_score_best):
                    print(f'Early stop / best model score improved from {early_stop_score_best} to {early_stop_score_current}.')
                    early_stop_score_best = early_stop_score_current
                    ckpt_fname = f"model-best{wandb_id}.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_outputs["loss"],
                    }, ckpt_dir / ckpt_fname)
                
    except KeyboardInterrupt:
        print(f"-" * 89)
        print(f"KeyboardInterrupt signal received. Exiting early from training.")

    # TEST SET
    evaluation(
        split='test', step=args.max_iterations, args=args, eval_loader=test_dataloader,
        model=model
    )

    return 0


if __name__ == "__main__":
    main(args)
