from typing import Optional, List
from dataclasses import dataclass, field

import torch
import wandb

from utils.utils_eval import ( 
    apply_clustering,
    calc_ari_score,
    get_ari_for_n_objects
)
from utils.utils_plot import (
    create_phase_colorcoded_groupings_and_radial_plots, 
    create_image_grids_for_logging,
    tensor_to_wandb_image
)


@dataclass
class SynCxMetricsRecorder:
    num_batches: int = 0
    loss_total: float = 0.
    loss_rec: float = 0.
    loss_phase: float = 0.
    loss_spatial: float = 0.
    loss_cw_magnitude: float = 0.
    total_norm: float = 0.
    ari_w_bg: float = 0.
    ari_wo_bg: float = 0.
    N_MAX_OBJECTS: int = 8  # In CLEVR max objects in the image is 7, dsprites it is 6.
    ari_w_bg_per_n_objects: List[float] = field(default_factory=lambda: [0. for _ in range(SynCxMetricsRecorder.N_MAX_OBJECTS)])
    ari_wo_bg_per_n_objects: List[float] = field(default_factory=lambda: [0. for _ in range(SynCxMetricsRecorder.N_MAX_OBJECTS)])
    n_samples_per_n_objects: List[int] = field(default_factory=lambda: [0. for _ in range(SynCxMetricsRecorder.N_MAX_OBJECTS)])
    inter_cluster_min: float = 0.
    inter_cluster_max: float = 0.
    inter_cluster_mean: float = 0.
    inter_cluster_std: float = 0.
    intra_cluster_dist: float = 0.
    intra_cluster_dist_safe: float = 0.
    intra_cluster_n_nan: float = 0.
    img_grid_all: Optional[wandb.Image] = None
    img_grid_main: Optional[wandb.Image] = None
    
    @staticmethod
    def get_init_value_for_early_stop_score():
        return float('-inf')
    
    def get_current_early_stop_score(self):
        return self.ari_wo_bg / self.num_batches
    
    def early_stop_score_improved(self, ari_current, ari_best):
        if ari_current > ari_best:
            return True
        return False
    
    def step(self, args, model_input, model_output):

        self.num_batches += 1
        self.loss_total += model_output["loss"].item()
        self.loss_rec += model_output["loss_rec"].item()
        if 'total_norm' in model_output:
            self.total_norm += model_output["total_norm"].item()

        # Determine output phase predictions
        gt_labels = model_input["labels"]
        output_phase = model_output["complex_phase"]
        img_resolution = tuple(output_phase.shape[-2:])
        # No clip and rescale used for SynCx eval
        output_magnitude_scaled = None
        
        output_labels_pred, cluster_metrics, n_objects = apply_clustering(
            args.phase_mask_threshold, output_phase, output_magnitude_scaled, 
            gt_labels, img_resolution, args.seed, args.use_eval_type,
            model_output,  args.features_to_cluster, args.phase_mask_type
        )

        # ARI score
        ari_w_bg, ari_w_bg_per_sample = calc_ari_score(gt_labels.shape[0], gt_labels, output_labels_pred, with_background=True)
        ari_wo_bg, ari_wo_bg_per_sample = calc_ari_score(gt_labels.shape[0], gt_labels, output_labels_pred, with_background=False)
        self.ari_w_bg += ari_w_bg
        self.ari_wo_bg += ari_wo_bg
        # Calculate ARI score wrt to the number of objects in the image
        ari_w_bg_per_n_objects, ari_wo_bg_per_n_objects, n_samples_per_n_objects = get_ari_for_n_objects(
            ari_w_bg_per_sample, ari_wo_bg_per_sample, n_objects, n_objects_max=self.N_MAX_OBJECTS)
        for i in range(self.N_MAX_OBJECTS):
            if n_samples_per_n_objects[i] > 0:
                self.ari_w_bg_per_n_objects[i] += ari_w_bg_per_n_objects[i]
                self.ari_wo_bg_per_n_objects[i] += ari_wo_bg_per_n_objects[i]
                self.n_samples_per_n_objects[i] += n_samples_per_n_objects[i]

        # Cluster metrics
        self.inter_cluster_min += cluster_metrics["inter_cluster_min"]
        self.inter_cluster_max += cluster_metrics["inter_cluster_max"]
        self.inter_cluster_mean += cluster_metrics["inter_cluster_mean"]
        self.inter_cluster_std += cluster_metrics["inter_cluster_std"]
        self.intra_cluster_dist += cluster_metrics["intra_cluster_dist"]
        self.intra_cluster_dist_safe += cluster_metrics["intra_cluster_dist_safe"]
        self.intra_cluster_n_nan += cluster_metrics["intra_cluster_n_nan"]

        # Visual logs - rendering images is costly, so render them only once
        if self.img_grid_all is None and args.n_images_to_log > 0:
            input_images = model_input["images"][:args.n_images_to_log]
            # reconstruction from the last cw_rnn iteration
            if args.step_loss_type == "multi_scale":
                reconstructed_images = model_output["reconstruction"][-1][:args.n_images_to_log]
            else:
                reconstructed_images = model_output["reconstruction"][:args.n_images_to_log, -1]
            gt_label = model_input["labels"][:args.n_images_to_log]
            # mask magnitudes/phases of background pixels in viz as in eval.
            bg_idxs = (gt_label == 0).type(torch.float)[:, None].detach().cpu().numpy()
            pred_label = output_labels_pred[:args.n_images_to_log]
            plot_resize_resolution = (args.plot_resize_resolution, args.plot_resize_resolution)
            # phase_map_iters
            # (b, n_iters, 1, 32, 32)
            plot_enc_layer_radial, plot_enc_layer_groups_phase, plot_enc_layer_groups_magnitude = [], [], []
            # Last iteration phase and magnitude
            plot_radial, plot_groups_phase, plot_groups_magnitude = None, None, None
            if isinstance(model_output["rnn_phase_map_iters"], torch.Tensor):
                phase_map_iters = model_output["rnn_phase_map_iters"]
                magnitude_map_iters = model_output["rnn_magnitude_map_iters"]
                n_iters = phase_map_iters.shape[1]
                for i in range(n_iters):
                    phase = phase_map_iters[:args.n_images_to_log, i]
                    phase = phase.detach().cpu().numpy()
                    phase = phase * (1. - bg_idxs)
                    magnitude = magnitude_map_iters[:args.n_images_to_log, i]
                    magnitude = magnitude.detach().cpu().numpy()
                    magnitude = magnitude * (1. - bg_idxs)
                    radial_plot, groups_plot_phase, groups_plot_magnitude = (
                        create_phase_colorcoded_groupings_and_radial_plots(
                            plot_resize_resolution, phase, magnitude, gt_label))
                    plot_enc_layer_radial.append(radial_plot)
                    plot_enc_layer_groups_phase.append(groups_plot_phase)
                    plot_enc_layer_groups_magnitude.append(groups_plot_magnitude)
                    # Last iteration phase and magnitude 
                    if i == n_iters-1:
                        plot_radial = radial_plot
                        plot_groups_phase = groups_plot_phase
                        plot_groups_magnitude = groups_plot_magnitude

            # pass None to plotting function if lists are empty
            if not plot_enc_layer_radial:
                plot_enc_layer_radial = None
                plot_enc_layer_groups_phase = None
            
            outputs_plots = create_image_grids_for_logging(
                plot_resize_resolution=plot_resize_resolution,
                img_in=input_images,
                img_rec=reconstructed_images,
                gt_label=gt_label,
                pred_label=pred_label,
                plot_radial=plot_radial,
                plot_groups_phase=plot_groups_phase,
                plot_enc_layer_radial=plot_enc_layer_radial,
                plot_enc_layer_groups_phase=plot_enc_layer_groups_phase,
            )

            self.img_grid_all = tensor_to_wandb_image(outputs_plots['img_grid_all'])
            self.img_grid_main = tensor_to_wandb_image(outputs_plots['img_grid_main'])
    
    def log(self):
        logs = {
            "loss": self.loss_total / self.num_batches,
            "loss_rec": self.loss_rec / self.num_batches,
            "total_norm": self.total_norm / self.num_batches,
            "ARI-FULL": self.ari_w_bg / self.num_batches,
            "ARI-FG": self.ari_wo_bg / self.num_batches,
            "inter_cluster_min": self.inter_cluster_min / self.num_batches,
            "inter_cluster_max": self.inter_cluster_max / self.num_batches,
            "inter_cluster_mean": self.inter_cluster_mean / self.num_batches,
            "inter_cluster_std": self.inter_cluster_std / self.num_batches,
            "intra_cluster_dist": self.intra_cluster_dist / self.num_batches,
            "intra_cluster_dist_safe": self.intra_cluster_dist_safe / self.num_batches,
            "intra_cluster_n_nan": self.intra_cluster_n_nan / self.num_batches,
            # Visual logs
            "img_grid_all": self.img_grid_all,
            "img_grid_main": self.img_grid_main,
        }
        for i in range(1, len(self.n_samples_per_n_objects)):
            logs[f"ARI-FG-{i}"] = self.ari_wo_bg_per_n_objects[i] / self.n_samples_per_n_objects[i] if self.n_samples_per_n_objects[i] > 0 else 0
            logs[f"ARI-FULL-{i}"] = self.ari_w_bg_per_n_objects[i] / self.n_samples_per_n_objects[i] if self.n_samples_per_n_objects[i] > 0 else 0
            logs[f"n_images-w-{i}-objects"] = self.n_samples_per_n_objects[i]
        return logs