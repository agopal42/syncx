from typing import Optional, List
from dataclasses import dataclass, field
import wandb

from utils.utils_eval import (
    apply_clustering,
    calc_ari_score,
    clip_and_rescale,
    get_ari_for_n_objects
    )
import utils.utils_plot as utils_plot


@dataclass
class RFMetricsRecorder:
    num_batches: int = 0
    loss_total: float = 0.
    loss_rec: float = 0.
    loss_phase: float = 0.
    loss_spatial: float = 0.
    ari_w_bg: float = 0.
    ari_wo_bg: float = 0.
    N_MAX_OBJECTS: int = 8  # In CLEVR max objects in the image is 7, dsprites it is 6.
    ari_w_bg_per_n_objects: List[float] = field(default_factory=lambda: [0. for _ in range(RFMetricsRecorder.N_MAX_OBJECTS)])
    ari_wo_bg_per_n_objects: List[float] = field(default_factory=lambda: [0. for _ in range(RFMetricsRecorder.N_MAX_OBJECTS)])
    n_samples_per_n_objects: List[int] = field(default_factory=lambda: [0. for _ in range(RFMetricsRecorder.N_MAX_OBJECTS)])
    inter_cluster_min: float = 0.
    inter_cluster_max: float = 0.
    inter_cluster_mean: float = 0.
    inter_cluster_std: float = 0.
    intra_cluster_dist: float = 0.
    intra_cluster_dist_safe: float = 0.
    intra_cluster_n_nan: float = 0.
    img_grid_main: Optional[wandb.Image] = None
    img_enc_gates: Optional[wandb.Image] = None
    img_dec_conv_gates: Optional[wandb.Image] = None
    img_dec_fc_gate: Optional[wandb.Image] = None

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
        if 'loss_spatial' in model_output:
            self.loss_spatial += model_output["loss_spatial"].item()
        if 'loss_phase' in model_output:
            self.loss_phase += model_output["loss_phase"].item()
        # Determine output phase predictions
        gt_labels = model_input["labels"]
        z_out = model_output["rotation_output"]
        img_resolution = tuple(z_out.shape[-2:])

        if args.use_eval_type == "rf":
            output_magnitude_scaled = model_output["rotation_magnitude"]
        else:
            output_magnitude_scaled = clip_and_rescale(model_output["rotation_magnitude"], args.phase_mask_threshold)

        output_labels_pred, cluster_metrics, n_objects = apply_clustering(
            args.phase_mask_threshold, z_out, output_magnitude_scaled, gt_labels,
            img_resolution, args.seed, args.use_eval_type
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
        if self.img_grid_main is None and args.n_images_to_log > 0:
            input_images = model_input["images"][:args.n_images_to_log]
            reconstructed_images = model_output["reconstruction"][:args.n_images_to_log]
            gt_label = model_input["labels"][:args.n_images_to_log]
            pred_label = output_labels_pred[:args.n_images_to_log]
            plot_resize_resolution = (args.plot_resize_resolution, args.plot_resize_resolution)

            outputs_plots = utils_plot.create_image_grids_for_logging(
                plot_resize_resolution=plot_resize_resolution,
                img_in=input_images,
                img_rec=reconstructed_images,
                gt_label=gt_label,
                pred_label=pred_label,
            )
            self.img_grid_main = utils_plot.tensor_to_wandb_image(outputs_plots['img_grid_main'])
            
    def log(self):
        logs = {
            "loss": self.loss_total / self.num_batches,
            "loss_rec": self.loss_rec / self.num_batches,
            "loss_phase": self.loss_phase / self.num_batches,
            "loss_spatial": self.loss_spatial / self.num_batches,
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
            "img_grid_main": self.img_grid_main,
        }
        if self.img_enc_gates is not None:
            logs["img_enc_gates"] = self.img_enc_gates
            logs["img_dec_conv_gates"] = self.img_dec_conv_gates
            logs["img_dec_fc_gate"] = self.img_dec_fc_gate

        for i in range(1, len(self.n_samples_per_n_objects)):
            logs[f"ARI-FG-{i}"] = self.ari_wo_bg_per_n_objects[i] / self.n_samples_per_n_objects[i] if self.n_samples_per_n_objects[i] > 0 else 0
            logs[f"ARI-FULL-{i}"] = self.ari_w_bg_per_n_objects[i] / self.n_samples_per_n_objects[i] if self.n_samples_per_n_objects[i] > 0 else 0
            logs[f"n_images-w-{i}-objects"] = self.n_samples_per_n_objects[i]
        return logs
