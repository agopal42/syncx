import numpy as np
from einops import rearrange
from sklearn.cluster import KMeans, SpectralClustering
import torch
from sklearn.metrics.cluster import adjusted_rand_score


def clip_and_rescale(input_tensor, clip_value):
    if torch.is_tensor(input_tensor):
        clipped = torch.clamp(input_tensor, min=0, max=clip_value)
    elif isinstance(input_tensor, np.ndarray):
        clipped = np.clip(input_tensor, a_min=0, a_max=clip_value)
    else:
        raise NotImplementedError

    return clipped * (1 / clip_value)


def spherical_to_cartesian_coordinates(x):
    # Second dimension of x contains spherical coordinates: (r, phi_1, ... phi_n).
    num_dims = x.shape[1]
    out = torch.zeros_like(x)

    r = x[:, 0]
    phi = x[:, 1:]

    sin_component = 1
    for i in range(num_dims - 1):
        out[:, i] = r * torch.cos(phi[:, i]) * sin_component
        sin_component = sin_component * torch.sin(phi[:, i])

    out[:, -1] = r * sin_component
    return out


def phase_to_cartesian_coordinates(phase_mask_threshold, phase, norm_magnitude):
    # Map phases on unit-circle and transform to cartesian coordinates.
    unit_circle_phase = torch.cat(
        (torch.ones_like(phase)[:, None], phase[:, None]), dim=1
    )
    if phase_mask_threshold != -1:
        # When magnitude is < phase_mask_threshold, use as multiplier to mask out respective phases from eval.
        unit_circle_phase = unit_circle_phase * norm_magnitude[:, None]
    # cartesian_form torch.Size([64, 2, 3, 32, 32])
    cartesian_form = spherical_to_cartesian_coordinates(unit_circle_phase)
    return cartesian_form


def syncx_phase_preprocess(features_to_cluster, img_resolution, model_output, gt_labels, 
    phase_mask_type, phase_mask_threshold):
    """ For SynCx evalulation we preprocess the phase here:
    1. set the magnitude-components to 1 of the complex-valued features
    2. elementwise convert from polar to cartesian the resultant complex-valued features from 1
    """

    # Take the features to cluster from the model output
    complex_features = model_output[features_to_cluster]
    magnitude = complex_features.abs()
    phase = complex_features.angle()
    # Optionally resize the feature maps
    if img_resolution != magnitude.shape[-2:]:
        magnitude = torch.nn.functional.interpolate(magnitude, size=img_resolution, mode="bilinear")
        phase = torch.nn.functional.interpolate(phase, size=img_resolution, mode="bilinear")

    # Map phases on unit-circle for each neuron-level activation [1, theta].
    polar_mag_and_phase = torch.cat(
        (torch.ones_like(phase)[:, None], phase[:, None]), dim=1
        )
    # Clip based on magnitudes (CAE-style)
    if phase_mask_type == "threshold":
        if torch.is_tensor(magnitude):
            clipped_mag = torch.clamp(magnitude, min=0, max=phase_mask_threshold)
        elif isinstance(magnitude, np.ndarray):
            clipped_mag = np.clip(magnitude, a_min=0, a_max=phase_mask_threshold)
        else:
            raise NotImplementedError("magnitude type not supported, expected torch.Tensor or np.ndarray")
        scaled_mag = clipped_mag * (1 / phase_mask_threshold)
        polar_mag_and_phase = polar_mag_and_phase * scaled_mag[:, None]
    # Filter out magnitudes of all background pixels
    elif phase_mask_type == "bg_only":
        bg_idxs = (gt_labels == 0).type(torch.float)[:, None].to(magnitude.device)
        scaled_mag = 1. - bg_idxs
        polar_mag_and_phase = polar_mag_and_phase * scaled_mag[:, None]

    # Spherical to cartesian coordinates
    cartesian_form = spherical_to_cartesian_coordinates(polar_mag_and_phase)
    # cartesian_form torch.Size([b, 2, c, h, w])
    return cartesian_form


def rotation_to_cartesian_coordinates(phase_mask_threshold, phase, magnitude, model_type, eps=1e-8):
    # Transform to cartesian coordinates.
    # torch.Size([bsz, p/r, c, h, w])
    assert model_type == "rf", "Only RF model is supported for rotation_to_cartesian_coordinates"
    cartesian_form = None
    if model_type in ["cae", "caev2"]:
        unit_circle_phase = torch.cat(
            (magnitude[:, None], phase[:, None]), dim=1
        )
        # torch.Size([bsz, p/r, c, h, w])
        cartesian_form = spherical_to_cartesian_coordinates(unit_circle_phase)
    elif model_type == "rf":
        cartesian_form = phase # rotation_output from RF model
    # determine the norm of the cartesian form
    norm_cartesian_form = torch.linalg.vector_norm(cartesian_form, dim=1)
    # w_eval
    weighted_mask = norm_cartesian_form >= phase_mask_threshold
    # z_norm
    normalized_cartesian_form = cartesian_form / (norm_cartesian_form[:, None] + eps)
    normalizing_coeff = torch.sum(weighted_mask[:, None], dim=2) + eps
    # weighted mean
    weighted_sum_cartesian_form = torch.sum(normalized_cartesian_form * weighted_mask[:, None], dim=2)
    # z_eval
    weighted_avg_cartesian_form = weighted_sum_cartesian_form / normalizing_coeff
    return weighted_avg_cartesian_form[:, :, None] # torch.Size([bsz, p/r, 1, h, w])


def apply_clustering(phase_mask_threshold, output_phase, output_norm_magnitude, 
        labels_true, img_resolution, seed, use_eval_type, model_outputs=None, 
        features_to_cluster="", phase_mask_type=""):
    
    input_phase = None
    if use_eval_type == "syncx":
        # Note: all the phase masking is handled by the function internally
        input_phase = syncx_phase_preprocess(
            features_to_cluster, img_resolution, model_outputs, 
            labels_true, phase_mask_type, phase_mask_threshold, 
            )
    elif use_eval_type == "rf":
        # Note: Pass unnormalized magnitudes as inputs.
        input_phase = rotation_to_cartesian_coordinates(
            phase_mask_threshold, output_phase, output_norm_magnitude, 
            use_eval_type
            )
    
    input_phase = input_phase.detach().cpu().numpy()
    input_phase = rearrange(input_phase, "b p c h w -> b h w (c p)")

    num_clusters = int(len(torch.unique(labels_true)))

    batch_size, num_angles = output_phase.shape[0:2]
    labels_pred = (
        np.zeros((batch_size, img_resolution[0], img_resolution[1]))
        + num_clusters
    )
    # Run k-means on each image separately.
    cluster_metrics_batch = {
        'inter_cluster_min': 0.,
        'inter_cluster_max': 0.,
        'inter_cluster_mean': 0.,
        'inter_cluster_std': 0.,
        'intra_cluster_dist': 0.,
        'intra_cluster_dist_safe': 0.,
        'intra_cluster_n_nan': 0.,
    }
    n_objects = []
    for img_idx in range(batch_size):
        in_phase = input_phase[img_idx]
        num_clusters_img = int(len(torch.unique(labels_true[img_idx])))
        n_objects.append(num_clusters_img - 1)  # -1 for background class
            
        # Remove areas in which objects overlap before k-means analysis.
        # filtering only for grayscale img
        if num_angles == 1: 
            label_idx = np.where(labels_true[img_idx].cpu().numpy() != -1)
            in_phase = in_phase[label_idx]        
        # flatten image before running k-means
        else:
            in_phase = rearrange(in_phase, "h w c -> (h w) c")
        
        # Apply clustering
        clustering = KMeans(
            n_clusters=num_clusters_img, n_init=5, random_state=seed
            ).fit(in_phase)
        # Calculate cluster inter- and intra-cluster distances (only for kmeans).
        cluster_metrics = calculate_cluster_metrics(in_phase, clustering)
        for key in cluster_metrics_batch.keys():
            cluster_metrics_batch[key] += cluster_metrics[key]

        # Create result image: fill in k_means labels & assign overlapping areas to class zero.
        cluster_img = (
            np.zeros((img_resolution[0], img_resolution[1])) + num_clusters
        )
        # for grayscale img -> assign cluster idxs to only non-overlapping regions
        if num_angles == 1:
            cluster_img[label_idx] = clustering.labels_
        # for colour img -> assign cluster idxs to all regions
        else:
            cluster_img = np.reshape(
                clustering.labels_, (img_resolution[0], img_resolution[1])
            )
        labels_pred[img_idx] = cluster_img
    n_objects = np.array(n_objects)

    # Calculate average cluster metrics.
    for key in cluster_metrics_batch.keys():
        cluster_metrics_batch[key] /= batch_size
    return labels_pred, cluster_metrics_batch, n_objects


def calc_ari_score(batch_size, labels_true, labels_pred, with_background):
    ari = 0
    per_sample_ari = []
    for idx in range(batch_size):
        if with_background:
            area_to_eval = np.where(
                labels_true[idx] > -1
            )  # Remove areas in which objects overlap.
        else:
            area_to_eval = np.where(
                labels_true[idx] > 0
            )  # Remove background & areas in which objects overlap.

        sample_ari = adjusted_rand_score(
            labels_true[idx][area_to_eval], labels_pred[idx][area_to_eval]
        )
        ari += sample_ari
        per_sample_ari.append(sample_ari)
    per_sample_ari = np.array(per_sample_ari)

    return ari / batch_size, per_sample_ari


def get_ari_for_n_objects(ari_w_bg_scores, ari_wo_bg_scores, n_objects, n_objects_max):
    ari_w_bg_per_n_objects = []
    ari_wo_bg_per_n_objects = []
    n_samples_per_n_objects = []
    for i in range(n_objects_max):
        idxs = n_objects == i
        n_samples_per_n_objects.append(np.sum(idxs))
        # take the sum because we want to average over all samples once we log the results
        ari_w_bg_per_n_objects.append(np.sum(ari_w_bg_scores[idxs]))
        ari_wo_bg_per_n_objects.append(np.sum(ari_wo_bg_scores[idxs]))
    return ari_w_bg_per_n_objects, ari_wo_bg_per_n_objects, n_samples_per_n_objects


def calculate_cluster_metrics(in_phase, k_means):

        # Calculate inter-cluster distances.
        centroids = k_means.cluster_centers_
        c1 = np.expand_dims(centroids, axis=0)
        c2 = np.expand_dims(centroids, axis=1)
        diff = (c1 - c2) ** 2  # n_clusters x n_clusters x RGB * 2 (euclidean coordinates)
        diff = np.mean(diff, axis=-1)
        # Create mask to ignore diagonal elements.
        mask = np.ones(diff.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        # Calculate inter-cluster distance metrics.
        inter_cluster_min = np.min(diff[mask])
        inter_cluster_max = np.max(diff[mask])
        inter_cluster_mean = np.mean(diff[mask])
        inter_cluster_std = np.std(diff[mask])

        # Calculate intra-cluster distance.
        centroids_expanded = np.expand_dims(centroids, axis=0)
        centroids_expanded = np.repeat(centroids_expanded, in_phase.shape[0], axis=0)
        intra_cluster_dist = []
        intra_cluster_dist_safe = []
        intra_cluster_n_nan = []
        for i in range(centroids.shape[0]):  # number of clusters
            # Mask out all points that are not in cluster i.
            arg1 = in_phase[k_means.labels_ == i]
            arg2 = centroids_expanded[k_means.labels_ == i]
            diff = (arg1 - arg2[:, i]) ** 2
            intra_cluster_dist.append(np.mean(diff))
            intra_cluster_dist_safe.append(np.nan_to_num(np.mean(diff)))
            intra_cluster_n_nan.append(np.count_nonzero(np.isnan(diff)))
        intra_cluster_dist = np.mean(intra_cluster_dist)
        intra_cluster_dist_safe = np.mean(intra_cluster_dist_safe)
        intra_cluster_n_nan = np.mean(intra_cluster_n_nan)
        
        return {
            'inter_cluster_min': inter_cluster_min,
            'inter_cluster_max': inter_cluster_max,
            'inter_cluster_mean': inter_cluster_mean,
            'inter_cluster_std': inter_cluster_std,
            'intra_cluster_dist': intra_cluster_dist,
            'intra_cluster_dist_safe': intra_cluster_dist_safe,
            'intra_cluster_n_nan': intra_cluster_n_nan,
        }