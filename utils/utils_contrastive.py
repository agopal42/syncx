from typing import Tuple

import random
import math

import torch
import torch.nn.functional as F

from einops import rearrange


def get_anchors_positive_and_negative_pairs(
    phase: torch.Tensor,
    magnitude: torch.Tensor,
    n_anchors_to_sample: int,
    n_positive_pairs: int,
    n_negative_pairs: int,
    top_k: int,
    bottom_m: int,
    use_avg_delta_theta: bool,
    use_patches: bool,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Colab used to prototype this function:
    # https://colab.research.google.com/drive/1tghM5STnbCePqV5dt8VDoCoWBUoox_xf?usp=sharing
    # phase: B, C, H, W
    # magnitude: B, C, H, W

    if use_patches:
        # divide phase and magnitude maps into patches of size p x p
        assert phase.size()[-1] % patch_size == 0, "Patch-size not a factor of output maps size!!!!"
        h_patched = w_patched = phase.size()[-1] // patch_size 
        phase = torch.nn.functional.unfold(phase, kernel_size=patch_size, stride=patch_size)
        phase = rearrange(phase, 'b c (h w) -> b c h w', h=h_patched, w=w_patched)
        magnitude = torch.nn.functional.unfold(magnitude, kernel_size=patch_size, stride=patch_size)
        magnitude = rearrange(magnitude, 'b c (h w) -> b c h w', h=h_patched, w=w_patched)
    
    anchor = phase.flatten(start_dim=-2)
    anchor = anchor.unsqueeze(-1).unsqueeze(-1)
    n_anchors_total = anchor.shape[-3]

    per_anchor_phase = torch.unsqueeze(phase, dim=-3)
    per_anchor_phase = torch.repeat_interleave(per_anchor_phase, repeats=n_anchors_total, dim=-3)

    per_anchor_magnitude = torch.unsqueeze(magnitude, dim=-3)
    per_anchor_magnitude = torch.repeat_interleave(per_anchor_magnitude, repeats=n_anchors_total, dim=-3)

    # Sample random anchors
    n_anchors_to_sample = min(n_anchors_to_sample, n_anchors_total)
    indices = random.sample(list(range(n_anchors_total)), k=n_anchors_to_sample)
    anchor = anchor[:, :, indices]
    anchor_features = magnitude.flatten(start_dim=-2)
    anchor_features = anchor_features[:, :, indices]
    per_anchor_phase = per_anchor_phase[:, :, indices]
    per_anchor_magnitude = per_anchor_magnitude[:, :, indices]

    # Compute cyclical differences
    # anchor: B, C, Na, 1, 1
    # per_anchor_phase: B, C, Na, H, W
    abs_diff = torch.abs(per_anchor_phase - anchor)
    delta_theta = torch.minimum(abs_diff, 2 * math.pi - abs_diff)

    # Average delta_theta over channels, but then repeat it (such that the gather below works properly)
    # TODO(astanic): this is actually a hack, we should probably have a better distance metric
    if use_avg_delta_theta:
        n_channels = delta_theta.shape[1]
        delta_theta = delta_theta.mean(dim=1, keepdim=True)
        delta_theta = torch.repeat_interleave(delta_theta, repeats=n_channels, dim=1)

    # Now sort
    delta_theta_for_sorting = torch.flatten(delta_theta, start_dim=-2)
    delta_theta_argsort = torch.argsort(delta_theta_for_sorting, dim=-1)

    # Pick 1/top-K and N-1/bottom-M
    # Note: we start at 1 because we don't want to pick the anchor itself
    positives_idcs = random.sample(list(range(1, top_k+1)), k=n_positive_pairs)
    n_phases = delta_theta_argsort.shape[-1]
    negatives_idcs = random.sample(range(n_phases - bottom_m, n_phases), k=n_negative_pairs)

    per_anchor_magnitude_flat = torch.flatten(per_anchor_magnitude, start_dim=-2)
    phase_positives = torch.take_along_dim(per_anchor_magnitude_flat, indices=delta_theta_argsort[..., positives_idcs], dim=-1)
    phase_negatives = torch.take_along_dim(per_anchor_magnitude_flat, indices=delta_theta_argsort[..., negatives_idcs], dim=-1)

    return anchor_features, phase_positives, phase_negatives


def get_anchors_positive_and_negative_pairs_mixed(
    addresses: torch.Tensor,
    features: torch.Tensor,
    n_anchors_to_sample: int,
    n_positive_pairs: int,
    n_negative_pairs: int,
    top_k: int,
    bottom_m: int,
    address_distance_type: str,
    use_avg_delta_theta: bool,
    use_patches: bool,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # phase: B, C, H, W
    # features: B, C, H, W

    if use_patches:
        # divide phase and magnitude maps into patches of size p x p
        assert addresses.size()[-1] % patch_size == 0, "Patch-size not a factor of output maps size!!!!"
        h_patched = w_patched = addresses.size()[-1] // patch_size 
        addresses = torch.nn.functional.unfold(addresses, kernel_size=patch_size, stride=patch_size)
        addresses = rearrange(addresses, 'b c (h w) -> b c h w', h=h_patched, w=w_patched)
        features = torch.nn.functional.unfold(features, kernel_size=patch_size, stride=patch_size)
        features = rearrange(features, 'b c (h w) -> b c h w', h=h_patched, w=w_patched)

    anchor = addresses.flatten(start_dim=-2)
    anchor = anchor.unsqueeze(-1).unsqueeze(-1)
    n_anchors_total = anchor.shape[-3]

    per_anchor_addresses = torch.unsqueeze(addresses, dim=-3)
    per_anchor_addresses = torch.repeat_interleave(per_anchor_addresses, repeats=n_anchors_total, dim=-3)

    per_anchor_features = torch.unsqueeze(features, dim=-3)
    per_anchor_features = torch.repeat_interleave(per_anchor_features, repeats=n_anchors_total, dim=-3)

    # Sample random anchors
    n_anchors_to_sample = min(n_anchors_to_sample, n_anchors_total)
    indices = random.sample(list(range(n_anchors_total)), k=n_anchors_to_sample)
    anchor = anchor[:, :, indices]
    anchor_features = features.flatten(start_dim=-2)
    anchor_features = anchor_features[:, :, indices]
    per_anchor_addresses = per_anchor_addresses[:, :, indices]
    per_anchor_features = per_anchor_features[:, :, indices]

    # anchor: B, C, Na, 1, 1
    # per_anchor_phase: B, C, Na, H, W
    n_channels = features.shape[1]
    if address_distance_type == 'cyclical_phase':
        # Compute cyclical differences
        abs_diff = torch.abs(per_anchor_addresses - anchor)
        delta_address = torch.minimum(abs_diff, 2 * math.pi - abs_diff)
        if use_avg_delta_theta:
            delta_address = delta_address.mean(dim=1, keepdim=True)
    elif address_distance_type == 'euclidean':
        sqr_norm_A = torch.sum(torch.pow(anchor, 2), dim=1, keepdim=True)
        sqr_norm_B = torch.sum(torch.pow(per_anchor_addresses, 2), dim=1, keepdim=True)
        inner_prod = torch.sum(anchor * per_anchor_addresses, dim=1, keepdim=True)
        delta_address = sqr_norm_A + sqr_norm_B - 2 * inner_prod
    elif address_distance_type == 'cosine':
        # TODO: check if this is correct
        anchor = F.normalize(anchor, dim=1)
        per_anchor_addresses = F.normalize(per_anchor_addresses, dim=1)
        delta_address = torch.sum(anchor * per_anchor_addresses, dim=1, keepdim=True)
        delta_address = 1 - delta_address
    else:
        raise ValueError(f'Unknown address_distance_type: {address_distance_type}')

    # Average delta_address over channels, but then repeat it (such that the gather below works properly)
    # TODO(astanic): this is actually a hack, we should probably have a better distance metric
    if use_avg_delta_theta or address_distance_type in ["cosine", "euclidean"]:
        delta_address = torch.repeat_interleave(delta_address, repeats=n_channels, dim=1)

    # Now sort
    delta_address_for_sorting = torch.flatten(delta_address, start_dim=-2)
    delta_address_argsort = torch.argsort(delta_address_for_sorting, dim=-1)

    # Pick 1/top-K and N-1/bottom-M
    # Note: we start at 1 because we don't want to pick the anchor itself
    positives_idcs = random.sample(list(range(1, top_k+1)), k=n_positive_pairs)
    n_phases = delta_address_argsort.shape[-1]
    negatives_idcs = random.sample(range(n_phases - bottom_m, n_phases), k=n_negative_pairs)

    per_anchor_features_flat = torch.flatten(per_anchor_features, start_dim=-2)
    positives = torch.take_along_dim(per_anchor_features_flat, indices=delta_address_argsort[..., positives_idcs], dim=-1)
    negatives = torch.take_along_dim(per_anchor_features_flat, indices=delta_address_argsort[..., negatives_idcs], dim=-1)

    return anchor_features, positives, negatives


def contrastive_soft_target_loss(anchors, positive_pairs, negative_pairs, temperature, min_temperature, learn_inverse_temperature, adjust_phase_range_for_the_loss):
    # anchors: B, C, Na
    # positive_pairs: B, C, Na, Np
    # negative_pairs: B, C, Na, Nn

    device = anchors.device
    batch_size, n_anchors = anchors.shape[0], anchors.shape[-1]
    n_positive_pairs = positive_pairs.shape[-1]
    n_negative_pairs = negative_pairs.shape[-1]

    pairs_concat = torch.cat([positive_pairs, negative_pairs], dim=-1)

    # Here the anchor and the pairs are in the [-pi, pi] range
    if adjust_phase_range_for_the_loss:
        # Broadcast anchors to the pairs
        anchors = torch.unsqueeze(anchors, dim=-1)
        # Repeat anchors to the pairs
        anchors = torch.repeat_interleave(anchors, repeats=pairs_concat.shape[-1], dim=-1)

        # Compute cyclical differences
        abs_diff = torch.abs(pairs_concat - anchors)

        # Create the first mask
        mask1 = abs_diff > math.pi

        # Create the second mask
        mask2 = anchors > 0

        # Merge masks
        mask_subtract_pi_from_anchors = mask1 * mask2
        mask_add_pi_to_anchors = mask1 * ~mask2

        anchors = torch.where(mask_subtract_pi_from_anchors, anchors - math.pi, anchors)
        pairs_concat = torch.where(mask_subtract_pi_from_anchors, pairs_concat + math.pi, pairs_concat)

        anchors = torch.where(mask_add_pi_to_anchors, anchors + math.pi, anchors)
        pairs_concat = torch.where(mask_add_pi_to_anchors, pairs_concat - math.pi, pairs_concat)

        # Normalize the anchors and pairs before the dot product
        anchors = F.normalize(anchors, dim=1)
        pairs_concat = F.normalize(pairs_concat, dim=1)

        logits = torch.einsum("bcAK, bcAK -> bAK", anchors, pairs_concat)
    else:
        # Normalize the anchors and pairs before the dot product
        anchors = F.normalize(anchors, dim=1)
        pairs_concat = F.normalize(pairs_concat, dim=1)

        logits = torch.einsum("bcA, bcAK -> bAK", anchors, pairs_concat)

    if learn_inverse_temperature:
        temperature = torch.clamp(temperature, max=1/min_temperature) if type(temperature) == torch.Tensor else temperature
        logits *= temperature
    else:
        temperature = torch.clamp(temperature, min=min_temperature) if type(temperature) == torch.Tensor else temperature
        logits /= temperature
    logits = rearrange(logits, "b A K -> (b A) K")

    labels_pos = torch.ones((batch_size * n_anchors, n_positive_pairs), dtype=torch.float32, device=device)
    labels_pos /= n_positive_pairs  # from (1 1 1 1 0 0 0 0 ..) to (.25 .25 .25 .25 0 0 0 0 ..)
    labels_neg = torch.zeros((batch_size * n_anchors, n_negative_pairs), dtype=torch.float32, device=device)
    labels = torch.cat([labels_pos, labels_neg], dim=-1)
    # Reference:
    # https://timm.fast.ai/loss.cross_entropy#SoftTargetCrossEntropy
    # https://github.com/huggingface/pytorch-image-models/blob/9a38416fbdfd0d38e6922eee5d664e8ec7fbc356/timm/loss/cross_entropy.py#L29
    loss = torch.sum(-labels * F.log_softmax(logits, dim=-1), dim=-1)
    loss = loss.mean()

    return loss


def pairwise_euclidean_distance(A, B):
    # batch size
    batchA = A.shape[0]
    batchB = B.shape[0]
    # number of vectors in each matrix
    nA = A.shape[1]
    nB = B.shape[1]

    sqr_norm_A = torch.reshape(torch.sum(torch.pow(A, 2), dim=-1), [batchA, 1, nA])
    sqr_norm_B = torch.reshape(torch.sum(torch.pow(B, 2), dim=-1), [batchB, nB, 1])

    inner_prod = torch.matmul(B, A.transpose(-2, -1))

    tile_1 = torch.tile(sqr_norm_A, [1, nB, 1])
    tile_2 = torch.tile(sqr_norm_B, [1, 1, nA])

    result = tile_1 + tile_2 - 2 * inner_prod
    return result


def pairwise_cosine_distance(A, B):
    normalized_A = torch.nn.functional.normalize(A, dim=-1)
    normalized_B = torch.nn.functional.normalize(B, dim=-1)
    prod = torch.matmul(normalized_A, normalized_B.transpose(-2, -1).conj())
    # 1 - prod because we want cosine *distance*, not cosine *similarity*
    result = 1 - prod
    return result


def contrastive_snn_loss(anchors, positive_pairs, negative_pairs, temperature, min_temperature, learn_inverse_temperature, distance_metric, use_vectorized_snn_loss):
    # anchors: B, C, Na
    # positive_pairs: B, C, Na, Np
    # negative_pairs: B, C, Na, Nn

    device = anchors.device
    batch_size, n_anchors = anchors.shape[0], anchors.shape[-1]
    n_positive_pairs = positive_pairs.shape[-1]
    n_negative_pairs = negative_pairs.shape[-1]

    pairs_concat = torch.cat([positive_pairs, negative_pairs], dim=-1)

    # Note: add +1 to the number of positive pairs to account for the anchor being a positive sample to itself
    labels_pos = torch.ones((batch_size * n_anchors, n_positive_pairs + 1), dtype=torch.float32, device=device)
    labels_neg = torch.zeros((batch_size * n_anchors, n_negative_pairs), dtype=torch.float32, device=device)
    labels = torch.cat([labels_pos, labels_neg], dim=-1)

    # Prepare anchors and pairs for the SNN loss
    # Concatenate anchors and pairs
    anchors = rearrange(anchors, "b c A -> b A 1 c")
    pairs_concat = rearrange(pairs_concat, "b c A K -> b A K c")
    anchors_and_pairs = torch.cat([anchors, pairs_concat], dim=-2)
    # Flatten batch and anchor dimension
    anchors_and_pairs = rearrange(anchors_and_pairs, "b A K c -> (b A) K c")

    # Note: all the above tensor manipulations are to interface SNN loss with the rest of the code
    loss = soft_nearest_neighbor_loss(anchors_and_pairs, labels, temperature, min_temperature, learn_inverse_temperature, distance_metric, use_vectorized_snn_loss)
    return loss


def soft_nearest_neighbor_loss(x, y, temperature, min_temperature, learn_inverse_temperature, distance_metric, use_vectorized_snn_loss):
    """
    Reference playground colab: https://colab.research.google.com/drive/1w1zWUal387oKoEYvR_Vzxi9mI3v7UEsD?usp=sharing
    x: a tensor of shape (batch_size, num_points, num_dims).
    y: a tensor of shape (batch_size, num_points).
    """
    STABILITY_EPS = 1e-5

    if distance_metric == 'euclidean':
        distance_matrix = pairwise_euclidean_distance(x, x)
    elif distance_metric == 'cosine':
        distance_matrix = pairwise_cosine_distance(x, x)
    else:
        raise ValueError(f'Unknown distance metric: {distance_metric}.')

    if learn_inverse_temperature:
        temperature = torch.clamp(temperature, max=1/min_temperature) if type(temperature) == torch.Tensor else temperature
        distance_matrix = torch.exp(-(distance_matrix * temperature))
    else:
        temperature = torch.clamp(temperature, min=min_temperature) if type(temperature) == torch.Tensor else temperature
        distance_matrix = torch.exp(-(distance_matrix / temperature))

    # Row normalized exponentiated pairwise distance between all the elements of x.
    # The probability of sampling a neighbor point for every element of x, proportional to the distance between the points.
    # By subtracting torch.eye we set the diagnoal entries to 0 (exp(0)-1=0), so that the probability of sampling the same point is 0.
    numerator = distance_matrix - torch.eye(x.shape[-2], device=x.device).unsqueeze(dim=0)
    denominator = STABILITY_EPS + torch.sum(numerator, dim=-1, keepdim=True)
    pick_probability = numerator / denominator

    # Masking matrix such that element i,j is 1 iff y[i] == y[j], for i,j=1..K.
    y_unsqueezed = torch.unsqueeze(y, dim=-1)
    equal = []
    for i in range(y.shape[0]):
        equal_mat_sample = (y[i] == y_unsqueezed[i])
        if use_vectorized_snn_loss:
            equal_mat_sample[1:,:] = False
        equal.append(equal_mat_sample)
    equal = torch.stack(equal)
    same_label_mask = torch.squeeze(equal).to(torch.float32)

    # The pairwise sampling probabilities for the elements of x for neighbor points which share labels.
    masked_pick_probability = pick_probability * same_label_mask
    summed_masked_pick_prob = torch.sum(masked_pick_probability, dim=-1)
    snn_loss = torch.mean(-torch.log(STABILITY_EPS + summed_masked_pick_prob))
    return snn_loss
