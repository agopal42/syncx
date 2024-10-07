import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from sklearn.manifold import TSNE
import umap

from typing import Tuple, Sequence, Optional

import numpy as np
import torch
from torch.nn import functional as F
import PIL
import wandb

from einops import rearrange, reduce
from utils.utils_eval import spherical_to_cartesian_coordinates


BG_COLOR = '#000000'
SLOT_COLORS = [
    '#1f77b4',  #tab:blue
    '#ff7f0e',  #tab:orange
    '#2ca02c',  #tab:green
    '#d62728',  #tab:red
    '#9467bd',  #tab:purple
    '#8c564b',  #tab:brown
    '#e377c2',  #tab:pink
    '#7f7f7f',  #tab:gray
    '#bcbd22',  #tab:olive
    '#17becf',  #tab:cyan
    '#ffffff',  #tab:white
]


def color_hex_to_int(
    hex_color: str
    ) -> np.ndarray:
    h = hex_color.lstrip('#')
    return np.asarray(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)), dtype=np.int32)


def color_hex_to_float(
    hex_color: str
    ) -> float:
    color_int = color_hex_to_int(hex_color)
    return color_int / 255.


def tensor_to_wandb_image(
    data: torch.Tensor
    ) -> wandb.Image:
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if data.shape[0] in [1, 3]:
        # CHW->HWC
        data = np.transpose(data, (1,2,0))
    if data.shape[-1] == 1:
        # HW1->HW3
        data = np.repeat(data, 3, axis=-1)
    data = np.clip(data, a_min=0, a_max=1)
    data = PIL.Image.fromarray(np.ascontiguousarray((data*255.0).astype(np.uint8)), mode="RGB")
    return wandb.Image(data)


def append_border_pixels(
    imgs: np.ndarray, 
    color: str
    ) -> np.ndarray:
    """Assumes last 3 dimensions of tensors are images HWC."""
    # Create pad_width for np.pad
    border_pixel_width = 2
    pad_width = [(0, 0) for _ in range(imgs.ndim)]
    pad_width[-3] = (border_pixel_width, border_pixel_width)
    pad_width[-2] = (border_pixel_width, border_pixel_width)
    pad_width = tuple(pad_width)
    
    # Pad with 1s
    imgs = np.pad(imgs, pad_width=pad_width, constant_values=1)

    # Use the appropriate color for the border
    color = color_hex_to_float(color)
    imgs[..., :border_pixel_width, :, :] = color
    imgs[...,-border_pixel_width:, :, :] = color
    imgs[..., :, :border_pixel_width, :] = color
    imgs[..., :,-border_pixel_width:, :] = color
    return imgs


def append_black_right_separator(imgs: np.ndarray) -> np.ndarray:
    """Assumes last 3 dimensions of tensors are images HWC."""
    border_pixel_width = 50
    pad_width = [(0, 0) for _ in range(imgs.ndim)]
    pad_width[-2] = (0, border_pixel_width)
    pad_width = tuple(pad_width)
    
    # Pad with 1s
    imgs = np.pad(imgs, pad_width=pad_width, constant_values=1)

    imgs[..., :,-border_pixel_width:, :] = color_hex_to_float(BG_COLOR)
    return imgs


def combine_slot_masks(
    slot_masks: torch.Tensor
    ) -> torch.Tensor:
    """Assumes last 3 dimensions of tensors are images HWC."""
    shp = list(slot_masks.shape)
    shp[-1] = 3
    slot_imgs = np.zeros(shp, dtype=np.float32)
    
    # Assumes -4 is the slot dimension
    for i in range(shp[-4]):
        slot_imgs[..., i, :, :, :] = color_hex_to_float(SLOT_COLORS[i % len(SLOT_COLORS)])
        # slot_imgs[..., i, :, :, :] *= slot_masks[..., i, :, :, :]
    
    combined_imgs = slot_imgs * slot_masks
    combined_imgs = combined_imgs.sum(axis=-4)
    return combined_imgs


def concat_imgs_in_rec_mask_slots_in_a_row(
    img_in: np.ndarray,
    img_rec: np.ndarray,
    img_slots: np.ndarray, 
    img_slot_masks: np.ndarray,
    img_slot_masks_multiplied: np.ndarray
    ) -> np.ndarray:
    """Image grid generation function for SlotAttention-like models."""
    # generate mask by combining slot masks depending on their RGB-color coded values
    img_combined_mask = combine_slot_masks(img_slot_masks)

    # first append color borders - black to input, rec, and mask - RGB-coded colors to slot images
    # and then sequentially concatenate (append) them along axis=1
    # input image
    img_res = append_border_pixels(img_in, BG_COLOR)
    
    # append reconstructed image
    img_tmp = append_border_pixels(img_rec, BG_COLOR)
    img_res = np.concatenate((img_res, img_tmp), axis=-2)
    
    # append combined slot masks image
    img_tmp = append_border_pixels(img_combined_mask, BG_COLOR)
    img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot images
    for i in range(img_slots.shape[-4]):
        # note: the slicing is (this) ugly due to not knowing if the input is a sequence (B, T, H, W, C)
        img_tmp = append_border_pixels(img_slots[..., i, :, :, :], SLOT_COLORS[i % len(SLOT_COLORS)])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot masks
    imgs = np.repeat(img_slot_masks, 3, axis=-1)
    for i in range(img_slot_masks.shape[-4]):
        img_tmp = append_border_pixels(imgs[:, i], SLOT_COLORS[i % len(SLOT_COLORS)])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot masks
    for i in range(img_slot_masks_multiplied.shape[-4]):
        img_tmp = append_border_pixels(img_slot_masks_multiplied[..., i, :, :, :], SLOT_COLORS[i % len(SLOT_COLORS)])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # img_res = torch.from_numpy(img_res)  # TODO: fix! do everything in torch instead of np?
    return img_res


def combined_labels_to_colormap(
    combined_labels: torch.Tensor, 
    number_of_classes: int
    ) -> np.ndarray:
    """Assumes last 3 dimensions of tensors are images HWC."""
    # Convert single-image label representation to one-hot representation (similar to what we have in slots)
    # undo the -1 class assignment for predicted labels
    combined_labels[combined_labels == -1] = 0  #
    # combined_labels = combined_labels + 1
    # number_of_classes = number_of_classes + 1
    labels_onehot = F.one_hot(combined_labels.to(torch.int64), number_of_classes)
    labels_onehot = labels_onehot.numpy()

    # Make it 3-colored representation (RGB)
    labels_onehot = np.repeat(np.expand_dims(labels_onehot, axis=-1), repeats=3, axis=-1)
    shp = list(labels_onehot.shape)
    colormaps = np.zeros(shp, dtype=np.float32)

    # Assumes -2 is the slot dimension
    slot_dimension = -2
    for i in range(shp[slot_dimension]):
        colormaps[..., i, :] = color_hex_to_float(SLOT_COLORS[i % len(SLOT_COLORS)])
        # colormaps[..., i, :] *= labels_onehot[..., i, :, :, :]
    
    combined_colormaps = colormaps * labels_onehot
    combined_colormaps = combined_colormaps.sum(axis=slot_dimension)
    return combined_colormaps


def resize_image_batch(image_batch, new_size):
    new_image_batch = []
    for batch_idx in range(image_batch.shape[0]):
        new_image_batch.append(cv2.resize(image_batch[batch_idx], new_size, interpolation=cv2.INTER_NEAREST))
    new_image_batch = np.stack(new_image_batch)
    return new_image_batch


def append_single_radial_plot_tuple(img_grid, plot_radial, plot_groups_phase):
    # append radial plot
    img_tmp = append_border_pixels(plot_radial, BG_COLOR)
    img_grid = np.concatenate((img_grid, img_tmp), axis=-2)
    # append image groups colorcoded by phase
    img_tmp = append_border_pixels(plot_groups_phase, BG_COLOR)
    img_grid = np.concatenate((img_grid, img_tmp), axis=-2)
    return img_grid


def append_multiple_radial_plot_tuples(img_grid, plot_radial_list, plot_groups_phase_list):
    for i in range(len(plot_radial_list)):
        radial_plot = plot_radial_list[i]
        groups_plot_phase = plot_groups_phase_list[i]
        # append radial plot
        img_tmp = append_border_pixels(radial_plot, BG_COLOR)
        img_grid = np.concatenate((img_grid, img_tmp), axis=-2)
        # append image groups colorcoded by phase
        img_tmp = append_border_pixels(groups_plot_phase, BG_COLOR)
        img_grid = np.concatenate((img_grid, img_tmp), axis=-2)
    return img_grid


def create_image_grids_for_logging(
    plot_resize_resolution: Tuple[int, int],
    img_in: torch.Tensor,
    img_rec: torch.Tensor,
    gt_label: torch.Tensor,
    pred_label: torch.Tensor,
    plot_radial: Optional[np.ndarray] = None,
    plot_groups_phase: Optional[np.ndarray] = None,
    plot_groups_magnitude: Optional[np.ndarray] = None,
    plot_enc_layer_radial: Optional[Sequence[np.ndarray]] = None,
    plot_enc_layer_groups_phase: Optional[Sequence[np.ndarray]] = None,
    plot_enc_layer_groups_magnitude: Optional[Sequence[np.ndarray]] = None,
    plot_dec_layer_radial: Optional[Sequence[np.ndarray]] = None,
    plot_dec_layer_groups_phase: Optional[Sequence[np.ndarray]] = None,
    plot_dec_layer_groups_magnitude: Optional[Sequence[np.ndarray]] = None,
    ) -> wandb.Image:
    # convert input image to numpy
    img_in = rearrange(img_in, "b c h w -> b h w c")
    img_in = img_in.cpu().numpy()
    if img_in.shape[-1] == 1:
        img_in = np.repeat(img_in, repeats=3, axis=-1)
    # convert reconstructed image to numpy
    img_rec = rearrange(img_rec, "b c h w -> b h w c")
    img_rec = img_rec.cpu().detach().numpy()
    if img_rec.shape[-1] == 1:
        img_rec = np.repeat(img_rec, repeats=3, axis=-1)

    # generate mask by combining phase masks depending on their RGB-color coded values
    pred_label = torch.Tensor(pred_label)
    # Reason for "+2": 1 addition is for the BG class (label 0)
    #                  1 addition is for the overlapping regions extra label added in the eval apply_kmeans function
    number_of_classes = int(torch.max(gt_label).item()) + 2
    gt_label_colormap = combined_labels_to_colormap(gt_label, number_of_classes=number_of_classes)
    phase_masks_colormap = combined_labels_to_colormap(pred_label, number_of_classes=number_of_classes)

    ##############################################################################################
    ## IMAGE GRID - ALL
    # first append color borders - black to input, rec, and mask - RGB-coded colors to slot images
    # and then sequentially concatenate (append) them along axis=1
    # input image
    img_in = resize_image_batch(img_in, plot_resize_resolution)
    img_grid_all = append_border_pixels(img_in, BG_COLOR)
    
    # append image reconstruction
    img_rec = resize_image_batch(img_rec, plot_resize_resolution)
    img_tmp = append_border_pixels(img_rec, BG_COLOR)
    img_grid_all = np.concatenate((img_grid_all, img_tmp), axis=-2)
    
    # append gt label
    gt_label_colormap = resize_image_batch(gt_label_colormap, plot_resize_resolution)
    img_tmp = append_border_pixels(gt_label_colormap, BG_COLOR)
    img_grid_all = np.concatenate((img_grid_all, img_tmp), axis=-2)
    
    # append phase masks
    phase_masks_colormap = resize_image_batch(phase_masks_colormap, plot_resize_resolution)
    img_tmp = append_border_pixels(phase_masks_colormap, BG_COLOR)
    img_grid_all = np.concatenate((img_grid_all, img_tmp), axis=-2)

    # Append output radial plot, groups_phase and groups_magnitude
    if plot_radial is not None and plot_groups_phase is not None:
        img_grid_all = append_single_radial_plot_tuple(img_grid_all, plot_radial, plot_groups_phase)

    # black separator
    img_grid_all = append_black_right_separator(img_grid_all)

    # append enc_layer_phase_maps plots
    if plot_enc_layer_radial is not None and plot_enc_layer_groups_phase is not None:
        img_grid_all = append_multiple_radial_plot_tuples(
            img_grid_all, plot_enc_layer_radial, plot_enc_layer_groups_phase)

    # black separator
    img_grid_all = append_black_right_separator(img_grid_all)

    # append dec_layer_phase_maps plots
    if plot_dec_layer_radial is not None and plot_dec_layer_groups_phase is not None:
        img_grid_all = append_multiple_radial_plot_tuples(
            img_grid_all, plot_dec_layer_radial, plot_dec_layer_groups_phase)
    
    ##############################################################################################
    ## IMAGE GRID - MAIN (input image, gt labels, predicted labels, output radial plots)
    # input image
    img_grid_main = append_border_pixels(img_in, BG_COLOR)

    # append image reconstruction
    img_tmp = append_border_pixels(img_rec, BG_COLOR)
    img_grid_main = np.concatenate((img_grid_main, img_tmp), axis=-2)
    
    # append gt label
    img_tmp = append_border_pixels(gt_label_colormap, BG_COLOR)
    img_grid_main = np.concatenate((img_grid_main, img_tmp), axis=-2)
    
    # append phase masks
    img_tmp = append_border_pixels(phase_masks_colormap, BG_COLOR)
    img_grid_main = np.concatenate((img_grid_main, img_tmp), axis=-2)
    
    # Append output radial plot, groups_phase and groups_magnitude
    if plot_radial is not None and plot_groups_phase is not None:
        img_grid_main = append_single_radial_plot_tuple(img_grid_main, plot_radial, plot_groups_phase)
    
    ##############################################################################################
    ## IMAGE GRID - ENCODER LAYERS (input image, encoder layer radial plots)
    # input image
    img_grid_enc_layers = None
    if plot_enc_layer_radial is not None and plot_enc_layer_groups_phase is not None:
        img_grid_enc_layers = append_border_pixels(img_in, BG_COLOR)
    
    # append enc_layer_phase_maps plots
    if img_grid_enc_layers is not None and plot_enc_layer_radial is not None and plot_enc_layer_groups_phase is not None:
        img_grid_enc_layers = append_multiple_radial_plot_tuples(
            img_grid_enc_layers, plot_enc_layer_radial, plot_enc_layer_groups_phase)

    ##############################################################################################
    ## IMAGE GRID - DECODER LAYERS (input image, decoder layer radial plots)
    # input image
    img_grid_dec_layers = None
    if plot_dec_layer_radial is not None and plot_dec_layer_groups_phase is not None:
        img_grid_dec_layers = append_border_pixels(img_in, BG_COLOR)
    
    # append dec_layer_phase_maps plots
    if img_grid_dec_layers is not None and plot_dec_layer_radial is not None and plot_dec_layer_groups_phase is not None:
        img_grid_dec_layers = append_multiple_radial_plot_tuples(
            img_grid_dec_layers, plot_dec_layer_radial, plot_dec_layer_groups_phase)

    ##############################################################################################

    # Flatten the batch dimension along rows and create the wandb image objects
    img_grid_main = batch_to_rowwise_image(img_grid_main.copy())
    if img_grid_all is not None:
        img_grid_all = batch_to_rowwise_image(img_grid_all.copy())
    if img_grid_enc_layers is not None:
        img_grid_enc_layers = batch_to_rowwise_image(img_grid_enc_layers.copy())
    if img_grid_dec_layers is not None:
        img_grid_dec_layers = batch_to_rowwise_image(img_grid_dec_layers.copy())

    return {
        'img_grid_all': img_grid_all,
        'img_grid_main': img_grid_main,
        'img_grid_enc_layers': img_grid_enc_layers,
        'img_grid_dec_layers': img_grid_dec_layers,
    }


def create_image_grids_for_logging_input_reconstruction(
    plot_resize_resolution: Tuple[int, int],
    img_in: torch.Tensor,
    img_rec: torch.Tensor,
    ) -> wandb.Image:
    # convert input image to numpy
    img_in = rearrange(img_in, "b c h w -> b h w c")
    img_in = img_in.cpu().numpy()
    if img_in.shape[-1] == 1:
        img_in = np.repeat(img_in, repeats=3, axis=-1)
    # convert reconstructed image to numpy
    img_rec = rearrange(img_rec, "b c h w -> b h w c")
    img_rec = img_rec.cpu().detach().numpy()
    if img_rec.shape[-1] == 1:
        img_rec = np.repeat(img_rec, repeats=3, axis=-1)

    ##############################################################################################
    ## IMAGE GRID - ALL
    # first append color borders - black to input, rec, and mask - RGB-coded colors to slot images
    # and then sequentially concatenate (append) them along axis=1
    # input image
    img_in = resize_image_batch(img_in, plot_resize_resolution)
    img_grid_all = append_border_pixels(img_in, BG_COLOR)
    
    # append image reconstruction
    img_rec = resize_image_batch(img_rec, plot_resize_resolution)
    img_tmp = append_border_pixels(img_rec, BG_COLOR)
    img_grid_all = np.concatenate((img_grid_all, img_tmp), axis=-2)
    
    # Flatten the batch dimension along rows and create the wandb image objects
    img_grid_all = batch_to_rowwise_image(img_grid_all.copy())
    img_grid_all = tensor_to_wandb_image(img_grid_all)

    return {
        'img_grid_all': img_grid_all,
    }


def batch_to_rowwise_image(
    imgs: np.ndarray
    ) -> np.ndarray:
    # Flatten the first (batch/time) and the second dimension (time/H)
    imgs = imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def batch_to_rowwise_video(
    videos: np.ndarray
    ) -> np.ndarray:
    # converts a Tensor of shape (B, T, H, W, C) to (1, T, B*H, W, C)
    videos = np.swapaxes(videos, 0, 1)
    shp = videos.shape
    videos = videos.reshape(shp[0], 1, np.prod(shp[1:3]), *shp[3:])
    videos = np.swapaxes(videos, 0, 1)
    return videos


def canvas_to_numpy(figure, canvas):
    width, height = figure.get_size_inches() * figure.get_dpi()
    width, height = int(width), int(height)
    np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    np_image = np_image.reshape(height, width, 3)
    # Make the figure square by removing the blank parts on the sides
    start = (width - height) // 2
    np_image = np_image[:, start:-start]
    return np_image


def create_phase_colorcoded_groupings_and_radial_plots(plot_resize_resolution, phase_batch, magnitude_batch, gt_labels_batch):

    batch_np_image_radial = []
    batch_np_image_groups_phase = []
    batch_np_image_groups_magnitude = []

    for batch_idx in range(phase_batch.shape[0]):

        gt_label = gt_labels_batch[batch_idx]

        phase = phase_batch[batch_idx, 0]
        magnitude = magnitude_batch[batch_idx,0]
        # Flip rows axis to make plots in line with the rest of the code
        phase = phase[::-1,:]
        magnitude = magnitude[::-1,:]
        colors = phase

        # Radial plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        canvas = FigureCanvas(fig)
        ax.scatter(phase, magnitude, c=colors, cmap='hsv', alpha=0.75)
        ax.set_xticklabels([])  # uncomment to remove angle labels
        # ax.set_yticklabels([])  # uncomment out to remove magnitude labels
        ax.spines['polar'].set_visible(False)
        canvas.draw()
        np_image_radial = canvas_to_numpy(fig, canvas)

        # Phase groupings plot
        fig, ax = plt.subplots(subplot_kw={'projection': None})
        canvas = FigureCanvas(fig)
        x = np.arange(phase.shape[0])
        y = np.arange(phase.shape[0])
        xx, yy = np.meshgrid(x, y, sparse=False)
        ax.set_box_aspect(1)
        ax.scatter(xx, yy, c=colors, cmap='hsv', s=100, marker='s')
        plt.axis('off')
        canvas.draw()
        np_image_groups_phase = canvas_to_numpy(fig, canvas)

        # Magnitude groupings plot
        fig, ax = plt.subplots(subplot_kw={'projection': None})
        canvas = FigureCanvas(fig)
        ax.set_box_aspect(1)
        colors = magnitude
        ax.scatter(xx, yy, c=colors, cmap='hsv', s=100, marker='s')
        plt.axis('off')
        canvas.draw()
        np_image_groups_magnitude = canvas_to_numpy(fig, canvas)

        plt.close('all')  # Clean up all figures

        # Prepare for merging with other images - cast to float and normalize to [0, 1] range
        np_image_radial = np_image_radial.astype(np.float32) / 255.
        np_image_groups_phase = np_image_groups_phase.astype(np.float32) / 255.
        np_image_groups_magnitude = np_image_groups_magnitude.astype(np.float32) / 255.

        np_image_radial = cv2.resize(np_image_radial, plot_resize_resolution, interpolation=cv2.INTER_NEAREST)
        np_image_groups_phase = cv2.resize(np_image_groups_phase, plot_resize_resolution, interpolation=cv2.INTER_NEAREST)
        np_image_groups_magnitude = cv2.resize(np_image_groups_magnitude, plot_resize_resolution, interpolation=cv2.INTER_NEAREST)

        batch_np_image_radial.append(np_image_radial)
        batch_np_image_groups_phase.append(np_image_groups_phase)
        batch_np_image_groups_magnitude.append(np_image_groups_magnitude)

    batch_np_image_radial = np.stack(batch_np_image_radial)
    batch_np_image_groups_phase = np.stack(batch_np_image_groups_phase)
    batch_np_image_groups_magnitude = np.stack(batch_np_image_groups_magnitude)

    return batch_np_image_radial, batch_np_image_groups_phase, batch_np_image_groups_magnitude


def process_layer_map(layer_map, mode='tsne'):
        bsz, channels, height, width = layer_map.shape
        layer_maps = []
        # convert spherical [r, theta_1, .... theta_n] to cartesian form
        layer_spherical_map = torch.cat(
            (torch.ones_like(layer_map.angle())[:, None], layer_map.angle()[:, None]), dim=1
        )
        layer_map_cartesian = spherical_to_cartesian_coordinates(layer_spherical_map)
        if mode in ["tsne", "umap"]:
            for i in range(bsz):
                layer_map_cartesian_i = layer_map_cartesian[i].detach().cpu().numpy()
                # apply t-SNE on each feature in the spatial grid separately
                layer_map_cartesian_i = rearrange(layer_map_cartesian_i, 'p c h w -> (h w) (p c)')
                if mode == "tsne":
                    ppl = 20 if layer_map_cartesian_i.shape[0] >= 64 else 15
                    tsne = TSNE(n_components=2, n_iter=500, perplexity=ppl, metric='euclidean')
                    layer_map_cartesian_i = torch.from_numpy(
                        tsne.fit_transform(layer_map_cartesian_i)
                        )
                elif mode == "umap":
                    um = umap.UMAP()
                    layer_map_cartesian_i = torch.from_numpy(
                        um.fit_transform(layer_map_cartesian_i)
                        )
                layer_map_cartesian_i = rearrange(
                    layer_map_cartesian_i, '(h w) p -> p h w', h=height, w=width
                )
                layer_maps.append(layer_map_cartesian_i)
            layer_map_cartesian = rearrange(layer_maps, 'b p h w -> b p h w')
            # convert back to polar form
            layer_map_phase = torch.atan2(
                layer_map_cartesian[:, 1], layer_map_cartesian[:, 0]
            )[:, None]
        elif mode == "mean":
            # average map (across channels)
            layer_map = reduce(layer_map.angle(), 'b c h w -> b 1 h w', 'mean')
            # upsample magnitude & phase maps if smaller than img_size
            if layer_map.shape[2] != height:
                layer_map = torch.nn.functional.interpolate(
                    layer_map, (height, width), mode='nearest'
                )
        else:
            raise ValueError(f"Unsupported processing mode: {mode}")
        return layer_map_phase