"""
Preprocessing scripts to generate Pytorch DataLoaders for multi-object-datasets. 
Adapted from https://github.com/pemami4911/EfficientMORL/blob/main/lib/datasets.py 
"""

import math
import warnings
from typing import List
import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import random


DATA_ROOT_PATH = "data"


DATASET_IMG_RESOLUTION = {
    "clevr": (96, 96),
    "multi_dsprites": (64, 64),
    "tetrominoes": (35, 35),
}


EMORL_DATASET_MAPPING = {
    'clevr': {
        'train': 'clevr6_with_masks_train.h5',
        'test': 'clevr6_with_masks_and_factors_test.h5'
    },
    'multi_dsprites': {
        'train': 'multi_dsprites_colored_on_grayscale.h5',
        'test': 'multi_dsprites_colored_on_grayscale_test.h5'
    },
    'tetrominoes': {
        'train': 'tetrominoes.h5',
        'test': 'tetrominoes_test.h5'
    },
}


class EMORLHdF5Dataset(torch.utils.data.Dataset):
    """
    The .h5 dataset is assumed to be organized as follows:
    {train|val|test}/
        imgs/  <-- a tensor of shape [dataset_size,H,W,C]
        masks/ <-- a tensor of shape [dataset_size,num_objects,H,W,C]
        factors/  <-- a tensor of shape [dataset_size,...]
    """
    def __init__(
        self,
        dataset_name: str, 
        split: str,
        return_masks: bool,
        return_factors: bool,
        make_background_black: bool,
        use_32x32_res: bool,
        use_64x64_res: bool,
        n_objects_cutoff: int,
        cutoff_type: str,
        use_grayscale: bool,
    ):    
        super(EMORLHdF5Dataset, self).__init__()
        h5_name = EMORL_DATASET_MAPPING[dataset_name][split]
        self.h5_path = str(Path(DATA_ROOT_PATH, h5_name))
        self.dataset_name = dataset_name
        assert not (use_32x32_res and use_64x64_res), "Cannot use both 32x32 and 64x64 resolutions"
        if use_32x32_res:
            self.img_resolution = (32, 32)
        elif use_64x64_res:
            self.img_resolution = (64, 64)
        else:
            self.img_resolution = DATASET_IMG_RESOLUTION[self.dataset_name]
        self.n_channels = 3
        self.split = split
        self.return_masks = return_masks
        self.return_factors = return_factors
        self.make_background_black = make_background_black
        self.n_objects_cutoff = n_objects_cutoff
        self.cutoff_type = cutoff_type
        self.use_grayscale = use_grayscale
        if self.use_grayscale:
            self.n_channels = 1

    def preprocess_image(
        self,
        img: np.ndarray,
        bg_mask: np.ndarray,
        ) -> np.ndarray:
        """
        img is assumed to be an array of integers each in 0-255 
        We preprocess them by mapping the range to 0-1        
        """
        if self.make_background_black:
            # img[np.repeat(bg_mask, repeats=3, axis=-1).astype(bool)] = 0
            img[bg_mask.astype(bool), :] = 0

        PIL_img = Image.fromarray(np.uint8(img))
        if self.dataset_name == "tetrominoes":
            PIL_img = PIL_img.resize(self.img_resolution)
        elif self.dataset_name == "multi_dsprites":
            PIL_img = PIL_img.resize(self.img_resolution)
        # square center crop of 192 x 192
        elif self.dataset_name == 'clevr':
            PIL_img = PIL_img.crop((64,29,256,221))
            PIL_img = PIL_img.resize(self.img_resolution)

        if self.use_grayscale:
            img = np.array(PIL_img)
            img = np.expand_dims(img, axis=-1)
            img = np.transpose(img, (2,0,1))
        else:
            # H,W,C --> C,H,W
            img = np.transpose(np.array(PIL_img), (2,0,1))

        # image range is 0,1
        img = img / 255. # to [0,1]
        return img

    def preprocess_mask(
        self, 
        mask: np.ndarray
        ) -> np.ndarray:
        """
        [num_objects, h, w, c]
        Returns the square mask of size 192x192
        """
        o,h,w,c = mask.shape
        masks = []
        for i in range(o):
            mask_ = mask[i,:,:,0]
            PIL_mask = Image.fromarray(mask_, mode="F")
            if self.dataset_name == "tetrominoes":
                PIL_mask = PIL_mask.resize(self.img_resolution)
            elif self.dataset_name == "multi_dsprites":
                PIL_mask = PIL_mask.resize(self.img_resolution)
            elif self.dataset_name == "clevr":
                # square center crop of 192 x 192
                PIL_mask = PIL_mask.crop((64,29,256,221))
                # TODO(astanic): this might be a hack, maybe fix later.
                # This resize was added such that we don't have to upsample predicted image in the evaluation
                # This also makes the learned phase evaluation easier.
                # Originaly Emami was resizing the predicted labels at evaluation time:
                # https://github.com/pemami4911/EfficientMORL/blob/main/eval.py#L336-L347
                PIL_mask = PIL_mask.resize(self.img_resolution, Image.NEAREST)
            masks += [np.array(PIL_mask)[...,None]]
        mask = np.stack(masks)  # [o,h,w,c]
        mask = np.transpose(mask, (0,3,1,2))  # [o,c,h,w]
        return mask

    def __len__(self) -> int:
        with h5py.File(self.h5_path,  'r') as data:
            data_size, _, _, _ = data[self.split]['imgs'].shape
            return data_size

    def __getitem__(self, i: int) -> dict:
        with h5py.File(self.h5_path,  'r') as data:
            if self.use_grayscale:
                imgs = data[self.split]['imgs'][i]
                pil_img = Image.fromarray(imgs)
                pil_img = pil_img.convert('L')
                imgs = np.asarray(pil_img).astype('float32')
            else:
                imgs = data[self.split]['imgs'][i].astype('float32')
            # exit(0)
            masks = data[self.split]['masks'][i].astype('float32')
            outs = {}
            outs['images'] = self.preprocess_image(img=imgs, bg_mask=masks[0,:,:,0]).astype('float32')
            if self.return_masks:
                masks = self.preprocess_mask(masks)
                # n_objects, 1, img_h, img_w (6, 1, 64, 64)
                # gt-segmentation masks for coloured datasets must be of shape (h, w)
                masks = np.squeeze(masks, axis=1)
                # the first element of the first axis is background label
                masks_argmax = np.argmax(masks, axis=0)
                
                # Check if we should filter out this sample based on the number of objects
                # If yes, then sample a new index from the dataset at random
                # Note: we can only do samples filtering here (and not at the start of the function)
                # because of cropping (in the case of CLEVR), the number of objects might be different
                # in the ground truth and the cropped mask
                if self.n_objects_cutoff > 0:
                    n_objects = len(np.unique(masks_argmax)) - 1  # subtract 1 for background
                    if self.cutoff_type == 'eq':
                        if n_objects != self.n_objects_cutoff:
                            return self.__getitem__(np.random.randint(0, self.__len__()))
                    elif self.cutoff_type == 'leq':
                        if n_objects > self.n_objects_cutoff:
                            return self.__getitem__(np.random.randint(0, self.__len__()))
                    elif self.cutoff_type == 'geq':
                        if n_objects < self.n_objects_cutoff:
                            return self.__getitem__(np.random.randint(0, self.__len__()))
                    else:
                        raise ValueError(f"Unknown n_objects cutoff type: {self.cutoff_type}")

                # find all overlaps and label them as -1. Note: the issue of correct object assignment
                # when 2 or more objects overlap should not exist in RGB data.
                masks_overlap = masks[1:] / 255
                masks_overlap = masks_overlap.sum(axis=0)
                outs['masks_argmax'] = masks_argmax
                outs['masks_overlap'] = masks_overlap
                masks_argmax[masks_overlap > 1] ==  -1
                outs['labels'] = masks_argmax
                
            if self.return_factors:
                outs['factors'] = data[self.split]['factors'][i]
            return outs


def seed_worker(worker_seed):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, num_workers, seed):
    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(seed)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return data_loader


def random_split(dataset, lengths, generator=torch.default_generator):
    """
    local version of torch.utils.data.random_split() for backward compatibility to 1.10
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [torch.utils.data.dataset.Subset(dataset, indices[offset - length : offset]) for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
