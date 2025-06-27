import argparse
import itertools
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from modules.mask_region_enhence import MaskAugmentor
import sys
import cv2

def gray2whiteblack(gray_image):
    
    img_array = np.array(gray_image)
    binary_img = (img_array > 128).astype(np.uint8) * 255
    mask_image = Image.fromarray(binary_img)
    
    return mask_image

def crop_white_instance(mask_image, paste_image):
    
    object_bbox = mask_image.getbbox()
    inverted_mask = ImageOps.invert(mask_image)
    white_background = inverted_mask.point(lambda x: 255 if x == 255 else 0)
    paste_clip_image = Image.composite(white_background, paste_image, inverted_mask)
    paste_clip_image = paste_clip_image.crop(object_bbox)
    
    return paste_clip_image

def draw_contour_on_image(image, mask, contour_color=(255, 255, 255), thickness=1):
  
    mask_np = np.array(mask)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_np = np.array(image)
    cv2.drawContours(image_np, contours, -1, contour_color, thickness)
    result_image = Image.fromarray(image_np)
    
    return result_image

def prepare_mask_and_image(image, mask, img_name):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    return mask, image

class RegionInpaintingDataset_Test(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        mask_mode=None,
        clip_processor=None,
        clip_vtext=None,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.clip_vtext = clip_vtext
        self.mask_mode = mask_mode
        self.mask_enhence_aug = MaskAugmentor()
        
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.gt_root = self.data_root / 'gt'
        self.mask_root = self.data_root / 'mask'
        self.paste_root = self.data_root / 'paste'
        
        self.image_names = list(Path(self.gt_root).iterdir())
        self.mask_images_path = list(Path(self.mask_root).iterdir())
        self.paste_images_path = list(Path(self.paste_root).iterdir())
        self.num_images = len(self.image_names)
        self._length = self.num_images
        
        self.image_transforms_resize = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        )
        
        self.mask_transforms_resize = transforms.Compose(
            [
                transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            ]
        )
        
        self.mask_transforms_resize_336 = transforms.Compose(
            [
                transforms.Resize((24, 24), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            ]
        )
        
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        print(index)
        example = {}
        
        img_name = os.path.basename(self.image_names[index % self.num_images])
        
        gt_image = Image.open(self.gt_root / img_name).convert("RGB")
        paste_image = Image.open(self.paste_root / img_name).convert("RGB")
        mask_image = Image.open(self.mask_root / img_name).convert("L")
        
        mask_image = gray2whiteblack(mask_image)
        
        gt_image = self.image_transforms_resize(gt_image)
        mask_image = self.image_transforms_resize(mask_image)
        paste_image = self.image_transforms_resize(paste_image)
        
        mask_image = self.mask_enhence_aug.augment_mask(mask_image, mode=self.mask_mode)
        mask_image = mask_image.convert('L')
        mask_image = gray2whiteblack(mask_image)
            
        paste_clip_image = crop_white_instance(mask_image, paste_image)
        paste_clip_image = self.clip_processor(images=paste_clip_image, return_tensors="pt")['pixel_values']
        example["paste_clip_image"] = paste_clip_image
        example["vtoken_prompt"] = self.tokenizer(self.clip_vtext, padding="max_length", max_length=7, truncation=True, return_tensors="pt")['input_ids']
        example["vtoken_prompt_uncond"] = self.tokenizer('', padding="max_length", max_length=7, truncation=True, return_tensors="pt")['input_ids']

        example["PIL_gt"] = gt_image
        example["PIL_paste"] = paste_image
        example["mask"] = mask_image
        example["img_name"] = img_name

        return example

    def collate_fn(self, examples):
        
        PIL_gt = [example["PIL_gt"] for example in examples]
        PIL_paste = [example["PIL_paste"] for example in examples]
        mask = [example["mask"] for example in examples]
        img_name = [example["img_name"] for example in examples]
        
        paste_clip_image = [example["paste_clip_image"] for example in examples]
        vtoken_prompt = [example["vtoken_prompt"] for example in examples]
        vtoken_prompt_uncond = [example["vtoken_prompt_uncond"] for example in examples]
        batch = {"PIL_gt": PIL_gt, "PIL_paste": PIL_paste, "mask": mask, "img_name": img_name, "paste_clip_image":paste_clip_image, "vtoken_prompt":vtoken_prompt, "vtoken_prompt_uncond":vtoken_prompt_uncond}
        
        return batch

class RegionInpaintingDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_root,
        split,
        tokenizer,
        size=512,
        clip_processor=None,
        clip_vtext=None,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.clip_vtext = clip_vtext
        self.mask_enhence_aug = MaskAugmentor()   


        self.data_root = Path(data_root) / split
        if not self.data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.gt_root = self.data_root / 'gt'
        self.mask_root = self.data_root / 'mask'
        self.paste_root = self.data_root / 'paste'
        
        self.image_names = list(Path(self.gt_root).iterdir())
        
        self.num_images = len(self.image_names)
        self._length = self.num_images

        self.image_transforms_resize = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            ]
        )

        self.mask_transforms_resize = transforms.Compose(
            [
                transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            ]
        )

        self.mask_transforms_resize_336 = transforms.Compose(
            [
                transforms.Resize((24, 24), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        
        example = {}
        
        img_name = os.path.basename(self.image_names[index % self.num_images])
        gt_image = Image.open(self.gt_root / img_name).convert("RGB")
        paste_image = Image.open(self.paste_root / img_name).convert("RGB")
        mask_image = Image.open(self.mask_root / img_name).convert("L")
        mask_image = gray2whiteblack(mask_image)
        
        gt_image = self.image_transforms_resize(gt_image)
        mask_image = self.image_transforms_resize(mask_image)
        paste_image = self.image_transforms_resize(paste_image)
        
        mask_image = self.mask_enhence_aug.augment_mask(mask_image)
        mask_image = mask_image.convert('L')
        mask_image = gray2whiteblack(mask_image)
        
        paste_clip_image = crop_white_instance(mask_image, paste_image)
        paste_clip_image = self.clip_processor(images=paste_clip_image, return_tensors="pt")['pixel_values']
        example["paste_clip_image"] = paste_clip_image
        example["vtoken_prompt"] = self.tokenizer(self.clip_vtext, padding="max_length", max_length=7, truncation=True, return_tensors="pt")['input_ids']
        example["vtoken_prompt_uncond"] = self.tokenizer('', padding="max_length", max_length=7, truncation=True, return_tensors="pt")['input_ids']

        example["gt"] = self.image_transforms(gt_image)
        example["PIL_paste"] = paste_image
        example["mask"] = mask_image
        example["img_name"] = img_name

        return example


    def collate_fn(self, examples):
        gt_values = [example["gt"] for example in examples]
        masks = []
        paste_images = []
        
        for example in examples:
            
            mask = example["mask"]
            pil_paste = example["PIL_paste"]
            # prepare mask and masked image
            mask, pil_paste = prepare_mask_and_image(pil_paste, mask, example["img_name"])
            masks.append(mask)
            paste_images.append(pil_paste)

        gt_values = torch.stack(gt_values)
        gt_values = gt_values.to(memory_format=torch.contiguous_format).float()
        
        masks = torch.stack(masks)
        paste_images = torch.stack(paste_images)
        
        paste_clip_image = [example["paste_clip_image"] for example in examples]
        paste_clip_images = torch.concat(paste_clip_image, dim=0)
        vtoken_prompt = [example["vtoken_prompt"] for example in examples]
        vtoken_prompt = torch.concat(vtoken_prompt, dim=0)
        vtoken_prompt_uncond = [example["vtoken_prompt_uncond"] for example in examples]
        vtoken_prompt_uncond = torch.concat(vtoken_prompt_uncond, dim=0)
        batch = {"gt_values": gt_values, "masks": masks, "paste_images": paste_images, "paste_clip_images":paste_clip_images, "vtoken_prompt":vtoken_prompt, 'vtoken_prompt_uncond':vtoken_prompt_uncond}
        
        return batch