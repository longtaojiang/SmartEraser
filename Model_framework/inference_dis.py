import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import random_split
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, SamProcessor
from modules.clip_visual_token import CLIPVisualPrompt
import json
import sys

from modules.pipeline.pipeline_stable_diffusion_inpaint_region import StableDiffusionInpaintRegionPipeline
from modules.eraser_dataset import RegionInpaintingDataset_Test



def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.3):

    if mask.mode != 'L':
        mask = mask.convert('L')
    
    color_layer = Image.new("RGB", mask.size, color)
    mask_with_color = Image.composite(color_layer, Image.new("RGB", mask.size), mask)
    mask_with_color = mask_with_color.convert("RGBA")
    mask_with_color.putalpha(int(alpha * 255))
    image = image.convert("RGBA")
    result_image = Image.alpha_composite(image, mask_with_color)
    result_image = Image.composite(result_image, image, mask)
    result_image = result_image.convert("RGB")

    return result_image

def images_composite(img_list, resolution=512):

    width, height = resolution, resolution
    total_width = len(img_list) * width
    max_height = height
    
    new_image = Image.new('RGB', (total_width, max_height))

    for i, img in enumerate(img_list):
        new_image.paste(img, (i*width, 0))
    
    return new_image

def prepare_mask_and_image(image, mask):
    
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

def init(args,rank=None):
    global sd_inpaint
    if rank is None:
        rank=torch.multiprocessing.current_process()._identity[0]-1
    print("init process GPU:",rank)
    device = torch.device('cuda:%s'%rank)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    sd_inpaint = StableDiffusionInpaintRegionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    sd_inpaint.to(rank)
    global vtext_model
    global ip_adapter336
    
    vtext_model = CLIPVisualPrompt(args.clip_visual_prompt)
    vtext_model.load_mlp_weight(os.path.join(args.pretrained_model_name_or_path, 'clip_mlp_weight.pth'))
    vtext_model.vision_model.eval()
    vtext_model.vision_model.to("cuda")
    vtext_model.text_model.eval()
    vtext_model.text_model.to("cuda")
    vtext_model.clip_mlp.eval()
    vtext_model.clip_mlp.to("cuda")
        
def run_inference(dataset, args):

    pre_img_path = os.path.join(args.output_dir, 'pre')
    gt_img_path = os.path.join(args.output_dir, 'gt')
    mask_img_path = os.path.join(args.output_dir, 'mask')
    mask_paste_img_path = os.path.join(args.output_dir, 'mask_paste')
    paste_img_path = os.path.join(args.output_dir, 'paste')
    concat_img_path = os.path.join(args.output_dir, 'concat')
    
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(gt_img_path, exist_ok=True)
    os.makedirs(mask_img_path, exist_ok=True)
    os.makedirs(mask_paste_img_path, exist_ok=True)
    os.makedirs(paste_img_path, exist_ok=True)
    os.makedirs(concat_img_path, exist_ok=True)
    
    for idx, example in enumerate(tqdm(dataset)):
        
        gt_img = example["PIL_gt"][0]
        paste_img = example["PIL_paste"][0]
        mask_img = example["mask"][0]
        img_name = example['img_name'][0]
        
        if os.path.exists(os.path.join(pre_img_path, img_name)):
            continue
        
        prompt_emb, uncondtion_emb = vtext_model.inference_vtoken(example["vtoken_prompt"][0].to(device=sd_inpaint.device), example["vtoken_prompt_uncond"][0].to(device=sd_inpaint.device), example["paste_clip_image"][0].to(device=sd_inpaint.device), sd_inpaint.text_encoder)
        pre_image = sd_inpaint(prompt_embeds=prompt_emb, negative_prompt_embeds=uncondtion_emb, image=paste_img, mask_image=mask_img, guidance_scale=args.inference_guidence_scale).images[0]
        
        mask_paste_img = overlay_mask_on_image(paste_img, mask_img)
        concat_img = images_composite([gt_img, mask_paste_img, mask_img, pre_image])
        
        pre_image.save(os.path.join(pre_img_path, img_name))
        gt_img.save(os.path.join(gt_img_path, img_name))
        mask_img.save(os.path.join(mask_img_path, img_name))
        mask_paste_img.save(os.path.join(mask_paste_img_path, img_name))
        paste_img.save(os.path.join(paste_img_path, img_name))
        concat_img.save(os.path.join(concat_img_path, img_name))
        

def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="dreambooth-inpaint-model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--test_mask_mode", type=int, default=1, choices=[1,2,3,4,5,6], help='ori, enrode, dilate, convex, bbox')
    parser.add_argument("--clip_vtext", type=str, default=None)
    parser.add_argument("--clip_ip_encoder", type=str, default=None)
    parser.add_argument("--clip_ip_ckpt", type=str, default=None)
    parser.add_argument("--inference_guidence_scale", default=1.0, type=float)
    parser.add_argument("--clip_visual_prompt", type=str, default=None,)
    
    args = parser.parse_args()
    
    mp.set_start_method('spawn',force=True)
    num_gpus=torch.cuda.device_count()
    print(f'Total have {num_gpus} gpus.')
    if num_gpus>1:
        pool = mp.Pool(processes=num_gpus, initializer=init, initargs=(args,))
    else:
        init(args, rank=0)

    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_visual_prompt)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_visual_prompt)
        
    print('Start loading dataset.')
    test_dataset = RegionInpaintingDataset_Test(
        data_root=args.data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        mask_mode=args.test_mask_mode,
        clip_processor=clip_processor,
        clip_vtext=args.clip_vtext,
    )
    
    num_samples = len(test_dataset)
    num_parts = num_gpus
    part_size = num_samples // num_parts
    if num_gpus>1:
        generator = torch.Generator().manual_seed(args.split_seed)
        test_dataset_parts = random_split(test_dataset, [part_size] * (num_parts - 1) + [num_samples - (part_size * (num_parts - 1))], generator=generator)
        test_dataloader_parts = [DataLoader(dataset_part, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn) for dataset_part in test_dataset_parts]
    else:
        test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    print('Start inference test dataset.')
    if num_gpus>1:
        with tqdm(total=num_gpus) as pbar:
            for i, res in enumerate(pool.starmap(run_inference, [(part, args) for part in test_dataloader_parts], 1)):
                results_json.append(res)
                pbar.update()
    else:
        results_json = run_inference(test_dataset, args)

    output_path_json = os.path.join(args.output_dir, 'result.json')
    with open(output_path_json,'w') as f:
        json.dump(results_json,f)

if __name__ == "__main__":
    main()