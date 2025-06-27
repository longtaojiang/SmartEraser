import argparse
import itertools
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tensorboard
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from modules.clip_visual_token import CLIPVisualPrompt
import sys
import random

from modules.pipeline.pipeline_stable_diffusion_inpaint_region import StableDiffusionInpaintRegionPipeline

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PNDMScheduler,    
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from modules.eraser_dataset import RegionInpaintingDataset_Test, RegionInpaintingDataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

def images_composite(img_list, resolution=512):

    width, height = resolution, resolution
    total_width = width + width + width + width
    max_height = height
    
    new_image = Image.new('RGB', (total_width, max_height))

    for i, img in enumerate(img_list):
        new_image.paste(img, (i*width, 0))
    
    return new_image
        

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--visual_example",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--clip_visual_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--clip_ip_ckpt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--clip_vtext",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--random_condition_abandon",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--inference_guidence_scale",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreambooth-inpaint-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_warmup_epochs", type=float, default=1, help="Number of epochs for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--test_mask_mode", type=int, default=1, choices=[1,2,3,4,5], help='ori, enrode, dilate, convex, bbox')
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=float,
        default=1,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir).resolve()
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    args.pipeline_output_dir = os.path.join(args.output_dir, 'pipeline')
    args.checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    args.checkpoints_total_limit = args.checkpoints_total_limit - 1
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.pipeline_output_dir, exist_ok=True)
            os.makedirs(args.checkpoints_dir, exist_ok=True)
            
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=args.output_dir.split('/')[-1], 
                                  config=vars(args))
        tensorboard_tracker = accelerator.get_tracker("tensorboard")
        tensorboard_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.unsqueeze(0))
                            ])

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vtext_model = CLIPVisualPrompt(args.clip_visual_prompt)
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_visual_prompt)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_visual_prompt)
    
    vae.requires_grad_(False)
    vtext_model.vision_model.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    params_to_optimize = (
        itertools.chain(unet.parameters(), vtext_model.clip_mlp.parameters(), vtext_model.text_model.parameters())
    )
        
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = RegionInpaintingDataset(
        data_root=args.data_dir,
        split='train',
        tokenizer=tokenizer,
        size=args.resolution,
        clip_processor=clip_processor,
        clip_vtext=args.clip_vtext,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn
    )

    if accelerator.is_main_process:
        visual_example_dataset = RegionInpaintingDataset_Test(
            data_root=args.visual_example,
            tokenizer=tokenizer,
            size=args.resolution,
            mask_mode=args.test_mask_mode,
            clip_processor=clip_processor,
            clip_vtext=args.clip_vtext,
        )
        visual_example_dataset = DataLoader(visual_example_dataset, batch_size=1, shuffle=False, collate_fn=visual_example_dataset.collate_fn)
        visual_example_output_dir = os.path.join(args.output_dir, 'visual_example')
        os.makedirs(visual_example_output_dir, exist_ok=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps / accelerator.num_processes)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.lr_warmup_epochs != 0:
        args.lr_warmup_steps = int(args.lr_warmup_epochs * num_update_steps_per_epoch)
    if args.checkpointing_epochs != 0:
        args.checkpointing_steps = int(args.checkpointing_epochs * num_update_steps_per_epoch)


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, vtext_model.clip_mlp, vtext_model.text_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vtext_model.clip_mlp, vtext_model.text_model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(lr_scheduler)
    
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    vtext_model.vision_model.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.checkpoints_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.checkpoints_dir, path))
            global_step = int(path.split("-")[1])
            
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        vtext_model.clip_mlp.train()
        vtext_model.text_model.train()
            
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                # batch = {"input_ids": input_ids, "gt_values": gt_values, "paste_values": paste_values, "masks": masks, "paste_images": paste_images}

                latents = vae.encode(batch["gt_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                paste_latents = vae.encode(
                    batch["paste_images"].reshape(batch["gt_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                paste_latents = paste_latents * vae.config.scaling_factor

                masks = batch["masks"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                ).to(dtype=weight_dtype)
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, paste_latents], dim=1)
                abandon_num = random.random()
                if abandon_num < args.random_condition_abandon:
                    encoder_hidden_states = vtext_model.uncondition_train_vtoken(batch["vtoken_prompt_uncond"], batch["paste_clip_images"].to(dtype=weight_dtype))
                else:
                    encoder_hidden_states = vtext_model.train_vtoken(batch["vtoken_prompt"], batch["paste_clip_images"].to(dtype=weight_dtype))
                
                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), vtext_model.clip_mlp.parameters(), vtext_model.text_model.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.checkpointing_steps == 0 or global_step == 10 or global_step == 20:
                # if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        all_checkpoints = os.listdir(args.checkpoints_dir)
                        if args.checkpoints_total_limit is not None and (len(all_checkpoints) > args.checkpoints_total_limit):
                            all_checkpoints_num = [int(checkpoints.split('-')[-1]) for checkpoints in all_checkpoints]
                            all_checkpoints_num.sort()
                            del_checkpoints_number = len(all_checkpoints_num) - args.checkpoints_total_limit
                            del_checkpoints_num = all_checkpoints_num[:del_checkpoints_number]
                            for del_check_num in del_checkpoints_num: 
                                del_check_file = os.path.join(args.checkpoints_dir, f"checkpoint-{del_check_num}")
                                shutil.rmtree(del_check_file)
                        save_path = os.path.join(args.checkpoints_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        accelerator.print(f"Saved state checkpoint-{global_step}")
                        
                        save_path = os.path.join(args.pipeline_output_dir, f"pipeline-{global_step}")
                        
                        pipeline = StableDiffusionInpaintRegionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        text_encoder=accelerator.unwrap_model(vtext_model.text_model).to(dtype=torch.float32),
                        unet=accelerator.unwrap_model(unet).to(dtype=torch.float32),
                        )
                            
                        pipeline.to(accelerator.local_process_index)
                        
                        os.makedirs(os.path.join(visual_example_output_dir, f'pipelin-{global_step}'), exist_ok=True)
                        os.makedirs(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'pre'), exist_ok=True)
                        os.makedirs(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'gt'), exist_ok=True)
                        os.makedirs(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'mask'), exist_ok=True)
                        os.makedirs(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'paste'), exist_ok=True)
                        os.makedirs(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'comp'), exist_ok=True)

                        for example in (tqdm(visual_example_dataset)):
                            
                            gt_img = example["PIL_gt"][0]
                            paste_img = example["PIL_paste"][0]
                            mask_img = example["mask"][0]
                            img_name = example['img_name'][0]

                            prompt_emb, uncondtion_emb = vtext_model.inference_vtoken(example["vtoken_prompt"][0].to(device=pipeline.device), example["vtoken_prompt_uncond"][0].to(device=pipeline.device), example["paste_clip_image"][0].to(dtype=weight_dtype, device=pipeline.device), pipeline.text_encoder)
                            pre_image = pipeline(prompt_embeds=prompt_emb, negative_prompt_embeds=uncondtion_emb, image=paste_img, mask_image=mask_img, guidance_scale=args.inference_guidence_scale).images[0]
                            
                            img_comp = images_composite([gt_img, paste_img, mask_img, pre_image])
                            pre_image.save(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'pre', img_name))
                            gt_img.save(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'gt', img_name))
                            mask_img.save(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'mask', img_name))
                            paste_img.save(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'paste', img_name))
                            img_comp.save(os.path.join(visual_example_output_dir, f'pipelin-{global_step}', 'comp', img_name))
                            
                            tensorboard_tracker.log_images({f"{img_name}": tensorboard_transform(img_comp)}, step=global_step)
                        
                        pipeline.save_pretrained(save_path)
                        del pipeline, example
                        clip_mlp_temp = accelerator.unwrap_model(vtext_model.clip_mlp)
                        torch.save(clip_mlp_temp.state_dict(), os.path.join(save_path, 'clip_mlp_weight.pth'))
                        del clip_mlp_temp, prompt_emb, uncondtion_emb
                            
                        accelerator.print(f"Saved pipeline to {save_path}")
                    accelerator.wait_for_everyone()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
