import argparse
import os
import random
import sys
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from modules.clip_visual_token import CLIPVisualPrompt
from transformers import CLIPTokenizer, CLIPImageProcessor
import datetime
import shutil

from modules.pipeline.pipeline_stable_diffusion_inpaint_region import StableDiffusionInpaintRegionPipeline
torch.set_grad_enabled(False)

def images_composite(img_list):

    width, height = img_list[0].size
    total_width = width + width + width + width
    max_height = height
    
    new_image = Image.new('RGB', (total_width, max_height))

    for i, img in enumerate(img_list):
        new_image.paste(img, (i*width, 0))
    
    return new_image

def save_input_image(input_image, result, input_mask):
    
    main_dir = './demo_save'
    filename = f"saved_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    input_image_dir = os.path.join(main_dir, 'input', filename)
    input_mask_dir = os.path.join(main_dir, 'mask', filename)
    mask_region_dir = os.path.join(main_dir, 'mask_region', filename)
    result_dir = os.path.join(main_dir, 'removed', filename)
    concat_dir = os.path.join(main_dir, 'concat', filename)
    
    input_image['image'].save(input_image_dir)
    shutil.copyfile(result[0]['name'], result_dir)
    shutil.copyfile(input_mask[1]['name'], mask_region_dir)
    shutil.copyfile(input_mask[0]['name'], input_mask_dir)
    
    input_image = input_image['image']
    mask_region_image = Image.open(input_mask[0]['name']).convert('RGB')
    mask_image = Image.open(input_mask[1]['name']).convert('RGB')
    result_image = Image.open(result[0]['name']).convert('RGB')
    concat_image = images_composite([input_image, mask_region_image, mask_image, result_image])
    concat_image.save(concat_dir)
    
    return f"Image saved as {filename}"

def resize_and_pad(image, mask, target_size=512):
    
    original_w, original_h = image.size
    
    if original_w > original_h:
        scale = target_size / original_w
        new_w = target_size
        new_h = int(original_h * scale)
    else:
        scale = target_size / original_h
        new_h = target_size
        new_w = int(original_w * scale)
    
    image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    image_padded = ImageOps.expand(image_resized, (pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h), (255, 255, 255))
    mask_padded = ImageOps.expand(mask_resized, (pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h), 0)
    
    return image_padded, mask_padded, (original_w, original_h), (scale, pad_w, pad_h)

def unpad_and_resize_back(image_padded, original_size, ori_mask, ori_image, params):
    original_w, original_h = original_size
    scale, pad_w, pad_h = params
    ori_image = ori_image.resize((int(original_w * scale), int(original_h * scale)))
    ori_image = ori_image.resize((original_w, original_h))

    image_unpadded = image_padded.crop((pad_w, pad_h, pad_w + int(original_w * scale), pad_h + int(original_h * scale)))
    image_original = image_unpadded.resize((original_w, original_h), Image.Resampling.BILINEAR)
    
    ori_mask = ori_mask.filter(ImageFilter.GaussianBlur(radius=5))
    image_final = Image.composite(image_original, ori_image, ori_mask)

    return image_final

def resize_and_crop(image, mask, target_size=512):

    original_w, original_h = image.size
    scale = target_size / min(original_w, original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    
    image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    mask_np = np.array(mask_resized)
    y_indices, x_indices = np.where(mask_np > 0)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    crop_x1 = max(0, min(min_x - (target_size - (max_x - min_x)) // 2, new_w - target_size))
    crop_y1 = max(0, min(min_y - (target_size - (max_y - min_y)) // 2, new_h - target_size))
    crop_x2 = crop_x1 + target_size
    crop_y2 = crop_y1 + target_size

    image_cropped = image_resized.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    mask_cropped = mask_resized.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    params = (original_w, original_h, scale, (crop_x1, crop_y1, crop_x2, crop_y2))
    return image_cropped, mask_cropped, params

def restore_image(image_cropped, params, image_original, ori_mask):
    original_w, original_h, scale, (crop_x1, crop_y1, crop_x2, crop_y2) = params
    
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    restored_image = image_original.resize((new_w, new_h))
    ori_mask = ori_mask.resize((new_w, new_h))
    
    restored_image.paste(image_cropped, (crop_x1, crop_y1))
    ori_mask = ori_mask.filter(ImageFilter.GaussianBlur(radius=5))
    final_image = Image.composite(restored_image, image_original.resize((new_w, new_h)), ori_mask)
    final_image = final_image.resize((original_w, original_h), Image.Resampling.BILINEAR)
    return final_image

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def select_tab_object_removal():
    return "object-removal"

class MVRemoverController:
    def __init__(self, weight_dtype, checkpoint_dir, clip_dir) -> None:
        self.checkpoint_dir = checkpoint_dir
        
        self.pipe = StableDiffusionInpaintRegionPipeline.from_pretrained(checkpoint_dir, torch_dtype=weight_dtype)
        self.pipe = self.pipe.to("cuda")

        self.clip_model = CLIPVisualPrompt(clip_dir)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
        self.clip_model.load_mlp_weight(os.path.join(checkpoint_dir, 'clip_mlp_weight.pth'))
        self.clip_model.vision_model.eval()
        self.clip_model.vision_model.to("cuda")
        self.clip_model.text_model.eval()
        self.clip_model.text_model.to("cuda")
        self.clip_model.clip_mlp.eval()
        self.clip_model.clip_mlp.to("cuda")
        
    def predict(
        self,
        input_image,
        ddim_steps,
        scale,
        seed,
    ):
        input_image["image"] = input_image["image"].convert("RGB")
        input_image["mask"] = input_image["mask"].convert("L")
        ori_image = input_image["image"]
        ori_mask = input_image["mask"]
        
        W, H = ori_image.size
        mask_bbox = ori_mask.getbbox()
        mask_w = mask_bbox[2] - mask_bbox[0]
        mask_h = mask_bbox[3] - mask_bbox[1]
        mask_max_l = max(mask_w, mask_h)
        if mask_max_l > min(W, H):
            radio = 'padding'
        else:
            radio = 'crop'
        
        if radio == 'padding':
            input_image["image"], input_image["mask"], original_size, params = resize_and_pad(input_image["image"], input_image["mask"])
        elif radio == 'crop':
            input_image["image"], input_image["mask"], params = resize_and_crop(input_image["image"], input_image["mask"])
            
        
        input_image["mask"] = gray2whiteblack(input_image["mask"])
        set_seed(seed)
        
        paste_clip_image = crop_white_instance(input_image["mask"], input_image["image"])
        paste_clip_image = self.clip_processor(images=paste_clip_image, return_tensors="pt")['pixel_values']
        vtoken_prompt = self.tokenizer('Remove the instance of', padding="max_length", max_length=7, truncation=True, return_tensors="pt")['input_ids']
        uncond_vtoken_prompt = self.tokenizer('', padding="max_length", max_length=7, truncation=True, return_tensors="pt")['input_ids']
        prompt_emb, uncondtion_emb = self.clip_model.inference_vtoken(vtoken_prompt.to(device=self.pipe.device), uncond_vtoken_prompt.to(device=self.pipe.device), paste_clip_image.to(device=self.pipe.device), self.pipe.text_encoder)
        result = self.pipe(prompt_embeds=prompt_emb, negative_prompt_embeds=uncondtion_emb, image=input_image["image"], mask_image=input_image["mask"], num_inference_steps=ddim_steps, guidance_scale=scale).images[0]

        if radio == 'padding':
            result = unpad_and_resize_back(result, original_size, ori_mask, ori_image, params)
        elif radio == 'crop':
            result = restore_image(result, params, ori_image, ori_mask)
        
        mask_np = np.array(ori_mask.convert("RGB"))
        red = np.array(ori_image).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        ori_image_m = np.array(ori_image)
        ori_image_m = Image.fromarray(
            (
                ori_image_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        dict_res = [ori_mask.convert("RGB"), ori_image_m]
        dict_out = [result]
        return dict_out, dict_res
    
    def infer(
        self,
        input_image,
        ddim_steps,
        scale,
        seed,
    ):
        
        return self.predict(
            input_image, ddim_steps, scale, seed,
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--weight_dtype", type=str, default="float32")
    args.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ppt-v1")
    args.add_argument("--clip_dir", type=str, default="./checkpoints/ppt-v1")
    args.add_argument("--share", action="store_true")
    args.add_argument("--port", type=int, default=7860)
    args = args.parse_args()

    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = MVRemoverController(weight_dtype, args.checkpoint_dir, args.clip_dir)
    
    # ui
    with gr.Blocks(css="style.css") as demo:
        
        gr.HTML(
        """
        <div style="text-align: center; max-width: 1600px; margin: 20px auto;">
        <h2 style="font-weight: 1200; font-size: 2.5rem; margin: 0rem">
            SmartEraser: Remove Anything from Images using Masked-Region Guidance
        </h2>     
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input image and draw mask")
                input_image = gr.Image(source="upload", tool="sketch", type="pil")

                # ["text-guided", "object-removal", "shape-guided", "image-outpainting"]
                task = gr.Radio(
                    ["object-removal"],
                    show_label=False,
                    visible=False,
                )

                # Object removal inpainting
                with gr.Tab("SmartEraser for Object removal") as tab_object_removal:
                    enable_object_removal = gr.Checkbox(
                        label="Enable remove object smartly",
                        value=True,
                        info="SmartEraser can “smartly” identify removal targets and remove it while preserving its surrounding region.",
                        interactive=False,
                    )
                tab_object_removal.select(fn=select_tab_object_removal, inputs=None, outputs=task)

                run_button = gr.Button(label="Run")
                save_button = gr.Button("Save Images")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                
                with gr.Accordion("Advanced options", open=True):
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale",
                        info="The recommended configuration for the Guidance Scale is 1.5.",
                        minimum=-5,
                        maximum=20.0,
                        value=1.5,
                        step=0.1,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        value=1767861908,
                        step=1,
                    )   
            
            with gr.Column():
                gr.Markdown("### Object removal result")
                inpaint_result = gr.Gallery(label="Generated images", show_label=False, columns=1, rows=2, object_fit="contain")
                gr.Markdown("### Mask")
                gallery = gr.Gallery(label="Generated masks", show_label=False, columns=1, rows=2, object_fit="contain")
        
        save_button.click(
            fn=save_input_image,
            inputs=[input_image, inpaint_result, gallery],
            outputs=save_status
        )
        
        run_button.click(
            fn=controller.infer,
            inputs=[
                input_image,
                ddim_steps,
                scale,
                seed,
            ],
            outputs=[inpaint_result, gallery],
        )
        

    demo.queue()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
