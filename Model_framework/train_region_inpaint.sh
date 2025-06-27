current_time=$(date +"%m-%d-%H-%M-%S")

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --config_file 'accelerate_config/default_config_8_gpus.yaml' \
train_region_inpaint.py \
--pretrained_model_name_or_path './ckpts/stable-diffusion-v1-5-inpainting' \
--resume_from_checkpoint "latest" \
--visual_example './visual_examples' \
--data_dir '/path/to/SAM_COCONut_paste' \
--output_dir "outputs/smarteraser_test" --train_batch_size 4 \
--num_train_epochs 25 --checkpointing_epochs 0.25 --checkpoints_total_limit 2 \
--learning_rate 5e-7 --scale_lr --lr_scheduler constant \
--gradient_accumulation_steps 1 \
--mixed_precision 'fp16' --enable_xformers_memory_efficient_attention \
--test_mask_mode 3 \
--random_condition_abandon 0.1 \
--inference_guidence_scale 1.5 \
--clip_visual_prompt './ckpts/clip-vit-large-patch14' \
--clip_vtext 'Remove the instance of object' \




