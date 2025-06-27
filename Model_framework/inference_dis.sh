current_time=$(date +"%m-%d-%H-%M-%S")
CUDA_VISIBLE_DEVICES='4,5' python inference_dis.py \
--pretrained_model_name_or_path "./ckpts/smarteraser-weights" \
--clip_visual_prompt './ckpts/clip-vit-large-patch14' \
--clip_vtext 'Remove the instance of object' \
--split_seed 42 \
--data_dir "./visual_examples" \
--output_dir './test' \
--test_mask_mode 3 \
--inference_guidence_scale 1.5 \