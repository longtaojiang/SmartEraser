CUDA_VISIBLE_DEVICES='4' python app_remove.py \
--weight_dtype float32 \
--share \
--port 7850 \
--clip_dir './ckpts/clip-vit-large-patch14' \
--checkpoint_dir "./ckpts/smarteraser-weights" \