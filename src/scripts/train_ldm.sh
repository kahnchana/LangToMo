export PYTHONPATH=$PYTHONPATH:$PWD

export OMP_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=2


accelerate launch --mixed_precision="fp16" src/train_online_ldm.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name=openx \
    --train_data_dir /export/home/data/openx \
    --sub_datasets fractal20220817_data \
    --val_sub_datasets ucsd_pick_and_place_dataset_converted_externally_to_rlds \
    --num_train_steps=15_000 \
    --train_batch_size=64 \
    --eval_batch_size=16 \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --validation_steps=1000 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --output_dir="experiments/ldm-003" \
    --report_to=wandb \
    --debug


accelerate launch --mixed_precision="fp16" --multi_gpu src/train_online_ldm.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name=openx \
    --train_data_dir /export/home/data/openx \
    --sub_datasets fractal20220817_data \
    --val_sub_datasets ucsd_pick_and_place_dataset_converted_externally_to_rlds \
    --num_train_steps=50_000 \
    --train_batch_size=64 \
    --eval_batch_size=16 \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --validation_steps=1000 \
    --learning_rate=5e-05 --lr_scheduler=cosine \
    --max_grad_norm=1 --lr_warmup_steps=500 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --output_dir="experiments/ox_ldm_002" \
    --report_to=wandb \
    --debug


accelerate launch --mixed_precision="fp16" --multi_gpu src/train_online_ldm.py \
 --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 \
 --dataset_name=fusing/instructpix2pix-1000-samples \
 --use_ema \
 --enable_xformers_memory_efficient_attention \
 --resolution=256 \
 --train_batch_size=64 --gradient_checkpointing \
 --max_train_steps=15000 \
 --validation_epochs=20 \
 --checkpointing_steps=200 --checkpoints_total_limit=1 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
 --validation_prompt="make the mountains snowy" \
 --seed=42 \
 --output_dir="experiments/ldm-002" \
 --report_to=wandb 