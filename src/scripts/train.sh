export PYTHONPATH=$PYTHONPATH:$PWD

export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2


# Default Training.
python src/train.py \
    --dataset calvin \
    --lr 1e-4 \
    --epochs 100 \
    --mask-ratio 0.50 \
    --mask-patch 16 \
    --crop-ratio 0.70 \
    --mask-crop-ratio 0.50 \
    --pretrained None \
    --output-dir test_017

# 2D UNet - Roboturk.
python src/train_online.py \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --dataset openx \
    --sub-datasets roboturk \
    --val-sub-datasets roboturk \
    --lr 1e-4 \
    --steps 300_000 \
    --prev-flow \
    --output-dir experiments/roboturk_001 \
    --resume \
    --wandb-id "pm6915gx" \
    --num-gpu 4 \
    --port 8807 \
    --debug

# 2D UNet - Metaworld.
python src/train_online.py \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --dataset metaworld \
    --lr 1e-4 \
    --steps 300_000 \
    --prev-flow \
    --output-dir experiments/mw_010 \
    --pretrained experiments/roboturk_001/model \
    --num-gpu 4 \
    --port 8807 \
    --debug

# 2D U-Net Metaworld - scratch
python src/train_online.py \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --dataset metaworld \
    --lr 1e-4 \
    --steps 300_000 \
    --prev-flow \
    --output-dir experiments/mw_011 \
    --num-gpu 4 \
    --port 8807 \
    --debug


# 3D U-NetMetaworld.
python src/train_online.py \
    --dataset metaworld \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --image-size 128 \
    --lr 1e-4 \
    --steps 300_000 \
    --temporal \
    --num-frames 2 \
    --output-dir experiments/mw_008 \
    --num-gpu 4 \
    --seed 7 \
    --port 8807 \
    --debug

# OpenX 7_sub datasets.
python src/train_online.py \
    --dataset openx \
    --sub-datasets split_7_ds \
    --val-sub-datasets taco_play \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --image-size 128 \
    --lr 1e-4 \
    --steps 300_000 \
    --prev-flow \
    --output-dir experiments/ox_ds7_001 \
    --resume \
    --wandb-id "lthj46nx" \
    --num-gpu 4 \
    --port 8809 \
    --debug

# OpenX ucsd_pick_and_place.
python src/train_online.py \
    --dataset openx \
    --sub-datasets ucsd_pick_and_place_dataset_converted_externally_to_rlds \
    --val-sub-datasets ucsd_pick_and_place_dataset_converted_externally_to_rlds \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --image-size 128 \
    --lr 1e-4 \
    --steps 300_000 \
    --prev-flow \
    --output-dir experiments/ox_xarm_001 \
    --num-gpu 4 \
    --port 8809 \
    --debug

# OpenX ucsd_pick_and_place 2-frame.
python src/train_online.py \
    --dataset openx \
    --sub-datasets ucsd_pick_and_place_2frame \
    --val-sub-datasets ucsd_pick_and_place_dataset_converted_externally_to_rlds \
    --train-batch-size 32 \
    --eval-batch-size 16 \
    --image-size 128 \
    --lr 1e-5 \
    --steps 10_000 \
    --prev-flow \
    --output-dir experiments/ox_xarm_002 \
    --pretrained experiments/ox_ds7_002/model \
    --num-gpu 4 \
    --port 8809 \
    --debug

# Real world fine-tune.
python src/train_online.py \
    --dataset realworld \
    --train-batch-size 32 \
    --eval-batch-size 4 \
    --image-size 128 \
    --lr 1e-4 \
    --steps 300_000 \
    --prev-flow \
    --output-dir experiments/rw_001 \
    --pretrained experiments/ox_ds7_001/model \
    --num-gpu 4 \
    --port 8809 \
    --debug


# Real world fine-tune v2.
python src/train_online.py \
    --dataset realworld \
    --train-batch-size 32 \
    --eval-batch-size 4 \
    --train-root /home/kanchana/data/human_jp_dataset/v1 \
    --val-root /nfs/ws2/kanchana/real_world/dataset_v1_val \
    --image-size 128 \
    --lr 1e-5 \
    --steps 50_000 \
    --prev-flow \
    --output-dir experiments/rw_004 \
    --pretrained experiments/ox_xarm_002/model \
    --num-gpu 4 \
    --port 8809 \
    --debug


# Real world fine-tune 2-frame.
python src/train_online.py \
    --dataset realworld_2f \
    --train-batch-size 32 \
    --eval-batch-size 4 \
    --image-size 128 \
    --lr 1e-5 \
    --steps 50_000 \
    --prev-flow \
    --output-dir experiments/rw_003 \
    --pretrained experiments/ox_xarm_002/model \
    --num-gpu 4 \
    --port 8809 \
    --debug


# Train policy 
python src/train_policy.py \
    --dataset metaworld \
    --train-batch-size 128 \
    --eval-batch-size 32 \
    --image-size 128 \
    --lr 1e-4 \
    --steps 100_000 \
    --output-dir experiments/mw_policy_002 \
    --num-gpu 3 \
    --port 8808 \
    --debug
