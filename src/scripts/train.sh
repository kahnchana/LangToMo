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
    --output-dir experiments/ox_ds7_002 \
    --num-gpu 8 \
    --port 8809 \
    --debug