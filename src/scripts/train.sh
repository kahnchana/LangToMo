
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

# More settings.
python src/train.py \
    --dataset calvin \
    --lr 1e-4 \
    --epochs 100 \
    --crop-ratio 0.70 \
    --pretrained "experiments/test_016" \
    --output-dir test_017