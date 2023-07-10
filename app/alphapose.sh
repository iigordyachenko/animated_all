#!/bin/sh

cd /home/ubuntu/proj/bvh_create/AlphaPose/  # Перейти в директорию исполнения

python3.10 scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video examples/IMG_9710.mp4 