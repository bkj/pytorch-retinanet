#!/bin/bash

# run.sh

PROJECT_ROOT=$(pwd)

# --
# Install

conda create -n retina_env python=3.6 pip -y
source activate retina_env

pip install -r requirements.txt
conda install pytorch==0.4.1 torchvision cuda91 -c pytorch -y

cd $PROJECT_ROOT/lib
bash build.sh
cd $PROJECT_ROOT

# --
# Download COCO dataset (2017)

mkdir -p $PROJECT_ROOT/data
cd $PROJECT_ROOT/data
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip

mkdir -p $PROJECT_ROOT/data/images
cd $PROJECT_ROOT/data/images
wget http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip && rm train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && rm val2017.zip

# --
# Run on COCO

mkdir -p results
python train.py \
    --batch-size 16 \
    --dataset coco \
    --coco-path data/coco \
    --depth 50 | tee results/coco_50_v0.jl