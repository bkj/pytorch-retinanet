#!/usr/bin/env python

"""
    train.py
"""

from __future__ import print_function

import sys
import json
import warnings
import argparse
import collections
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, \
    Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'
assert torch.cuda.is_available()
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco-path', help='Path to COCO directory')
    parser.add_argument('--csv-train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv-classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv-val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--device-ids', help='Device IDs', type=str, default='0,5,6,7')
    
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr-patience', type=float, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-grad-norm', type=float, default=0.1)  # !! This seems small?
    parser.add_argument('--config-path', type=str, default='./.config.json')
    
    args = parser.parse_args()
    
    assert args.depth in [18, 34, 50, 101, 152]
    if args.dataset == 'coco':
        assert args.coco_path is not None
    elif args.dataset == 'csv':
        assert args.csv_train is not None
        assert args.csv_classes is not None
    else:
        raise Exception('args.dataset not in ["coco", "csv"]')
    
    return args

if __name__ == "__main__":
    args = parse_args()
    json.dump(vars(args), open(args.config_path, 'w'))
    
    # Create the data loaders
    if args.dataset == 'coco':
        dataset_train = CocoDataset(
            root_dir=args.coco_path,
            set_name='train2017', 
            transform=transforms.Compose([
                Normalizer(),
                Augmenter(),
                Resizer()
            ])
        )
        
        dataset_val = CocoDataset(
            root_dir=args.coco_path,
            set_name='val2017', 
            transform=transforms.Compose([
                Normalizer(),
                Resizer()
            ])
        )
        
    elif args.dataset == 'csv':
        dataset_train = CSVDataset(
            train_file=args.csv_train,
            class_list=args.csv_classes,
            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        )
        
        if args.csv_val is not None:
            dataset_val = CSVDataset(
                train_file=args.csv_val,
                class_list=args.csv_classes, 
                transform=transforms.Compose([Normalizer(), Resizer()])
            )
        else:
            dataset_val = None
            print('No validation annotations provided.')
    
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=4,
        collate_fn=collater,
        batch_sampler=AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False),
    )
    
    if dataset_val is not None:
        dataloader_val = DataLoader(
            dataset_val,
            num_workers=4,
            collate_fn=collater,
            batch_sampler=AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False),
        )
        
    # --
    # Define model
    
    device_ids = [int(device_id) for device_id in args.device_ids.split(',')]
    
    model_fn  = getattr(model, 'resnet%d' % args.depth)
    retinanet = model_fn(num_classes=dataset_train.num_classes(), pretrained=True).cuda(device=device_ids[0])
    retinanet  = torch.nn.DataParallel(retinanet, device_ids=device_ids).cuda(device=device_ids[0])
    retinanet.training = True
    
    opt = optim.Adam(retinanet.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=args.lr_patience, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    
    retinanet.train()
    retinanet.module.freeze_bn()
    
    print('Num training images: %d' % len(dataset_train), file=sys.stderr)
    
    for epoch_num in range(args.epochs):
        
        retinanet.train()
        retinanet.module.freeze_bn()
        
        epoch_loss = []
        
        start_time = time()
        for iter_num, data in enumerate(dataloader_train):
            
            opt.zero_grad()
            
            img, annot = data['img'], data['annot']
            img = data['img'].cuda(device=device_ids[0]).float()
            cls_loss, reg_loss = retinanet([img, annot])
            
            cls_loss, reg_loss = cls_loss.mean(), reg_loss.mean()
            loss = cls_loss + reg_loss
            
            # if loss == 0:  # !! Seems unnecessary
            #     continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), args.max_grad_norm)
            
            opt.step()
            
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            
            print(json.dumps({
                "epoch_num"  : epoch_num,
                "iter_num"   : iter_num,
                "img_num"    : iter_num * args.batch_size,
                "cls_loss"   : float(cls_loss),
                "reg_loss"   : float(reg_loss), 
                "loss_hist"  : float(np.mean(loss_hist)),
                "elapsed"    : time() - start_time,
            }))
            sys.stdout.flush()
            
            del cls_loss
            del reg_loss
        
        print('Evaluating dataset')
        if args.dataset == 'coco':
            coco_eval.evaluate_coco(dataset_val, retinanet)
        elif args.dataset == 'csv' and args.csv_val is not None:
            _ = csv_eval.evaluate(dataset_val, retinanet)
        
        lr_scheduler.step(np.mean(epoch_loss))
        
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(args.dataset, epoch_num))
    
    retinanet.eval()
    torch.save(retinanet, 'model_final.pt'.format(epoch_num))

