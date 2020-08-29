---
layout:     post
title:      MMDetection
subtitle:   Open MMLab Detection Toolbox and Benchmark
date:       2020-05-21
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Tools

---



## 1. Introduction

MMdetection的特点：

- 模块化设计：将不同网络的部分进行切割，模块之间具有很高的复用性和独立性（十分便利，可以任意组合）
- 高效的内存使用
- 支持多种框架
- SOTA



## 2. Support Frameworks

- 单阶段检测器

  > SSD(2015)、RetinaNet(2017)、GHM(2019)、FCOS(2019)、FSAF(2019)

- 两阶段检测器

  > Faster R-CNN(2015)、R-FCN(2016)、Mask R-CNN(2017)、Grid R-CNN(2018)、Mask Scoring R-CNN(2019)、Double-Head R-CNN(2019)

-  多阶段检测器

  > Cascade R-CNN(2017)、Hybrid Task Cascade(2019)

- 通用模块和方法

  > - soft-NMS(2017)、DCN(2017)、OHEN(2016)、DCN2(2018)、Train from Scratch(2018)、ScratchDet(2018)、M2Det (2018)、GCNet(2019) 、Generalized Attention(2019)、SyncBN(2018)、Group Normalization(2018)、Weight Standardization(2019)、HRNet(2019) 、Guided Anchoring(2019)、Libra R-CNN(2019)

## 3. Architecture

模型表征：划分为以下几个模块：

> Backbone（ResNet等）、Neck（FPN）、DenseHead（AnchorHead）、RoIExtractor、RoIHead（BBoxHead/MaskHead）



结构图如下：

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/mmdetection.png)

## 4. Benchmarks

- Datasets

  > 支持COCO-sytle和 VOC-style数据集

- Implementation details

  > Images are resized to a maximum scale of 1333 × 800,without changing the aspect ratio.
  >
  > 
  >
  > “1x” and “2x” means 12 epochs and 24 epochs respectively. “20e” is adopted in cascade models, which denotes 20 epochs.

- Evaluation metrics

  > Adopting standard evaluation metrics for COCO dataset, where multiple IoU thresholds from 0.5 to 0.95 are applied. 
  >
  > 
  >
  > The results of region proposal network (RPN) are measured with Average Recall (AR) and detection results are evaluated with mAP.

## 5. Config文件说明

> model
>
> - backbone：通常是FCN，用于提取特征图，例如ResNet
>
> - neck：
>
> - rpn_head
> - box_roi_extractor：
> - bbox_head
>
> train_cfg
>
> test_cfg

数据管道

> train_pipeline
>
> test_pipeline

```python
# model settings
model = dict(
    type='FastRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
  	# 数据加载
    dict(type='LoadImageFromFile'),   
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    
  	# 预处理
  	dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    
  	# 格式化
  	dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        proposal_file=data_root + 'proposals/rpn_r50_fpn_1x_train2017.pkl',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        proposal_file=data_root + 'proposals/rpn_r50_fpn_1x_val2017.pkl',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        proposal_file=data_root + 'proposals/rpn_r50_fpn_1x_val2017.pkl',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fast_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]

```

