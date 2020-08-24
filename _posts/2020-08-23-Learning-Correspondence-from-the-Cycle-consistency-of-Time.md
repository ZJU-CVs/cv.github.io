---
layout:     post
title:      Learning Correspondence from the Cycle-consistency of Time
subtitle:   
date:       2020-08-23
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Video Analysis
---



#### 1. Introduction

> 介绍了一种**自监督方法**来学习未标记视频中的视觉对应关系。
>
> 其主要思想是利用时间上的周期一致性(cycle-consistency) 作为自由监督信号，从头开始学习视觉表征。



#### 2. Related Works

`Correspondence`

##### Visual Tracking

> 获得box-level correspondence
>
> 但是目前训练模型做tracking需要标注视频的每一帧进行训练，大大限制了训练样本的数量



**Optical Flow Estimation**

> 获得pixel-level correspondence
>
> 但通常训练模型计算optimal flow需要在synthetic dataset 上进行，使得训练出来的网络很难泛化到真实数据中 (generalization to real data)。而且 optical flow 对于局部的变化过于敏感，很难处理长距离或者 large motion 的视频



#### 3. Method

> 本文所提出的是介于tracking和optical flow的 mid-level correspondence/semi-dense correspondence

