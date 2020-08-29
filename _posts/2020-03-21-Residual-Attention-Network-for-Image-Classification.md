---
layout:     post
title:      Residual Attention Network for Image Classification
subtitle:   
date:       2020-03-21
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Classification
    - Attention Mechanism
---



#### 1. Introduction

- 在视觉领域中Attention机制具有重大作用。Attention能使运算聚焦于特定区域，同时也可以使该部分区域的特征的到增强
- `very deep`的网络结构结合残差连接在图像分类等任务中表现出极好的性能
- 所提出的残差注意力网络（residual attention network）具有以下属性：
  - 增加更多注意力模型可以线性提升网络的分类性能，基于不同深度的特征图可以提取额外的注意力模型
  - 残差注意力模型可以结合到目前的大部分深层网络中，做到end-to-end训练结果，因为残差结构的存在，可以很容易将网络扩展到百数层。并且使用该种策略可以在达到其他大网络的分类准确率的同时显著降低计算量（计算量基本上为ResNet大网络的69%左右）
  
  

#### 2. Method

##### soft mask

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/attention-residual2.png" alt="attention-residual" style="zoom:50%;" />

- 注意力模块为主干网络以某一个特征图为节点的分叉自网络。不同层特征图响应的注意力不同。在浅层结构中，网络的注意力集中于背景等区域；而在深层结构中，网络的注意力特征聚焦于待分类的物体（因为深层次的特征图具有更高的抽象性和语义表达能力，对于物体分类相比浅层特征有较大作用）

  > ![attention-residual](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/attention-residual.png)

  

##### 整体结构

![attention-residual](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/attention-residual1.png)

