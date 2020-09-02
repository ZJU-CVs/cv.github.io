---
layout:     post
title:      Learning Memory-guided Normality for Anomaly Detection
subtitle:   
date:       2020-09-02
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Anomaly Detection

---



#### 1. Introduction

- 异常检测通常视为无监督学习问题，利用没有异常样本的训练集学习一个描述正常样本的模型；在测试阶段，模型未描述的事件和活动被视为异常

- Related works

  > - 基于重构误差的异常检测方法：在训练时候，给定正常的视频帧作为输入，模型提取特征表示并尝试再次重建输入。在测试时将具有较大重构误差的视频帧视为异常（前提假设时异常样本不能很好地重构，因为在模型训练期间从未见过）
  >
  >   > 缺点：基于CNN的自编码器等模型具有强大的特征表示能力能够更好的提取特征，异常帧的CNN特征可能会通过结合正常帧的CNN特征来重构，因此在这种情况下重构误差较小
  >
  > - 基于预测未来帧的方法：在训练时将预测的未来帧与真实帧的差异最小化，是一种间接的异常检测方法
  >
  > `以上两种方法训练的模型提取的是更接近于general特征表示，而不是正常模式`
  >
  > 
  >
  > - 基于Deep SVDD的方法：利用单类分类目标将正常样本映射到超球体，在训练过程中最大程度减少超球体的体积，从而使正常样本紧密地映射到球的中心。这个中心代表正常数据的通用特征（即原型prototypical）
  >
  >   > 缺点：只有单个中心点，未考虑正常样本的各种模式



#### 2. Method

- 本文提出了一种在视频序列中进行异常检测的无监督学习方法

- 本文认为单个原型特征不足以表示正常数据的各种模式，即视频的特征空间存在多个原型。因此设计了异常检测的存储模块，用于存储正常数据中不同的原型

  ><img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/28.png" alt="img" style="zoom:40%;" />



##### Network architecture

**输入：**四个连续的视频帧

**输入：**预测第五个视频帧



**模型组件：**主要包括编码器、存储模块和解码器

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/26.png" alt="img" style="zoom:30%;" />

> 编码器输入正常的视频帧并提取query features
>
> 使用memory items检索对应的正常模式原型并更新memory
>
> 将query features和memory聚合后输入到解码器
>
> 解码器重构得到预测帧



##### Encoder and Decoder

> 采用Unet的结构，但是去掉了skip connections
>
> 输入视频帧$I_t$



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/27.png" alt="img" style="zoom:30%;" />



#### 3. Experiments



