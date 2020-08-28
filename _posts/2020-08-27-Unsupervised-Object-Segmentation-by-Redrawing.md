---
layout:     post
title:      Unsupervised Object Segmentation by Redrawing
subtitle:   
date:       2020-08-27
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Segmentation
---



#### 1. Introduction

- 提出了ReDO模型，能够以无监督的方式从图中提取对象而无需任何注释。

- 模型基于对抗结构。对于输入图像，生成器能够提取object mask，然后在同一位置重绘新对象。鉴别器确保生成图像的分布与原始图像的分布对齐

  

#### 2. Model——ReDO

- 掩膜生成器f——基于PSPnet

  - 金字塔场景分析网络，对不同区域的语境进行聚合，使模型拥有理解全局语境信息的能力。（因为在深层网络中，越深层次的特征包含越多的语义和更少的位置信息，结合多尺度特征可以提高网络性能）
  - 全局平均池化（GAP）通常用于图像分类任务，将它用于提取全局语境信息是一种比较好的方式

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/image-20190901222250077.png)

  - 具体步骤：
    - 输入图像后，使用预训练的带空洞卷积ResNet提取特征图。最终的特征映射大小是输入图像的1/8，如(b)所示

    - 在特征图上，我们使用(c)中的金字塔池化模块来收集上下文信息。使用4层金字塔结构，池化内核覆盖了图像的全部、一半和小部分。它们被融合为全局先验信息

    - 在(c)的最后部分将之前的金字塔特征映射与原始特征映射concate起来

    - 再进行卷积，生成(d)中的最终预测图

      

- 区域生成器$G_k$，鉴别器D和重建z的网络δ——基于SAGAN 

  

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/34.png)  



- loss定义

$$
\begin{array}{l}
\max_{\mathrm{G}_{\mathrm{F}, \delta}} \mathcal{L}_{\mathrm{G}}=\mathbb{E}_{\mathbf{I} \sim p_{\text {data}}, \mathbf{i} \sim \mathcal{U}(n), z_{\mathrm{i}} \sim p(z)}\left[\mathrm{D}\left(\mathrm{G}_{\mathrm{F}}\left(\mathbf{I}, \mathbf{z}_{\mathbf{i}}, \mathbf{i}\right)\right)-\lambda_{z}\left\|\delta_{\mathbf{i}}\left(\mathrm{G}_{\mathrm{F}}\left(\mathbf{I}, \mathbf{z}_{\mathbf{i}}, \mathbf{i}\right)\right)-\mathbf{z}_{\mathbf{i}}\right\|_{2}\right] \\
\max _{\mathrm{D}} \mathcal{L}_{\mathrm{D}}=\mathbb{E}_{\mathbf{I} \sim p_{\text {data}}}\left[\min (0,-1+\mathrm{D}(\mathbf{I})]+\mathbb{E}_{\mathbf{I} \sim p_{\text {data}}, \mathbf{i} \sim \mathcal{U}(n), \mathbf{z}_{\mathrm{i}} \sim p(\mathbf{z})}\left[\min \left(0,-1-\mathrm{D}\left(\mathrm{G}_{\mathrm{F}}\left(\mathbf{I}, \mathbf{z}_{\mathbf{i}}, \mathbf{i}\right)\right)\right]\right.\right.
\end{array}
$$

