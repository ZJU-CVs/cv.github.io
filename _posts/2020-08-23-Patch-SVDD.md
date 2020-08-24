---
layout:     post
title:      Patch SVDD
subtitle:   Patch-level SVDD for Anomaly Detection and Segmentation
date:       2020-08-23
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Paper
    - Anomaly Detection
    - FSL
---



#### 1. Introduction

> 异常检测和分割：
>
> - 异常检测包括对输入图像是否包含异常进行二元判断
> - 异常分割的目的是对异常进行像素级的定位



> `One-class support vector machine (OC-SVM)`和`Support vector data description (SVDD)`都是用于one-class classification的经典算法
>
> - 在给定核函数的情况下，OC-SVM从内核空间中从原点寻找一个最大边缘超平面
> - SVDD在内核空间中搜索一个data-enclosing hypersphere
>   - 基于深度学习提出了deep-SVDD，通过在核函数的位置部署一个深度神经网络，基于数据表示，无需手动选择合适的核函数



#### 2. Method

> 本文将deep-SVDD拓展到一种patch-wise的检测方法中，并结合自监督学习，实现了异常分割并提高异常检测的性能

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/5.png" alt="img" style="zoom:40%;" />



##### Patch-wise Deep SVDD

> - 训练一个编码器，该编码器将整个训练数据映射为位于特征空间中的small hypersphere内的特征
>
> - 使用以下损失函数训练编码器$f_{\theta}$以最小化特征与超球面中心之间的欧式距离：
>
> $$
> \mathcal{L}_{SVDD}=\sum_i \Vert f_{\theta}(x_i)-c \Vert_2
> $$
>
> - 在测试时，将输入的表示与**中心** $c$ 之间的距离作为异常分数
>
> $$
> \mathbf{c} \doteq \frac{1}{N} \sum_{i}^{N} f_{\theta}\left(\mathbf{x}_{i}\right)
> $$

> 本文将Deep SVDD这种方法拓展到patch-wise
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/6.png" alt="img" style="zoom:40%;" />
>
> - 编码器对每个batch进行编码，而不是整张图
>
> - Patch-wise inspection具有以下优点：
>   - 可以在每个位置获得检测结果，从而定位缺陷位置
>   - 这种细粒度的检测提高了整体检测的性能

> - 由下图可见，对于相对简单的图像，使用$\mathcal{L}_{SVDD}$和$\mathcal{L}_{Patch \ SVDD}$训练的编码器都能很好地定位缺陷；然而对于较复杂的图像，$\mathcal{L}_{SVDD}$无法定位
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/7.png" alt="img" style="zoom:40%;" />
>
> - 但是由于
> - 因此，
>
> $$
> \mathcal{L}_{SVDD'}=\sum_{i,i'} \Vert f_{\theta}(p_i)- f_{\theta}(p_{i'}) \Vert_2
> $$
>
> > 其中$p_{i'}$是$p_i$附近的patch



##### Self-supervised learning

> 为了捕获patch的语义，使用自监督学习
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/8.png" alt="img" style="zoom:40%;" />



##### Hierarchical encoding

> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/9.png" alt="img" style="zoom:40%;" />



##### Generating anomaly maps