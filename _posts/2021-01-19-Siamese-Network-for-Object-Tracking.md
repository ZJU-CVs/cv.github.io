---
layout:     post
title:      Siamese Network for Object Tracking
subtitle:   
date:       2021-01-19
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Object Tracking
    - Overview
---

https://blog.csdn.net/WZZ18191171661/article/details/88369667



### 1. Background

#### Object Tracking

> 使用视频序列第一帧的图像(包括bounding box的位置)，来找出目标在后面序列帧中的位置

#### Siamese Network

> 孪生网络是一种基于度量学习(metric learning)的方法
>
> - 孪生网络不是一种网络结构，而是一种网络架构(框架)
> - 孪生网络的backbone可以是CNN(用于图像相似性度量)，也可以是LSTM(用于自然语言语义的相似性分析)
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/siamese-network.jpg" alt="img" style="zoom:67%;" />
>
> - **伪孪生网络(pseudo siamese network)**
>
>   > (两个输入的网络权重不共享)
>   >
>   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/1.png" alt="img" style="zoom:33%;" />
>
> 
>
> - 损失函数
>
>   - Contrastive Loss
>
>     > 《Dimensionality Reduction by Learning an Invariant Mapping》
>     > $$
>     > D_{W}\left(\vec{X}_{1}, \vec{X}_{2}\right)=\left\|G_{W}\left(\vec{X}_{1}\right)-G_{W}\left(\vec{X}_{2}\right)\right\|_{2} \\
>     > \mathcal{L}(W)=\sum_{i=1}^{P} L\left(W,\left(Y, \vec{X}_{1}, \vec{X}_{2}\right)^{i}\right)\\
>     > L\left(W,\left(Y, \vec{X}_{1}, \vec{X}_{2}\right)^{i}\right)=(1-Y) L_{S}\left(D_{W}^{i}\right)+Y L_{D}\left(D_{W}^{i}\right)\\
>     > \begin{array}{l}
>     > L\left(W, Y, \vec{X}_{1}, \vec{X}_{2}\right)= (1-Y) \frac{1}{2}\left(D_{W}\right)^{2}+(Y) \frac{1}{2}\left\{\max \left(0, m-D_{W}\right)\right\}^{2}
>     > \end{array}
>     > $$
>     >
>     > - $D_W$是$X_1$和$X_2$在隐空间的欧式距离
>     > - Y为成对标签，若$X_1$和$X_2$为一类则$Y=0$；否则$Y=1$
>     > - 当两个不同类的$X_1$和$X_2$在隐空间中的距离大于m，则不再做优化
>
>   - Triplet Loss
>
>     > 《FaceNet: A Unified Embedding for Face Recognition and Clustering》

### 2. Methods

#### Siam-FC

