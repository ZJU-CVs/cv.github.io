---
layout:     post
title:      Drop an Octave
subtitle:   Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
date:       2020-08-18
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Backbone
---



#### 1. Introduction

> - 在自然图像中，信息以不同的频率传递，其中较高的频率通常用精细的细节编码，较低的频率通常用全局结构编码。(低频部分对应灰度图中变化平缓的部分，高频部分对应灰度图中变化剧烈的部分。具体表现出来，低频部分对应的图片整体结构，而高频部分对应的边缘细节)
> - 同样，卷积层的输出特征图也可以看作是不同频率下信息的混合。
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Octave Convolution2.png" alt="img" style="zoom:50%;" />
>
> - 基于卷积层的输出特征映射也可以分解为不同空间频率的特征，提出了一种新的多频特征表示方法，将高频和低频特征映射存储到不同的组中。
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Octave Convolution1.png" alt="img" style="zoom:50%;" />
>
>   > - **卷积层的输出图可以根据其空间频率分解和分组**
>   > - **其中可以将平滑变化的低频映射存储在低分辨率张量中(H,W减半)，以减少空间冗余。**
>   > - Octave Convolution**更新每个组内的信息，并进一步支持组之间的信息交换**



#### 2. Details

> - 对于普通卷积，所有的输入和输出特征图具有相同的空间分辨率
>
> - 设计了一种新的Octave Convolution(OctConv)操作来存储和处理空间分辨率较低且空间变化较慢的特征图，从而降低了内存和计算成本。

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Octave Convolution.png" alt="img" style="zoom:50%;" />

> 设X, Y为因式分解的输入和输出张量。那么高和低频特征图的输出$Y=\{Y^H,Y^L\}$由$Y^{H}=Y^{H \rightarrow H}+Y^{L \rightarrow H}$和$Y^{L}=Y^{L \rightarrow L}+Y^{H \rightarrow L}$。具体来说，$Y^{H \rightarrow H}, Y^{L \rightarrow L}$表示intra-frequency信息更新，而$Y^{H \rightarrow L}, Y^{L \rightarrow H}$表示inter-frequency沟通。

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Octave Convolution kernel.png" alt="img" style="zoom:50%;" />

> 将卷积核W分成两个分量$W=\left[W^{H}, W^{L}\right]$，分别负责与$X^{H}$和$X^{L}$进行卷积。将各分量进一步划分为频率内分量和频率间分量：$W^{H}=\left[W^{H \rightarrow H}, W^{L \rightarrow H}\right]$]和$W^{L}=\left[W^{L \rightarrow L}, W^{H \rightarrow L}\right]$



