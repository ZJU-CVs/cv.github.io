---
layout:     post
title:      Attention U-Net
subtitle:   Learning Where to Look for the Pancreas
date:       2020-04-25
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Segmentation
    - Attention Mechanism
---



#### 1. Introduction 

- 提出了一种注意力门模型(Attention Gate, AG)，用该模型进行训练时，能够抑制模型学习与任务无关的部分，同时加重学习与任务有关的特征
- Attention gate可以很容易地集成到标准CNN体系结构中



#### 2. Method

- Attention-Unet模型是以Unet模型为基础，其和unet的区别在于解码时，从编码部分提取的特征不是直接用于解码，而是先进行attention gate再进行解码。

  - Unet只是单纯的把同层的下采样层的特征直接concat到上采样层
  - Attention-Unet使用attention模块，对**下采样层同层和上采样层上一层的特征图**进行处理后再和上采样后的特征图进行concat

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/unet.png" alt="img" style="zoom:50%;" />

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/attention-unet.jpeg" alt="img" style="zoom:50%;" />

  - 绿色箭头是g信号
  - 从编码部分延伸的虚线是$\widehat{x}^l$
  - 粉红色箭头是输出

  

- **attention模块：Attention Gate**

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/attention gate.png)
  - F代表某一层的channel数

  - 使用下采样层同层的特征图和上采样层上一层的特征图进行一个pixel权重图的构建，然后再把这个权重图对下采样同层的特征图进行处理，得到进行权重加权的特征图。

    - 下采样同层的特征图$g_{i}\left(F_{g} \times H_{g} \times W_{g} \times D_{g}\right)$，

      进行1$\times$1$\times$1卷积运算得到$W_{g}^{T} g_{i}$

    

    - 上采样同层的特征图$x_{i}^l\left(F_{l} \times H_{x} \times W_{x} \times D_{x}\right)$，

      进行1$\times$1$\times$1卷积运算得到$W_{x}^{T} x_{i}^{l}$

    
    
    - 将上两步得到的特征图$W_{g}^{T} g_{i}$和$W_{x}^{T} x_{i}^{l}$进行相加后，在进行ReLU得到：
    
    $$
    \sigma_{1}\left(W_{x}^{T} x_{i}^{l}+W_{g}^{T} g_{i}+b_{g}\right)\left(F_{i n t }\times H_{g} W_{g} D_{g}\right)
    $$
    
    > $\sigma_1$为ReLU激活函数
    
    
    
    - 随后在使用1$\times$1$\times$1卷积运算得到
    
      
    
    $$
    q_{a t t}^{l}=\psi^{T}\left(\sigma_{1}\left(W_{x}^{T} x_{i}^{l}+W_{g}^{T} g_{i}+b_{g}\right)\right)+b_{\psi}
    $$
    
    
    
    - 最后对$q_{a t t}$进行sigmoid激活函数得到最终的attention coefficient ($\alpha_i^l$)
    
     

