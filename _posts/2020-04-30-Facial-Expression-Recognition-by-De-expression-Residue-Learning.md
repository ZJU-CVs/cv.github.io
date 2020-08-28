---
layout:     post
title:      Facial Expression Recognition by De-expression Residue Learning
subtitle:   
date:       2020-04-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Facial
---



#### 1. Introduction

> 核心思想：一个人的人脸表情是由表情和中性脸组成
> $$
> Expression = Neutral+ Expressive
> $$
> 



方法：通过*De-expression Residue Learning*，提取面部表情组成部分的信息。



#### 2. DeRL Model

- 首先通过cGAN训练一个生成模型，来学习用expression的图像生成中立图像

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/page1image6056512.png" alt="page1image6056512.png" style="zoom:50%;" />

- 生成器中各个中间层保留了表情中的expressive component特征，因此可以用来训练分类器classifier，从而对人脸表情进行分类 

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/)

#### 3. Details

- CGAN：

  输入图像对<$I_ {input}$ , $I_ {target}$>，训练cGAN。输入目标是显示任何表情的脸部图像，目标是同一主题的中性脸部图像。训练后的生成器为任何输入重建相应的中性面部图像，同时保持身份信息不变。从表情面部图像到中性面部图像，表达信息被记录在中间层中的表达成分

  -  鉴别器D目标表示：

  $$
  \begin{aligned}
  L_{c G A N}(D)=\frac{1}{N} & \sum_{i=1}^{N}\left\{\log D\left(I_{\text {input}}, I_{\text {target}}\right)+\right.\log (1-D(\text {Iinput}, G(\text {Iinput})))\}
  \end{aligned}
  $$

  

  > 其中Ｎ是训练图相对的总数

  

  - 生成器G的结构设计采用Autoencoder的形式，从而保证了会存在相同尺寸的中间层，目标表示：

  $$
  \begin{array}{r}
  L_{c G A N}(G)=-\frac{1}{N} \sum_{i=1}^{N}\left\{\log \left(D\left(I_{\text {input}}, G\left(I_{\text {input}}\right)\right)\right)+\right. \left.\theta_{1} \cdot\left\|I_{\text {target}}-G\left(I_{\text {input}}\right)\right\|_{1}\right\}
  \end{array}
  $$

  

  > 其中使用L1损失来获得图像相似度而不是L2，因为L2损失倾向于过度模糊输出图像

  

  -  最终目标

  $$
  G^{\star} = arg \min_G \max_D L_{cGAN}(D)+ \theta_2 \cdot L_{cGAN}(G)
  $$

  

-  分类器

   -  用I代表查询图像，在输入生成模型G后，生成中性表达图像：

   $$
I^{id=A}_{exp=neutral}=G(I^{id=A}_{exp=E})
   $$

   > 其中，G是生成器，E属于六种基本原型面部表情中的任何一种

   
   
   - 为了从发生器的中间层学习去表达残差，这些层的所有滤波器都是固定的，并且具有相同大小的所有层被连接并输入到本地CNN中，对于每个本地CNN模型，代价函数被标记为损失i，$i\in[1,2,3,4]$
   
   - 每个本地CNN模型的最后全完连接的层被进一步连接并与用于面部表情分类的最后编码层组合。
   
   $$
    Total\ loss = λ_1loss_1 + λ_2loss_2 + λ_3loss_3 + λ_4loss_4 + λ_5loss_5
   $$
   
   

