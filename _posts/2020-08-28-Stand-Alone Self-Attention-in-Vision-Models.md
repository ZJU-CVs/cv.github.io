---
layout:     post
title:      Stand-Alone Self-Attention in Vision Models
subtitle:   
date:       2020-08-28
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Attention Mechanism
---



> 可与[Attention Augmented Convolutional Networks](https://zju-cvs.github.io/2020/06/25/Attention-Augmented-Convolutional-Networks/)对比阅读



#### 1、Introduction

- 卷积层缺点：
  - 捕获长距离交互能力比较差，因为卷积的感受野大时缩放特性弱
  - 针对上述问题，往往将卷积层引入注意力机制，主要有基于通道的注意力机制（squeeze  and excite）和基于空间的注意力机制（spatial space），然而这些方法有一个特点就是用**全局注意力层作为卷积的附加模块**；还有一个限制是因为关注输入的所有位置，**要求输入比较小，否则计算成本大**。
- 本文提出了**stand-alone self-attention** 自注意力层用来替换卷积
  - 之前的注意力机制是和卷积层结合起来作为卷积层的拓展来用的
  - 独立自注意力层不依赖卷积层单独作为一层

#### 2、Method

- 提出了独立自注意力层和空间感知独立自注意力层，用空间感知独立自注意力层替换初始层卷积，用独立自注意力层替换其余卷积层

- **独立自注意力层**

  - 定义了query, key,value三个概念。**自注意层的运算是局域的**，因此不用限制输入的大小。

  - 自注意力层的参数个数与感受野的大小无关，卷积的参数个数与感受野的大小成平方关系。

  - 自注意力层的运算量增长速度也比卷积缓慢

  - 自注意力层的公式如下：
    
    > $$
    > y_{i j}=\sum_{a, b \in \mathcal{N}_{k}(i, j)} \operatorname{softmax}_{a b}\left(q_{i j}^{\top} k_{a b}\right) v_{a b}
    > $$
>
    > 
>
    > 其中：$q_{ij}=W_Qx_{ij}$, 
>
    > ​			$k_{ij}=W_Kx_{ij}$
>
    > ​			$v_{ij}=W_Vx_{ij}$, 
    >
    > ​			$W_Q$,$W_K$和$W_V\in \mathbb{R}^{d_{o u t} \times d_{i n}}$  为学习到的参数


​    
  - 但是上述公式没有包含位置信息，因此对于一个query，其邻域的位置关系无法体现出来。因此通过嵌入向量表示相对位置，把位置信息添加到自注意力运算，**有位置的自注意力**公式如下：
    
    > $$
    > y_{i j}=\sum_{a, b \in \mathcal{N}_{k}(i, j)} \operatorname{softmax}_{a b}\left(q_{i j}^{\top} k_{a b}+q_{i j}^{\top} r_{a-i, b-j}\right) v_{a b}
    > $$
    >
    > 
    >
    > 其中$r_{a-i，b-j}$为位置嵌入,如下图矩阵
    
    <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA6.png" alt="img" style="zoom:50%;" />

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA.png" alt="img" style="zoom:50%;" />



- **空间独立自注意力层**

  - 将卷积层的初始层和其余层在卷积网络中起着不同的作用

    > 初始层有两个方面的任务：
    >
    > （1）学习局部特征，如边缘等，用于后续的全局目标识别或检测
    >
    > （2）由于输入图像通常较大，初始层还有下采样的任务

  - 提出了空间感知(spatially aware)自注意力层来<u>解决标准的自注意力层无法包含空间信息的问题</u>

  - 空间感知自注意力层的定义如下：

  $$
  \begin{aligned} q_{i j} &=W_{Q} x_{i j} \\ k_{i j} &=W_{K} x_{i j} \\ v_{i j} &=W_{V} x_{i j} \\ y_{i j} &=\sum_{a, b \in \mathcal{N}_{k}(i, j)} \operatorname{softmax}_{a b}\left(q_{i j}^{\top} k_{a b}\right) v_{a b} \end{aligned}
  $$

  

  > 其中$v_{ij}$的定义与标准的自注意力层不同，定义为：
  >
  > 
  > $$
  > v_{a b}=\left(\sum_{m} p(a, b, m) W_{V}^{m}\right) x_{a b}\\
  > p(a, b, m)=\operatorname{softmax}_{m}\left(\left(\operatorname{emb}_{r o w}(a)+\mathrm{emb}_{c o l}(b)\right)^{\top} \nu^{m}\right)
  > $$
  > 
  >
  > - 表示在一个窗口中每个位置的$v_{ij}$都通过$x_{ab}$与不同的$W_V$相乘得到。其中$v_{ij}$是多值的m维矩阵，$p(a,b,m)$是向量的m维元素，为标量，a和b是相对于窗口的行和列位置。

  
