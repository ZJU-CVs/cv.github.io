---
layout:     post
title:      Attention Augmented Convolutional Networks
subtitle:   
date:       2020-06-25
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Paper
    - Attention Mechanism
---



可与[Stand-Alone Self-Attention in Vision Models ](Stand-Alone Self-Attention in Vision Models.md)对比阅读

#### 1、Introduction

- 卷积操作具有显著的**弱点**，因为它仅在本地邻域上操作，缺少全局信息（only operates on a local neighborhood，thus missing global information）
- Self-attention能够capture long range interactions
- 本文考虑将Self-attention用于discriminative visual tasks，代替卷积的作用。



#### 2、Methods

（1）引入**two-dimensional relative self-attention mechanism**，并通过实验证明了可行性

**（2）模型细节**

> $H$: height 
>
> $W$: weight
>
> $F_{in}$: number of input filters of an activation map
>
> $N_h$: number of heads, $N_h$ divides $d_v$ and $d_k$
>
> $d_v$: depth of the values 
>
> $d_k$: depth of the queries/keys



- single-head attention
  

  - 以$3\times 3$为例，经过6个filter得到$3\times 3 \times 6$的input

  - 以单个head attention为例（蓝色部分），attention map中包括query map、key map和value map

    <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA4.png" alt="img" style="zoom:50%;" />

  - 如input中的位置6为self-attention target,对应的attention map中$q_6,k_6,v_6$

    <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA3.png" alt="img" style="zoom:50%;" />

  - 通过$QK^T$可以得到$9\times9$的矩阵。

	<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA5.png" alt="img" style="zoom:40%;" />

  
  
  - 得到的a single head attention可以用以下公式表示：
  
    
  
  $$
  O_{h}=\operatorname{Softmax}\left(\frac{\left(X W_{q}\right)\left(X W_{k}\right)^{T}}{\sqrt{d_{k}^{h}}}\right)\left(X W_{v}\right)
  $$
  
  
  
  > where $W_q,W_k \in \mathbb{R}^{F_{i n} \times d_{k}^{h}}，W_v \in \mathbb{R}^{F_{i n} \times d_{v}^{h}}$
  
  

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA1.png" alt="img" style="zoom:50%;" />

---



- multi-head 融合公式：

  

$$
\mathrm{MHA}(X)=\text { Concat }\left[O_{1}, \ldots, O_{N h}\right] W^{O}
$$


> where $W^O \in \mathbb{R}^{d_{v} \times d_{v}}$,最终得到$(H,W,d_v)$的tensor

---



- 增加了位置信息
  
  
  $$
  l_{i, j}=\frac{q_{i}^{T}}{\sqrt{d_{k}^{h}}}\left(k_{j}+r_{j_{x}-i_{x}}^{W}+r_{j_{y}-i_{y}}^{H}\right)
  $$
  
  $$
  O_{h}=\operatorname{Softmax}\left(\frac{Q K^{T}+S_{H}^{r e l}+S_{W}^{r e l}}{\sqrt{d_{k}^{h}}}\right) V
  $$
  
  where $S_H^{rel},S_W^{rel}\in \mathbb{R}^{H W \times H W}$,其中$S_H^{rel}[i,j]=q_i^Tr^H_{j_y-i_y}$and$S_W^{rel}[i,j]=q_i^Tr^W_{j_x-i_x}$ 
  
  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SASA2.png" alt="img" style="zoom: 33%;" />
---

- 但是直接使用self-attention会导致破坏了平移性，因此可以将self-attention与CNN结合使用

  
$$
\operatorname{AAConv}(X)=\text { Concat }[\operatorname{Conv}(X), \operatorname{MHA}(X)]
$$



