---
layout:     post
title:      PAY LESS ATTENTION WITH LIGHTWEIGHT AND DYNAMIC CONVOLUTIONS
subtitle:   
date:       2020-06-18
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper

---

#### 1. Introduction

##### *lightweight convolutons*

> - 轻量级卷积利用深度可分离卷积为原型，通过在通道维度上的共享参数大大减少参数量，降低算法的复杂度（如下图，以$3\times 3$卷积核为例），其思想在于将通道和区域分开考虑
>
>   > - 传统卷积参数：$k^2 \times d_{in} \times d_{out}$ (其中$k$为卷积核大小，$d_{in}$为输入通道数量，$d_{out}$为输出通道数量)
>   > - 深度可分离卷积（depthwise convolutions）中一个卷积核负责一个通道，一个通道只被一个卷积核卷积；传统卷积中每个卷积核同时操作输入图片的每个通道。深度可分离卷积参数：$k^2 \times d_{in}$
>   > - 深度可分离卷积后往往使用Pointwise Convolution进行深度方向的加权组合，pointwise convolution参数：$1^2 \times d_{in} \times d_{out}$
>
>   ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/17.jpeg)
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/18.jpeg" alt="img" style="zoom:50%;" />



##### *dynamic convolutions 动态卷积*

> - 由于深度可分离卷积
>
> - 从宏观上讲，动态卷积也可以看作是self-attention的一种，以与SEnet对比为例：
>
>   - SEnet：通过输入内容自动学习到不同通道特征的重要程度，生成通道维度的权重，对每个通道的特征图进行加权
>   - 动态卷积：处理对象为卷积核，通过输入内容自动学习对应卷积核的权重
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/19.png" alt="img" style="zoom:40%;" />
>
>   ref: [Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/abs/1912.03458v2)

##### *self-attention*

> - self-attention是一种有效的机制，应用于NLP、CV等各种任务中并都有很好的性能提升
>
> - 但面对长序列，self-attention受限于其$O(n^{2})$算法复杂度，且self-attention可以高效捕捉长期依赖的特性最近也被学者质疑（存在大量冗余）

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/16.png)

#### 2. Model

`（以NLP数据为例）`

> $d_{in}=d,d_{out}=c$, *kernel width=k*
>
> - self-attention:
>
>   > $$
>   > \text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
>   > $$
>
> - transitional convolutions: $k\times d^2$
>
> - depth-wise convolutions: $k\times d$
>
>   >   $$
>   >   O_{i, c}=\operatorname{Depthwise} \operatorname{Conv}\left(X, W_{c,:}, i, c\right)=\sum_{j=1}^{k} W_{c, j} \cdot X_{\left(i+j-\left\lceil\frac{k+1}{2}\right\rceil\right), c}
>   >   $$
>   >   *c* is the output dimension, $W\in R^{d\times k}$, $O \in R^{n\times d}$
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/20.jpg" alt="img" style="zoom:50%;" />
>
> - light convolutions:
> 		> 将通道分为H组，对每组的子通道上实现参数共享，参数量为$k \times H$，并对参数进行归一化处理
>   > $$
>   > \left.\operatorname{LightConv}\left(X, W_{[\frac{c H}{d}]}, ; i, c\right)=\operatorname{Depthwise} \operatorname{Conv}(X, \operatorname{softmax}\left(W_{[\frac{c H}{d}]},:\right), i, c\right)
>   > $$
>   >
>   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/21.jpg" alt="img" style="zoom:50%;" />
>   
> - dynamic convolutions:
>
>   > 不同组的权重根据输入的$X_i$得到，进行加权组合得到Dynamic Convolution
>   > $$
>   > \text { DynamicConv }(X, i, c)=\operatorname{Light} \operatorname{Conv}\left(X, f\left(X_{i}\right)_{h,:}, i, c\right)
>   > $$
>   > $$
>   > f(X_i)=\sum_{c=1}^{d} W_{h, j, c}^{Q} X_{i, c}
>   > $$
>   >
>   > 
>   >
>   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/22.jpg" alt="img" style="zoom:30%;" />
>
> 

#### 

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/15.png)



#### 3. Conclusion



#### 4. Conclusion

- 所提出的动态卷积在深度可分离卷积的基础上构建，与常规的动态卷积相比，减少了参数量
- 将所提的Dynamic convolution 代替transformer中的self-attention，效果更好（实验表明）