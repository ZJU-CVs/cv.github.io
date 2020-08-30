---
layout:     post
title:      Learning Attentive Pairwise Interation for Fine-Grained Classification
subtitle:   
date:       2020-08-21
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-grained

---



#### 1. Introduction

- 细粒度分类问题的大多数方法都是通过学习单个输入图像的区别性表征 (discriminative representation)来解决
- 而人类可以通过比较图像对来有效识别对比信息，如下图中的两个不同品种的海燕图，如果仅查看单张图片，很难识别其属于哪一类，尤其是当对象被嘈杂的背景遮挡时。而通过图片对的相互对比，则可以很好的区分两种品类
- 因此，本文提出了一种简单有效的注意力对交互网络 (Attentive Pairwise Interaction Network)



#### 2. Method

> Attentive Pairwise Interaction Network (API-Net)由`mutual vector learning`, `gate vector generation` 和`pairwise interaction`三个子模块组成



整体流程如下：

> - 在训练阶段，输入一对图像到backbone中，分别提取特征。利用得到的特征向量$x_1,x_2\in \mathbb{R}^D$得到一个mutual vector $x_m\in \mathbb{R}^D$
> - 将$x_m$与$x_i$按通道进行点乘，即利用$x_m$查找哪个通道可能包含对比信息，然后通过sigmoid函数得到gate vector $g_i\in \mathbb{R}^D$
> - 在gate vector的指导下进行成对的交互，交互后的向量输入softmax classifier得到损失函数

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/8.png" alt="img" style="zoom:67%;" />



##### Mutual Vector Learning

`互向量学习`

> 输入是一对细粒度图像(具有较高的相似性)，通过CNN分别生成两个特征向量$x_1$和$x_2$，之后通过concat层将两个特征向量进行拼接，再通过MLP生成一个互向量$x_m$ (Mutual Vector)
>
> 
> $$
> x_m=f_m(\mid x_1,x_2 \mid)
> $$
>



##### Gate Vector Learning

`门向量生成`

> $x_m$分别对$x_1$和$x_2$作通道积(channel-wise)，利用$x_m$作为指导来发现单个$x_i$的哪些通道可能包含**对比信息**，然后经过sigmoid函数生成对应门向量$g_1$和$g_2$
>
> 
> $$
> g_i=\operatorname{sigmoid}\left(x_{m} \odot x_{i}\right) ,i \in\{1,2\}
> $$
>
> > $g_i$为discriminative attention，突出每个$x_i$不同的语义差异。如$g_1$的关键区域为身体，$g_2$的关键区域为嘴巴



##### Pairwise Interaction

`成对交互`

> 为了捕获一对细粒度图像中的细微差异，不仅要检查每个图像的突出部分，还要检查彼此不同的部分。因此通过残差注意力引入一种交互机制
>
> 
> $$
> x_1^{self}=x_1+x_1 \odot g_1 \\
> x_2^{self}=x_2+x_2 \odot g_2 \\
> x_1^{other}=x_1+x_1 \odot g_2 \\
> x_2^{other}=x_2+x_2 \odot g_1
> $$
>
> > $x_i^{self}$通过自身的门向量突出自身的特征；$x_i^{other}$通过另一图像的门向量激活其他部分

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/25.png" alt="img" style="zoom:50%;" />

##### Training

> API模块最后输入4个attentive feature $x_i^j, \ where \ i \in \{1,2\},j\in\{ self,other\}$，然后将其输入到softmax classifer中
>
> 
> $$
> p^j_i = softmax(Wx^j_i+b)
> $$
>
> > $p^j_i\in \mathbb{R}^C$为预测的得分向量，$C$为预测类别数



> **Loss function**:
>
> 
> $$
> \mathcal{L}=\mathcal{L}_{ce}+\lambda \mathcal{L}_{rk}
> $$
>
> > (1) Cross Entropy Loss
> >
> > 
> > $$
> > \mathcal{L}_{c e}=-\sum_{i \in\{1,2\}} \sum_{j \in\{\text {self,other}\}} \mathbf{y}_{i}^{\top} \log \left(\mathbf{p}_{i}^{j}\right)
> > $$
> > 在标签$y_i$的监督下，识别出所有的attentie features $x_i^j$
>
> 
>
> > (2) Score Ranking Regularization
> >
> > 
> > $$
> > \mathcal{L}_{r k}=\sum_{i \in\{1,2\}} \max \left(0, \mathbf{p}_{i}^{\text {other}}\left(c_{i}\right)-\mathbf{p}_{i}^{\text {self}}\left(c_{i}\right)+\epsilon\right)
> > $$
> > $p^j_i(c_i)$是预测的向量$p^j_i$中获得的分数，$c_i$为对应的真实标签
> >
> > 与$x_i^{other}$相比，$x_i^{self}$对于识别相应的图像应更有判别能力，因此利用分数差异$p^{self}_i(c_i)-p^{other}_i(c_i)$应该大于margin $\epsilon$ 
> >
> > 通过$\mathcal{L}_{rk}$让$x^{self}_i$仅通过自己的gate vector进行激活



> **Pair Construction**
>
> - 在一个batch中随机采样$N_{cl}$个类，对于每个类别随机采样$N_{im}$个图像
> - 将这些图像输入到backbone中提取特征向量，根据欧式距离将其特征与batch中的其他特征进行比较
> - 根据欧式距离为每个图像构造两个对intra pair(其特征和该批次中类内最相似特征组成) 和 inter pair (其特征和该批次中类间最相似特征组成)
> - 因此每批次有$2\times N_{cl}\times N_{im}$输入到API模块中，并累加所有对的损失进行端对端训练
>
> `该设计能够使API-Net学习区分哪些是高度混淆或真正相似的图像对`



#### 3. Experiments

##### Comparison with SOTA

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/27.png" alt="img" style="zoom:50%;" />



##### Visualization

> 如红色虚线框所示，Baseline的许多特征图是混乱的或有噪声的，例如目标区域模糊，或某些背景区域被激活。
>
> 相反，API-Net可以有效地发现并区分有区别的线索

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/26.png" alt="img" style="zoom:50%;" />