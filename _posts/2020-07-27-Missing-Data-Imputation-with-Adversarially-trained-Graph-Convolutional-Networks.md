---
layout:     post
title:      Missing Data Imputation with Adversarially-trained Graph Convolutional Networks
subtitle:   
date:       2020-07-27
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Note

---

```markdown
看到一篇基于图神经网络的数据缺失填充文章，做一下记录
```



#### 1. Introduction

> - 缺失数据插补（Missing data imputation, MDI)是用预测值代替缺失值的任务
> - 本文利用图神经网络（GNNs）提出了一种通用的MDI框架



#### 2. Method

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/11.png" alt="img" style="zoom:50%;" />

> - 首先使用欧式距离成对计算所有特征向量的相似矩阵，但每次仅使用两个向量的非缺失元素计算
>   $$
>   S_{i j}=d\left(\mathbf{x}_{i} \odot\left(\mathbf{M}_{i} \odot \mathbf{M}_{j}\right), \mathbf{x}_{j} \odot\left(\mathbf{M}_{i} \odot \mathbf{M}_{j}\right)\right)
>   $$
>
> - 将相似矩阵经过剪枝步骤稀疏化，得到图结构（邻接矩阵）。在得到的图中，数据集中的每个特征向量被编码为图的一个节点，邻接矩阵a由特征向量的相似矩阵$S$导出
>
> - 构建图自编码器，由一个将输入映射到不同维空间$h=encode(x)$的编码器和一个将$h\in \mathbb{R}^m$映射到原始d维空间的$\hat{x}=decode(h)$解码器组成。其中$m>d$，从而将输入映射到更高维的空间，以帮助数据恢复。因此GINN定义为：
>   $$
>   \begin{array}{l}
>   \mathbf{H}=\operatorname{ReLU}\left(\mathbf{L X \Theta}_{1}\right) \\
>   \hat{\mathbf{X}}=\operatorname{Sigmoid}\left(\mathbf{L H} \Theta_{2}\right)
>   \end{array}
>   $$
>
>   > $\Theta_1$和$\Theta_2$为适应性系数矩阵，起到前馈的作用
>   >
>   > $L$允许每个节点的近邻之间传播信息
>
> - 训练阶段：
>
>   - 因为缺失值在训练阶段是未知的，不能简单地训练自动编码器丢失的值，因此采用降噪自编码的训练方式
>
>     <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/13.png" alt="img" style="zoom:50%;" />
>
>     > - 在训练过程中，首先随机将输入中一些值设置为0，得到$\tilde{x}$，然后将$\tilde{x}$作为输入，计算出重构z，将重构值z与原始值x进行比较，计算误差。这种经过假造并进行降噪的训练，能够使网络学习到更加鲁棒性的不变特征
>     >
>     > - 因此在每个优化步骤中，直接对网络输入应用inverted dropout layer，在输入上添加额外的masking noise
>
>   - 损失函数定义为数值变量的均方误差(MSE)和分类变量的交叉熵(CE)的组合
>
>   $$
>   L_A=\alpha MSE(X,\hat{X})+(1-\alpha)CE(X,\hat{X})
>   $$
>
>   - 为了加快训练速度，使用了一种额外的对抗性训练策略。设计一个前向网络作为判别器，学习如何区分输入数据和真实数据（使重建向量接近原始模式的自然分布），在本文汇总，GCN自编码器每5次优化步骤都要接受
>     $$
>     L_D=L_A-\underset{\mathbf{x} \sim \mathbb{P}_{imp}}{\mathbb{E}}[C(\hat{x})]
>     $$
>
> - Tricks：
>
>   - 引入一个额外的skip layer，编码器层变为：
>     $$
>     \hat{X} = Sigmoid(LH\Theta_2+\tilde{\mathbf{L}} \mathbf{X} \Theta_{3})
>     $$
>     
>
>     <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/12.png" alt="img" style="zoom:50%;" />
>
>   - 考虑全局信息g
>     $$
>     \hat{X} = Sigmoid(LH\Theta_2+\tilde{\mathbf{L}} \mathbf{X} \Theta_{3}+\Theta_4g)
>     $$
>
>     $$
>     L=L_d+\gamma MSE(global(\hat{X}),global(X))
>     $$
>
>     



