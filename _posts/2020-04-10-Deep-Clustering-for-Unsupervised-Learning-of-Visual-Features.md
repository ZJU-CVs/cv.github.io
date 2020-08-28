---
layout:     post
title:      Deep Clustering for Unsupervised Learning of Visual Features
subtitle:   
date:       2020-04-10
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Cluster
---



[code](https://github.com/facebookresearch/deepcluster)

#### 1. Introduction

- 提出了一种聚类方法`DeepCluster`，将端对端学习和聚类结合起来，同时学习神经网络的参数和网络输出的特征进行聚类
- `DeepCluster`使用标准聚类算法`Kmeans`迭代地对特征进行分组，并基于聚类结果作为监督来更新网络权重



#### 2. Method

##### Preliminaries

> 通过Convnet将原始图像映射到固定维度的向量空间
>
> - 用$f_\theta$表示convnet映射，其中$\theta$是相应的参数的集合，通过将该映射应用于图像作为特征或表示(feature or representation)而获得向量
> - 在给定$N$个图像的训练集$X={x_1,x_2,...x_N}$，找到参数$\theta^*$，使映射$f_{\theta^*}$得到良好的通用(general-purpose)特征
> - 传统做法:
>   - 这些参数通过监督学习得到，即每个图像$x_n$与{$0,1^k$}中的标签$y_n$相关联。此标签表示图像为$K$个预定义类之一
>   
>   - 参数分类器$g_W$可预测$f_\theta(x_n)$特征之上的正确标签。分类器参数$W$和映射参数$\theta$通过优化**网络损失**进行共同学习
>   
>     
> $$
> \min _{\theta, W} \frac{1}{N} \sum_{n=1}^{N} \ell\left(g_{W}\left(f_{\theta}\left(x_{n}\right)\right), y_{n}\right)....(1)
> $$
>
> > 其中$\ell$是多项逻辑损失，即`negative log-softmax`函数，使用mini-batch随机梯度下降和反向传播来计算梯度，最大限度的降低cost function



##### Unsupervised learning by clustering

> - 当从高斯分布中采样$\theta$时，若没有任何学习，$f_{\theta}$不会产生好的特征。然而这种随机特征在标准迁移(Transfer)任务中的表现远远高于随机(chance)水平
>
> - 随机网络的良好性能与它们的卷积结构密切相关，先对特征进行聚类。
>
>   > 将convnet产生的特征$f_{\theta}(x_n)$作为输入，使用标准聚类算法K-means将它们聚类成k个不同的组
>   >
>   > （通过联合学习$d{\times}k$质心矩阵C和每个图像$n$的聚类分配$y_{n}$来优化**聚类损失**）
>
>   $$
>   \min _{C \in \mathbb{R}^{d \times k}} \frac{1}{N} \sum_{n=1}^{N} \min _{y_{n} \in\{0,1\}^{k}}\left\|f_{\theta}\left(x_{n}\right)-C y_{n}\right\|_{2}^{2} \quad ,\text { such that } \quad y_{n}^{\top} 1_{k}=1  {....(2)}
>   $$
>
> - 学习后得到最优分配$\left(y_{n}^{\star}\right)_{n \leq N}$和质心矩阵$C^{\star}$，
>
>   将此分配结果作为伪标签（先验），通过优化`Eq(1)`来更新convnet的参数
>
> **Overall：** 
>
> - `DeepCluster`交替进行以下两个步骤：
>   - 使用`Eq(2)`聚类特征来生成伪标签
>   - 使用`Eq(1)`来更新网络参数

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/image1.jpg)

##### Avoiding trivial solutions

`Problem:特征和聚类同时进行学习会产生无效解和无效参数`

> Solutions:
>
> - Empty clusters  (避免无效解)
>   - Reason：使用模型来预测伪标签，可能使得网络产生的特征经过聚类都位于某个簇心周围，而使得其他簇心没有样本。这个问题是由于没有限制某个簇心不能没有样本。
>
>   - Solve：
>
>     > 当某个簇心为空时，随机选择一个非空的簇心，在其上加一些小的扰动作为新的簇心，然后让属于非空簇心的样本重新分配给两个结果簇（即同时让属于非空簇心的样本也属于新的簇心）
>
> - Trivial parametrization  (避免无效参数)
>   - Reason: 当每个类别的图像数量高度不平衡时，大量的数据被聚类到少量的几类上，一种极端场景是被聚类到一类上，这种情况下网络可能对于任意的输入都产生相同的输出。
>
>   - Solve：
>
>     > 根据类别（或伪标签）对样本进行均匀采样，相当于将输入对`Eq(2)`损失函数的贡献率乘以其指定簇大小的倒数
>



3、实现细节

> **结构**：VGG16+BN
>
> **训练数据**：ImageNet；数据使用了一个基于Sobel算子进行处理去除了颜色的信息
>
> **优化**：聚类的时候使用center crop的样本特征，训练模型时使用数据增强（左右翻转、随机大小和长宽比的裁剪）；同时聚类时使用了PCA降维

