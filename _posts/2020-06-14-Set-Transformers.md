---
layout:     post
title:      Set Transformer
subtitle:   A Framework for Attention-based Permutation-Invariant Neural Networks
date:       2020-06-14
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Transformer
---

#### 1. Introduction

- NLP的输入存在有序性，但是还有很多机器学习任务的输入与顺序无关（set-strcture data），如：

  - multiple instance learning
  - 3D shape recognition
  - sequence ordering
  - meta-learning

- 以上任务的模型有两类特性：

  - permutation invariant，即输入结果与输出顺序无关
  - 能够处理任何大小的输入

  

#### 2. Methods

##### (1) Multihead Attention Block (MAB)

$$
\begin{array}{c}
\operatorname{MAB}(X, Y)=\text { LayerNorm }(H+\operatorname{rFF}(H)) \\
H=\text { LayerNorm }(X+\text { Multihead }(X, Y, Y ; \omega))
\end{array}
$$

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/set_transformer3.png" alt="img" style="zoom:67%;" />



##### (2) Set Attention Block (SAB)

$$
\operatorname{SAB}(X):=\operatorname{MAB}(X, X)
$$

> 计算复杂度为$O(n^2)$，复杂度较高

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/set_transformer2.png" alt="img" style="zoom:67%;" />

##### (3) Induced Set Attention Block (ISAB)

引入inducing points矩阵$I\in R^{n\times d}$，将原来的attention拆分为两步，复杂度从$O(n^2)$优化为$O(mn)$：

- 首先用$I$对输入$X$做self-attention
- 用得到的结果对输入做attention

$$
\mathrm{ISAB}_{m}(X)=\operatorname{MAB}(X, H) \in \mathbb{R}^{n \times d} \\
\text { where } H=\operatorname{MAB}(I, X) \in \mathbb{R}^{m \times d}
$$



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/set_transformer1.png" alt="img" style="zoom:67%;" />

##### Pipeline

- Encoder由SAB或ISAB堆栈构成：$X\in R^{n\times dx}$为输入，$Z\in R^{n\times d}$为经过Encoder之后的特征表示

$$
Encoder(X)=SAB(SAB(X))
$$

$$
Encoder(X)=ISAB_m(ISAB_m(X))
$$



- Decoder:

  > $$
  > \operatorname{Decoder}(Z ; \lambda)=\operatorname{rFF}\left(\operatorname{SAB}\left(\operatorname{PMA}_{k}(Z)\right)\right)
  > $$
  >
  > - PMA为Pooling by Multihead Attention，用来聚合多个特征:
  >
  >   $$\operatorname{PMA}_{k}(Z)=\operatorname{MAB}(S, \operatorname{rFF}(Z)), S\in R^{k\times d}$$
  >
  > - SAB用于建模k个输出之间的关系



#### 3. Experiments

##### (1) Maximum Value Regression

> 给定一组实数${x_1,x_2,...,x_N}$,目标为返回$max(x_1,x_2,...,x_N)$



##### (2) Counting Unique Characters

> 不同字符计数



##### (3) Amortized Clustering with Mixture of Gaussians

$$
\log p(X;\theta)=\sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_{j} \mathcal{N} \left(x_{i} ; \mu_{j}, diag(\sigma_{j}^{2})\right)
$$

> 给定一个数据集$X = \{x_{1:n}\}$的混合高斯的最大似然估计                         
>
> - 不使用EM算法，直接用神经网络学习到最大值参数的映射



##### (4) Set Anomaly Detection

> 文章使用`CelebA`进行异常检测。该数据集有202599张图片，共40个属性。随机选定两个属性，对这两个属性随机选定七张图片，它们都含有该属性，再选定另一张图片，不含这两个属性，将这八张图片作为一个集合去找出与众不同的图片 （有点meta-learning的感觉）



##### (5) Point Cloud Classiﬁcation

> 文章使用`ModelNet40`作为数据集，该数据集包含许多三维物体数据，共40类，每个物体用一个点云表示，即一个有$n$个三维向量的集合，作为分类的属性



#### 4. Conclusion

- 贡献点：

  > - 使用self-attention处理数据集中的每个元素，构成类似Transformer的结构，用于建模set类型的数据
  > - 将计算时间从$O(n^2)$变为$O(mn)$，$m$为预定义的参数
  > - 在**最大值回归**、**计数不同字符**、**混合高斯**、**集合异常检测**和**点云分类**五个任务上有较好表现