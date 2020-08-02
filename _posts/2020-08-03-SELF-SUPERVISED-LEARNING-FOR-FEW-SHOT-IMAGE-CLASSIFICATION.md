---
layout:     post
title:      Self Supervised Learning for Few Shot Image Classification
subtitle:   
date:       2020-08-03
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - FSL
    - SSL
---



### 1. Introduction

> - 少样本图像分类的目的是用有限的标记样本对unseen的类进行分类，常用meta-learning的方法，能够快速适应从训练到测试的分类
> - 元学习中的初始嵌入网络是元学习的一个重要组合部分，由于每个任务的样本数量有限，在实际应用中对其性能有很大影响。因此提出了许多预先训练的方法，但**大多数是以监督的方式进行训练**，对unseen classes的迁移能力有限

### 2. Method

- 本文是用自监督学习(SSL)来训练一个更通用的嵌入网络，可以通过从数据本身学习来为下游任务提供“slow and robust”表征。

- 模型包括self-supervised learning和meta-learning两个阶段

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/33.png)

#### Self-supervised learning stage

> - 用**Augmented Multiscale Deep InfoMax(AMDIM)**作为自监督模型，其核心思想是从同一幅图像的两个视图$(x_a,x_b)$中最大化全局特征和局部特征之间互信息，可以通过最小化基于负样本的Noise Contrastive Estimation(NCE) loss来最大化互信息的下界
>
>   - 具体而言，最大化$<f_g(x_a),f_5(x_b)>$，$<f_g(x_a),f_7(x_b)>$和$<f_5(x_a),f_5(x_b)>$，其中$f_g$为全局特征，$f_5$为编码器的$5\times 5$局部特征映射，$f_7$为编码器的$7\times 7$特征映射，以$f_g(x_a)$和$f_5(x_b)$间的NCE loss为例：
>     $$
>     \begin{array}{l}
>     \mathcal{L}_{\text {amdim}}\left(f_{g}\left(x_{a}\right), f_{5}\left(x_{b}\right)\right)= -\log \frac{\exp \left\{\phi\left(f_{g}\left(x_{a}\right), f_{5}\left(x_{b}\right)\right)\right\}}{\sum_{\widetilde{x}_{b} \in \mathcal{N}_{x} \cup x_{b}} \exp \left\{\phi\left(f_{g}\left(x_{a}\right), f_{5}(\tilde{x}_b)\right)\right\}}
>     \end{array}
>     $$
>
>     > $\mathcal{N}_x$为image $x$的负样本，$\phi$为距离度量函数



#### Meta-learning stage

> 在基于上述自监督学习得到嵌入网络的情况下，将元学习应用于网络fine-tune，以满足少样本分类的类变化
>
> - 典型的元学习可以看作是一个具有多个任务的K-way C-shot episodic 分类问题，对于每个分类任务$T$，有$K$个类，每个类有$C$个样本。
> - 

### 3. Experiments

- 数据集：采用MiniImageNet(64类为训练，16类为验证，20类为测试)和CUB-200-2011(100类为训练，50类为验证，50类为测试)

- 定量比较：所提的方法通过一个large network 的自监督预训练，能够显著改进少样本分类任务

  **MiniImageNet**

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/34.png" alt="img" style="zoom:50%;" />

  > - **Mini80_SL:**
  > - **Mini80_SSL$^-$:**
  > - **Mini80_SSL:**
  > - **Image900_SSL:**

  

  **CUB-200-2011**

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/35.png" alt="img" style="zoom:50%;" />

  > - **CUB150_SL:**
  > - **CUB150_SSL$^-$:**
  > - **CUB150_SSL:**
  > - **Image1K_SSL:**

  

### 4. Conclusion

> 提出利用自监督学习来有效地训练一个鲁棒的嵌入网络来进行少镜头图像分类。与其他基线相比，所得到的嵌入网络更具通用性和可移植性。通过元学习过程进行微调后，所提出的方法的性能可以显著优于基于两个常见的少量快照分类数据集的定量结果的所有基线。目前的框架可以在未来以多种方式扩展。例如，一个方向是将这两个阶段结合起来，并为此任务开发一个端到端的方法。另一个方向是研究所提出的方法在少镜头检测等其他少数任务上的有效性