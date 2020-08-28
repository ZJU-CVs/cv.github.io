---
layout:     post
title:      Attention-Gated Networks for Improving Ultrasound Scan Plane Detection
subtitle:   
date:       2020-04-20
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Segmentation
    - Attention Mechanism

---

[code](https://github.com/ozan-oktay/Attention-Gated-Networks)



#### 1. Introduction

- AG-Sononet在Sononet的基础上加入注意力机制

- [Sononet](https://arxiv.org/pdf/1612.05601.pdf)介绍：

  > Sononet是一个CNN架构，包含两个组件:特征提取模块和自适应模块。
  >
  > - 在特征提取模块，利用VGG网络的前17层(包括Map pooling)提取判别特征
  > - 在自适应模块，将通道的数量减少到目标类别$K$的数量

  > 随后通过通道全局平均池化(GAP)来flatten空间信息，最后将Softmax操作应用于所得到的向量，并选择具有最大激活的entry作为预测结果。由于网络基于reduced vector进行分类，因此网络为每个类提取最显著的特征

  > Sononet在池化层之前获得了良好定位的特征映射，因此也可以用于高精度的弱监督定位(weakly-supervised localisation)

  

- Sononet使用Global Average Pooling能够快速聚合空间文本信息，但它们没有**保存局部信息(local)**的能力，因此如果两个图像具有相似的全局特点，则不能很好的区分

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Sononet.png)



#### 2. Method

- 在Sononet框架基础上引入了注意力机制

- 参考《LEARN TO PAY ATTENTION》中的attention mechanism

  > - 所选层$s \in\{1, \ldots, S\}$的激活图(activation map)为：
  >
  > $$
  > \mathcal{F}^{s}=\left\{\mathbf{f}_{i}^{s}\right\}_{i=1}^{n}
  > $$
  >
  > > 其中每个$f^s_i$表示长度为$C_s$(通道数)的pixel-wise的特征向量。
  >
  > 
  >
  > - 设$\mathbf{g} \in \mathbb{R}^{C_{g}}$是在标准CNN分类器最后softmax前提取的全局特征向量(global feature vector)，在这种情况下，g编码的ROI是全局的、有区别的相关信息
  >
  > - 其思想是：考虑每一个$f^s_i$和g的关联，来处理与每个尺度$s$相关的特征，这些特征由与$g$表示的粗尺度特征(如ojbect型)相关。为此，定义了兼容性得分$\mathcal{C}(\mathcal{F}^s,g)=\{c^s_i\}^n_{i=1}$，由additive 注意力模型给出
  >
  >   
  >
  > $$
  > c_{i}^{s}=\left\langle\Psi, \mathbf{f}_{i}^{s}+\mathbf{g}\right\rangle
  > $$
  > 
  >
  > > 其中$<·,·>$是点积，$\Psi \in \mathbb{R}^{C_{s}}$是可学习的参数
  >
  > - 文中$\mathbf{f}_{i}^{s}$和$\mathbf{g}$有不同的维度
  >
  >   > 权重$W_g \in \mathbb{R}^{C_{s} \times C_{g}}$用于匹配 $\mathbf{f}_{i}^{s}$ 和$g$的维度
  >   >
  >   > 一旦计算得到兼容分数，它们通过softmax来获得归一化的注意力系数：
  >   >
  >   > 
  >   > $$
  >   > a_i^l=e^{c_i^l}/\sum_i e^{c_i^l}
  >   > $$
  >   > 
  >
  > - 对于每个尺度$s$，计算得到加权和$g^s=\sum^n_{i=1}a^s_if_i^s$，然后通过拟合全连接层{$g^1...g^S$}得到最终预测。通过约束从加权和进行的预测，网络被迫学习对该类有贡献的最显着特征。
  >
  > - 因此注意力系数${\alpha^l_i}$识别显著的图像区域，放大它们的影响，并抑制背景区域中的无关信息。

  

- 提出了一个更为通用的attention mechanism：

  

  > $$
  > c_{i}^{s}=\boldsymbol{\Psi} \sigma_{1}\left(\mathbf{W}_{f} \mathbf{f}_{i}^{s}+\mathbf{W}_{g} \mathbf{g}+\mathbf{b}_{g}\right)+\mathbf{b}_{\psi}
  > $$
  >
  > 
  >
  > - 引入注意机制，特征$\mathcal{F}^s$分支到两个路径：一个用于提取全局特征向量，另一个用于通过门控进行预测。 
  > - 首先，推测引入$W_f$允许精细尺度层更少关注于生成与g兼容的信号，这有助于其专注于学习判别特征。 
  > - 其次，通过引入$W_f$，$W_g$和$σ_1$，允许网络学习向量之间的非线性关系(因为图像本质上是有噪声的，并且感兴趣的区域通常是高度不均匀的。 因此，线性兼容性功能可能对这种波动过于敏感) 
  > - {$W_f$}和{$Ψ，b_ψ$}为1×1卷积层，{$W_g，b_g$}是全连接层。

  

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/4.png" alt="img" style="zoom:80%;" />

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/5.png" alt="img" style="zoom:50%;" />

- AG-Sononet介绍

  > 目的：引入注意力机制，更好的利用局部信息
  >
  > 具体细节：
  >
  > - 移去了自适应模块。提取模块的最后一层用作网格化全局特征图g。
  > - 对第11层和14层池化前应用注意力机制
  > - 在获得注意力图{$\alpha^s_i$}后，计算特征图中每个通道的空间轴上的加权平均值， 在尺度$s$产生长度为$C^s$的向量。此外，还在最粗糙的尺度上执行全局平均池化，并将其用于最终分类。
  
  

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/AG-Sononet.png)