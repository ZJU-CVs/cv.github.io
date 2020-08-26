---
layout:     post
title:      Broad Learning System
subtitle:   An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture
date:       2020-03-23
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Paper
    - Learning Paradigm
---



#### 1. Introduction

- 深度学习中的模型待优化参数的数量庞大，通常需要耗费大量时间和机器资源来优化。
- 宽度学习是一种不需要深度结构的高效增强学习系统



#### 2. Method

- 宽度学习系统(BLS)的前身是随机向量函数连接网络(random vector functional-link neural network)

  > 其中RVFLNN只有**增强层**是真正意义上的神经网络单元，因为只有它带了激活函数，而网络的其他部分(输入层，输出层)均是线性的

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/BLS.png" alt="img" style="zoom:50%;" />



- 宽度学习对输入层做出来一点改进，不直接使用原始数据作为输入，而是对数据进行特征提取，因此第一层不叫输入层，而叫特征层

- 把增强层和特征层排成一行，将它们视为一体，网络就成了由 **A**（特征层+增强层）到 **Y** 的线性变换了，线性变换对应的权重矩阵 **W** 就是*输入层* 加增强层到*输出层* 之间的线性连接

  > - 当给定特征Z，直接计算增强层H，将特征层和增强层合并成$A=[Z\mid H]$，$\mid$表示合并成一行
  >
  > - 固定输入层到增强层之间的权重，那么对整个网络的训练就是求出 A 到 Y 之间的变换 W，$W=A^{-1} Y$。
  >
  > - 实际计算时，使用岭回归来求解权值矩阵(其中取$\sigma_{1}=\sigma_{2}=u=v=2$)
  >
  > $$
  > \underset{\mathbf{W}}{\arg \min } :\|A W-Y\|_{p}^{\sigma_{1}}+\lambda\|W\|_{u}^{\sigma_{2}}
  > $$
  > $$
  > \boldsymbol{W}=\left(\lambda \boldsymbol{I}+\boldsymbol{A} \boldsymbol{A}^{T}\right)^{-1} \boldsymbol{A}^{T} \boldsymbol{Y}
  > $$
  >
  > `当数据固定，模型结构固定，可以直接找到最优的参数W`

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/BLS1.png)

- 增量学习

  - 在大数据时代，数据固定是不可能的，数据会源源不断地来。模型固定也是不现实的，因为时不时需要调整数据的维数，比如增加新的特征。因此需要应用网络增量学习算法
  - 增量学习的核心是，利用上一次的计算结果和新加入的数据，只需要少量计算就能得到更新的权重
  - 增量学习算法包括：增强节点增强，特征节点增强和输入数据增强
  - 宽度学习系统可以高效重建需要在线学习的模型

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/BLS2.png)

