---
layout:     post
title:      Dual Attention Network for Scene Segmentation
subtitle:    
date:       2020-08-18
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Attention Mechanism
---



####  1. Introduction

> 提出了双重注意力网络（DANet）来**自适应**集成局部特征和全局依赖
>
> 在FCN上附加两种类型的注意力模块，分别模拟空间和通道维度中的语义相互依赖性。
>
> - 位置注意力模块通过所有位置处的特征的加权和来选择性地聚合每个位置的特征。无论距离如何，类似的特征都将彼此相关
> - 通道注意力模块通过整合所有通道映射之间的相关特征来选择性地强调存在相互依赖的通道映射

#### 2. Methods

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/DANet.png" alt="img" style="zoom:50%;" />

##### 2.1 Position Attention Module

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/DANet1.png" alt="img" style="zoom:50%;" />

> - 特征图**A**(C×H×W)首先分别通过3个卷积层得到3个特征图**B,C,D,**然后将**B,C,D** 都 reshape为C×N，其中N=H×W
> - 之后将 reshape后的**B**的转置(NxC)与reshape后的C(CxN)相乘，再通过softmax得到spatial attention map **S**(N×N)
> - 接着在reshape后的**D**(CxN)和 **S**的转置(NxN)之间执行矩阵乘法，再乘以尺度系数α，再reshape为原来形状，<u>最后与**A**相加得到最后的输出**E</u>**
> - 其中α初始化为0，并逐渐的学习得到更大的权重

##### 2.2 Channel Attention Module

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/DANet2.png" alt="img" style="zoom:50%;" />

> - 分别对**A**做reshape(CxN)和reshape与transpose(NxC)
> - 将得到的两个特征图相乘，再通过softmax得到channel attention map **X**(C×C)
> - 接着把**X**的转置(CxC)与reshape的**A**(CxN)做矩阵乘法，再乘以尺度系数β，再reshape为原来形状，最后与**A**相加得到最后的输出**E**
> - 其中β初始化为0，并逐渐的学习得到更大的权重



##### 2.3  Attention Module Embedding with Networks

> - 两个注意力模块的输出先进行元素求和以完成特征融合
> - 再进行一次卷积生成最终预测图