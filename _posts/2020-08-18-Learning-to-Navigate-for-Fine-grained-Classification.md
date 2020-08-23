---
layout:     post
title:      Learning to Navigate for Fine-grained Classification
subtitle:   
date:       2020-08-18
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-grained
---



### 1. Introduction

- 由于很难找到区分性的特征(discriminative features)用于充分表征物体的细微特征，因此细粒度分类具有挑战

- 细粒度分类模型的设计的关键在于准确识别图像中的信息区域。之前的一些方法采用human annotation，需要较高的标注成本。另一些方法采用无监督学习来定位信息区域，消除了对昂贵注释的需求，但是缺少一种机制来确保模型聚焦于正确的区域，通常会导致准确性下降。

- 本文提出了一种新颖的自监督机制(self-supervision mechanism)，可以有效地定位信息区域，而无需使用bounding-box/part 标注。

  > **假设：**有意义的局部信息可以辅助分类，局部信息加全局信息可以进一步提高分类效果
  >
  > 基于以上假设，首先需要一个方法来给出每个局部位置的信息量$I$，信息量越大表明此局部用于预测此类别的概率$\mathcal{C}$越高，此局部区域可以提升细粒度识别的效果；然后取$M$个信息量最大的区域加上整张图，输入预测网络来预测类别



### 2. Method

> 提出了NTS-Net (Navigator-Teacher-Scrutinizer)，采用multi-agent协同学习的方案来解决准确识别图像中信息区域的问题。
>
> - 假设所有区域都是矩阵，并将$\mathbb{A}$表示为给定图像中所有区域的集合
> - 定义信息函数: $\mathcal{I}\rightarrow (-\infty, \infty)$用于评估区域$R\in \mathbb{A}$
> - 定义置信函数$\mathcal{C}:\mathbb{A} \rightarrow [0,1]$作为分类器，以评估该区域所属真实类别的置信度
> - 信息量更多的区域具有较高的置信度，因此满足以下
>
> 



##### Navigator

> 使用类似faster-rcnn的RPN来生成局部位置的信息量$I$，

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/17.png)

### 3. Experiments

