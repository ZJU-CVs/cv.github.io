---
layout:     post
title:      Self-Supervised Prototypical Transfer Learning for Few-Shot Classiﬁcation
subtitle:   
date:       2020-08-04
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - FSL
---



### 1. Introduction

- 大多数few-shot classification方法仍然需要大量的预训练标注数据获取先验知识
- 本文将**无监督学习**应用在一种基于prototype network的few-shot learning方法中，在一个未标记的训练域上执行自监督的域训练任务(pretext task)，并可以转移到few-shot target domain tasks

### 2. Method: ProtoCLR

#### Preliminaries

- few-shot classification 的目标是根据给定的少量标记示例(*the support set*) 预测一组未标记点(*the query set*)，support set和query set中的数据标签集相同。

- few-shot learning通常包括两个后续学习阶段：
  - 第一个学习阶段利用训练集$D_b=\{(x,y)\} \subset I \times Y_{b}$，其中$x\in I$是标签为$y \in Y_b$的样本。在第一阶段无监督学习的设置意味着无法获取每个样本的标签信息、类别分布以及标签集大小等信息，而进行预训练，为第二阶段在target domain进行few-shot learning做准备
  - 第二个学习阶段包含$N$个新的类别，$D_n=\{(x,y)\} \subset I \times Y_{n}$，类别在$Y_n$标签集的样本很少



