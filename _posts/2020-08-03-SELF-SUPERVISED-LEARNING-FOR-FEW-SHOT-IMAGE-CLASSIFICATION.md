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

> - 用**Augmented Multiscale Deep InfoMax(AMDIM)**作为自监督模型，其
>
>   (The pretext task is designed to maximize the mutual information between features extracted from multiple views of a shared context）
>
> - 

#### Meta-learning stage

> 

### 3. Experiments



### 4. Conclusion