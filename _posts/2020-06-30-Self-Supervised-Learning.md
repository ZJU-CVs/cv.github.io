---
layout:     post
title:      Self-Supervised Learning
subtitle:   自监督学习
date:       2020-06-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Overview
---

更新中...

ref: https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html

### 1. Introduction

##### **背景**

> 大规模标注数据集的出现是深度学习在计算机视觉领域取得巨大成功的关键因素之一。然而监督式学习过于依赖大规模标注数据集，数据集的收集和人工标注需消耗大量的人力成本
>
> 近年来，自监督学习作为一种新的学习方法，在表征学习方面取得了不错的成绩，能够从大规模未标记数据中学习图像特征，无需使用任何人工标注数据。



##### **自监督学习介绍**

`自监督学习系统学会通过输入的其他部分预测输入的一部分。——LeCun`

> 自监督和监督学习之间的主要区别在于标注的来源：自监督学习利用输入数据本身作为监督不需要人工标注，其学习到的特征可以作为知识转移到所有类型的下游任务(downstream task)
>
> 在自监督学习中，把用于预训练的任务称为“pretext task”，把用于fine-tune的任务称为“downstream task”。在计算机视觉中，自监督学习的关键是使用使用什么**pretext task**。
>
> > pretext task所提取的数据特征是解决downstream task所需要的
> >
> > downstream task其实是一个迁移学习的问题，可以是任何监督问题，目的是用自监督特征改善下游任务的性能
> >
> > 通常下游任务最大的问题是**数据有限和过度拟合**，通过自监督训练可以在大型数据库上进行预训练而无需担心人为标签
> >
> > pretext task和普通分类任务差别：在纯分类任务中，网络学习表征是为了分离特征空间中的类；而在自监督学习中，pretext tasl通常会促使网络学习更多的通用概念
>
> 自监督学习能够在没有大规模标注数据中获得优质的表征，然后可以使用这些特征来学习缺乏数据的新任务



##### **自监督学习在计算机视觉中的相关工作**

> - Colorization
>
>   > 《Colorful Image Colorization》
>   > 《Learning Representations for Automatic Colorization》
>   > 《Tracking Emerges by Colorizing Videos》
>
> - Placing image patches in the right place
>
>   > 《Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles》
>   > 《Unsupervised Visual Representation Learning by Context Prediction》
>
> - Placing frames in the right order
>
>   > 《Unsupervised Representation Learning by Sorting Sequences》
>   > 《Shuffle and Learn: Unsupervised Learning using Temporal Order Verification》
>
> - Classify corrupted images
>
>   > 《Self-Supervised Feature Learning by Learning to Spot Artifacts》



##### 自监督学习在图像任务上的特征表示

ref: https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html

- Distortion (扭曲、变形)

  《Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks》

  > 用图像中的patch创建数据集：
  >
  > - 从包含大量梯度的位置选取"exemplary" patch
  > - 每个patch都通过施加各种随机变换（平移、旋转、缩放等），所有变换得到的失真patch都被认为属于同一代理类
  > - pretext task是区分一组代理类（每一个patch就是一个代理类）

- Rotation

  《Unsupervised Representation Learning by Predicting Image Rotations》

  > - 一副图像中提取多个patch，并要求模型

  《Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles》

  > 

  《Representation Learning by Learning to Count》