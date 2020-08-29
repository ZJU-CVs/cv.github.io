---
layout:     post
title:      Object-Part Attention Model for Fine-grained Image Classification
subtitle:   
date:       2020-07-21
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-grained

---



#### 1. Introduction

- 解决问题：细粒度图像分类(Fine-grained Image Classification)
- 特点：类内差异大，类间差异小
- 现有方法：首先定位对象或部分(locate the objects or parts)，然后区分图像属于那个子类别。但是有两个局限：
  - 依赖大量人工标注数据
  - 忽略对象与其他各部分之间以及这些部分之间的空间关系
  
  

#### 2. Method

- 弱监督细粒度图像分类的对象注意模型（OPAM）

  - Object-level attention model

    利用CNN中的平均池化(Global average pooling)来提取用于定位图像对象的显著性映射(saliency map)，即学习对象特征。该模型由两部分组成：patch filtering和saliency extraction

    - patch filtering：利用selective search从原始图片中产生大量的patches，然后经过在ImageNet 1K上预训练过的网络(FilterNet)，过滤掉噪声图像块patches，并保留与object相关的图像块patches。
    - saliency extraction：将这些patches用来从头训练一个CNN网络，称为ClassNet，是一个可用于细粒度分类的分类器，以学习特定子类别的多视图和多尺度特征，并通过CNN的全局平均平均池化来提取显著性映射，以便定位图像对象。对于这些定位了的图片，训练一个网络称为ObjectNet，作为object-level的预测结果。

    ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/OPAM1.png)

    

  - Part-level attention model

    首先选择判别部分，然后基于神经网络的聚类模式对其部分(aligns the parts)，即学习细微和局部特征。由两部分组成object-part spatial constraint model 和 part alignment

    - object-part spatial constraint model

      object-part constraint model分为两个部分，一个是object spatial constraint，用来使选择的part位于object region，另一个是part spatial constraint，使选择的parts之间的重复性降低，减少冗余，增加parts之间的区分性

    - part alignment

      part alignment的目的使让同样语义的parts在一起。方法：对Classnet的中间层的神经元进行聚类的模式，以构建用于对齐所选part的part集群。
  
    - 选取Classnet的卷积层倒数第二层的神经元，首先计算相似矩阵S，其中S (i, j)表示两个中间层神经元ui和uj权值的余弦相似度，然后对相似矩阵S进行谱聚类，将中间层神经元划分为m组
      
      - 将所选部分的图像形变为倒数第二个卷积层中神经元输入图像上感受野大小
  - 将所选部分前馈到倒数第二个卷积层，以产生每个神经元的激活分数
    
      - 对每个cluster内的神经元的得分进行求和，得到聚类得分，把part归到得分最高的那一类(将所选部分与具有最高聚类得分的聚类对齐）
- 将这些parts分好类之后，使用他们训练一个CNN，称为PartNet，也是一个细粒度分类器。
  

  
- 最终的结果是将图片在这个分类器(ClassNet，ObjectNet，PartNet)的得分各取一个权重相加得到最终得分，归为得分高的那一类作为最终预测结果。
  
    
  
    ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/OPAM.png)
    
    `Object-level attention model侧重于representative object appearance，而part-level attention model侧重于区分子类别之间的部分特定差异。它们共同用于促进多视图和多尺度特征学习，并增强它们的相互促进以实现细粒度图像分类的良好性能`
    
    
    