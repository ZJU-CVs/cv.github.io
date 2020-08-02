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

### 2. Method

#### Preliminaries

- few-shot classification 的目标是根据给定的少量标记示例(*the support set*) 预测一组未标记点(*the query set*)，support set和query set中的数据标签集相同。

- few-shot learning通常包括两个后续学习阶段：
  - 第一个学习阶段利用训练集$D_b=\{(x,y)\} \subset I \times Y_{b}$，其中$x\in I$是标签为$y \in Y_b$的样本。在第一阶段无监督学习的设置意味着无法获取每个样本的标签信息、类别分布以及标签集大小等信息，而进行预训练，为第二阶段在target domain进行few-shot learning做准备
  - 第二个学习阶段包含$N$个新的类别，$D_n=\{(x,y)\} \subset I \times Y_{n}$，类别在$Y_n$标签集的样本很少



#### ProtoCLR

- 将每个ProtoCLR预训练步骤视为一个N-way 1-shot的分类任务，通过对比损失函数进行优化（inspired from unsupervised meta-learning and self-supervised visual contrastive learning of representations）
  - 《Unsupervised Meta-Learning for FewShot Image Classiﬁcation》
  - 《A Simple Framework for Contrastive Learning of Visual Representations》（SimCLR）

- ProtoCLR步骤如下：

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/29.png)

  > - 批处理过程（4-10）：每个mini-batch包含$N$个随机样本$\{x_i\}$，由于自监督设置不假设任何关于基类标签$y_b$的信息，因此将每个样本视为独立的一类，每个样本$x_i$作为1-shot 支持集样本，并通过不同的增强方式$Q$得到$\tilde{x}_{i,q}$作为查询集样本
  > - 优化对比原型损失（11-13）：使增强的样本集$\{\tilde{x}_{i,q}\}$的嵌入特征聚合在对应的原型$x_i$嵌入特征中
  > - 最终目标是训练得到嵌入函数$f_{\theta}(\cdot)$



#### ProtoTune

> 上一步ProtoCLR通过无监督学习的方式得到预训练的嵌入函数$f_\theta(\cdot)$，利用[ProtoNet](https://zju-cvs.github.io/2020/03/25/Prototypical-Networks-for-Few-shot-Learning/)的训练方式，对$f_\theta(\cdot)$进行fine-tuning
>
> - 首先计算类原型$c_n$:
>   $$
>   \boldsymbol{c}_{n}=\frac{1}{\left|S_{n}\right|} \sum_{\left(\boldsymbol{x}_{i}, y_{i}=n\right) \in S} f_{\theta}\left(\boldsymbol{x}_{i}\right)
>   $$
>
> - 训练一个线性分类器，初始化$W_n=2c_n,b_n=-\|c_n\|^2$，在保持嵌入函数参数$\theta$固定的情况下，通过softmax交叉熵损失对线性分类器进行fine-tuning
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/28.png)



### 3. Experiments

#### In-Domain Few-shot Classification

> 利用Omniglot和 mini-Imagenet进行域内实验(base classes和novel classes来自同一分布)
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/30.png)

#### Cross-domain Few-shot Classfication

> 利用CDFSL benchmark进行跨域实验
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/31.png)

### 4. Conclusion

- 提出了ProtoTransfer用于少样本分类，只需要几个带标签的示例就可以从未标记的源域向目标域执行迁移学习
- 在mini-ImageNet上，性能大大优于之前的无监督少镜头学习方法；在更具挑战的跨域少样本分类基准测试中，表现出与完全监督方法相近的性能
- **大批量**是得到良好的表征用于下游少样本分类任务的关键，在目标任务进行参数微调能显著提高性能

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/32.png)



