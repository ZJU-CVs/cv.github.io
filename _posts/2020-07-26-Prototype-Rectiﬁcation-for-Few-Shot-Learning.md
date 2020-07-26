---
layout:     post
title:      Prototype Rectiﬁcation for Few-Shot Learning
subtitle:   
date:       2020-07-26
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---



#### 1. Introduction

传统的原型网络将support set中每个类的所有样本representation平均作为类的原型表示，通过query set 中的特征representation与support set中每个类别的原型representation进行欧式距离计算，在经过softmax得出最后所属的类别

- 本文认为简单的求平均会产生很大的bias，因此提出从intra-class bias 和cross-class bias 对原型网络进行修正

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/model/1.png" alt="img" style="zoom:50%;" />

训练阶段：对于一个包含$\mathcal{C}_{base}$基类的标记数据集$\mathcal{D}$，训练特征提取器$F_\theta(\cdot)$和余弦分类器$C(\cdot \mid W)$

推理阶段：基于每个类有K个标记的图像集，识别少量的$\mathcal{C}_{few}$类

> Episodic采样N-way K-shot tasks，每个episode包含一个support set $\mathcal{S}$ 和一个query set $\mathcal{Q}$

> - 在support set中，所有样本$x$被标记，使用特征提取器提取特征$X=F_\theta(x)$来计算few-shot classes的原型$P$，计算和样本的余弦相似度进行分类
> - 在query set 中的无标记样本用于测试
> - 由于基础原型和预期原型存在偏差，需要消除intra-class bias和cross-class bias
>   - intra-class bias：



#### 2. Method

**Cosine Similarity Based Prototypical Network （CSPN）**

基于余弦相似度的原型网络

> CSPN用于计算 few-shot classes的basic prototypes
>
> - 首先在base classes的基础上训练特征提取器$F_\theta(\cdot)$，该特征提取器具有余弦相似性的分类器$C(.\mid W)$
>   $$
>   C\left(F_{\theta}(x) \mid W\right)=\operatorname{Softmax}\left(\tau \cdot \operatorname{Cos}\left(F_{\theta}(x), W\right)\right)
>   $$
>
>   > $W$是基类的可学习权重，$\tau$是标量参数
>   >
>   > 目标是尽可能减少监督分类任务的负对数似然损失$$L(\theta, W \mid \mathcal{D})=\mathbb{E}\left[-\log C\left(F_{\theta}(x) \mid W\right)\right]$$
>
> - 在推理阶段，重新训练$F_\theta(\cdot)$和分类权重对$C_{few}$少数类数据可能会出现过拟合，因此直接计算n类的基本原型$P_n$如下:
>   $$
>   P_{n}=\frac{1}{K} \sum_{i=1}^{K} \bar{X}_{i, n}
>   $$
>
>   > 其中$\bar{X}$是support samples的标准化特征，query samples可以根据余弦相似性找到最近的原型来分类



**Bias Diminishing for Prototype Rectiﬁcation**

> 在样本较少的情况下（如k=1或k=5），通过平均support samples的特征来获得基本的原型的方式与预期原型存在偏差。减少偏差可以提高类原型的表征能力，从而提高分类的准确率
>
> - 定义了两种影响因素：类内偏差和跨类偏差，并提出了偏差递减方法
>
>   > 类内偏差定义：
>
>   $$
>   B_{\text {intra}}=\mathbb{E}_{X^{\prime} \sim p_{X^{\prime}}}\left[X^{\prime}\right]-\mathbb{E}_{X \sim p_{X}}[X]
>   $$
>
>   - $p_{X'}$是某个类所有数据的分布，$p_X$是该类的可用标记数据的分布
>
>   > 跨类偏差定义：support 和 query数据集之间的距离，为域适应问题
>
> - 为了减少偏差，采用伪标记样本来增加support samples，即根据未标记数据的预测可信度为其分配临时标签
>   
>   > 首先通过计算query set中的样本和基础类原型之间的余弦相似度获得query sample的伪标签，然后将top-z confident的query sample加入support set中，并根据下式重新计算得到修正后的类原型$P'_n$
>   
> $$
>   P_{n}^{\prime}=\sum_{i=1}^{Z+K} w_{i, n} \cdot \bar{X}_{i, n}^{\prime}
> $$
> 
> $$
>   w_{i, n}=\frac{\exp \left(\varepsilon \cdot \operatorname{Cos}\left(X_{i, n}^{\prime}, P_{n}\right)\right)}{\sum_{j=1}^{K+Z} \exp \left(\varepsilon \cdot \operatorname{Cos}\left(X_{j, n}^{\prime}, P_{n}\right)\right)}
> $$
> 
- 跨类偏差是由于有标数据集support set和无标注数据集query set之间存在偏差，因此在无标注数据中加入偏移量$\xi$
> 
> $$
> B_{\text {cross}}=\mathbb{E}_{X_{s} \sim p_{S}}\left[X_{s}\right]-\mathbb{E}_{X_{q} \sim p_{\mathcal{Q}}}\left[X_{q}\right]
> $$
> 
> $$
> \xi=\frac{1}{|\mathcal{S}|} \sum_{i=1}^{|S|} \bar{X}_{i, s}-\frac{1}{|\mathcal{Q}|} \sum_{j=1}^{|\mathcal{Q}|} \bar{X}_{j, q}
> $$
> 



#### 3. Experiments

本文在两个公开数据集miniImagenet和tiredImagenet上进行了实验，与其他方法相比，所提出的BD-CSPN在1-shot和5-shot上均达到了最佳效果

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/model/10.png" alt="img" style="zoom:30%;" />

通过**消融实验**进一步验证模型每一部分的有效性

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/model/11.png" alt="img" style="zoom:30%;" />

