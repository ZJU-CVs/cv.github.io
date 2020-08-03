---
layout:     post
title:      Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions
subtitle:   
date:       2020-08-02
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---



### 1. Introduction

- 目前许多少样本学习方法通过可见类学习一个实例嵌入函数，并将其应用于标签有限的不可见类的实例中。但是这种迁移方式的任务无关（Task-agnostic）的：相对于不可见类，嵌入函数的学习不是最佳的判别式，因此会影响模型在目标任务的性能。

- 如上所述，当前少样本学习方法缺少一种适应策略，即将从seen classes中提取的视觉知识调整为适合目标任务中的unseen classes。因此对于少样本学习模型，需要单独的嵌入空间，其中每个嵌入空间都是自定义的，因此对于给定的任务，使视觉特征最具有区分性。

- 本文提出了一种**few-shot model-based embedding adpatation**方法，该方法基于不同任务的不同seen classes调整实例嵌入模型。

  - 通过一个*set-to-set*函数使实例嵌入与目标分类任务相适应，产生具有任务特定性和区分性的嵌入。该函数映射从few-shot 支持集中获得所有实例，并输出适应后的support instance嵌入，集合汇总的元素相互配合

  - 然后将函数的输出嵌入作为每个视觉类别的原型，并用作最近邻的分类器。(如下图所示，FEAT的嵌入自适应步骤将支持嵌入从混乱的位置推向了自己的簇，从而可以更好地拟合其类别的测试数据)

    ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/36.png)

  

### 2. Method

#### Learning Embedding for Task-agnostic FSL

> 在FSL中，一个任务表示为*M-shot N-way*分类问题
>
> 在只有少量训练实例的情况下，构造复杂的$f(\cdot)$具有挑战性，因此常常通过元学习的方式
>
> - 在可见类的数据集进行采样来生成许多M-shot N-way FSL tasks，使用训练集$\mathcal{D}_{train}=\{x_i,y_i\}^{NM}_{i=1}$来学习$f(\cdot)$。
>
> $$
> f^{*}=\underset{f}{\arg \min } \sum_{\left(\mathbf{x}_{\mathbf{t} \operatorname{est}}^{S}, \mathbf{y}_{\mathbf{t} \mathbf{e s t}}^{S}\right) \in \mathcal{D}_{\mathbf{t} \in \mathbf{e t}}}^{\mathcal{L}} \ell\left(f\left(\mathbf{x}_{\mathbf{t} \mathbf{e s t}}^{\mathcal{S}} ; \mathcal{D}_{\mathbf{t r a i n}}^{\mathcal{S}}\right), \mathbf{y}_{\mathbf{t e s t}}^{\mathcal{S}}\right)
> $$
>
> **可以看出嵌入函数的学习是任务无关的**



#### Adapting Embedding for Task-speciﬁc FSL

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/38.png" alt="img" style="zoom:30%;" />



##### Adapting to Task-Speciﬁc Embeddings

##### Embedding Adaptation via Set-to-set Functions

##### Contrastive Learning of Set-to-Set Functions

> $$
> \hat{\mathbf{y}}_{\text {test }}=f\left(\phi_{\mathbf{x}_{\text {test }}} ;\left\{\psi_{\mathbf{x}}, \forall(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{\text {train }}\right\}\right)
> $$
>
> 

### 3. Experiments

