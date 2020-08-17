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

`Step 1`

> - 用**Augmented Multiscale Deep InfoMax(AMDIM)**作为自监督模型，其核心思想是从同一幅图像的两个视图$(x_a,x_b)$中最大化全局特征和局部特征之间互信息，可以通过最小化基于负样本的Noise Contrastive Estimation(NCE) loss来最大化互信息的下界
>
>   - 具体而言，最大化$<f_g(x_a),f_5(x_b)>$，$<f_g(x_a),f_7(x_b)>$和$<f_5(x_a),f_5(x_b)>$，其中$f_g$为全局特征，$f_5$为编码器的$5\times 5$局部特征映射，$f_7$为编码器的$7\times 7$特征映射，以$f_g(x_a)$和$f_5(x_b)$间的NCE loss为例：
>   
>   $$
>   \begin{array}{l}
>   \mathcal{L}_{\text {amdim}}\left(f_{g}\left(x_{a}\right), f_{5}\left(x_{b}\right)\right)= -\log \frac{\exp \left\{\phi\left(f_{g}\left(x_{a}\right), f_{5}\left(x_{b}\right)\right)\right\}}{\sum_{\widetilde{x}_{b} \in \mathcal{N}_{x} \cup x_{b}} \exp \left\{\phi\left(f_{g}\left(x_{a}\right), f_{5}(\tilde{x}_b)\right)\right\}}
>   \end{array}
> $$
>   
>   
>   
>   > $\mathcal{N}_x$为image $x$的负样本，$\phi$为距离度量函数



#### Meta-learning stage

`Step 2`

在基于上述自监督学习得到嵌入网络的情况下，将元学习应用于网络fine-tune，以满足少样本分类的类变化

> - 典型的元学习可以看作是一个具有多个任务的K-way C-shot episodic 分类问题，对于每个分类任务$T$，有$K$个类，每个类有$C$个样本。对于$K$类，用训练样本嵌入的质心表示:
>
> $$
> c_{k}=\frac{1}{|S|} \sum_{\left(x_{i}, y_{i}\right) \in S} f_{g}\left(x_{i}\right)
> $$
>
> 
>
> - 本文使用一种基于度量学习的方法，使用距离函数d，并从查询集$Q$中给定一个查询样本$q$，在所有类进行距离计算，并经过softmax得到：
>
> $$
> p(y=k \mid q)=\frac{\exp \left(-d\left(f_{g}(q), c_{k}\right)\right)}{\sum_{k^{\prime}} \exp \left(-d\left(f_{g}(q), c_{k^{\prime}}\right)\right)}
> $$
>
> 
>
> - 因此元学习阶段的损失为：
>
> $$
> \mathcal{L}_{meta} = -log(p(y=k \mid q))=d(f_g(q),c_k)+log \sum_{k^{\prime}} d(f_g(q),c_k')
> $$
>
> 

### 3. Experiments

- 数据集：采用MiniImageNet(64类为训练，16类为验证，20类为测试)和CUB-200-2011(100类为训练，50类为验证，50类为测试)

- 定量比较：所提的方法通过一个large network 的自监督预训练，能够显著改进少样本分类任务

  **MiniImageNet**

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/34.png" alt="img" style="zoom:50%;" />

  > - **Mini80_SL:** 利用AmdimNet和标签交叉熵损失进行监督训练
  > - **Mini80_SSL$^-$:** 表示自监督训练后没有进行元学习
  > - **Mini80_SSL:** 从80个类(训练+验证)中进行自监督训练，没有标签
  > - **Image900_SSL:** 从ImageNet1K中除了miniimagenet中的900类数据进行自监督训练

  

  **CUB-200-2011**

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/35.png" alt="img" style="zoom:50%;" />

  

### 4. Conclusion

> - 提出利用自监督学习来有效地训练一个鲁棒的嵌入网络来进行少镜头图像分类。与其他基线相比，所得到的嵌入网络更具通用性和可移植性。
>
>   
>
> - 通过元学习过程进行微调后，所提出的方法的性能可以显著优于基于两个常见的少样本分类数据集的定量结果的所有基线。



`可以改进的地方`

- 模型框架的改进：如将这两个阶段结合起来，并为此任务开发一个端到端的方法。
- 少样本任务拓展：验证所提出的方法在其他少数任务上的有效性
- 模型细节改进：如修改模型fine-tune方式，将global feature和local feature 拼接，参考[prototune]([https://zju-cvs.github.io/2020/08/04/Self-Supervised-Prototypical-Transfer-Learning-for-Few-Shot-Classi%EF%AC%81cation/](https://zju-cvs.github.io/2020/08/04/Self-Supervised-Prototypical-Transfer-Learning-for-Few-Shot-Classiﬁcation/))，冻结Amdimnet部分，在后面加线性分类器进行fine-tuning