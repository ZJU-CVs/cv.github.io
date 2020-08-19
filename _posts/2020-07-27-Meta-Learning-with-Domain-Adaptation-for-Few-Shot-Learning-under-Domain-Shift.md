---
layout:     post
title:      Meta-Learning with Domain Adaptation for Few-Shot Learning under Domain Shift
subtitle:   
date:       2020-07-27
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Meta-Learning
---

### 1. Introduction

- 元学习是Few-shot learning的一种方法，模型学习如何学习一个有效的模型来进行few-shot learning。其主要思想是从一组训练任务中获得有效先验知识，然后用于执行 (few-shot) 测试任务。

- 但是现有的meta-learning work假设training task和test task都来自同一分布($\tau_{\text {train}}=\tau_{\text {test}}$)，并且训练任务中有大量的标记数据

- **上述这一假设限制了元学习策略在现实世界中的应用**，在现实世界中，没有足够的具有充足标记的训练任务，且遵循与测试任务相同的分布

- 本文提出了一种新的元学习范式，**通过学习少量的学习模型，同时通过对抗性的域适应克服训练和测试任务之间的域转移**

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/20.png" alt="img" style="zoom:43%;" />

  

### 2. Method

##### Problem Setting

> 总体目标是学习一个meta-learner，使其能够利用$ \tau_{train}$获得一个好的少样本先验知识，克服task-level 域偏移，以学习$\tau_{test}$中未观测到的少样本任务        
>
> 
> $$
> \left(\mathcal{D}_{train}^{m}, \mathcal{D}_{train}\right) \sim \tau_{train},\left(\mathcal{D}_{test}^{m}, \mathcal{D}_{test}\right) \sim \tau_{test}
> $$
> 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/16.png" alt="img" style="zoom:43%;" />



##### **Meta-Learning with Domain Adaptation (MLDA)**

有两个目标需要同时优化：

- 学习一个特征提取器，能够区分特征是否有助于新任务的快速学习
- 希望这些特征对在训练任务的分布和测试任务的分布是不变的，即对于任务$T_i \sim \left(\mathcal{D}_{ train}^m, \mathcal{D}_{train}\right)$，希望其类似于从$\left(\mathcal{D}_{test}^{m}, \mathcal{D}_{test}\right)$中提取的任务



> 具体实现：
>
> - 在元学习阶段，考虑先把meta-training里的图像先通过G转换为meta-testing里面的图像domain，然后再做meta-learning，该步骤通过GAN来训练（**这里用到了meta-testing的数据，有点违背meta learning的基本设定**）
>
>   > 特征提取器$F$将$X_{train}$输入映射为$d$维嵌入，$F(x)=\hat{F}(G(x))$，其中$G: \mathcal{X}^{train} \rightarrow \mathcal{X}^{test},\hat{F}=\mathcal{X}^{test} \rightarrow \mathbb{R}^d$
>
>   
>
> - 损失函数：
>
>   - $\mathcal{L}_{fs}$为预测损失，表示仅使用从$\tau_{train}$中的任务所标记的训练数据进行优化
>   - $\mathcal{L}_{da}$表示域适应损失，从$\tau_{train}$和$\tau_{test}$中的任务中未标记的数据进行优化，$\mathcal{L}_{da}=\mathcal{L}_{GAN}+\mathcal{L}_{cycle}$
>
> $$
> \min _{\hat{\mathbf{F}}, \mathbf{G}, \mathbf{G}^{\prime}} \max _{D} \mathcal{L}_{f s}+\mathcal{L}_{d a}
> $$
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/17.png" alt="img" style="zoom:43%;" />



##### Few-shot Learning

本文选择基于原型网络来实现具有域适应的元学习框架

> - 对于给定的任务$T_i \sim (\mathcal{D}_{train}^m,\mathcal{D}_{train})$，原型网络使用特征提取器$F$为每个实例计算得到一个$d$维的嵌入，计算得到原型:
>
> $$
> c_n==\frac{1}{S_{n}^{s u p p o r t}} \sum_{\left(\mathbf{x}_{i}, y_{i}\right) \in S_{n}^{s u p p o r t}} \mathbf{F}\left(\mathbf{x}_{i}\right)
> $$
>
> 
>
> - 对于给定的查询实例$x$,原型网络在类别上生成概率分布：
>
> $$
> p(y=n \mid \mathbf{x})=\frac{\exp \left(-\operatorname{dist}\left(\mathbf{F}(\mathbf{x}), \mathbf{c}_{n}\right)\right)}{\sum_{(j=1)}^{N} \exp \left(-\operatorname{dist}\left(\mathbf{F}(\mathbf{x}), \mathbf{c}_{j}\right)\right)}
> $$
>
> > 其中$dist$用于衡量查询实例的嵌入与类原型之间的距离
>
> - $\mathcal{L}_{f s}=-\log p(y=k \mid \mathbf{x})$计算在query set上的损失，并反向传播更新特征提取器$F$



##### Adversarial task-level domain adaptation

> How to perform task-level domain adaptation and learn the mapping parameters $G$.

- 使用GAN loss学习生成器$G:\mathcal{X}^{\text {train}} \rightarrow \mathcal{X}^{\text {test}}$和对应的判别器$D$
  $$
  \mathcal{L}_{G A N}\left(\mathbf{G}, D, \mathcal{X}^{\text {train}}, \mathcal{X}^{\text {test}}\right)=\mathbb{E}_{\mathbf{x}^{\text {test}} \sim \mathcal{D}_{\text {test}}}\left[\log D\left(\mathbf{x}^{\text {test}}\right)\right]+\mathbb{E}_{\mathbf{x}^{t r a i n} \sim \mathcal{D}_{\text {train}}}\left[\log \left(1-D\left(\mathbf{G}\left(\mathbf{x}^{\text {train}}\right)\right)\right)\right]
  $$

- 使用生成对抗网络能够产生与测试域$\mathcal{X}^{train}$相似的输出，但是网络将会将训练域$\mathcal{X}^{train}$中同一组输入图像映射到测试域的任意随机排列，因为GAN loss的目标是高度不受约束的。因此需要使用周期一致损失cycle-consistency loss，使用一个新的映射$G':\mathcal{X}^{test}\rightarrow \mathcal{X}^{\text {train}}$恢复原始实例
  $$
  \mathcal{L}_{\text {cycle}}=\mathbb{E}_{\mathbf{x}^{\text {train}} \sim \mathcal{D}_{\text {train}}}\left[\left\|\mathbf{G}^{\prime}(\mathbf{G}(\mathbf{x}))-\mathbf{x}\right\|_{1}\right]
  $$

- 因此域适应目标为：$$\mathcal{L}_{d a}=\mathcal{L}_{G A N}+\mathcal{L}_\text {cycle}$$



##### Additional Improvements

> 考虑了两个advanced variants，有助于提高域自适应的性能
>
> - **Identity loss**: encourage $G$ to behave like an identity mapping when it receives an instance from $\mathcal{X}^{test}$
> - **Reverse direction mapping** and **cycle loss**: map instances from test tasks $\mathcal{X}^{test} \rightarrow \mathcal{X}^{train}$ and reconstruct back the instance in $\mathcal{X}^{test}$
>
> 理解advanced variants中的identity loss, cycle loss等，需要了解[CycleGAN](https://zju-cvs.github.io/2020/07/27/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/)

### 3. Experiments

- 使用了Omniglot和Ofﬁce-Home Dataset，元训练和元测试在不同的领域，用于域适应的数据与元测试数据之间没有重叠

- 对比方法：

  > - **域适应方法：**RevGrad、ADDA和CyCADA，在元训练数据集上训练得到多类分类器，未标记的目标域数据用于域自适应。在元测试过程中，模型被用作特征提取器，并用KNN进行预测
  >
  > - **元学习方法：**MAML、PN，在元训练数据上进行训练，并未考虑域适应问题，在元测试数据上进行测试
  >
  > - **元学习+域适应方法：** Meta-RevGrad，MLDA，MLDA+idt (considering the previous objective and an identity loss)，MLDA+idt+revMap (adding an additional component of (reverse) mapping testing tasks to train tasks)
  >
  >   - MLDA: remove the losses related to **target $\rightarrow$ source** mapping and set $\lambda_{idt}=0$
  >
  >   - MLDA+idt: set $\lambda_{idt}$=0.1
  >   - MLDA+idt+revMap: 

  

##### Omniglot 

> 域迁移：Omniglot ，Omniglot-M
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/23.png" alt="img" style="zoom:43%;" />

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/21.png" alt="img" style="zoom:43%;" />

##### Office-Home Dataset

> 域迁移：Clipart，Product
>
> 分为元训练的标记数据25类，域适应的未标记数据20类，以及元测试数据20类
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/24.png" alt="img" style="zoom:43%;" />

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/22.png" alt="img" style="zoom:43%;" />



