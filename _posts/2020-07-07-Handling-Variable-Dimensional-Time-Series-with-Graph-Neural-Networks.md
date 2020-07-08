---
layout:     post
title:      Handling Variable-Dimensional Time Series with Graph Neural Networks
subtitle:   
date:       2020-07-07
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---

### 1. Introduction

> - 在IOT场景，涉及到多个传感器采集数据，会产生多传感器时间序列。现有的基于NN方法的多变量时间序列建模方法输入维数（即传感器数量）固定
> - 但是在实际场景中，往往存在不同种类、不同数量传感器的多种组合，因此模型需要考虑不同(可变)的输入维度

### 2. Related Works

> **解决方案1:** 变量维度的变化可以看作是缺失值，常用的处理时间序列中缺失值的方法（如平均值、插值等）性能会随着缺失率的增加而迅速下降，同时该类方法不适合整条缺失（rely on the availability of at least one value for each dimension in the time series）
>
> 
>
> **解决方案2:** 为每个可能的多变量组合训练一个不同的网络，但存在三点缺陷：
>
> - 随着可能组合数量成指数型增长而不可拓展
> - 该方法的前提假设是每个组合都要有足够的训练数据可用
> - 跨组合之间的知识没有有效保留和利用
>
> 
>
> 可以看到，以上两种方案都有一定缺陷，在GNN、transfer-learning、meta-learning的启发下，提出了一种新的神经网络结构，适用于zero-shot 迁移学习，能够实现在测试时对多变量时间序列进行鲁棒的推理，且这些多变量时间序列中的有效维度（即传感器组合）是未知的。（模型具体细节见下一节描述）



### 3. Method

**问题定义：**

> - 考虑一个训练集$\mathcal{D}={(x_i,y_i)}^N_{i=1}$，具有N个多变量时间序列$x_i \in \mathcal{X}$和对应的相关目标$y_i \in \mathcal{Y}$
> - $\mathcal{S}$表示所有传感器的集合，共$d$维；$\mathcal{S}_i \subseteq \mathcal{S}$表示不同的传感器组合，共$d_i$维
> - $$x_i=\{x^t_i\}^{T_i}_{t=1}$，$x_i^t\in \mathbb{R}^{d_{i}}$$

**模型架构：**由一个conditioning module和一个core dynamics module构成

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/31.png)

> **conditioning module:**
>
> > 该模型基于GNN构建，能够生成一个"conditioning vector"并作为附加输入传递给core dynamics module
>
> - **Details**
>
>   - 每个传感器$s\in \mathcal{S}$与一个可学习的嵌入向量$v_s\in \mathbb{R}^{d_{s}}$相关联，对应于传感器组合$\mathcal{S}$，考虑图$\mathcal{G}(\mathcal{V}, \mathcal{E})$，每个$s\in\mathcal{S}$对应一个节点$v_s\in \mathcal{V}$，每个节点$v_s$的邻节点用$\mathcal{N}_{\mathcal{G}}\left(v_{s}\right)$表示
>
>   - 只有已知组合$\mathcal{S}_i$的中的对应节点被激活，图中每个active节点都接连接到其他active节点，因此每个边两端的节点都是active的
>
>   - 节点更新和边更新（可以理解为是信息传递）：
>
>     - $f_n$ 为节点前馈网络，$f_e$为边前馈网络，$f_n$和$f_e$在图中的节点和边上共享，也就是说这两个网络训练的不是特定点或特定边的更新方式，而是通用的更新方式
>
>     - 对于任意active 节点$v_k$，对应的节点向量$\mathbf{v}_{k}$更新如下：
>       $$
>       \begin{aligned}
>       \mathbf{v}_{k l} &=f_{e}\left(\left[\mathbf{v}_{k}, \mathbf{v}_{l}\right] ; \boldsymbol{\theta}_{e}\right), \quad \forall v_{l} \in \mathcal{N}_{\mathcal{G}}\left(v_{k}\right) \\
>       \tilde{\mathbf{v}}_{k} &=f_{n}\left(\left[\mathbf{v}_{k}, \sum_{\forall l} \mathbf{v}_{k l}\right] ; \boldsymbol{\theta}_{n}\right)
>       \end{aligned}
>       $$
>
>       > $\theta _e$和$\theta _n$分别为$f_e$和$f_n$的可学习参数，$f_e$将邻节点信息传递到待更新节点，$f_n$利用邻节点的聚合信息更新相应节点
>     
>     - 最后根据更新后的节点向量得到conditioning vector $\mathbf{v}_{\mathcal{S}_{i}} \in \mathbb{R}^{d_{s}}$
>     
>       $$\mathbf{v}_{\mathcal{S}_{i}}=\max \left(\left\{\tilde{\mathbf{v}}_{k}\right\}_{v_{k} \in \mathcal{V}_{i}}\right)$$
>     
>       
>
> - 使用GNN的优势在于可以处理除训练过程中传感器组合之外的其他unseen组合，从而实现组合泛化
>
> 
>
> 
>
> **core dynamics module:**
>
> > 该模型基于GRU构建（当然也可以用LSTM等），输入为固定维度的多变量，对于缺失的变量，**用一个常量或平均值对缺失值进行填补**
>
> **Details:**
>
> - 对于不同多变量组合的时间序列$x_i$转化为固定维度d的$\tilde{\mathbf{x}}_{i}$，对于缺失变量用均值填充
>
> - 将$\tilde{\mathbf{x}}_{i}$输入GRU进行训练，在最后一个时间步骤为$T_i$的module的输出特征向量$z^{T_i}$
>   $$
>   \mathbf{z}_{i}^{t}=G R U\left(\left[\tilde{\mathbf{x}}_{i}^{t}, \mathbf{v}_{\mathcal{S}_{i}}\right], \mathbf{z}_{i}^{t-1} ; \boldsymbol{\theta}_{G R U}\right), \quad t: 1, \ldots, T_{i}
>   $$
>
> - 根据下游任务(分类或回归)确定$f_o$，训练得到$\hat{y}_i$
>   $$
>   \hat{y}_{i}=f_{o}\left(\mathbf{z}_{i}^{T_{i}} ; \boldsymbol{\theta}_{o}\right)
>   $$
>   
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/index.png" alt="img" style="zoom:50%;" />
>
>  

**训练方式：**

> 通过随机梯度下降(SGD)以end-to-end的方式联合训练
>
> **损失函数：**
>
> - 分类任务：$$\mathcal{L}_{c}=-\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i}^{k} \log \left(\hat{y}_{i}^{k}\right)$$
> - 回归任务：$$\mathcal{L}_{r}=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}$$
>
> 值得注意的是：模型参数学习方式为mini-batch SGD，在每个小批量内输入为具有相同可用传感器集的时间序列，即每个小批量中所有时间序列的活动节点都相同。(可以看这篇blog：https://www.imooc.com/article/details/id/48566)

### 4. Experiments

> 数据集和参数设置就不详细描述了，主要介绍一下实验结果

##### 实验一

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/32.png" alt="img" style="zoom:50%;" />

实验设计了zero-shot和fine-tuning两个测试场景，两个测试场景下使用的测试集是在训练时看不到的传感器组合

> -  在zero-shot中，训练的网络直接用于推理
> - 在fine-tuning中，先使用测试实例中一部小部分标记的时间序列进行网络微调



实验将所提方法GRU-CM与GRU和GRU-SE方法进行对比

> GRU: 只有时序模块，没有调节模块，不会给GRU提供额外的条件向量信号
>
> GRU-SE: 忽略GNN中信息传递和交换，直接利用max operate得到条件向量嵌入GRU中



> GRU-A: 对于所有的训练和测试实例，假设所有传感器都可用，并训练一个与所提模型使用相同参数的GRU网络，相当于给出一个评估上限（即GRU特征提取能力的上限，提供参照值）



**对比结果：**

- 在两个场景中，GRU-CM在三个数据集上的性能始终优于GRU，**因此可以认为GRU-CM对未知传感器组合的适应能力**，同时也具有更好适应少数据fine-tuning的能力
- 随着$f_{te}$的增加，即测试时不可用传感器比例的增加，GRU-CM下降更为缓慢，**体现了调节模块的优势**

- 大多数情况下，GRU-CM的性能优于GRU-SE，GRU-SE的性能有时候比GRU还差，因此可以证明**在可用传感器之间传递信息能够提供更有意义的调节向量**



##### 实验二

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/33.png" alt="img" style="zoom:50%;" />

> 考虑使用与测试实例中传感器组合由最高重叠的现有训练实例，对训练好的模型进行fine-tuning，而不是像上个实验依赖测试集中的新数据进行模型微调

可以看到这种方式的fine-tuning相比zero-shot，对模型的性能有明显改善，且GRU-CM优于GRU，**可见GRU-CM对新的传感器组合具有更好的适应能力**

`idea:可以考虑找多个高重叠实例进行集成学习fine-tuning,这样可以互相补充缺失`



