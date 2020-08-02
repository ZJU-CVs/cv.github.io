---
layout:     post
title:      Graph Models
subtitle:   图模型总结
date:       2020-08-01
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
	- Overview
	- Upgrade
---



### Overview

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/1.png)

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/2.png)

### GCN

> - 对于一个图$G=(V,E)$，输入$X$是一个$N\times D$的矩阵，表示每个节点的特征（$N$为节点数量，每个节点使用$D$维的特征向量进行表示），各个节点之间的关系会形成$N\times N$的矩阵$A$，即**邻接矩阵**。$X$和$D$是GCN模型的输入，希望得到一个$N\times F$的特征矩阵$Z$，表示学得的每个节点的特征表示，其中$F$是希望得到的表示维度。
>
> - 对于$L$层神经网络，可以表示为：
>   $$
>   H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
>   $$
>   加入两个trick：
>
>   - 对于每个节点，都加入自环：$\tilde{A}=A+I_{N}$
>   - $\tilde{D}$是度矩阵，$\tilde{D}_{ii}=\sum_j \tilde{A}_{ij}$，$\tilde{D}$是对角矩阵（图为无向图）
> - 正则化邻接矩阵，使得每一行的和都为1：$\hat{A}=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$
>     - $W^{(l)}$是第$l$层的权重矩阵，维度为$F^l\times F^{l+1}$，$F^{l}$为当前层的特征维度，$F^{l+1}$为下一层的特征维度
>   
> - 输入经过一个两层的GCN网络，得到每个标签的预测结果
>   $$
>   Z=f(X, A)=\operatorname{softmax}\left(\hat{A} \operatorname{ReLU}\left(\hat{A} X W^{(0)}\right) W^{(1)}\right)
>   $$
>
>   > - $W^{(0)}\in \mathcal{R}^{C\times H}$为第一层的权值矩阵，用于将节点的特征表示映射为相应的隐层状态
>   >
>   > - $W^{(1)}\in \mathcal{R}^{C\times H}$为第二层的权值矩阵，用于将节点的隐层表示映射为相应的输出（$F$对应节点标签的数量）
>   >
>   > - 最后每个节点的表示通过一个softmax函数得到每个标签的预测结果

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/GCN.png)

**优缺点**

> - 优点：
>   - $W^l$的维度可以调节，与顶点的数目无关，使得模型可以用于大规模的graph数据集中
> - 缺点：
>   - 对于同阶的邻域上分配给不同邻节点的权重是完全相同的，限制了模型对于空间信息相关性的捕捉能力
>   - GCN聚合邻节点特征的方式和图的结构相关，这局限了训练所得的模型在其他图结构上的泛化能力

 



### GAT

`GAT`提出了用注意力机制对邻节点特征加权求和，邻节点特征的**权重完全取决于节点的特征，独立于图结构**

> - 输入与输出
>
>   - 输入是一个节点特征向量集$h=\{\left.\vec{h}_{1}, \vec{h}_{2}, \ldots, \vec{h}_{N}\right\}, \vec{h}_{i} \in \mathbb{R}^{F}$，其中$N$为节点个数，$F$为节点特征的维度
>   - 输出是一个新的节点特征向量集$h'=\{\left.\vec{h'_{1}}, \vec{h'_{2}}, \ldots, \vec{h'_{N}}\right\}, \vec{h'_{i}} \in \mathbb{R}^{F'}$，其中$F'$表示新的节点特征向量维度
>
> - 特征提取与注意力机制
>
>   - 对于输入输出的转换(即根据输入的节点特征更新得到节点的新特征)，需要对所有节点训练一个权重矩阵$W\in \mathbb{R}^{F'\times F}$
>
>   - 对每个节点执行self-attention机制，注意力系数为$$\vec{h'_{i}},W\vec{h'_{j}}$$，其中$a(\cdot)$为一个函数，用于得到注意力系数，体现节点$j$对节点$i$的重要性，而不需要考虑图结构信息
>
>   - 论文中通过masked attention将这个注意力机制引入图结构中，即**仅将注意力分配到节点$i$的邻节点集$N_i$上，$j\in N_i$**
>
>   - 为了使得注意力系数更容易计算和便于比较，引入softmax进行正则化
>     $$
>     a_{i j}=\operatorname{softmax}_{j}\left(e_{i j}\right)=\frac{\exp \left(e_{i j}\right)}{\sum_{k \in N_{i}} \exp \left(e_{i k}\right)}
>     $$
>     
>
>   - 论文的实验中，$a(\cdot)$采用单层的前馈神经网络+LeakyReLu，因此得到完整的注意力机制如下：
>     $$
>     a_{i j}=\frac{\exp \left(L eakyReLu(\vec{a}^T[W \vec{h_i}\|W\vec{h_j}])\right)}{\sum_{k \in N_{i}} \exp \left(LeakyReLu(\vec{a}^T[W \vec{h_i}\|W\vec{h_j}]\right)}
>     $$
>     <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/4.png" alt="img" style="zoom:50%;" />
>
> - 输出特征计算
>
>   - 通过得到的不同节点之间的注意力系数，来预测每个节点的输出特征。下面这个公式表示**节点$i$的输出特征和它的相邻的所有节点特征有关，由它们的线性和非线性激活后得到**：
>     $$
>     \vec{h'_i}=\sigma(\sum_{j \in N_{i}} a_{i j} W \vec{h}_{j})
>     $$
>
> - multi-head attention
>
>   - 为了稳定self-attention学习过程，采用多头注意力机制。考虑$K$个注意力机制
>     $$
>     \overrightarrow{h^{\prime}}_{i}=\sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in N i} a_{i j}^{k} W^{k} \vec{h}_{j}\right)
>     $$
>     

**优缺点**

> - 优点：
>   - 图中的每个节点可以根据邻节点的特征，为其分配不同的权值
>   - 引入注意力机制后，指导聚合的信息只与邻节点有关，不需得到整张图的信息: （1）图不需要是无向的；（2）直接适用于*inductive learning*

### GraphSAGE

> 

### GAE

