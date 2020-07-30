---
layout:     post
title:      Inductive Representation Learning on Large Graphs 
subtitle:   
date:       2020-07-29
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper

---



ref: https://blog.csdn.net/yyl424525/article/details/100532849

#### 1. Introduction

##### 回顾GCN：

> 基本思想：把一个节点在图中的高维度邻接信息降维到一个低维的向量表示
>
> > 假设一个无向图有$N$个节点，每个节点都有$D$维的特征，图的邻接矩阵记为$A$，节点的特征组成$N\cdot D$维矩阵记为$X$。则对于GCN网络而言，层与层之间的传播可以用如下公式表达：
> > $$
> > H^{(l+1)}=\sigma\left(\widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}} H^{l} W^{l}\right)
> > $$
> >
> > - $H^l$为第$l$层图的特征，对于第一层网络的输入有$H^0=X$
> > - $\widetilde{A}=A+I$，$I$是单位矩阵
> > - $\widetilde{D}$是$\widetilde{A}$的度矩阵，使用$\widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}}$进行矩阵$\widetilde{A}$归一化
> > - $\sigma$为激活函数
> > - $W^l$为第$l$层的网络权值，起到特征映射的作用
>
> 优点：可以捕捉graph的全局信息，从而很好地表示node的特征
>
> 缺点：是Transductive learning的方式，需要把所有节点都参与训练才能得到node embedding，无法快速得到新的node的embedding
>
> > **Transductive Learning** vs **Inductive Learning**
> >
> > - transductive和inductive的区别在于想要预测的样本，是不是在训练的时候已经见过
> > - GCN是transductive的：
> >   - 根据GCN的公式可以看到，图结构是固定的，训练时在固定的图上进行的。若图是变动的，如突然加入一个新的节点，那么会导致前面计算的归一化矩阵整体变化。因此GCN对于大规模的图数据集其可拓展性是很低的，其中归一化矩阵是导致GCN为Transductive的根源。



#### 2. Method: GraphSAGE

> - GraphSAGE的核心思想是学习如何从节点的局部邻域中聚合特征信息（GCN学习的是每个节点的一个唯一确定的embedding；而GraphSAGE学习的是node embedding，即根据node的邻节点关系的变化而变化）
>
> - 将GCN传播的步骤用aggregate函数表示，在训练GraphSAGE时，只需要学习Aggregate函数即可实现Inductive Learning

##### Embedding generation

> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/10.png" alt="img" style="zoom:40%;" />
>
> - 上图为节点embedding生成(前向传播)算法:
>
>   > - 假设已完成GraphSAGE的训练，模型包括$K$个聚合器$AGGREGATE_k,\forall k \in\{1, \ldots, K\}$的参数，聚合器用来将邻域的embedding信息聚合到节点上(随着国车过的迭代，节点会从越来越远的地方获得信息)
>   > - 一系列的权重矩阵$W^k,\forall k \in\{1, \ldots, K\}$被用作在模型层与层之间传播embedding时做非线性变换
>   >
>   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/12.jpg" alt="img" style="zoom:40%;" />
>
> - 学习过程如下：
>
>   > - 对邻节点采样
>   > - 采样后的邻节点embedding传到节点，使用一个聚合函数聚合这些邻域信息，以更新节点的embedding
>   > - 根据更新后的embedding预测节点的标签
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/11.png" alt="img" style="zoom:40%;" />
>
> - 采样邻节点方式：
>
>   > 对每个节点采样一定数量的邻节点作为待聚合信息的顶点。设需要采样数量为$S$，若邻节点数少于$S$，则采用有放回的抽样方法，直到采样出$S$个顶点；若邻节点的数大于$S$，则采用无放回的抽样（若不考虑计算效率，完全可以对每个顶点利用其所有的邻居顶点进行信息聚合，这样是信息无损的）
>   >
>   > `论文里常常出现的“固定长度的随机游走”其实就是指随机采样了固定数量的邻节点`



##### Aggregator Architectures

`聚合函数的选取`

> 由于在图中节点的邻节点是无序的，因此够早的聚合函数应该具有对称性，确保模型能够应用于任意顺序的节点邻域特征集合上

- Mean Aggregator

  > 将目标顶点和邻节点的第$k-1$层向量拼接起来，然后对向量的每个维度进行球均值的操作，并将得到的结果做一次非线性变换得到目标节点的第k层表示向量
  >
  > 

- LSTM aggregator

  > 基于LSTM的复杂聚合器和均值聚合器相比具有更强的表达能力，但是原始的LSTM模型不是symmetric的，因此需要先对邻节点随机排序，然后将邻域的embedding作为LSTM的输入
  >
  > 

- Pooling aggregator

  > 先对目标节点的邻节点的embedding向量进行一次非线性变换，然后进行一次pooling操作，将结果域目标节点的表示向量拼接，最后再经过一个非线性变换得到目标节点的第k层表示向量



##### Learning the parameter

`定义好聚合函数之后，需要对函数中的参数进行学习，论文分别提出了无监督学习和监督学习两种方式`

**基于图的无监督损失**

> 

**基于图的有监督损失**

> 

**参数学习**

> 

**新节点embedding的生成**

> 