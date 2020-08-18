---
layout:     post
title:      Dynamic Graph Representation Learning via Self-attention Network
subtitle:   
date:       2020-07-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - GCN
---



---
### **Abstract**

​		学习图中节点的潜在表示是一项重要而普遍的任务，具有广泛的应用，如链接预测、节点分类和图形可视化等。以往的图表示学习方法主要集中在**静态图**上，而现实世界中的许多图都是动态的，并随着时间的推移而演化。在这篇论文中，提出了动态自注意网络（DySAT），这是一种新的神经架构，它可以在动态图上运作，并学习同时捕捉结构特性和时间演化模式的节点表示。具体来说，DySAT通过沿着两个维度（结构邻域和时间动态）联合使用自注意层来计算节点表示。论文在动态图数据集上进行了链接预测实验，实验结果表明，与几种不同的最新图形嵌入baseline相比，DySAT具有显著的性能增益。

---



### 1. Introduction

#### 图网络表示学习 & 图网络嵌入

- 近年来，随着图神经网络表示学习的兴起，网络嵌入(Network Embedding) 成为了学术界和工业界日益关注的研究热点。

- 网络嵌入的目的是为每个节点学习一个低维向量，该向量为节点的潜在表示，包含节点自身属性、邻域的结构属性等信息，可以帮助捕捉网络的结构和性质，完成大量的图分析任务（下游任务），如节点分类、链接预测、推荐和图可视化等

- 大多数基于网络嵌入的方法主要都是针对**静态图网络**，静态图包含一组固定的节点和边。而现实世界中很多图网络结构都是动态的、时变的（如学术合作网络，作者可以周期性地改变合作行为，电子邮件通信网络结构可能会因突发事件而发生巨大变化，因此在这种情况下，对时间演化模式进行建模对于准确预测节点属性和未来节点链接非常重要）

- 动态图网络相比静态图网络而言，强调了网络中节点和边的出现顺序和时间（即节点的邻域不是同时形成的，节点间的信息传递也不是同时发生的），如下图所示。**因此要求学习的嵌入不仅要保持节点之间的结构近似性，还要联合捕获时间变化下的时序依赖关系**。

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/3.png" alt="img" style="zoom:33%;" />
  
  

#### 自注意力机制

- 注意机制近年来在许多序列学习任务中取得了巨大的成功，如机器翻译和阅读理解。

- 注意力机制的关键是学习一个函数能够**聚合可变大小的输入**，同时关注与特定上下文最相关的部分

- 当注意力机制使用单一序列作为输入和上下文时，通常被称为自注意力机制

  > attention函数的本质可以被描述为一个查询（query）到一系列（键key-值value）对的映射，self-attention中，query、key和value往往都来自同一输入

- 《Attention is all you need》，提出的Transformer是基于self-attention的经典模型，在机器翻译任务汇总实现了最先进的性能，可以参考 [Link](https://zju-cvs.github.io/2020/05/14/Transformer/)

- 《Graph attention networks》，将self-attention与GCN结合，用注意力机制对邻近节点特征进行加权求和。邻近节点特征的权重完全取决于节点特征，也就是说**注意力权值通过两个链接节点信息的非线性组合得到**。加入自注意力机制的图网络实现了静态图中半监督节点分类任务的最佳性能。

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/4.png" alt="img" style="zoom:50%;" />



### 2. Method

> 基于[Introduction](#1. Introduction)中对动态图性质的分析以及自注意力机制的介绍，本文提出了一种新的神经网络架构，即动态自注意力网络(DySAT)来学习动态图上的节点表示：
>
> - 在**结构邻域和时序动态**两个维度上使用了自注意力，即考虑节点的邻域和历史表征，遵循自注意力机制，生成一个节点的动态表征
> - 动态节点的表示反映了图结构在不同历史snapshots下的时间演化，且反映时间相关性的注意力权重在node-level的细粒度上捕获



#### Problem Definition

> - 动态图被定义为一系列的observed snapshots $\mathbb{G}=\{\mathcal{G}^1,\mathcal{G}^2,...\mathcal{G}^T\}$
> - 每个snapshot $\mathcal{G}_t=(\mathcal{V,\mathcal{E}^{t}})$是一个加权无向图，在$t$时刻，有一个共享节点集$V$，一个链接集$\mathcal{E}^t$和一个加权邻接矩阵$A^t$，本文所提方法允许删除图节点链接（以前的方法链接只能随着时间的推移而添加）
> - 动态图表示学习的目标是在每个时间节点$t=1,2,...,T$，为每个节点$v\in \mathcal{V}$ 学习潜在表示$e^t_v\in \mathbb{R}^d$，即$e^t_v$**既保留了以$v$作为中心节点的局部图结构，又保留了$t$之前的演化行为。**



#### Model Structure

> - DySAT主要由两个部分构成：*structural and temporal self-attention layers*，可以通过堆叠(stack)的方式来构造任意的图神经网络，同时采用多头注意力机制来提高模型的能力和鲁棒性
> - DySAT由一个**结构块**和一个**时间块**组成，其中每个块可以包含对应层类型的多个堆叠层
>   - 结构块通过自注意力机制聚合从局部邻域中提取的特征，计算每个snapshot的节点表示，并将这些表示输入到时间块中
>   - 时间块经过多个时间步，捕捉图结构中的时间变化特性
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/5.png" alt="img" style="zoom:40%;" />



`下面对结构自注意力模块和时序自注意力模块进行具体介绍`

##### Structural Self-Attention

> - 该层的输入是一个graph snapshot $\mathcal{G}\in \mathbb{G}$和一组输入节点表示$\{x_v \in \mathbb{R}^D,\forall v \in \mathcal{V}\}$，其中D是输入嵌入的维数
>
> - 输出是一组新的节点表示$\{\mathcal{z_v \in \mathbb{R}^F},\forall v \in \mathcal{V}\}$，其中$F$表示具有捕捉局部结构特性的维数
>
> $$
> \boldsymbol{z}_{v}=\sigma\left(\sum_{u \in \mathcal{N}_{v}} \alpha_{u v} \boldsymbol{W}^{s} \boldsymbol{x}_{u}\right), \quad \alpha_{u v}=\frac{\exp \left(\sigma\left(A_{u v} \cdot \boldsymbol{a}^{T}\left[\boldsymbol{W}^{s} \boldsymbol{x}_{u} \| \boldsymbol{W}^{s} \boldsymbol{x}_{v}\right]\right)\right)}{\sum_{w \in \mathcal{N}_{v}} \exp \left(\sigma\left(A_{w v} \cdot \boldsymbol{a}^{T}\left[\boldsymbol{W}^{s} \boldsymbol{x}_{w}|| \boldsymbol{W}^{s} \boldsymbol{x}_{v}\right]\right)\right)}
> $$
>
> > 其中$\mathcal{N}_v=\{u\in \mathcal{V}:(u,v)\in \mathcal{E}\}$是snapshot G中节点$v$的邻节点集；    
> >
> > $W^s\in \mathbb{R}^{D\times F}$是应用于图中每个节点的共享权值；   
> >
> > $a\in \mathbb{R}^{2D}$是一个权重向量参数化的注意力函数，通过前馈层实现；
> >
> > $\Vert$ 表示级联操作；
> >
> > $\sigma(\cdot)$是一个函数非线性激活；
> >
> > $A_{uv}$是当前snapshot G中链接$(u,v)$的权重；
> >
> > 通过softmax在每个节点的邻域上获得学习系数$a_{uv}$表示节点$u$对节点$v$在当前snapshot G中的重要性或贡献度；
> >
> > 使用**LeakyRELU**来计算注意权重$a_{uv}$，然后用**ELU**计算输出表示$z_v$



##### Temporal Self-Attention

> - 为了进一步捕捉动态网络中的时序演化模式，设计了时间自注意力层，该层的输入是特定节点$v$在不同时间戳下的序列表示
>
> - 对应每个节点v，将输入定义为$\{x^1_v,x^2_v,...,x^T_v\},x^t_v\in \mathbb{R^{D'}}$
>
>   > 其中T表示时间步，$D'$为输入表示的维度；
>   >
>   > 层输出是每个时间步的一个新的v表示序列，即$z_v=\{z_v^1,z_v^2,...,z_v^T\},z_v^t\in \mathbb{R}^{F'}$；
>   >
>   > 最终，分别用$X_v\in \mathbb{R}^{T\times D'}$和$Z_v\in \mathbb{R}^{T\times F'}$表示v的输入输出表示
>
> - 时间自注意层的关键目标是捕捉图结构在多个时间步长上的时间变化。节点$v$在时间戳$t$的输入表示为$x^t_v$，对$v$周围的当前局部结构进行编码。使用$x^t_v$作为query来关注其历史表示($<t$)，追溯$v$节点周围局部领域的演化。因此自注意有助于学习跨越不同时间步的节点的不同表示之间的依赖关系。
>
>   - 为了计算节点$v$在$t$的输出表示，使用标量点积(scaled dot-product)，其中query, key和value被设置为输入节点表示。首先利用线性投影矩阵$W_q\in \mathbb{R}^{D'\times F'}$，$W_k\in \mathbb{R}^{D'\times F'}$，$W_v\in \mathbb{R}^{D'\times F'}$将query, key和value转化为不同的空间。
>
>     > $$
>     > Z_{v}=\boldsymbol{\beta}_{\boldsymbol{v}}\left(\boldsymbol{X}_{v} \boldsymbol{W}_{v}\right), \quad \beta_{v}^{i j}=\frac{\exp \left(e_{v}^{i j}\right)}{\sum_{k=1}^{T} \exp \left(e_{v}^{i k}\right)}, \quad e_{v}^{i j}=\left(\frac{\left(\left(\boldsymbol{X}_{v} \boldsymbol{W}_{q}\right)\left(\boldsymbol{X}_{v} \boldsymbol{W}_{k}\right)^{T}\right)_{i j}}{\sqrt{F^{\prime}}}+M_{i j}\right)
>     > $$
>     >
>     > - $\beta_v \in \mathbb{R}^{T\times T}$是由 dot-product (multiplicative) attention 得到的注意力权重矩阵
>
>   - 为了防止信息向左流动并保持自回归性质，允许每个时间步骤$t$都参与到$t$之前(包括$t$)的所有时间步骤中，对时间顺序进行编码，将$M$定义为：
>
>     > $$
>     > M_{i j}=\left\{\begin{array}{ll}
>     > 0, & i \leq j \\
>     > -\infty, & \text { otherwise }
>     > \end{array}\right.
>     > $$
>     >
>     > - 当$M_{ij}= -\infty$时，softmax函数会导致注意力权重为0，即$\beta^{ij}_v=0$
>
>   

`注：上面这些公式不熟悉Scaled Dot-Product Attention 计算公式的可能比较难理解，下面给出《Attention is all you need》中的定义帮助理解`

> > $$
>> \text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
> > $$
>> 其中因子$\sqrt{d_k}$ 起到调节作用，使得上方内积不至于太大（太大的话 softmax 后就非 0 即 1 了，不够“soft”了），其中$\sqrt{d_k}$ 指的是键向量维度的平方根。



##### Multi-Head Attention

>采用多头注意机制来共同关注每个输入的不同子空间，分别在结构和时序自注意层均使用了多个注意力头，然后进行拼接(concatenation):
>
>- Structural multi-head self-attention:
>
>$$
>\boldsymbol{h}_{v}=\operatorname{Concat}\left(\boldsymbol{z}_{v}^{1}, \boldsymbol{z}_{v}^{2}, \ldots, \boldsymbol{z}_{v}^{H}\right) \quad \forall v \in V
>$$
>
>- Temporal multi-head self-attention:
>
>$$
>\boldsymbol{H}_{v}=\operatorname{Concat}\left(\boldsymbol{Z}_{v}^{1}, \boldsymbol{Z}_{v}^{2}, \ldots, \boldsymbol{Z}_{v}^{H}\right) \quad \forall v \in V
>$$
>
>> 其中$H$表示注意力头的数目，$h_v \in \mathbb{R^F}$和$H_v\in \mathbb{R}^T\times F'$分别是结构和时间多头部注意力的输出



##### Overview of Architecture 

DySAT自上而下有三个模块：(1) 结构注意力模块; (2) 时间注意力模块; (3) 图上下文预测



> **Structural attention block**
>
> 该模块由多层结构自注意力层堆叠组成，从不同距离的节点($t_1,t_2,...,t_T$)中提取特征。使用共享参数在不同snapshots上独立的利用每一层，在每个时间步捕捉节点$v$周围的局部邻域结构。最终将结构注意力块输出的节点表示为$\{h_v^1,h^2_v,...,h^T_v\}，h^t_v \in \mathbb{R}^f$，输入到时序注意力模块。
>
> - Note：层的嵌入输入在不同的snapshots可能会有不同



> **Temporal attention block** 
>
> 通过位置嵌入(position embedding)体现时序注意力模块输入的时序性，$\{p^1,...,p^T\},p^t\in \mathbb{R}^f$，其中嵌入了每个snapshot的绝对时间位置，然后将位置嵌入与结构注意力模块的输出相结合，得到一系列输入表示$\{h^1_v+p^1,h^2_v+p^2,...,h^T_v+p^T\}$表示节点$v$跨越多个时间步长。
>
> - 时序注意力模块由多层时序自注意力层堆叠而成，最终层的输出进入位置前馈层，给出最终的节点表示$\{e^1_v,e^2_v,...,e^T_v\}$



> **Graph context prediction**
>
> 为了确保学习到的表示同时捕获结构和时序信息，定义了一个目标函数，该函数跨多个时间步仍保持节点周围的局部结构。
>
> - 使用节点$v$在时间步为$t$的动态表示$e_v^t$来预测$t$时刻在节点$v$附近局部邻域节点的出现
>
> - 在每个时间步使用BCE loss来激励相关节点在固定长度随机游走中同时出现，因此获得以下类似的表示：
>   $$
>   L_{v}=\sum_{t=1}^{T} \sum_{u \in \mathcal{N}_{w a l k}^{t}(v)}-\log \left(\sigma\left(<e_{u}^{t}, e_{v}^{t}>\right)\right)-w_{n} \cdot \sum_{u^{\prime} \in P_{n}^{t}(v)} \log \left(1-\sigma\left(<e_{u^{\prime}}^{t}, e_{v}^{t}>\right)\right)
>   $$
>
>   > $\sigma$表示sigmoid函数，$<\cdot>$表示内积运算体现节点间的相似性，$\mathcal{N}^t_{walk}$表示在$t$时snapshot中与固定长度随机游走同时出现的节点集；$P^t_n$是snapshot $\mathcal{G}^t$的负采样分布，$w_n$为负采样比，用于平衡正样本和样本的可调超参数
>
> **注：**
>
> > `这里补充一下随机游走(Random Walk)的相关知识`
> >
> > - 随机游走是指给定一个图和一个起始节点，随机选择一个邻节点，走到该处后在随机选择一个邻节点，重复length次，length是指随机游走的长度
> >
> > - 使用随机游走从起始节点到终止节点的概率值，可以用来表示相似度，即**从u节点到v节点的概率值，应该正比于u节点与v节点embedding之后的点乘结果$z_v^Tz_u ∝P(v∣u)$**



### 3. Experiments

#### Datasets

> **Communication network:** 
>
> - Enron: 员工间电子邮件交互
> - UCI: 社交平台上用户间发送的信息
>
> **Rating network:**
>
> - Yelp: 由用户和企业之间的链接组成，这些链接根据一段时间内观察得到的评级得出
> - MovieLens: 由一个用户标签交互网络组成，用户标签链接将用户与他们在某些电影上应用的标签链接起来
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/6.png" alt="img" style="zoom:40%;" />

#### Experimental Setup

> 对动态图中的**链路预测任务**进行实验，学习snapshots $\{\mathcal{G}^1,\mathcal{G}^2,...,\mathcal{G}^t\}$中的动态节点表示，并用$\{e^t_v,\forall v \in \mathcal{V}\}$来预测$\mathcal{G}^{t+1}$的链接。
>
> - 基于正确地将每个节点对**(node pair)**分为链接和非链接的能力，训练一个logistic回归分类器作为动态链接预测，来评估不同模型的性能
> - 为了进一步分析预测能力，评估了新链路预测能力，重点关注每个时间步出现的新链路
> - 使用Hadamard运算符$e_{u}^{t} \odot e_{v}^{t}$来计算一对节点的特征表示，用ROC曲线下的面积(AUC)分数来评估链路预测的性能



#### Results

**Baseline：**

> - 比较了几种最先进的无监督静态嵌入方法：node2vec、GraphSAGE 和 graph autoencoder
>
>   - GraphSAGE：GraphSAGE的实验中使用不同的聚合器(GCN、mean pooling、max pooling和 LSTM)并取性能最好的聚合器结果
>   - GraphSAGE+GAT：在GraphSAGE中实现了一个图注意力层作为额外的聚合器
>   - GCN-AE和GAT-AE：表示训练GCN和GAT作为链路预测的自动编码器
>
>   注：对GraphSAGE不太了解的可以看[Link](https://zju-cvs.github.io/2020/07/29/Inductive-Representation-Learning-on-Large-Graphs/)
>
> - 比较了动态图嵌入的方法：DynAERNN、DynamicRiad、DynGEM



**Model Performance：**

> - 在每个时间步骤$t$通过训练不同的模型直到snapshot $t$来评估模型，并在每个$t=1,...,T$，评估$t+1$预测的准确性
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/7.png" alt="img" style="zoom:40%;" />
>
> 

> - 比较了每个时间步的模型性能，课件DySAT的性能比其他方法更加稳定
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/16.png" alt="img" style="zoom:40%;" />

### 4. Conclusion

> DySAT利用自注意力机制在结构邻域和历史节点的表示计算动态节点表示，从而有效捕获图结构的时间演化模式。





### Appendix

`选了一些比较有意义的parts`

#### Effectiveness of Self-Attention

> - 通过消融实验分别验证自注意机制在结构和时间模块的有效性
> - 验证多头注意力机制的有效性

#### Dynamic New Link Prediction

> - 报告了动态链接预测的结果，即只在每个时间步对新链接进行评估，对不同方法预测相对不可见链接能力进行了深入分析，验证了DySAT在准确捕捉时序上下文以用于新链路预测的有效性
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/8.png" alt="img" style="zoom:40%;" />

#### Impact of Unseen Nodes on Dynamic Link Prediction

> - 分析了不同图表示学习对$t$时刻新出现的unseen节点链路预测的敏感性

#### Incremental Self-Attention Network

> - 论文在附录中还提出了增量的DySAT结构IncSAT，使用$t-1$中学习的嵌入作为初始化来学习$t$时的嵌入
>   - 通过存储结构块的中间输入$\{h^T_v,\forall v \in \mathcal{V}\}$来实现增量学习
>   - 在历史snapshots($1 \leqslant t<T$)的中间输出表示可以从先前保存的结果中直接加载，表示先前历史snapshots中的结构信息。
>   - 时间自注意力只应用于当前的snapshot $\mathcal{G}^T$，在每个节点的历史表示上计算最终的$\{e^T_v,\forall v \in \mathcal{V}\}$
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/graph-models/9.png" alt="img" style="zoom:40%;" />