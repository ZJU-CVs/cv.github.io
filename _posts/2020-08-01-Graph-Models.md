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

> - 对于一个图$G=(V,E)$，输入$X$是一个$N\times D$的矩阵，表示每个节点的特征（$N$为节点数量，每个节点使用$D$维的特征向量进行表示），希望得到一个$N\times F$的特征矩阵$Z$，表示学得的每个节点的特征表示，其中$F$是希望得到的表示维度
>
> - 对于$L$层神经网络，可以表示为：
>   $$
>   H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
>   $$
>   加入两个trick：
>
>   - 对于每个节点，都加入自环：$\tilde{A}=A+I_{N}$
>   - 正则化邻接矩阵，使得每一行的和都为1：$\hat{A}=D^{-\frac{1}{2}} \tilde{A} D^{-\frac{1}{2}}$
>
> - 输入经过一个两层的GCN网络，得到每个标签的预测结果
>   $$
>   Z=f(X, A)=\operatorname{softmax}\left(\hat{A} \operatorname{ReLU}\left(\hat{A} X W^{(0)}\right) W^{(1)}\right)
>   $$
>
>   > $W^{(0)}\in \mathcal{R}^{C\times H}$为第一层的权值矩阵，用于将节点的特征表示映射为相应的隐层状态
>   >
>   > $W^{(1)}\in \mathcal{R}^{C\times H}$为第二层的权值矩阵，用于将节点的隐层表示映射为相应的输出（$F$对应节点标签的数量）
>   >
>   > 最后每个节点的表示通过一个softmax函数得到每个标签的预测结果

 



### GAT

> 

### GraphSAGE

> 

### GAE

