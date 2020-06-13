---
layout:     post
title:      SOLO Segmenting Objects by Locations
subtitle:   SOLO分割模型
date:       2020-06-09
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---

#### 1. Introduction

> 实例分割主要有两大类别：
>
> - 自上而下：先检测后分割（依赖目标检测的准确率）
> - 自下而上：为每个像素学习一个嵌入向量，然后通过聚类、度量学习等方法区分不同的实例（依赖每个像素的嵌入学习和分组后处理）

> **创新点**：利用中心点和大小两个信息区别实例



#### 2. SOLOv1

`SOLO: Segmenting Objects by Locations`

##### Network Architecture

> 量化中心点位置和物体大小，根据实例位置和大小为实例中的每个像素分配类别
>
> - Location: 将图片划分为$S\times S$的网格，得到$S\times S$个位置，将定义的实例中心位置的类别放在channel维度上，若目标中心（质心）落入grid中，则这个grid要输出实例类别和分割mask
> - Size: 使用FPN将不同尺寸的物体分配到不同层级的特征图上，作为实例的尺寸类别
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/solo.png)



> SOLO在Backbone网络后使用了FPN，用来对应不同的尺寸。FPN的每一层后都接了category和mask两个分支，进行类别和位置的预测。
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/solo-head.png)
>
> (1) Category Branch
>
> - 每个网格预测类别，得到$C$维的置信得分，因此输出为$S\times S\times C $（前提假设：每个grid只属于一个单独的实例）
>
> (2) Mask Branch
>
> - 每个positive grid cell都会输出对应类别的instance mask，其中第k个通道负责预测第$(i,j)$个网格的instance mask，$k=i \cdot S+j$，因此输出为$H\times W\times S^2$
>
> 最后在semantic category和class-agnostic mask之间建立对应关系



##### Loss function

> $$
> L =L_{cate}+\lambda L_{mask}
> $$
>
> $$
> L_{mask}=\frac{1}{N_{p o s}} \sum_{k} \mathbb{1}_{\left\{\mathbf{p}_{i, j}^{*}>0\right\}} d_{m a s k}\left(\mathbf{m}_{k}, \mathbf{m}_{k}^{*}\right)
> $$
>
> 其中$d_{mask}$采用Dice loss



##### Improved

> **Decoupled Solo**: 
>
> - 通过将原始的输入$M\in R^{H\times W\times S^2}$替换成$X\in R^{H\times W\times S}$和$Y\in R^{H\times W\times S}$的element-wise multiplication，大大降低了输出的维度，同时在精度上没有损失
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/decoupled-solo.png)
>



#### 3. SOLOv2

`SOLOv2: Dynamic, Faster and Stronger`

> - 引入动态机制，学习卷积核权重
>
>   ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/solov2.png)
>
>   > kernel branch：输出$G$为$S\times S\times D$表示预测卷积核的权重，其中$D$表示参数的个数（若为$1\times 1$卷积的权重，则$D=E$；若为$3\times 3$卷积的权重，则$D=9E$）。这些权重取决于位置。
>   >
>   > feature branch：输出$F$为$H\times W\times E$，其中$E$是mask特征的维度
>
>   $G$和$F$做卷积得到mask为$H\times W\times S^2$
>
>   
>
> - 提出Matrix NMS，减少前向推理时间

#### 4. Conclusion

> - 只需要mask的标注信息，无需bbox标注信息
> - 将坐标回归转化为分类问题