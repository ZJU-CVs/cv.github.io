---
layout:     post
title:      Learning to Navigate for Fine-grained Classification
subtitle:   
date:       2020-08-18
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-grained
---



code: https://github.com/yangze0930/NTS-Net





### 1. Introduction

- 由于很难找到区分性的特征(discriminative features)用于充分表征物体的细微特征，因此细粒度分类具有挑战

- 细粒度分类模型的设计的关键在于准确识别图像中的信息区域。之前的一些方法采用human annotation，需要较高的标注成本。另一些方法采用无监督学习来定位信息区域，消除了对昂贵注释的需求，但是缺少一种机制来确保模型聚焦于正确的区域，通常会导致准确性下降。

- 本文提出了一种新颖的自监督机制(self-supervision mechanism)，可以有效地定位信息区域，而无需使用bounding-box/part 标注。

  > **假设：**有意义的局部信息可以辅助分类，局部信息加全局信息可以进一步提高分类效果
  >
  > 基于以上假设，首先需要一个方法来给出每个局部位置的信息量$I$，信息量越大表明此局部用于预测此类别的概率$\mathcal{C}$越高，此局部区域可以提升细粒度识别的效果；然后取$M$个信息量最大的区域加上整张图，输入预测网络来预测类别



### 2. Method

> 提出了NTS-Net (Navigator-Teacher-Scrutinizer)，采用multi-agent协同学习的方案来解决准确识别图像中信息区域的问题。
>
> - 假设所有区域都是矩阵，并将$\mathbb{A}$表示为给定图像中所有区域的集合
> - 定义信息函数: $\mathcal{I}\rightarrow (-\infty, \infty)$用于评估区域$R\in \mathbb{A}$
> - 定义置信函数$\mathcal{C}:\mathbb{A} \rightarrow [0,1]$作为分类器，以评估该区域所属真实类别的置信度
> - 信息量更多的区域具有较高的置信度，因此满足以下条件：
>
> $$
> for \ any \ R_1,R_2\in \mathbb{A},\ if \ \mathcal{C}(R_1)>\mathcal{C}(R_2), \ \mathcal{I}(R_1)>\mathcal{I}(R_2)
> $$
>
> 
>
> - 其中使用Navigator网络逼近信息函数$\mathcal{I}$，使用Teacher网络逼近之心函数$\mathcal{C}$



> 为了简化，在区域空间$\mathbb{A}$中选择$M$个区域$\mathbb{A}_M$，对于每个区域$R_i \in \mathbb{A}_M$
>
> - Navigator网络评估其信息量$\mathcal{I}(R_i)$
> - Teacher网络评估其置信度$\mathcal{C}(R_i)$。
> - 通过优化Navigator网络使{$\mathcal{I}(R_1),\mathcal{I}(R_2),...,\mathcal{I}(R_M)$}和{$\mathcal{C}(R_1),\mathcal{C}(R_2),...,\mathcal{I}(C_M)$}具有相同的顺序
>
> - Navigator网络随着Teacher网络的改进，将产生更多有效信息区域，以帮助Scrutinizer网络获得更好的细粒度分类结果



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/17.png" alt="img" style="zoom:50%;" />



##### Navigator and Teacher

- 使用FPN结构的RPN (region proposal network)，将图像作为输入，输出每个矩形区域的信息量$\mathcal{I}(R)$，并进行排序

$$
\mathcal{I}(R_1) \geq \mathcal{I}\left(R_{2}\right) \geq \cdots \geq \mathcal{I}\left(R_{A}\right)
$$

> 为了减少区域冗余，对区域采用非最大抑制 (NMS)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/18.png" alt="img" style="zoom:50%;" />

- 选取前$M$个信息区域{$R_1,R_2,\cdots,R_M$}，将其输入Teacher网络以获取置信度{$\mathcal{C}(R_1),\mathcal{C}(R_2),\cdots,\mathcal{C}(R_M)$}

- 通过优化Navigator网络使{$\mathcal{I}(R_1),\mathcal{I}(R_2),...,\mathcal{I}(R_M)$}和{$\mathcal{C}(R_1),\mathcal{C}(R_2),...,\mathcal{I}(C_M)$}具有相同的顺序

- 通过优化Teacher网络使得区域映射到真实的类别

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/19.png" alt="img" style="zoom:50%;" />



##### Scrutinizer

- 随着Navigator网络训练逐渐收敛，将产生信息丰富的对象特征区域，以帮助Scrutinizer网络作出决策
- 使用前K个信息区域以及完整图像作为输入来训练Scrutinizer网络，这K个区域用于促进细粒度识别（可以减少类内方差，并可能在正确的标签上产生更高的一致性分数）



##### Network architecture

> Navigator Network (网络参数为$W_\mathcal{I}$)

- 使用具有横向连接的自上而下结构来检测多尺度区域
- 得到一系列不同空间分辨率的特征图，使用不同图层的多尺度特征图，可以得到不同尺度和比例的区域信息（较大的特征图中锚点对应与较小的区域）

> Teacher Network (网络参数为$W_C$)

- 除了共享的特征提取器外，连接一个全连接层



> Scrutinizer Network (网络参数为$W_S$)

- 从Navigator网络接收到前K个信息区域后，将K个区域的大小调整为预先定义的大小，并输入特征提取器生成特征向量
- 将K个局部特征与输入突袭那个的特征连接起来，并将其输入全连接层



`Navigator`、`Teacher`、`Scrutinizer`中的特征提取器共享参数

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/20.png" alt="img" style="zoom:50%;" />





##### Loss function and Optimization

(1) Navigation Loss

> $$
> L_{I}(I, C)=\sum_{(i, s): C_{i}<C_{s}} f\left(I_{s}-I_{i}\right)
> $$
>
> > 其中函数$f(x)=max${$1-x,\ 0$}
> >
> > 该损失函数表示如果置信度$C_{i}<C_{s}$，那么信息量$I_s$就要比$I_i$大，以保证置信度高的信息量大



(2) Teaching Loss

> $$
> L_{c}=-\sum_{i=1}^{M} \log C\left(R_{i}\right)-\log C(X)
> $$
>
> > $-\sum_{i=1}^{M} \log C\left(R_{i}\right)$表示所有区域的交叉熵损失的总和
> >
> > $-\log{C}(X)$表示整张图的交叉熵损失



(3) Scrutinizing Loss

> $$
> L_s=-\log {S}(X,R_1,R_2,\cdots,R_K)
> $$
>
> > 表示将各个区域和整张图的特征进行拼接后计算交叉熵损失



(4) Total Loss

> $$
> L_{total}=L_I+\lambda L_s+\mu L_c
> $$
>
> 



##### Training Process

> 对于图像X，生成anchors，计算每个anchor区域的信息量，使用NMS技术减少冗余；选择M个信息量最大的区域，计算每个区域的置信度，将每个区域和整个图像的特征进行拼接，计算总体损失；误差反向传播，对$W_{\mathcal{I}},W_{\mathcal{C}},W_{\mathcal{S}}$进行梯度更新

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/24.png" alt="img" style="zoom:40%;" />





### 3. Experiments

##### Dataset

> Caltech-UCSD Birds
>
> Stanford Cars 
>
> FGVC Aircraft

##### Quantitative Results

> 进行定量实验，与其他方法进行对比，性能最优
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/21.png" alt="img" style="zoom:50%;" />



##### Ablation Study

> 进行消融实验，在没有Teacher网络时准确率下降到83.3%
>
> NS-Net准确率从下降的原因可能是在没有Teacher网络的情况下，Navigator提取的区域都是噪声，影响了整个网络。
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/22.png" alt="img" style="zoom:50%;" />



##### Qualitative Results

> 使用红色，橙色，黄色，绿色矩形来表示Navigator网络建议的前四个信息区域，其中红色矩形表示最多。 内容丰富的一个。 可以看出，局部区域确实可以为细粒度分类提供信息。
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/23.png" alt="img" style="zoom:30%;" />
>
> 