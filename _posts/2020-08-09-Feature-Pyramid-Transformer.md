---
layout:     post
title:      Feature Pyramid Transformer
subtitle:   
date:       2020-08-09
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Transformer
---



**论文地址：https://arxiv.org/abs/2007.09451**

**代码地址：https://github.com/ZHANGDONG-NJUST/FPT**



**Abstract:**

> 跨空间和尺度的特征交互是现代视觉识别系统的基础，因为它们引入了有益的视觉环境。通常空间上下文信息被动地隐藏在卷积神经网络不断增加的感受野中，或者被non-local卷积主动地编码。但是，**non-local空间交互作用并不是跨尺度的，因此它们无法捕获在不同尺度中的对象（或部分）的非局部上下文信息。**为此，本文提出了一种在空间和尺度上完全活跃的特征交互，称为特征金字塔Transformer（FPT）。它**通过使用三个专门设计的Transformer，以自上而下和自下而上的交互方式，将任何一个特征金字塔变换成另一个同样大小但具有更丰富上下文的特征金字塔。**FPT作为一个通用的视觉框架，具有合理的计算开销。最后，本文在实例级（即目标检测和实例分割）和像素级分割任务中进行了广泛的实验，使用不同的主干和头部网络，并观察到和所有baseline和最先进的方法相比均有所改进。



### 1. Introduction

- Modern visual recognition systems

  > 由于卷积神经网络（CNN）的层次结构，通过pooling池化、stride或空洞卷积等操作，在逐渐变大的感受野（绿色虚线矩形）中对上下文进行编码。因此，通过最后一个特征图进行预测是基于丰富的上下文信息。如图(a)

  

- Scale also matters

  > - 传统的解决方案是对同一图像进行多尺度图像金字塔的堆积，其中较高/较低的层次采用较低/较高分辨率的图像进行输入。因此，不同尺度的物体在其相应的层级中被识别。然而，图像金字塔增加了CNN前向传递的耗时，因为每个图像都需要一个CNN来识别。
  >
  > - CNN提供了一种特征金字塔FPN (Feature Pyramid Network)，即通过低/高层次的特征图代表高/低分辨率的视觉内容，而不需要额外的计算开销。可以通过使用不同级别的特征图来识别不同尺度的物体，即小物体在较低层级中进行识别，大物体在较高层级中进行识别。如图(b)



- Sometimes the recognition

  > 尤其是像语义分割这样的像素级标签，需要结合多个尺度的上下文。要对显示的帧区域的像素赋予标签，从较低的层次上看，实例本身的局部上下文就足够了；但对于类外的像素，需要同时利用局部上下文和较高层次的全局上下文。如图(c)



- 上述non-local context通过non-local卷积和自注意力进行建模，希望捕获多个对象同时发生时的相互关系，如图(d)，有助于利用其他对象的信息来帮助当前对象的识别
- 本文认为non-local交互应该在交互对象的相应尺寸上发生，而不是仅在一个统一的尺度上发生，如图(e)

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/1.png)

---

- 本文提出了一种称为特征金字塔转换器Transformer（FPT）的新颖特征金字塔网络，用于视觉识别任务，例如实例级（即目标检测和实例分割）和像素级分割任务。

  > - FPT的输入是一个特征金字塔，而输出是一个变换的金字塔，其中**每个level都是一个更丰富的特征图，跨空间和尺度的non-local交互进行了编码。**
  > - 将特征金字塔附加到任何特定任务的头部网络。即FPT中特征之间的交互采用了 transformer-style。具有query，key和value操作，在选择远程信息进行交互时非常有效，从而可以调整我们的目标：**以适当的规模进行non-local交互**。另外，像其他任何transformer模型一样，使用TPU可以减轻计算开销。



- FPT主要是由三种transformer的设计构成：

  > 1)**自变换器Self-Transformer(ST)：**基于经典的同级特征图内的non-local交互，输出与输入具有相同的尺度。
  >
  > 2) **Grounding Transformer(GT)：**以**自上而下**的方式，输出与下层特征图具有相同的比例。直观地说，将上层特征图的 "concept "与下层特征图的 "pixels "关联。特别是，由于没有必要使用全局信息来分割对象，而局部区域内的上下文在经验上更有参考价值，因此，还设计了一个**locality-constrained的GT，以保证语义分割的效率和准确性。**
  >
  > 3) **Rendering Transformer(RT)**。以**自下而上**的方式，输出与上层特征图具有相同的比例。直观地说，将上层 "concept "与下层 "pixels"的视觉属性进行渲染。这是一种**局部交互**，因为用另一个远处的 "pixels "来渲染一个 "object "是没有意义的。**每个层次的转换特征图（红色、蓝色和绿色）被重新排列到相应的map大小，然后与原始map连接，然后再输入到卷积层，将它们调整到原始 "thickness"。**



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/2.png" alt="img" style="zoom:50%;" />



### 2. Method

> **Feature Pyramid Transformer** (FPT)使特征能够在空间和尺寸上进行交互，具体包括三个transformers：
>
> - self-transformer
> - grounding transformer
> - rendering transformer
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/3.png)



#### Non-Local Interaction Revisited

`传统的Non-Local Interaction 回顾`

> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/4.png" alt="img" style="zoom:50%;" />
>
> > $q_i=f_q(X_i)\in Q$表示第$i$个query；$k_j=f_k(X_j)\in K$和$v_j=f_v(X_j)\in V$表示第$j$个key-value对
> >
> > $f_q(\cdot)$，$f_k(\cdot)$和$f_v(\cdot)$分别表示query，key和value的转换函数
> >
> > $X_i$和$X_j$分别是$X$中的第$i$和第$j$个特征位置
>
> 
>
> 用提出non-local的[论文](https://arxiv.org/pdf/1711.07971v3.pdf)表示如下：     
> $$
> y_{i}=\frac{1}{\mathcal{C}(x)} \sum_{\forall j} f\left(x_{i},x_{j}\right)g\left(x_{j}\right)
> $$
>
> > $f(x_i,x_j)$用来计算$i$与所有可能关联的位置$j$之间pairwise关系
> >
> > $g(x_j)$用于计算输入信号在位置$j$的特征值
> >
> > $\mathcal{C}(x)$为归一化参数



#### I. Self-Transformer

`Self-Transformer(ST)旨在捕获一个特征图上同时出现的对象特征`

> ST是一种修改后的non-local交互，输出的特征图与其输入特征图的尺度相同。
>
> - 使用Mixture of Softmaxes(MoS)作为归一化函数，事实证明它比标准的Softmax在图像上更有效。具体来说，首先将query和key划分为N个部分，然后使用$F_{sim}$计算每对图像的相似度分数。MoS的本质是使用多个不同的softmax来增加模型的表达能力
> - 基于MoS的归一化函数$F_{mos}$表达式如下：
>
> $$
> F_{\operatorname{mos}}\left(s_{i, j}^{n}\right)=\sum_{n=1}^{\mathcal{N}} \pi_{n} \frac{\exp \left(s_{i, j}^{n}\right)}{\sum_{j} \exp \left(s_{i, j}^{n}\right)}
> $$
>
> > $\pi_n=Softmax(w_n^T\overline{\mathbf{k}})$，其中$w_n$是可学习的用于归一化的线性向量，$\overline{\mathbf{k}}$是$k_j$的所有位置的算术平均值
>
> - Self-Transformer可以表达为：
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/5.png" alt="img" style="zoom:50%;" />



#### II. Grounding Transformer

`Grounding Transformer(GT)为自上而下的non-local交互`

> 将上层特征图$X^c$中的 "concept "与下层特征图$X^f$中的 "pixels "进行对接。输出特征图$\hat{X}^f$与$X^f$具有相同的尺度。
>
> - 一般来说，不同尺度的图像特征提取的语义或上下文信息不同，或者两者都有。
> - 根据经验，当两个特征图的语义信息不同时，**euclidean距离的负值比点积更能有效地计算相似度**，因此使用euclidean距离$F_{edu}$作为相似度函数，其表达方式为：
>
> $$
> F_{e u d}\left(\mathbf{q}_{i}, \mathbf{k}_{j}\right)=-\left\|\mathbf{q}_{i}-\mathbf{k}_{j}\right\|^{2}
> $$
>
> - Grounding Transformer可以表示为：
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/6.png" alt="img" style="zoom:50%;" />
>
>   > - $q_i=f_q(X_i^f)$，$k_j=f_k(X^c_j)$，$v_j=f_v(X^c_j)$
>   > - $X^f_i$是$X^f$中的第$i$个特征位置，$X^c_j$是$X^c$中的第$j$个特征位置
>
>   
>
> - 在特征金字塔中，高/低层次特征图包含大量全局/局部图像信息。从经验上讲，**查询位置周围的局部区域内的上下文会提供更多信息**，这些常规的跨尺度交互（如 summation/concatenation)）在现有的分割方法中体现出了有效性。然而**对于通过跨尺度特征交互的语义分割，没有必要使用全局信息来分割图像中的两个对象，**因此引入了<u>局域性GT转换</u>。



`Locality-constrained  Grounding  Transformer`

> - 引入了局域性GT转换进行语义分割，每个q（即低层特征图上的红色网格）在中心区域的局部正方形区域内与k和v的一部分（即高层特征图上的蓝色网格）相互作用，对于k和v超出索引的位置，改用0值。



#### III. Rendering Transformer

`Rendering Transformer（RT）以自下而上的方式交互`

> 旨在通过将视觉属性合并到low-level 的 “pixels” 中来渲染high-level的“concept”。
>
> - RT是一种局部交互，其中该局部是基于**渲染具有来自另一个遥远对象的特征或属性的“object”是没有意义的**这一假设。
>
> - RT不是按像素进行的，而是按整个特征图进行的。具体步骤如下：
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/7.png" alt="img" style="zoom:60%;" />
>
>   > - 高层特征图定义为$Q$，低层特征图定义为$K$和$V$
>   > - $Q$和$K$之间的交互是以channel-wise的注意力方式进行，$K$首先通过全局平均池化(GAP)计算权重$w$对$Q$进行加权得到$Q_{att}$
>   > - $Q_{att}$通过3×3卷积进行优化，$V$通过3×3步长卷积(stride convolution)来缩小特征尺寸(灰色方块)，若$Q$和$V$的尺寸相同，则stride=1。
>   > - 最后，将优化后的$Q_{att}$和$V_{dow}$**相加**，得到输出$\hat{X}^c$



### 3. Experiment

`建立特定的FPT网路来处理目标检测、实例分割和语义分割，每个FPT网络都有四个部分组成：用于特征提取的backbone(如ResNet-50)、特征金字塔构建模块(如UFP、BFP)、所提的feature transformer和用于特定任务的head network(如Faster R-CNN head)`



#### Instance-Level Recognition

`Object Detection & Instance Segmentation`

> Dataset: MS-COCO 2017
>
> Backbone: 
>
> - ResNet-50 -------in ablation study
> - ResNet-101, Non-local Network (NL-ResNet-101), Global Context Network (GC-ResNet-101) and Attention Augmented Network (AA-ResNet-101)    -------- to compare to SOTA

- 方法对比，将FPT与最新的跨尺度特征金字塔交互 (cross-scale feature pyramid interactions) 进行比较

  > 最新的跨尺度特征金字塔交互包括FPN、Bottom-up Path Aggregation(BPA)、Bi-direction Feature Interaction(BFI)
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/15.png" alt="img" style="zoom:50%;" />

- 进行消融实验，评估三个单独的tranformer(ST, GT,RT)和组合后的精度和模型效率(eﬃciency)，其中BFP (Bottom-up Feature Pyramid) 为baseline

	> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/8.png" alt="img" style="zoom:80%;" />

- 进行消融实验，研究FPT上加入tricks

  > SBN (SyncBN)和DropBlock (卷积正则化)的影响
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/9.png" alt="img" style="zoom:80%;" />

- 在数据集MS-COCO 2017测试集上对模型与SOTA方法进行对比将FPT与最新的跨尺度特征金字塔交互 (cross-scale feature pyramid interactions) 进行比较
	
	> 最新的跨尺度特征金字塔交互包括FPN、Bottom-up Path Aggregation(BPA)、Bi-direction Feature Interaction(BFI)，同时加入了Augmented Head(AH) 和 Multi-scale Training(MT)进行实验
	>
	> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/10.png" alt="img" style="zoom:80%;" />

- 可视化结果

  > 红色矩形突出显示了该区域FPT优越的预测性能
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/12.png" alt="img" style="zoom:80%;" />



#### Pixel-Level Recognition

`semantic segmentation`

> Dataset: Cityscapes; ADE20K; LIP; PASCAL VOC 2012
>
> Backbone: ResNet-101

- 消融实验，UFP作为baseline，分别应用所提的tranformers (+ST，+LGT和+RT )进行消融实验，评估精度和模型效率

  > 测试数据集为Cityscape validation set
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/14.png" alt="img" style="zoom:50%;" />

- 与SOTA 方法进行比较
	
	> 将FPT应用于三种特征金字塔模型（UFP, PPM和ASPP），与最先进的(SOTA)像素级特征金字塔交互方法，如Object Context Network (OCNet).
	>
	> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/11.png" alt="img" style="zoom:70%;" />
	>
	> - 红色表示best，蓝色表示second best

- 可视化结果

  > 红色矩形突出显示了该区域FPT优越的预测性能
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/13.png" alt="img" style="zoom:70%;" />

### 4. Conclusion

> - 提出了一种有效的特征交互方法(FPT)，该方法由三个carefully-designed transformers组成，分别对特征金字塔中的explicit self-level、top-down和bottom-up信息进行编码。
>
> - FPT不会更改特征金字塔的大小，因此具有通用性，易于即插即用（plug-and-play）
> - FPT在baseline和SOTA均取得了改进，证明了其高效性和强大的应用能力

