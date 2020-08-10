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
  > **3) Rendering Transformer(RT)**。以**自下而上**的方式，输出与上层特征图具有相同的比例。直观地说，将上层 "concept "与下层 "pixels"的视觉属性进行渲染。这是一种**局部交互**，因为用另一个远处的 "pixels "来渲染一个 "object "是没有意义的。**每个层次的转换特征图（红色、蓝色和绿色）被重新排列到相应的map大小，然后与原始map连接，然后再输入到卷积层，将它们调整到原始 "thickness"。**

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/2.png)

### 3. Experiment

> FPT可以极大地改善传统的检测/分割网络：1）在MS-COCO test-dev数据集上，用于框检测的百分比增益为8.5％，用于遮罩实例的mask AP值增益为6.0％；2）对于语义分割，分别在Cityscapes和PASCAL VOC 2012 测试集上的增益分别为1.6％和1.2％mIoU；在ADE20K 和LIP 验证集上的增益分别为1.7％和2.0％mIoU。

### 1、Non-Local Interaction Revisited

**传统的Non-Local Interaction**



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyEpNMMW42NEPhTGcKnhvLibjFo9WPzonFHxy9wnEly0et772E8IacKCA/640?wxfrom=5&wx_lazy=1&wx_co=1)

**2、Self-Transformer**

**
**

自变换器(Self-Transformer，ST)的目的是**在同一张特征图上捕获共同发生的对象特征**。如图3(a)所示，ST是一种修改后的非局部non-local交互，输出的特征图与其输入特征图的尺度相同。与其他方法区别在于，作者部署了Mixture of Softmaxes(MoS)作为归一化函数，事实证明它比标准的Softmax在图像上更有效。具体来说，首先将查询q和键k划分为N个部分。然后，使用Fsim计算每对图像的相似度分数。基于MoS的归一化函数Fmos表达式如下：



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyscxeSdqgIY3eQpFzGnIfT0Zxt4ehVY7IO14yYRRLB1e9nA6KJjbkmA/640?wxfrom=5&wx_lazy=1&wx_co=1)

自变换器可以表达为：



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyAGL3zfD4qMibMjpOfCg8jwKur4pwjicR3shnpySHTkgPDibmKibdMf7ibkQ/640?wxfrom=5&wx_lazy=1&wx_co=1)



**3、Grounding Transformer**



Grounding Transformer(GT)可以归类为**自上而下的非局部non-local交互**，它将上层特征图Xct中的 "概念 "与下层特征图Xf中的 "像素 "进行对接。输出特征图与Xf具有相同的尺度。一般来说，不同尺度的图像特征提取的语义或语境信息不同，或者两者兼而有之。此外，根据经验，当两个特征图的语义信息不同时，**euclidean距离的负值比点积更能有效地计算相似度**。所以我们更倾向于使用euclidean距离Fedu作为相似度函数，其表达方式为：



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyibdgdsiaZ2tZozxiauzUUB2Z6CAsXX7aVmd1CyGBug8HeflQOJIqmcicdA/640?wxfrom=5&wx_lazy=1&wx_co=1)



于是，Grounding Transformer可以表述为：

![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyQwtJHckgkCOXjklkibjnHgibFczJRzyNp93YFjMbXr66eE8yibibuy9cNQ/640?wxfrom=5&wx_lazy=1&wx_co=1)

在特征金字塔中，高/低层次特征图包含大量全局/局部图像信息。然而，**对于通过跨尺度特征交互的语义分割，没有必要使用全局信息来分割图像中的两个对象。**从经验上讲，**查询位置周围的局部区域内的上下文会提供更多信息**。这就是为什么常规的跨尺度交互（例如求和和级联）在现有的分割方法中有效的原因。如图3（b）所示，它们本质上是隐式的局部non-local样式，但是本文的默认GT是**全局交互的**。



![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/backbone/3.png)



Locality-constrained  Grounding  Transformer。因此，作者引入了局域性GT转换进行语义分割，这是一个明确的局域特征交互作用。如图3（c）所示，每个q（即低层特征图上的红色网格）在中心区域的局部正方形区域内与k和v的一部分（即高层特征图上的蓝色网格）相互作用。坐标与q相同，边长为正方形。特别是，对于k和v超出索引的位置，改用0值。

**4、Rendering Transformer**

Rendering Transformer（RT）**以自下而上的方式工作**，旨在通过将视觉属性合并到低层级“像素”中来渲染高层级“概念”。如图3（d）所示，RT是一种局部交互，其中该局部是基于渲染具有来自另一个遥远对象的特征或属性的“对象”是没有意义的这一事实。

在本文的实现中，RT不是按像素进行的，而是按整个特征图进行的。具体来说，高层特征图定义为Q，低层特征图定义为K和V，为了突出渲染目标，**Q和K之间的交互是以通道导向的关注方式进行的，K首先通过全局平均池化(GAP)计算出Q的权重w。然后，加权后的Q(即Qatt)通过3×3卷积进行优化，V通过3×3卷积与步长来缩小特征规模**(图3(d)中的灰色方块)。最后，将优化后的Qatt和下采样的V(即Vdow)**相加**，再经过一次3×3卷积进行细化处理。



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyZV8mwtWXa1LiabtzeUwZz8KDe2LZcRp4JgZQrSTicZ3uiaYgYDr2MCfXQ/640?wxfrom=5&wx_lazy=1&wx_co=1)

实验与结果







**消融实验：**



**![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyRcQicaiaicIZX3KZLsTzuy3TdUWm5ib1YXibfwRAMg6tnwkCXqLXwZcicPCw/640?wxfrom=5&wx_lazy=1&wx_co=1)**



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyRxiaYegGBNAlkQwCfiaiaJTATKCYiaVNnNgOL9GlVlu0K2P6z67olfduuw/640?wxfrom=5&wx_lazy=1&wx_co=1)

**对比实验**



![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyYEfPZqGfUvviapBdL0LdyeEyutHEBIIxCiaelMMZHVMoOdRjLKLcaRkg/640?wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyTAkUFbqaXV7Fj6vtaSl3FhYRqnKibQGmV4BMNnAJlwHPknszLrXo97g/640?wxfrom=5&wx_lazy=1&wx_co=1)



**可视化对比**



**![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyuZYpeh9hWKSShVfTUl5fRwrBGg3lVN9L6cUmRzLpxyABmM2LhfkAWg/640?wxfrom=5&wx_lazy=1&wx_co=1)**

**![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyePd96t2VsTEysM9ER2wKsNicdZKnU0oZQwPibkZicUVq9YickC3S3NcnWg/640?wxfrom=5&wx_lazy=1&wx_co=1)**

**![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZywo1Eic81QL9jaq4z5kVRicibUQnxWbNTGtjPZALUYFMFDArNwiaWzooJZQ/640?wxfrom=5&wx_lazy=1&wx_co=1)**

**![img](https://mmbiz.qpic.cn/mmbiz_png/Z8w2ExrFgDwhE9yjDyeymx8SXwCUlUZyRlGYGQX4423QcMiaY6wk0ATo8vQ5fZbHWkaFzjEGSTHO2c8224qia6SQ/640?wxfrom=5&wx_lazy=1&wx_co=1)**

#### 