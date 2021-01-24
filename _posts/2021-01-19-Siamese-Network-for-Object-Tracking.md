---
layout:     post
title:      Siamese Network for Object Tracking
subtitle:   
date:       2021-01-19
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Object Tracking
    - Overview
---

### 1. Background

#### Object Tracking

> 使用视频序列第一帧的图像(包括bounding box的位置)，来找出目标在后面序列帧中的位置

#### Siamese Network

> 孪生网络是一种基于度量学习(metric learning)的方法
>
> - 孪生网络不是一种网络结构，而是一种网络架构(框架)
> - 孪生网络的backbone可以是CNN(用于图像相似性度量)，也可以是LSTM(用于自然语言语义的相似性分析)
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/siamese-network.jpg" alt="img" style="zoom:67%;" />
>
> - **伪孪生网络(pseudo siamese network)**
>
>   > (两个输入的网络权重不共享)
>   >
>   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/1.png" alt="img" style="zoom:33%;" />
>
> 
>
> - 损失函数
>
>   - Contrastive Loss
>
>     > 《Dimensionality Reduction by Learning an Invariant Mapping》
>     > $$
>     > D_{W}\left(\vec{X}_{1}, \vec{X}_{2}\right)=\left\|G_{W}\left(\vec{X}_{1}\right)-G_{W}\left(\vec{X}_{2}\right)\right\|_{2} \\
>     > \mathcal{L}(W)=\sum_{i=1}^{P} L\left(W,\left(Y, \vec{X}_{1}, \vec{X}_{2}\right)^{i}\right)\\
>     > L\left(W,\left(Y, \vec{X}_{1}, \vec{X}_{2}\right)^{i}\right)=(1-Y) L_{S}\left(D_{W}^{i}\right)+Y L_{D}\left(D_{W}^{i}\right)\\
>     > \begin{array}{l}
>     > L\left(W, Y, \vec{X}_{1}, \vec{X}_{2}\right)= (1-Y) \frac{1}{2}\left(D_{W}\right)^{2}+(Y) \frac{1}{2}\left\{\max \left(0, m-D_{W}\right)\right\}^{2}
>     > \end{array}
>     > $$
>     >
>     > - $D_W$是$X_1$和$X_2$在隐空间的欧式距离
>     > - Y为成对标签，若$X_1$和$X_2$为一类则$Y=0$；否则$Y=1$
>     > - 当两个不同类的$X_1$和$X_2$在隐空间中的距离大于m，则不再做优化
>
>   - Triplet Loss
>
>     > 《FaceNet: A Unified Embedding for Face Recognition and Clustering》
>     >
>     > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/2.png" alt="img" style="zoom:67%;" />
>     > $$
>     > L = \sum_{i}^{N}\left[\left\|f\left(x_{i}^{a}\right)-f\left(x_{i}^{p}\right)\right\|_{2}^{2}-\left\|f\left(x_{i}^{a}\right)-f\left(x_{i}^{n}\right)\right\|_{2}^{2}+\alpha\right]_{+}
>     > $$
>     >
>     > - 最小化锚点和具有相同身份的正样本之间的距离，最大化锚点和具有不同身份的负样本之间的距离



### 2. Methods

#### Single Object Tracking（SOT）

https://blog.csdn.net/WZZ18191171661/article/details/88369667

##### Siam-FC

> 主要构成部分：
>
> - 特征提取网络AlexNet
> - 互相关运算网络
>
> 
>
> 引入FC (fully-convolutional)的优势：候选图像的尺寸可以大小不同
>
> 
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/siam-FC.png)
>
> > - 输入为**模板图像z**（大小为127x127x3) + **搜索图像x** (大小为255x255x3)
> > - 经过特征提取网络 ，论文中采用了较为简单的**AlexNet**，输出为 $\varphi(z)$以及 $\varphi(x)$
> > - 互相关运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) （论文中的cross-correlation)，实质上是以$\varphi(x)$为**特征图**，以 $\varphi(z)$ 为**卷积核**进行的**卷积互运算**
> >   
> > - 输出 score map ，大小为（17x17x1)，score map反映了$\varphi(z)$与$\varphi(x)$中每个对应关系的相似度，相似度越大越有可能是同一个物体
> >   
> > - 损失函数细节：
> >   $$
> >   \begin{array}{l}
> >   \ell(y, v)=\log (1+\exp (-y v)) \\
> >   L(y, v)=\frac{1}{|\mathcal{D}|} \sum_{u \in \mathcal{D}} \ell(y[u], v[u]) \\
> >   y[u]=\left\{\begin{array}{ll}
> >   +1 & \text { if } k\|u-c\| \leq R \\
> >   -1 & \text { otherwise }
> >   \end{array}\right.
> >   \end{array}
> >   $$
> >
> >   - R表示只要在正确的一个半径内，都算预测正确
>
> 缺点：
>
> - 跟踪对象的外观不是一成不变的，可能会有放大，缩小，变形等等，但是bbox却是一直不变的，这就让跟踪效果不理想
> - 只把首帧的一个框作为GT，供提取特征的数据过于单一，很容易误检，比如跟踪人时，当多个人出现在画面中时，很容易跟丢



##### Siam-RPN

> Siam-FC存在的问题：
>
> - *没有边界框回归，因此需要进行多尺度测试*
> - *性能（精度和鲁棒性）不及最新的**相关滤波器方法***
>
> Siam-RPN的创新：
>
> - Siam-FC把输出直接用来进行相关滤波（xorr），而Siam-RPN接入的是一个RPN（分类和回归）
> - 将RPN的思路应用到目标跟踪领域中，在提速的同时提升了精度；
> - 引入1x1卷积层对网络的通道进行升维处理；
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/siam-RPN.png)
>
> - 特征图会通过一个卷积层进入到两个分支中
>
>   - 在Classification Branch中
>
>     > `17*17*2k`做的是区分目标和背景，其会被分为k个groups，每个group会有一层正例，一层负例。最后会用`softmax + cross-entropy loss`进行损失计算。
>
>   - 在Regression Branch中
>
>     > `17*17*4k`同样会被分为k groups，每个group有四层，分别用于预测`dx,dy,dw,dh`（偏移量）			
>   
>    `注：k为生成的anchors的数目`
>
> - Training的一些细节
>
>   - 在anchor选择上，相比目标检测任务，跟踪任务可以选择少一点的anchor。因为同一个物体在两帧里的形变不大。因此文中选择了一种尺寸的anchor，并设置4种长宽比[0.33, 0.5, 1, 2, 3]
>
>   - 正负样本选择：通过设置阈值，比较anchor与GT的IOU，如果超过阈值，判别为正样本，反之为负样本（为了防止正负样本失衡，将正负样本的比例设置为1:3）
>
> - Testing
>
>   > 测试阶段，模版分支的第一帧特征图计算好保存起来，作为RPN中的分类和回归卷积核（视频每次都是对第一帧中的目标框进行跟踪）
>   >
>   > - 此时的检测分支变成了一个简单的检测网络，在经过特征提取网络后，分别经过两个卷积层，可以得到网络预测出的所有框及其对应分数。
>   > - 从中选出分数最高的对应的框，作为最终网络预测的目标位置。
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/3.png" alt="img" style="zoom:67%;" />
>
>   **one-shot的体现：**在detection任务重，所有的类别在训练中都出现过多次；但对于跟踪任务，所跟踪的目标在训练集中可能没有见过，只有在推理时才第一次见。



##### Siam-RPN++

> 创新点：**提出了一种打破平移不变性限制的采样策略**
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/siam-RPN-pp.png)



#### Multiple Object Tracking（MOT）

- 在单目标跟踪中，目标的出现是预先知道的，而在多目标跟踪中，需要一个检测步骤来识别出进入或离开场景的目标。同时跟踪多个目标的主要困难来自于各种各样的遮挡以及物体之间的相互重叠，有时物体间也会有相似的外观。因此，仅仅使用SOT模型直接解决MOT问题，往往会导致目标漂移和大量的ID切换错误，因为此类模型往往难以区分外观相似的类内物体
- 单目标跟踪的主流做法基本上是在一个局部小区域上操作，而多目标跟踪是全图操作

- 绝大多数MOT算法主要包括四个步骤：**①检测 ②特征提取、运动预测 ③相似度计算 ④数据关联**



https://www.bilibili.com/read/cv6731832/