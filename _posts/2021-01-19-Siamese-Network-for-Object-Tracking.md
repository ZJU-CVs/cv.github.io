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

## 1. Background

#### Object Tracking

##### 目标跟踪分类

> 根据特定任务可分为：
>
> - **单目标跟踪** - 给定一个目标，追踪这个目标的位置。
> - **多目标跟踪** - 追踪多个目标的位置
> - **Person Re-ID** - 行人重识别，是利用计算机视觉技术判断图像或者视频序列中是否存在特定行人的技术。广泛被认为是一个图像检索的子问题。给定一个监控行人图像，检索跨设备下的该行人图像。旨在弥补固定的摄像头的视觉局限，并可与行人检测/行人跟踪技术相结合。
> - **MTMCT** - 多目标多摄像头跟踪（Multi-target Multi-camera Tracking），跟踪多个摄像头拍摄的多个人
> - **姿态跟踪** - 追踪人的姿态
>
> 
>
> 按照任务计算类型可分为：
>
> - **在线跟踪** - 在线跟踪需要实时处理任务，通过过去和现在帧来跟踪未来帧中物体的位置。
> - **离线跟踪** - 离线跟踪是离线处理任务，可以通过过去、现在和未来的帧来推断物体的位置，因此准确率会在线跟踪高
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/overview.png" alt="img" style="zoom:33%;" />

##### 目标跟踪难点

>- **形态变化**:  姿态变化是目标跟踪中常见的干扰问题。运动目标发生姿态变化时, 会导致它的特征以及外观模型发生改变, 容易导致跟踪失败。例如:体育比赛中的运动员、马路上的行人。
>- **尺度变化**:  尺度的自适应也是目标跟踪中的关键问题。当目标尺度缩小时, 由于跟踪框不能自适应跟踪, 会将很多背景信息包含在内, 导致目标模型的更新错误:当目标尺度增大时, 由于跟踪框不能将目标完全包括在内, 跟踪框内目标信息不全, 也会导致目标模型的更新错误。因此, 实现尺度自适应跟踪是十分必要的。
>- **遮挡与消失**:  目标在运动过程中可能出现被遮挡或者短暂的消失情况。当这种情况发生时, 跟踪框容易将遮挡物以及背景信息包含在跟踪框内, 会导致后续帧中的跟踪目标漂移到遮挡物上面。若目标被完全遮挡时, 由于找不到目标的对应模型, 会导致跟踪失败。
>- **图像模糊**: 光照强度变化, 目标快速运动, 低分辨率等情况会导致图像模型, 尤其是在运动目标与背景相似的情况下更为明显。因此, 选择有效的特征对目标和背景进行区分非常必要。

##### 目标跟踪算法

> - 经典跟踪算法：光流法、粒子滤波...
>
> - 基于核相关滤波的算法
>
> - **基于深度学习的算法**
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/object-tracking/method.png" alt="img" style="zoom:50%;" />



#### Siamese Neural Network (SNN)

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



## 2. Methods

### Single Object Tracking（SOT）

#### Introduction

单目标跟踪任务：使用视频序列第一帧的图像(包括bounding box的位置)，来找出目标在后面序列帧中的位置



#### Methods

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



### Multiple Object Tracking（MOT）

#### Introduction

- 多目标追踪是指输入一段视频，在没有任何对目标的先验知识的前提下，追踪其中一类或多类物体（如行人追踪、车辆追踪）

- 与单目标追踪(SOT)的区别

  - 多目标追踪不仅需要输出每一帧中每个目标的Bbox，还需要对每个box标注target ID，以此来区分intra-class objects
  - 在单目标跟踪中，目标的出现是预先知道的（训练集会给出第一帧的bbox），而在多目标跟踪中，需要一个检测步骤来识别出进入或离开场景的目标。
  - 单目标跟踪的主流做法基本上是在一个局部小区域上操作，而多目标跟踪是全图操作

  

- 因此，仅仅使用SOT模型直接解决MOT问题，此类模型往往难以区分外观相似的类内物体，从而导致目标漂移和大量的ID切换错误。

- MOT的难点：
  - 各种各样的遮挡问题以及物体之间的相互重叠
  - 同类物体间ID标注错误



#### Methods

(1) 算法一般可以分为四部分构成：

> - Detection stage：检测当前帧中的物体以给出检测对象的bounding boxes。一般可采用Faster R-CNN、SSD、YOLO等检测端。
> - Feature extraction/motion prediction stage：根据检测端给出的定位框提取特征。
> - Affinity stage：计算前后帧之间各定位框所框对象之间的相似度。
> - Association stage：根据相似度矩阵给出当前帧各检测框所对应的ID。

(2) 在计算MOT算法的benchmark时，需要考虑以下信息：

> - 最多跟踪路径（Mostly Tracked）：number of ground-truth trajectories that are correctly tracked in at least 80% of the frames
> - 最多丢失路径（Mostly Lost）：number of ground-truth trajectories that are correctly tracked in less than 20% of the frames
> - Fragments：trajectory hypotheses which cover at most 80% of a ground-truth trajectory
> - False trajectories
> - ID switches：当对象被正确跟踪，但相应的ID与其他对象错误交换的总次数

(3) benchmark datasets

- **MOT challenge** :行人检测
- **KITTY** : 行人和车辆检测，moving camera，通过在城市里开车收集
- **UA-DETRAC**：车辆检测数据集



**ref：**https://zhuanlan.zhihu.com/p/108670114

