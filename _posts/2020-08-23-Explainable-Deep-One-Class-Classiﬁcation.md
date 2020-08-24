---
layout:     post
title:      Explainable Deep One-Class Classiﬁcation
subtitle:  
date:       2020-08-23
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Paper
    - Anomaly Detection

---



#### 1. Introduction

> 异常检测任务是识别一组数据中的异常，而目前基于异常分数的一些方法在可解释性方面研究有限

- DSVDD作为一种常见的异常检测方法，能够将正常数据聚集到预定的中心，而异常数据则为离群点位于其他位置
- 本文提出了FCDD，是对DSVDD的一种改进。通过使用卷积层和池化层，实现对输入图像中异常像素点的判别，从而识别异常区域



#### 2. Method

##### Deep One-Class Classification

> Deep One-Class Classification 通过学习神经网络将正常样本映射到输出空间中心$c$附近，从而使异常被映射出去。



本文使用一个Hypershere Classifier (HSC) 

> 设$X_1,\cdots,X_n$表示一组样本，$y_1,\cdots,y_n$表示标签，其中$y_i=1$表示异常，$y_i=0$表示正常



**HSC objective:**


$$
\min _{\mathcal{W}} \frac{1}{n} \sum_{i=1}^{n}\left(1-y_{i}\right) h\left(\phi\left(X_{i} ; \mathcal{W}\right)-\mathbf{c}\right)-y_{i} \log \left(1-\exp \left(-h\left(\phi\left(X_{i} ; \mathcal{W}\right)-\mathbf{c}\right)\right)\right)
$$

> $c\in \mathbb{R}^d$表示一个预先确定的中心
>
> $\phi:\mathbb{R}^{c\times h \times w }\rightarrow \mathbb{R}^d $是一个权值为$\mathcal{W}$的神经网络
>
> $h$为pseudo-Huber loss: $h(\mathbf{a})=\sqrt{\|\mathbf{a}\|_{2}^{2}+1}-1$

HSC loss鼓励$\phi$映射正常样本到中心 $c$ 附近而使异常样本远离中心 $c$



##### Fully Convolutional Architecture

> 使用FCN通过交替卷积和池化层将图像映射到特征矩阵，即$\phi:\mathbb{R}^{c\times h \times w }\rightarrow \mathbb{R}^{1\times u\times v} $
>
> FCN能够保留空间信息



##### Fully Convolutional Data Description

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/3.png" alt="img" style="zoom:50%;" />

> 结合FCNs和HSC，提出了FCDD

FCDD使用标记为正常或异常的样本进行训练，经过FCN: $\phi:\mathbb{R}^{c\times h \times w }\rightarrow \mathbb{R}^{u\times v} $得到输出矩阵$\phi(X;\mathcal{W})$

利用输出矩阵得到pseudo-Huber loss：


$$
A(X)=\sqrt{\phi(X;\mathcal{W})^2+1}-1
$$
FCDD的目标函数定义为：


$$
\min _{\mathcal{W}} \frac{1}{n} \sum_{i=1}^{n}\left(1-y_{i}\right) \frac{1}{u \cdot v}\left\|A\left(X_{i}\right)\right\|_{1}-y_{i} \log \left(1-\exp \left(-\frac{1}{u \cdot v}\left\|A\left(X_{i}\right)\right\|_{1}\right)\right)
$$


##### Heatmap Upsampling

> $A(X)\in\mathbb{R}^{u\times v}$的空间维度比原始的输入图片维度$h\times w$ 小，因此为了得到全分辨率的热图，引入了一种基于感受域的上采样方案：
>
> 即使用固定高斯核的跨步反卷积进行上采样

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/2.png" alt="img" style="zoom:50%;" />



#### 3. Experiment

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/1.png" alt="img" style="zoom:50%;" />



> 与基于重构的方法相比，FCDD的一个主要优点是可以很容易地用于半监督异常检测的设置中 (**Semi-Supervised FCDD**)
>
> 在训练集中加入少量异常样本数据，利用ground truth标注，训练一个像素级的模型，目标函数如下：
>
> 
> $$
> \min _{\mathcal{W}} \frac{1}{n} \sum_{i=1}^{n}\left(\frac{1}{m} \sum_{j=1}^{m}\left(1-\left(Y_{i}\right)_{j}\right) A^{\prime}\left(X_{i}\right)_{j}\right)-\log \left(1-\exp \left(-\frac{1}{m} \sum_{j=1}^{m}\left(Y_{i}\right)_{j} A^{\prime}\left(X_{i}\right)_{j}\right)\right)
> $$
>
> > $m=h\cdot w$

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/4.png" alt="img" style="zoom:50%;" />