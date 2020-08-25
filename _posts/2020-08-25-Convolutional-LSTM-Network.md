---
layout:     post
title:      Convolutional LSTM Network
subtitle:   A Machine Learning Approach for Precipitation Nowcasting
date:       2020-08-25
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Paper
    - Video Analysis
---



### 1、LSTM

LSTM网络由input gate, forget gate, cell, output gate, hidden五个模块组成

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/video-analysis/4.png" alt="image-20191023142137607" style="zoom:47%;" />


$$
\begin{aligned} i_{t} &=\sigma\left(W_{x i} x_{t}+W_{h i} h_{t-1}+W_{c i} \circ c_{t-1}+b_{i}\right) \\ f_{t} &=\sigma\left(W_{x f} x_{t}+W_{h f} h_{t-1}+W_{c f} \circ c_{t-1}+b_{f}\right) \\ c_{t} &=f_{t} \circ c_{t-1}+i_{t} \circ \tanh \left(W_{x c} x_{t}+W_{h c} h_{t-1}+b_{c}\right) \\ o_{t} &=\sigma\left(W_{x o} x_{t}+W_{h o} h_{t-1}+W_{c o} \circ c_{t}+b_{o}\right) \\ h_{t} &=o_{t} \circ \tanh \left(c_{t}\right) \end{aligned}
$$



> 空心小圆圈表示矩阵对应元素相乘，又称为Hadamard乘积



LSTM(FC-LSTM)对于时序数据可以很好地处理，但是对于空间数据来说，将会带来冗余性，原因是空间数据具有很强的局部特征，但是FC-LSTM无法刻画此局部特征



### 2、ConvLSTM

将FC-LSTM中input-to-state和state-to-state部分由前馈式计算替换成卷积的形式

输入与各个门之间的连接由前馈式替换成了卷积，同时状态与状态之间也换成了卷积运算。这里的$X,C,H,f,o$都是三维tensor，后两个维度表示行和列的空间信息，第一维的为时间维


$$
\begin{aligned} i_{t} &=\sigma\left(W_{x i} * \mathcal{X}_{t}+W_{h i} * \mathcal{H}_{t-1}+W_{c i} \circ \mathcal{C}_{t-1}+b_{i}\right) \\ f_{t} &=\sigma\left(W_{x f} * \mathcal{X}_{t}+W_{h f} * \mathcal{H}_{t-1}+W_{c f} \circ \mathcal{C}_{t-1}+b_{f}\right) \\ \mathcal{C}_{t} &=f_{t} \circ \mathcal{C}_{t-1}+i_{t} \circ \tanh \left(W_{x c} * \mathcal{X}_{t}+W_{h c} * \mathcal{H}_{t-1}+b_{c}\right) \\ o_{t} &=\sigma\left(W_{x o} * \mathcal{X}_{t}+W_{h o} * \mathcal{H}_{t-1}+W_{c o} \circ \mathcal{C}_{t}+b_{o}\right) \\ \mathcal{H}_{t} &=o_{t} \circ \tanh \left(\mathcal{C}_{t}\right) \end{aligned}
$$


<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/video-analysis/5.png" alt="image-20191023142543935" style="zoom:67%;" />

> ![image-20191023140822275](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/video-analysis/6.png)





#### 3、Compare

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/video-analysis/7.png" alt="img" style="zoom:50%;" />

#### 4、Prediction

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/video-analysis/8.png" alt="img" style="zoom:50%;" />

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/video-analysis/9.png" alt="img" style="zoom:50%;" />