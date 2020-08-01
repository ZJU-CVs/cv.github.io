---
layout:     post
title:      Small Data Challenges in Big Data Era: A Survey of Recent Progress on Unsupervised and Semi-Supervised Methods
subtitle:   
date:       2020-07-31
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Notes
    - FSL
---



### 1. Introduction

- small data challenges：

  > - 深度神经网络的成功通常取决于大量的标记数据，而这些数据收集起来很昂贵。为了解决这个问题，出现了大量以无监督和半监督的方式训练带有小数据的复杂模型的工作
  > - Unlabeled data的作用：(1) 未标记数据的分布可以学习更加鲁棒的表征，以推广到新的学习任务; (2) 未标记的数据还可以帮助模型缩小不同任务之间的domain gap
  >
  > - Auxiliary tasks的作用：以ZSL (无监督方法)、FSL (半监督方法)问题为例，可以transfer语义知识或学习知识(元学习)等辅助信息从源任务转移到目标任务

- Overview:

  > ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/31.png)

### 2. Unsupervised Methods

> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/32.png)

### 3. Semi-Supervised Methods

> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/33.png)



### 4. Domain Adaptation

> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/34.png)

