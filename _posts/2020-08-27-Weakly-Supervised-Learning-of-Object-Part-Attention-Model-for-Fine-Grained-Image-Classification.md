---
layout:     post
title:      Weakly Supervised Learning of Object-Part Attention Model for Fine-Grained Image Classification
subtitle:   
date:       2020-08-27
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-Grained

---





#### 1. Introduction

​       由于细粒度间差异较小，细粒度类别之间的类内距离较大，因此细粒度分类具有挑战性。 解决此问题的关键是在图像中定位判别部分。 本文提出了一种弱监督方法，它只需要图像级标签进行细粒度分类。



#### 2. Method 

- 模型由三个部分组成，使用第一个CNN从具有object-level关注的原始图像中查找对象位置，然后使用第二个CNN来学习object-level特征并定位判别部分，此外，part-level图像被输入到最后一个CNN以进行特征学习

- Attention Network for Fine-Grained Classification

  采用 SE module

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/SEnet1.png)

- Loss

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/1.png)