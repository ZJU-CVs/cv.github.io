---
layout:     post
title:      Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy
subtitle:   
date:       2020-08-01
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---



### 1. Introduction

- 异常检测方法训练集中正常样本的比例往往比异常图像大得多，因此模型设计需要考虑非均衡问题

- 目前常用方法：

  - 针对正常图像训练可以重构正常
  - 分布的学习，在测试时，相对于正常分布较远的样本被分类为异常

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/46.png" alt="img" style="zoom:30%;" />

