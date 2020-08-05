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

  - 设计一种解决非平衡学习的训练方法(a)，但仍需要相对大量的异常训练样本 
  - 针对正常图像训练可以重构正常图像的条件生成模型，并根据测试图像的重建误差检测异常(b)。然而，这类方法可能会误分类离inliers较近的outliers

- 本文提出了一种使用高度不平衡训练集进行训练的few-shot 异常检测方法网络(FSAD-NET)，其中包含大量正常图像和少量异常图像(c)。

  - 首先学习一个特征编码器，该编码器使用正常图像进行训练，以最大化训练图像与特征嵌入之间的互信息(MI)
  - 然后训练一个分数推断网络(SIN)，该网络将正常图像的特征嵌入朝向特征空间的特定区域拉近，并将异常图像的嵌入推离正常特征的区域
  - 

  

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/46.png" alt="img" style="zoom:30%;" />



### 2. Method

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/48.png" alt="img" style="zoom:30%;" />

以全局和局部的方式最大化正常训练图像和嵌入之间的MI，并最小化嵌入的差异和先验分布来对编码器建模；然后使用编码器产生的嵌入，使用类似对比的损失训练SIN

> **1)** 特征编码器$z=f_E(x;\theta_E)$的预训练，以学习图像嵌入，从而最大化正常样本图像$x\in \mathcal{D}_N$和它们的嵌入$z=f_E(x\in \mathcal{D}_N;\theta_E)$之间的互信息(MI)，具体如下：
> $$
> \begin{aligned}
> \theta_{E}^{*}, \theta_{G}^{*}, \theta_{L}^{*}=\arg \max _{\theta_{E}, \theta_{G}, \theta_{L}}(&\left.\alpha \hat{I}_{\theta_{G}}\left(\mathbf{x} ; f_{E}\left(\mathbf{x} ; \theta_{E}\right)\right)+\frac{\beta}{|\mathcal{M}|} \sum_{\omega \in \mathcal{M}} \hat{I}_{\theta_{L}}\left(\mathbf{x}(\omega) ; f_{E}\left(\mathbf{x}(\omega) ; \theta_{E}\right)\right)\right) \\
> &+\gamma \arg \min _{\theta_{\mathrm{n}}} \arg \max _{\mathcal{A}} \hat{D}_{\phi}\left(\mathbb{V}|| \mathbb{U}_{\mathbb{P}, \theta_{E}}\right)
> \end{aligned}
> $$
>
> $$
> \hat{I}_{\theta_{G}}\left(\mathbf{x} ; f_{E}\left(\mathbf{x} ; \theta_{E}\right)\right)=\mathbb{E}_{\mathbb{J}}\left[f_{G}\left(\mathbf{x}, f_{E}\left(\mathbf{x} ; \theta_{E}\right) ; \theta_{G}\right)\right]-\log \mathbb{E}_{\mathbb{M}}\left[e^{f_{G}\left(\mathbf{x}, f_{E}\left(\mathbf{x} ; \theta_{E}\right) ; \theta_{G}\right)}\right]
> $$
>
> 
>
> 
>
> **2)** 训练SIN $f_s(f_E(x;\theta_E);\theta_S)$ ，具有类似对比的损耗，使用$\mathcal{D}_N$和$\mathcal{D}_A$达到目标
> $$
> f_S(f_E(x\in \mathcal{D}_A;\theta_E);\theta_S)>f_S(f_E(x\in \mathcal{D}_N;\theta_E);\theta_S)
> $$
> 



在inference阶段