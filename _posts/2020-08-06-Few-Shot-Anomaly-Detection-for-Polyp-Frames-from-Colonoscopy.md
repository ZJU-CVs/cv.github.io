---
layout:     post
title:      Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy
subtitle:   
date:       2020-08-06
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - FSL
    - Anomaly Detection
---



### 1. Introduction

- 异常检测方法训练集中正常样本的比例往往比异常图像大得多，因此模型设计需要考虑非均衡问题

- 目前常用方法：

  - 设计一种解决非平衡学习的训练方法，如图 (a) 所示，但仍需要相对大量的异常训练样本 
  - 针对正常图像训练可以重构正常图像的条件生成模型，并根据测试图像的重建误差检测异常，如图 (b) 所示。然而，这类方法可能会误分类离inliers较近的outliers

- 本文提出了一种使用高度不平衡训练集进行训练的few-shot 异常检测方法网络(FSAD-NET)，其中包含大量正常图像和少量异常图像，如图 (c) 所示。

  - 首先学习一个特征编码器，该编码器使用正常图像进行训练，以最大化训练图像与特征嵌入之间的互信息(MI)

  - 然后训练一个分数推断网络(SIN)，该网络将正常图像的特征嵌入朝向特征空间的特定区域拉近，并将异常图像的嵌入推离正常特征的区域

    

  

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/46.png" alt="img" style="zoom:30%;" />



### 2. Method

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/48.png" alt="img" style="zoom:30%;" />

**1).** 特征编码器$z=f_E(x;\theta_E)$的预训练，以学习图像嵌入，从而最大化正常样本图像$x\in \mathcal{D}_N$和它们的嵌入$z=f_E(x\in \mathcal{D}_N;\theta_E)$之间的互信息(MI)，具体如下：

$$
\theta_{E}^{*}, \theta_{G}^{*}, \theta_{L}^{*}=\arg \max _{\theta_{E}, \theta_{G}, \theta_{L}}(\alpha \hat{I}_{\theta_{G}}\left( x ; f_{E}\left( x ; \theta_{E}\right)\right)+\frac{\beta}{\mid M \mid} \sum_{\omega \in M } \hat{I}_{\theta_{L}}\left( x (\omega) ; f_{E}\left( x (\omega) ; \theta_{E}\right)\right))\\+\gamma \arg \min _{\theta_{ E }} \arg \max _{\phi} \hat{D}_{\phi}\left( V \Vert U _{ P , \theta_{E}}\right)
$$


> $x(w)$表示局部图像区域，$x$表示整张图片
> 
> $\hat{I}_G(\cdot)$和$\hat{I}_L(\cdot)$表示MI下界（基于Donsker-Varadhan表示的KL散度） 


$$
\hat{I}_{\theta_{G}}\left(\mathbf{x} ; f_{E}\left(\mathbf{x} ; \theta_{E}\right)\right)=\mathbb{E}_{\mathbb{J}}\left[f_{G}\left(\mathbf{x}, f_{E}\left(\mathbf{x} ; \theta_{E}\right) ; \theta_{G}\right)\right]-\log \mathbb{E}_{\mathbb{M}}\left[e^{f_{G}\left(\mathbf{x}, f_{E}\left(\mathbf{x} ; \theta_{E}\right) ; \theta_{G}\right)}\right]
$$

> $\mathbb{J}$表示联合分布，$\mathbb{M}$表示the product of the marginals的图像嵌入
> 
> `注：正样本定义为来自joint distribution，负样本定义为来自the product of marginals`


$$
\left.\arg \min _{\theta_{E}} \arg \max _{\phi} \hat{D}_{\phi}\left(\mathbb{V}|| \mathbb{U}_{\mathbb{P}, \theta_{E}}\right)=\mathbb{E}_{\mathbb{V}}[\log d(\mathbf{z} ; \phi)]+\mathbb{E}_{\mathbb{P}}\left[\log \left(1-d\left(f_{E}\left(\mathbf{x} ; \theta_{E}\right)\right) ; \phi\right)\right)\right]
$$

> $\mathbb{V}$表示嵌入$z$的先验分布$\mathcal{N}(.;\mu_V,\sum_V)$，$\mathbb{P}$表示嵌入$z=f_E(x\in \mathcal{N}_N;\theta_E)$的分布



**2).** 训练SIN $f_s(f_E(x;\theta_E);\theta_S)$ ，具有类似对比的损失，使用$\mathcal{D}_N$和$\mathcal{D}_A$达到目标：        


$$
f_S(f_E(x\in\mathcal{D}_A;\theta_E);\theta_S)\gt f_S(f_E(x\in \mathcal{D}_N;\theta_E);\theta_S)
$$



> - 通过计算$z=f_E(x\in \mathcal{D}_A \cup \mathcal{D}_{N};\theta_{E}^{*})$ 训练 $f_S(z;\theta_S)$        
>
> $$
> \ell_{S}=\mathbb{I}(y \text { is } \text {Normal})\left|s\left(f_{S}\left(\mathbf{z} ; \theta_{S}\right)\right)\right|+\mathbb{I}(y \text { is } \text {Abnormal}) \max \left(0, a-s\left(f_{S}\left(\mathbf{z} ; \theta_{S}\right)\right)\right)
> $$
>
> > 其中，$s(x)=\frac{x-\mu_{S}}{\sigma_{S}}$
>



**3).**  在inference阶段，对于一张测试图像$x$，使用$f_E(x;\theta_E)$计算特征嵌入，然后使用$s=f_S(z;\theta_S)$计算分数，将分数结果$s$与阈值$\mathcal{T}$进行比较，确定测试图像是正常还是异常



### 3. Experiments 

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/47.png" alt="img" style="zoom:30%;" />

> 与之前的零样本异常检测方法和不平衡学习方法相比，所提方法实现了最新的异常检测性能



### 4. Conclusion

> 提出了FSAD-NET少样本异常检测框架，通过最大化正常训练图像和嵌入之间全局和局部的互信息，并最小化嵌入和先验分布的差异，训练得到编码器。利用训练好的编码器产生的嵌入，使用类似对比学习的损失来训练Score Inference Network ( $f_S$ )用于计算异常分数。



