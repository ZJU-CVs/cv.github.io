---
layout:     post
title:      Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
subtitle:   
date:       2020-07-27
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:

   - paper
---

```
因为最近在看few-shot和meta-learning，一些方法需要与域适应相结合，提升模型性能。其中域适应常用的方法是基于cycleGAN的应用和改进，因此在这里简单介绍一下cycleGAN
```



#### 1. Introduction

> **cycleGAN**主要目的与pix2pix相似，都是用于image to image translation。与pix2pix不同的是：
>
> - pix2pix网络训练需要提供image pair
> - cycleGAN不要求提供pairs，即unpaired。因此cycleGAN的创新点在于能够在源域和目标域之间，无需建立训练数据间一对一映射，即可实现迁移（如风格转换等）
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/1.png" alt="img" style="zoom:50%;" />



#### 2. Method

普通的GAN loss：

> $$
> L_{\mathrm{GAN}}\left(F, D_{Y}, X, Y\right)=E_{y \sim p_{\mathrm{data}}(y)}\left[\log D_{Y}(y)\right]+E_{x \sim p_{\mathrm{data}}(x)}\left[\log \left(1-D_{Y}(F(x))\right)\right]
> $$
>
> 映射F完全可以将所有x都映射到y空间的同一张图片，使损失无效化



CycleGAN Loss:

> CycleGAN Loss = Adversarial loss + Cycle-consistency loss + (Identity loss)
> $$
> \begin{aligned}
> \mathcal{L}\left(G, F, D_{X}, D_{Y}\right)=& \mathcal{L}_{\mathrm{GAN}}\left(G, D_{Y}, X, Y\right)+\mathcal{L}_{\mathrm{GAN}}\left(F, D_{X}, Y, X\right)+\lambda \mathcal{L}_{\mathrm{cyc}}(G, F)
> \end{aligned}
> $$

- Adversarial loss：

  > 两个分布X, Y，生成器G和F分别是X->Y和Y->X的映射，两个判别器$D_x$和$D_y$可以对转换后的图片进行判别
  > $$
  > \begin{aligned}
  > \mathcal{L}_{\mathrm{GAN}}\left(G, D_{Y}, X, Y\right) &=\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\log D_{Y}(y)\right] +\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\log \left(1-D_{Y}(G(x))\right]\right.
  > \end{aligned}
  > $$
  >
  > $$
  > \begin{aligned}
  > \mathcal{L}_{\mathrm{GAN}}\left(F, D_{X}, Y, X\right) &=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\log D_{X}(x)\right] +\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\log \left(1-D_{X}(F(y))\right]\right.
  > \end{aligned}
  > $$
  >
  > 
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/2.png" alt="img" style="zoom:50%;" />

  

- Cycle-consistency loss

  > 循环一致性损失，保留x中content成分，只改变style
  > $$
  > \begin{aligned}
  > \mathcal{L}_{\mathrm{cyc}}(G, F) &=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(G(x))-x\|_{1}\right]+\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(F(y))-y\|_{1}\right]
  > \end{aligned}
  > $$
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/3.png" alt="img" style="zoom:40%;" /><img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/4.png" alt="img" style="zoom:40%;" />



- Identity loss

  > $$
  > L_{\text {Identity}}(G, F)=\mathbb{E}_{y \sim p_{\text {data}}(y)}\left[\|G(y)-y\|_{1}\right]+\mathbb{E}_{x \sim p_{\text {data}}(x)}\left[\|F(x)-x\|_{1}\right]
  > $$
  >
  > > 生成器G用来生成y风格图像，则把y输入G，应该仍然生成y，只有这样才能证明G具有生成y风格的能力，因此G(y)和y应该尽可能接近。如下图，若无Identity loss，生成器G和F会自主地修改色调，使得整体的颜色产生变化。
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/5.png" alt="img" style="zoom:40%;" />

