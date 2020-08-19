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
    - GAN
---

`因为最近在看few-shot和meta-learning，一些方法需要与域适应相结合，提升模型性能。其中域适应常用的方法是基于cycleGAN的应用和改进，因此在这里简单介绍一下cycleGAN`




#### 1. Introduction

> **cycleGAN**主要目的与pix2pix相似，都是用于image to image translation。与pix2pix不同的是：
>
> - pix2pix网络训练需要提供image pair
> - cycleGAN不要求提供pairs，即unpaired。因此cycleGAN的创新点在于能够在源域和目标域之间，无需建立训练数据间一对一映射，即可实现迁移（如风格转换等）
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/1.png" alt="img" style="zoom:50%;" />



#### 2. Method

普通的GAN loss：

$$
L_{\mathrm{GAN}}\left(F, D_{Y}, X, Y\right)=E_{y \sim p_{\mathrm{data}}(y)}\left[\log D_{Y}(y)\right]+E_{x \sim p_{\mathrm{data}}(x)}\left[\log \left(1-D_{Y}(F(x))\right)\right]
$$


> 映射F完全可以将所有x都映射到y空间的同一张图片，使损失无效化



CycleGAN Loss:

$$
\begin{aligned}
\mathcal{L}\left(G, F, D_{X}, D_{Y}\right)=& \mathcal{L}_{\mathrm{GAN}}\left(G, D_{Y}, X, Y\right)+\mathcal{L}_{\mathrm{GAN}}\left(F, D_{X}, Y, X\right)+\lambda \mathcal{L}_{\mathrm{cyc}}(G, F)
\end{aligned}
$$

> CycleGAN Loss = Adversarial loss + Cycle-consistency loss + (Identity loss)            
>
> - Adversarial loss：
>   - 两个分布X, Y，生成器G和F分别是X->Y和Y->X的映射，两个判别器$D_x$和$D_y$可以对转换后的图片进行判别                       
>
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
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/2.png" alt="img" style="zoom:50%;" />

> - Cycle-consistency loss
>   - 循环一致性损失，保留x中content成分，只改变style             
>
> $$
> \begin{aligned}
>   \mathcal{L}_{\mathrm{cyc}}(G, F) &=\mathbb{E}_{x \sim p_{\text {data }}(x)}\left[\|F(G(x))-x\|_{1}\right]+\mathbb{E}_{y \sim p_{\text {data }}(y)}\left[\|G(F(y))-y\|_{1}\right]
>   \end{aligned}
> $$
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/3.png" alt="img" style="zoom:40%;" />
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/4.png" alt="img" style="zoom:40%;" />

> - Identity loss
>
>   - 生成器G用来生成y风格图像，则把y输入G，应该仍然生成y，只有这样才能证明G具有生成y风格的能力，因此G(y)和y应该尽可能接近。如下图，若无Identity loss，生成器G和F会自主地修改色调，使得整体的颜色产生变化。      
>
>     
> $$
> L_{\text {Identity}}(G, F)=\mathbb{E}_{y \sim p_{\text {data}}(y)}\left[\|G(y)-y\|_{1}\right]+\mathbb{E}_{x \sim p_{\text {data}}(x)}\left[\|F(x)-x\|_{1}\right]
> $$
>
> 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/5.png" alt="img" style="zoom:40%;" />



#### 3. Application in Domain Adaptation

[CYCADA:](https://arxiv.org/pdf/1711.03213.pdf)循环一致性对抗域适应方法

> - 通过在多个损失函数上训练模型，同时进行了特征级和像素级的对齐：
>   - 特征级域适应方法：通过对齐从源域和目标域中提取出的特征完成域适应
>   - 像素级域适应方法：执行与特征级域适应类似的分布对齐，但是不是在特征空间中进行对齐，而是在原始像素空间中将源域数据转化为目标域中的风格
>
> - 更关注在对齐过程中保留数据的语义信息，即将数据的类别信息考虑进来



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/6.png" alt="img" style="zoom:40%;" />

> 如上图所示，模型共有5个损失函数，指导学习**分类器f，生成器G和域判别器D**，最终达到域适应的目的：
>
> - 分类判别损失
> - 原样本映射的目标样本对抗损失
> - 特征级域适应损失
> - 重构的原样本的循环损失
> - 源图像和转化为目标图像后的语义一致性损失



**Step 1:**  首先学习一个源域分类器$f_s$，在源域数据上进行分类判别，损失函数如下：        
$$
\mathcal{L}_{task}(f_S,X_S,Y_S)=-\mathbb{E}_{\left(x_{s}, y_{s}\right)\sim\left(X_{S}, Y_{S}\right)}\sum_{k=1}^{K}\mathbb{1}_{\left[k=y_{s}\right]}\log\left(\sigma\left(f_{S}^{(k)}\left(x_{s}\right)\right)\right)
$$


**Step 2**: 学习目标域分类器$f_T$

- 使用生成器$G_{S\rightarrow T}$，通过源域样本生成与目标样本类似的结果。对抗判别器$D_t$用于判别是原始的目标样本还是由源域生成的虚假目标样本。进行对抗域适应训练$G_{S\rightarrow T}$和$D_t$，损失函数为：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{GAN}}\left(G_{S \rightarrow T}, D_{T}, X_{T}, X_{S}\right) &=\mathbb{E}_{x_{t} \sim X_{T}}\left[\log D_{T}\left(x_{t}\right)\right]+\mathbb{E}_{x_{s} \sim X_{S}}\left[\log \left(1-D_{T}\left(G_{S \rightarrow T}\left(x_{s}\right)\right)\right)\right]
\end{aligned}
$$



- 通过$G_{S\rightarrow T}$让源数据近似于目标数据，但为了**保证源域数据的结构和内容在对齐过程中保留下来**，在CyCADA中加入了一个循环一致性限制，即将目标域映射到源域

$$
\begin{array}{l}
\mathcal{L}_{\mathrm{cyc}}\left(G_{S \rightarrow T}, G_{T \rightarrow S}, X_{S}, X_{T}\right)= \mathbb{E}_{x_{s} \sim X_{S}}\left[\left\|G_{T \rightarrow S}\left(G_{S \rightarrow T}\left(x_{s}\right)\right)-x_{s}\right\|_{1}\right]+\mathbb{E}_{x_{t} \sim X_{T}}\left[\left\|G_{S \rightarrow T}\left(G_{T \rightarrow S}\left(x_{t}\right)\right)-x_{t}\right\|_{1}\right]
\end{array}
$$



- 为了在对齐过程中保留数据语义信息，需要将源数据的类别信息考虑进来

  - 具体来说就是首先预训练一个源域分类器$f_S$，固定其权值，这样对于转换前的图像以及经过转换后的图像能够通过固定权值的分类器以相同的方式进行分类。
  - 定义通过固定权值分类器f得到预测标签，即$p(f,X)=\arg max(f(X))$

  

  因此语义一致性损失为：
  $$
  \begin{aligned}
  \mathcal{L}_{\mathrm{sem}}\left(G_{S \rightarrow T}, G_{T \rightarrow S}, X_{S}, X_{T}, f_{S}\right) &=\mathcal{L}_{\mathrm{task}}\left(f_{S}, G_{T \rightarrow S}\left(X_{T}\right), p\left(f_{S}, X_{T}\right)\right) +\mathcal{L}_{\mathrm{task}}\left(f_{S}, G_{S \rightarrow T}\left(X_{S}\right), p\left(f_{S}, X_{S}\right)\right)
  \end{aligned}
  $$
  
  
- 此外，还考虑了特征级域适应，通过任务网络$f_T$的**输出特征**判断是否是来自两个图像集的特征或语义

$$
\mathcal{L}_{\mathrm{GAN}}\left(f_{T}, D_{\mathrm{feat}}, f_{S}\left(G_{S \rightarrow T}\left(X_{S}\right)\right), X_{T}\right)
$$



- **Total Loss：**

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CyCADA}} &\left(f_{T}, X_{S}, X_{T}, Y_{S}, G_{S \rightarrow T}, G_{T \rightarrow S}, D_{S}, D_{T}\right) \\
&=\mathcal{L}_{\mathrm{task}}\left(f_{T}, G_{S \rightarrow T}\left(X_{S}\right), Y_{S}\right) \\
&+\mathcal{L}_{\mathrm{GAN}}\left(G_{S \rightarrow T}, D_{T}, X_{T}, X_{S}\right)+\mathcal{L}_{\mathrm{GAN}}\left(G_{T \rightarrow S}, D_{S}, X_{S}, X_{T}\right) \\
&+\mathcal{L}_{\mathrm{GAN}}\left(f_{T}, D_{\mathrm{feat}}, f_{S}\left(G_{S \rightarrow T}\left(X_{S}\right)\right), X_{T}\right) \\
&+\mathcal{L}_{\mathrm{cyc}}\left(G_{S \rightarrow T}, G_{T \rightarrow S}, X_{S}, X_{T}\right)+\mathcal{L}_{\mathrm{sem}}\left(G_{S \rightarrow T}, G_{T \rightarrow S}, X_{S}, X_{T}, f_{S}\right)
\end{aligned}
$$

> - 其中第一项为经过转换后的源域图像，再使用转化前的标签得到分类损失；第二项为源域到目标域到对抗损失以及重构损失；第三项为特征级的对抗损失；第四项为循环一致性与语义一致性
> - 第一项、第二项和第四项都为了尽可能多的在域适应中保留结构信息

