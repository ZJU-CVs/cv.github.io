---
layout:     post
title:      GAN Compression
subtitle:   Efﬁcient Architectures for Interactive Conditional GANs
date:       2020-05-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Model Compression

---

#### 1. Introduction

> 本文提出了一种通用的条件生成GAN模型的压缩算法，在pix2pix, Cyclegan, GauGAN等常见的condition GAN上进行应用，计算量减少了9～21倍



#### 2. Methods

##### **主要过程：**

> (1) 给定一个与训练的teacher **G'**。先从teacher G'中蒸馏一个“once-for-all”的student **G**（通过权重共享的策略），student G在每次training step中选择不同的通道数
>
> (2) 从once-for-all 的student G中sample一些sub generators，然后评估每个generator的性能，不需要retrain （通过神经网络架构搜索 NAS）
>
> (3) 从（2）中选择满足压缩比与效果条检测的模型作为候选模型，fine-tune候选模型，将其作为最终的压缩模型

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/GAN_compress.png" alt="img" style="zoom:67%;" /> 



##### Training Objective 

`蒸馏过程`

- Unifying unpaired and paired learning.

  > 生成器$G^{\prime} \rightarrow G$的蒸馏

  $$
  \mathcal{L}_{\text {recon }}=\left\{\begin{array}{ll}
  \mathbb{E}_{\mathbf{x}, \mathbf{y}}\|G(\mathbf{x})-\mathbf{y}\|_{1} & \text { if paired cGANs } \\
  \mathbb{E}_{\mathbf{x}}\left\|G(\mathbf{x})-G^{\prime}(\mathbf{x})\right\|_{1} & \text { if unpaired cGANs }
  \end{array}\right.
  $$



- Inheriting the teacher discriminator.

  > 判别器$D^{\prime} \rightarrow D$ 的蒸馏

  student D采用与teacher D'相同的模型架构，使用teacher D中的pre-train weights，并与G 一起作为compressed model进行 fine-tune
  $$
  \mathcal{L}_{\mathrm{cGAN}}=\mathbb{E}_{\mathbf{x}, \mathbf{y}}[\log D(\mathbf{x}, \mathbf{y})]+\mathbb{E}_{\mathbf{x}}[\log (1-D(\mathbf{x}, G(\mathbf{x})))
  $$



- Intermediate feature distillation.

  > cGAN 通常输出确定性的图像，而不是概率分布。因此很难从teacher model的输出像素中提取dark knowledge
  >
  > 特别是对于成对的训练设置，与真实目标图像相比，由教师模型生成的输出图像实质上不包含任何其他信息。

  通过匹配生成器的中间表示实现蒸馏（中间层包含更多的通道，能够提供更丰富的信息）

$$
\mathcal{L}_{\text {distill }}=\sum_{t=1}^{T}\left\|G_{t}(\mathbf{x})-f_{t}\left(G_{t}^{\prime}(\mathbf{x})\right)\right\|_{2}
$$

​		*其中 $G_t(x)$ 和 $G'_t(x)$ 是student和teacher model中第t个选定层的中间特征激活，$T$表示层数, $f_t$表示$1\times1$可学习卷积层，将学生模型的特征映射到教师模型的特征中相同数量的通道*

​		共同优化$G_t$和$f_t$，以使蒸馏损失$L_{distill}$最小化



- Full objective

  最终目标：
  $$
  \mathcal{L}=\mathcal{L}_{\mathrm{cGAN}}+\lambda_{\mathrm{recon}} \mathcal{L}_{\mathrm{recon}}+\lambda_{\mathrm{distill}} \mathcal{L}_{\mathrm{distill}}
  $$



##### Efﬁcient Generator Design Space

> 选择设计良好的student architecture对于最终的知识蒸馏性能十分关键。
>
> 一味地缩小教师模型的通道数不能产生compact的学生模型（在4倍以上的计算减缩后，性能开始显著下降）



Convolution decomposition and layer sensitivity

- 现有的生成器通常采用vanilla convolutions来设计CNN分类和分割，广泛采用卷积的分解形式（depthwise+pointwise）

- 因此使用decomposed convolution进行生成器设计，然而将Decomposition直接应用于所有卷积层会大大降低图像质量。Decomposition一些层会立刻损害性能，而其他层则更健壮（灵敏度不同）

  > Example: ResBlock层消耗了大部分模型参数和计算成本，而几乎不受分解的影响；而上采样层的参数要少得多，但对模型压缩相当敏感

  

Automated channel reduction with NAS

- 为了进一步提高压缩率，使用通道修建（channel pruning）在生成器自动选择通道宽度，以消除冗余，从而减少二次计算量

  > 给定可能的通道配置${c_1,c_2,...,c_K}$，其中K是要修剪的层数，使用神经网络架构搜索来找到最佳的通道配置$\{c_1^*,c_2^*,...,c_K^*\}=argmin_{\{c_1,c_2,...,c_K\}}L, s.t.MACs<F_t$



##### **Decouple Training and Search**

采用one-shot NAS方法，将模型训练与架构搜索分离[1]

```
[1]《Single path one-shot neural architecture search with uniform sampling》
```



> 首先训练一个支持不同通道数的“one-for-all” network，具有不同数量通道的每个子网络Sub-networks都经过同等训练，可以独立运行。子网络之间共享权重。

- 假设原始的teacher 生成器具有$\{c_k^0\}^K_{k=1}$个通道，对于给定的通道数配置$\{c_k\}^K_{k=1},c_k\leq c_k^0$，通过“once-for-all”网络中相应权重向量中提取第一个$\{c_{k=1}^K\}$个通道来获得子网络的权重
- 在每一training step，随机采样一个确定通道数配置的子网络，使用learning objective计算输出和梯度，并更新提取的权重。由于前几个通道的权重被频繁更新，因此在所有权重中起到更为关键的作用
- 在"once-for all"网络训练好后，通过直接评估性能找到最佳的子网络

> 通过这种方式，可以将训练和搜索生成器体系结构分离开来：只需要训练一次，可以在无需进一步训练的情况下评估所有可能的通道配置，并选择最佳的搜索结果。 同时还可以微调所选架构，以进一步提高性能。