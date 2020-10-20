---
layout:     post
title:      A Framework For Contrastive Self-Supervised Learning And Designing A New Approach
subtitle:   
date:       2020-10-18
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - SSL
---



#### 1. Introduction

- 本文介绍了<u>CPC、AMDIM、BYOL、SimCLR、Swav</u>等最近较为著名的对比学习方法，并提出了 <u>YADIM</u> 新型对比学习算法
- 为描述对比自监督学方法，形式化定义了一个**概念框架**，并使用该框架分析了三种对比学习的示例：SimCLR、CPC、AMDIM
- 通过实验分析了对比自监督学习方法框架中各个部分的作用，为新方法设计提供方向和思路



#### 2. Contrastive self-supervised learning

##### main idea:

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/3.gif)

> 通过使用三个关键的元素（正样本、anchor、负样本的表征）来实现上述思想。
>
> 为了创建一个正样本对，需要两个相似的样本，而当创建一个负样本对时，将使用第三个与两个正样本不相似的样本



##### CSL Framework:

- Data augmentation pipeline 

  > 一个数据增强过程 A(x) 对同一个输入应用一系列随机变换
  >
  > ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/4.gif)
  >
  > 
  >
  > 在深度学习场景中，<u>数据增强</u>旨在构建对于原始输入中的噪声具有不变性的表征。
  >
  > 在对比学习场景中，通过<u>数据增强</u>生成anchor、正样本和负样本
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/2.png" alt="img" style="zoom:50%;" />

  

  

- Encoder 

  > $$
  > f_{\theta}: \mathbb{R}^{s \times w \times h} \rightarrow \mathbb{R}^{k \times c}
  > $$
  >
  > 
  >
  > 大多数方法使用了具有各种各样深度和宽度的ResNet 类网络，常用Resnet-50作为网络架构

  

- Representation extraction

  > $r^{+}=f_{\theta}\left(v^{+}\right):$ positive represention，$r^{a}=f_{\theta}\left(a^{+}\right)$：anchor representation，$r^{-}=f_{\theta}\left(v^{-}\right)$：negative representation
  >
  > 不同对比学习，采取的提取表征策略不同

  

- Similarity measure 

  > 使用了点积或余弦进行相似度计算

  

- Loss function

  > - CPC、AMDIM、SimCLR、Swav都选择使用了一种噪声对比估计损失（包括NCE loss、triplet loss和InfoNCE），NCE 损失函数包含两个部分：一个分子、一个分母。分子鼓励相似的向量靠近，分母推动所有其它的向量远离。
  >
  > - BYOL 并不需要该分母，而是依赖于第二个编码器的权重更新机制来提供对比信号

  

##### Special Case 1: AMDIM

> **Data augmentation pipeline**: 
>
> - AMDIM的数据增强方式包括：random flip，image jitter，color jitter，random gray scale and normalization of mean and standard deviation
>
> - 对于每一张图像，都会通过将数据增强过程$A$在图像$x$上应用两次得到两个版本的变换图像，并将其输入encoder得到positive $v^{+}\sim A(x)$和anchor $v^{a}\sim A(x)$特征，negative $v^{-}$通过对其他不同的图片应用变换并提取特征得到$v^{-} \sim A\left(x^{\prime}\right)$。
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/10.png" alt="img" style="zoom: 67%;" />
>
>   
>
> **Encoder:** 
>
> - 基于ResNet
> - 网络设计主要考虑：
>   - 控制感受域的信息重叠，当正样本对中两个特征分布的距离过近时，它们信息重叠的部分较多，此时优化任务会变得很简单，模型得不到有效训练
>   - 通过避免padding，来使特征分布达到平稳
>   - 增加网络宽度，如ResNet-34有64个、128个、256个和512个feature maps，而AMDIM ResNet有320个、640个、1280个和2560个feature maps
>
> 
>
> **Representation extraction:**
>
> - 考虑Multiscale
> - 训练的目标是**源图像各层级的输出特征应该和同源的特征相关度最大，和其他图片的各级特征相关度最小**
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/8.gif" alt="img" style="zoom: 67%;" />
>
> **Similarity measure:** 	
>
> - 使用点积$\Phi(a,b)=a\cdot b$
>
> 
>
> **Loss function:**
> $$
> \mathcal{N}_{\theta}\left(r^{a}, R^{+}, R^{-}\right)=-\log \frac{\sum_{r_{i}^{+} \in R^{+}} \exp \left(\Phi\left(r^{a}, r_{i}^{+}\right)\right)}{\sum_{r_{i}^{-} \in R^{-}} \exp \left(\Phi\left(r^{a}, r_{i}^{-}\right)\right)}
> $$
>
> $$
> \mathcal{L}_{\mathrm{AMDIM}}=-\frac{1}{3}\left[\mathcal{N}_{\theta}\left(r_{j}^{a}, R_{k-1}^{+}, R_{k-1}^{-}\right)+\mathcal{N}_{\theta}\left(r_{j}^{a}, R_{k-2}^{+}, R_{k-2}^{-}\right)+\mathcal{N}_{\theta}\left(r_{j-1}^{a}, R_{k-1}^{+}, R_{k-1}^{-}\right)\right]
> $$
>
> ref: 《SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION》
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/15.png" alt="img" style="zoom: 25%;" />



##### Special Case 2: CPC

> **Data augmentation pipeline**: 
>
> - 应用色彩抖动、随机灰度、随机翻转等变换的处理流程（与AMDIM相同）
>
> - 此外还引入了一种特殊的变换：将一张图像划分为一些重叠的子图块，$\mathbb{R}^{w \times h \times d} \rightarrow \mathbb{R}^{p \times q \times q \times d}$
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/9.gif" alt="img" style="zoom:67%;" />
>
> - 通过这一过程，生成多组正负样本对
>
>   
>
> **Encoder:** 
>
> - 采用ResNet-101的改进版
>   - 将residual blocks从23增加到46
>   - 将bottleneck layer维度从256拓宽到512
>   - 将feature maps数量从1024增加到4096，同时用layer normalization代替batch normalization以最小化来自同一图像的两个特征向量之间的信息共享。
>
> 
>
> **Representation extraction:**
>
> - 基于$H_{i,j}$特征向量预测$H_{i+k,j}$，$H=f_\theta(v),\ where\ v\sim A(x)\ and \  H\in \mathbb{R}^{c\times h \times w}$
>
> - 预测任务首先通过一个上下文编码器$g_{\psi}$将masked卷积应用于$H$，生成$C=g_\psi(H),\ where \ C\in \mathbb{R}^{c\times h \times w}$。每个$c_{i,j}$总结每个$H_{i,j}$周围的上下文
>
> - 对于每个$c_{i,j}$，通过预测矩阵$W_k$预测$\hat{H}_{i+k, j}=W_{k} c_{i, j}$
>
>   > anchor representation: $r^{a}=\hat{H}_{i+k, j}$
>   >
>   > positive representation: $r^{+}={H}_{i+k, j}$
>   >
>   > negative representations: 所有其他的${H}_{i, j}$为$r^{-}$
>
> 
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/7.gif)
>
> **Similarity measure:** 
>
> - 使用点积衡量$\hat{H}_{i+k,j}$和$H_{i+k,j}$
>
> 
>
> **Loss function:**
> $$
> \mathcal{L}_{\theta}\left(r^{a}, r^{+}, R^{-}\right)=-\log \frac{\exp \left(\Phi\left(r^{a}, r_{i}^{+}\right)\right)}{\exp \left(\Phi\left(r^{a}, r_{i}^{+}\right)\right)+\sum_{r_{i}^{-} \in R^{-}} \exp \left(\Phi\left(r^{a}, r_{i}^{-}\right)\right)}
> $$
>
> - 可以看作是从正确的$H_{i+k,j}$和干扰因子



##### Special Case 3: SimCLR

> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/simclr_arch.png)
>
> **Data augmentation pipeline**:
>
> - 在AMDIM的数据增强变换方法基础上，加入了高斯模糊 (Gaussian Blur)
>
> 
>
> **Encoder:**
>
> - SimCLR中的编码器$f_θ$是一个宽度和深度可变的ResNet。S
> - imCLR中的resnet使用batch normalization。
>
> 
>
> **Representation extraction:**
>
> - 编码器的输出$r=f_\theta(v), \ where \ r\in \mathbb{R}^{c\times h \times w}$
>
> - reshaping: $r: \mathbb{R}^{c\times h \times w} \rightarrow \mathbb{R}^{c\cdot h \cdot w}$
>
>   > anchor representation: $r^{a}=f_\theta(v^a)$
>   >
>   > positive representation: $r^{+}=f_\theta(v^+)$
>   >
>   > negative representations: 所有从其他样本$x'$生成的$r^{-}$
>
>   
>
> **Similarity measure:** 
>
> - SimCLR使用$z=f_\phi$将表示向量从编码器映射到另一个向量空间，即$f_\phi：\mathbb{R}^c \rightarrow \mathbb{R}^c$
> - 用$(z_i，z_j）$对之间的余弦相似性作为相似性得分。
> - 映射和余弦相似度的合成可以看作是一种参数化的相似性度量。
>
> 
>
> **Loss Function:**
> $$
> \mathcal{L}_{\theta}\left(z^{a}, z^{+}, Z^{-}\right)=-\log \frac{\exp \left(\Phi\left(z^{a}, z^{+}\right) / \tau\right)}{\sum_{z_{i}^{-} \in Z^{-}} \exp \left(\Phi\left(z^{a}, z_{i}^{-}\right) / \tau\right)}
> $$
>
> - 加入$\tau \in \mathbb{R}$ 调节 the similarity scores





##### Others (MoCo, BYOL, Swav)

- **Moco (v2)**

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/simclr_arch.png)

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/moco_v2_arch.png)

  > - 修改了SimCLR的projection head $g_\theta(x)$和数据增强的pipeline，减小了批次大小（从4096减少到256）并提高了性能
  >
  > - MoCo将单个网络拆分为参数化的在线网络（顶层）和参数化的动量网络（底层）
  >   -  在线网络通过随机梯度下降进行更新
  >   - 动量网络根据在线网络权重的指数移动平均值进行更新
  >   - 动量网络使MoCo可以**有效地使用过去的预测存储库作为对比损失的负面示例**。 此存储库可实现小得多的批处理大小

  

- **BYOL**

  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/byol_arch.png)

  > - BYOL建立在MoCo的动量网络概念的基础上，并添加了一个MLP以根据z预测z’ (prediction)
  > - BYOL而不是使用对比损失，而是使用归一化预测p和目标z'之间的L2误差，因为此损失函数不需要负示例

  

  > 1）BYOL 用到了两个编码器。第二个编码器实际上完全是第一个编码器的副本，但是它不会在每一轮更新权重，而是使用一种滚动均值（rolling average）进行更新。
  >
  > 2）BYOL 并没有用到负样本，而是依靠滚动权值更新作为一种为训练提供对比信号的方式
  >
  > 3) 批量标准化对于培训BYOL至关重要 (ref: https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html)

  

- **Swav**

  > ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/6.gif)
  >
  > - 通过应用从图像变换集合$T$采样的变换$t$，将每个图像$x_n$变换为增强视图$x_{nt}$
  >
  > - 通过将非线性映射$f_\theta$应用于$x_{nt}$，将扩充后的视图映射到矢量表示，然后将特征映射到单位球面，即
  >   $$
  >   z_{nt}=f_{\theta}\left(\mathbf{x}_{n t}\right) /\left\|f_{\theta}\left(\mathbf{x}_{n t}\right)\right\|_{2}
  >   $$
  >
  > - 通过将$z_{nt}$映射到一组K个可训练的原型向量$\{c_1,...,c_k\}$,
  >
  > 
  >
  > SwAV和contrastive instance learning方法的异同主要是在feature对比上：
  >
  > ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/self-supervised/14.png)
  >
  > 
  >
  > SwAV使用了一个code来表达feature进而来保持一致性，可以看到训练主要包括两部分
  >
  > - z(feature)通过c(prototype)映射得到Q(code)
  >
  >   > $$
  >   > \ell\left(\mathbf{z}_{t}, \mathbf{q}_{s}\right)=-\sum_{k} \mathbf{q}_{s}^{(k)} \log \mathbf{p}_{t}^{(k)}, \quad \text { where } \quad \mathbf{p}_{t}^{(k)}=\frac{\exp \left(\frac{1}{\tau} \mathbf{z}_{t}^{\top} \mathbf{c}_{k}\right)}{\sum_{k^{\prime}} \exp \left(\frac{1}{\tau} \mathbf{z}_{t}^{\top} \mathbf{c}_{k^{\prime}}\right)}
  >   > $$
  >   >
  >   > 
  >
  > - Q通过Swapped Prediction完成loss计算
  >
  >   > $$
  >   > L\left(\mathbf{z}_{t}, \mathbf{z}_{s}\right)=\ell\left(\mathbf{z}_{t}, \mathbf{q}_{s}\right)+\ell\left(\mathbf{z}_{s}, \mathbf{q}_{t}\right)
  >   > $$
  >   >
  >   > 函数$\ell(z,q)$用于衡量特征z和code Q之间的拟合，而对比学习中直接比较特征
  >
  > 



##### Experiments

> - 经过消融实验发现，数据增强阶段对变换的选取对于对比方法最终的性能是十分关键的
> - 通过消融实验发现：更宽的编码器在对比学习任务中性能要好得多
> - 根据实验结果发现，CPC 和 AMDIM 策略对于结果的影响可以忽略不计，反而增加了计算复杂度。**使这些对比方法奏效的主要驱动力是数据增强过程。**
> - 根据实验结果说明：对于相似度的选择在很大程度上是无关紧要的。



#### 4. YADIM 

> (1) Data Augmentation Pipeline:
>
> - We deﬁne a data augmentation pipeline for YADIM as the union of the CPC and AMDIM pipelines. This new pipeline applies all six transforms sequentially to an input twice to generate two version of the same input $v^a\sim A(x)$, $v^ + \sim A(x)$. The same pipeline generates the negative sample from a different input $v^ − \sim A(x')$.
>
> (2) Encoder:
>
> - We use the wide ResNet-34 from AMDIM, although the choice of any other encoder would not have a signiﬁcant impact the ﬁnal performance.
>
> (3) Representation Extraction:
>
> - YADIM compares the triplets from the last feature maps generated a + by the encoder $(r^a_{−1},R^+ _{−1}, R^-_{−1})$
>
> (4) Similarity measure:
>
> - 使用点积$\Phi(a,b)=a\cdot b$
>
> (5) Loss Function:
> $$
> \mathcal{N}_{\theta}\left(r^{a}, R^{+}, R^{-}\right)=-\log \frac{\sum_{r_{i}^{+} \in R^{+}} \exp \left(\Phi\left(r^{a}, r_{i}^{+}\right)\right)}{\sum_{r_{i}^{-} \in R^{-}} \exp \left(\Phi\left(r^{a}, r_{i}^{-}\right)\right)}
> $$
> 