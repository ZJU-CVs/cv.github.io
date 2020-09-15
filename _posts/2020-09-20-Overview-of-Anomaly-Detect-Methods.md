---
layout:     post
title:      Overview of Anomaly Detect Methods 
subtitle:   
date:       2020-09-20
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Anomaly Detection
---



#### P-Net

《Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images》

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/29.png" alt="img" style="zoom:50%;" />

- 异常检测方法主要分为SVDD类（学习判别超平面以将异常样本和正常样本区分来开）和Anomaly类（引入生成对抗网络来进行异常检测）

- 本文认为，纹理和结构信息有助于异常检测，因此提出了P-Net网络结构
  - 从原始图像中提取结构，然后将结构特征与图像特征进行融合以重构图像
  - 使用重构图像进一步提取特征，可以作为正则化器，有助于改善前一阶段的图像重构
  - 同时，通过衡量原始图像和重构图像间的内容误差和结构误差得到异常分数，用于判别正常/异常



##### Details

> 以医学图像为例，对于健康人群，视网膜的脉管系统分布和组织学是规则的，而对于患有疾病的受试者，病变将破坏脉管系统和组织学的规则性



> **模型细节：**
>
> - 网络架构包括三个模块：
>
>   1) 结构提取模块$G_s$，从原始图像$I$中提取结构$S$; 
>
>   2) 图像重构模块$G_r$，利用图像编码器最后一层输出的特征和结构来重构。通过最小化$I$和$\hat{I}$的差异，将纹理和结构之间的关系编码到网络中
>
>   3) 从重构图像模块中提取结构$\hat{S}$，通过最小化$S$和$\hat{S}$的差异。该模块使原始图像被$G_r$正确地重构



> **Structure extraction network with domain adaptation**
>
> 不同数据集间域适应
>
> 
> $$
> \mathcal{L}_{seg}(I_{src})= - \sum S_{src} log(G_s(I_{src}))\\
> \mathcal{L}_{seg}(I_{tar})= \mathbb{E}[log(1-D(G_s(I_{tar})))]+\mathbb{E}[logD(G_s(I_{src}))]
> $$
>
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/34.png" alt="img" style="zoom:80%;" />



> **Image Reconstruction Module**
>
> 分别用En1和En2对原始图像及其结构进行编码，然后将两个特征串联起来，输入解码器De以重建原始图像
>
> 在结构编码器En1和解码器之间为同一级别的特征引入了skip connection，从而避免了由于结构中的下采样池化而导致的信息丢失；
>
> 而在图像编码器En2和解码器之间没有skip connection，防止影响结构对于原始图像重构的作用
>
> 为了提高重建图像的质量，应用PatchGAN中的重构误差
>
> 
> $$
> \mathcal{L}_{\mathrm{rec}}(\mathbf{I})=\Vert\mathbf{I}-\hat{\mathbf{I}}\Vert_{1}\\
> \mathcal{L}_{\mathrm{adv}}(\mathbf{I})=\mathbb{E}\left[\log \left(1-\mathbf{D}\left(\mathbf{G}_{r}(\mathbf{I}, \mathbf{S})\right)\right)\right]+\mathbb{E}[\log \mathbf{D}(\mathbf{I})]
> $$
>



> **Structure Extraction From Reconstructed Image Module**
>
> 进一步将结构提取器$G_s$附加到重建图像上，有两个目的：
>
> 1）通过使从原始图像提取的结构与从重建图像提取的结构相同，可以更好地重建原始图像。 从这个意义上说，从重建的图像模块重建图像的行为像一个正则化器。  
>
> 2）一些病变在结构上更具区分性，分别从原始图像和重建图像中提取结构，并利用它们的差异进行异常检测
>
> 
> $$
> \mathcal{L}_{\mathrm{str}}(\mathbf{I})=\Vert\mathbf{S}-\hat{\mathbf{S}}\Vert_{1}
> $$



> **Objective Function**
>
> 
> $$
> \mathcal{L}=\lambda_1 \mathcal{L}_{adv}+\lambda_2 \mathcal{L}_{rec}+\lambda_s \mathcal{L}_{str}
> $$



> **Anomaly Detection for Testing Data**
>
> 
> $$
> \mathcal{A}(I)=(1-\lambda_f)\Vert I-\hat{I}\Vert_1+\lambda_f\Vert S-\hat{S}\Vert_1
> $$
>



#### DifferNet

《Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows》

> 产品缺陷检测中，许多缺陷都是先验未知的。
>
> 本文提出了**Differnet**，利用卷积神经网络提取的特征描述性来使用归一化流估计其密度。
>
> Normalizing flows非常适合处理**低维数据分布**，本文考虑高维信息，使用多尺度的特征提取器。该提取器能够对流进行归一化处理，从而为图像分配meaningful likehoods
>
> 此外，还开发了指示缺陷的评分函数，通过将分数回传回图像能够进行像素定位
>
> 本方法不需要大量的训练样本，并且在少于16张图像的情况下仍表现良好

##### Related Works

> Detection with Pretrained Network
>
> - 使用预训练网络的特征空间来检测异常
> - 大多数情况下，都是使用简单的传统机器学习方法来获取异常分数，将正常特征的分布建模为单峰高斯分布，从而实现异常检测(如One-Class-SVM，SVDD等)
> - 该技术仅适用于缺陷检测中的特定类别



> Generative Models
>
> - 生成模型能够从训练数据的流形中生成样本，采用的异常检测思想为：由于训练集中不存在异常，因此无法生成异常图像
> - 生成模型适用于各种缺陷检测方案，但是很大程度上取决于异常类型，如缺陷区域的大小和结构会严重影响异常评分

##### Normalizing Flow

> 归一化流是能够学习数据分布和well-defined密度之间的转换神经网络，它的映射是双向的，并且可以在两个方向上进行评估
>
> 首先，对给定样本分配可能性；然后从建模分布中采样来生成数据
>
> 双映射性（bijectivity）通过堆叠固定或自回归的affine transforms layers来保证



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/31.png" alt="img" style="zoom:50%;" />

> Differnet基于从正常的训练图像$x\in X$的特征$y\in Y$中进行密度估计
>
> 其中$f_{ex}:X \rightarrow Y$是未进一步优化的预训练特征提取器的映射
>
> 由$f_{ex}(x)$获得估计$p_Y(y)$，通过应用归一化流：$f_{NF}:Y\rightarrow Z$，将$Y$映射到潜空间$Z$，该空间中具有明确定义的分布$p_Z(z)$
>
> 图像样本的Likehood直接根据$p_Z(z)$得到，异常样本的特征应该out of distribution，因此比正常样本的likehood更低
>
> 为了获得不同比例的结构，并在$y$中有更多的描述性特征，因此将$f_{ex}$定义为三个尺寸的级联



##### Details

Architecture:

> 应用scale和shift操作
>
> 将输入$y_{in}$分为$y_{in,1}$和$y_{in,2}$，通过对子网络$s$和$t$中的乘法和加法分量进行回归来相互操作。这些操作依次应用于各自的对应对象
>
> **scale**和**shift**操作描述为：
>
> 
> $$
> \begin{array}{c}
> y_{\text {out }, 2}=y_{\text {in }, 2} \odot e^{s_{1}\left(y_{\text {in }, 1}\right)}+t_{1}\left(y_{\text {in }, 1}\right) \\
> y_{\text {out }, 1}=y_{\text {in }, 1} \odot e^{s_{2}\left(y_{\text {out }, 2}\right)}+t_{2}\left(y_{\text {out }, 2}\right)
> \end{array}
> $$
> 
>
> - 内部函数$s$和$t$可以是任何可微函数
>
> - 为了保持模型的稳定性，实现更好的收敛性，对$s$的值进行soft-clamping，即对$s$的最后一层使用激活函数，将$s(y)$限制在$(-\alpha,\alpha)$来防止较大的缩放比例分量
>
>   
>
> $$
> \sigma_{\alpha}(h)=\frac{2 \alpha}{\pi} \arctan \frac{h}{\alpha}
> $$
>
> 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/32.png" alt="img" style="zoom:50%;" />



Training:

> 训练过程中的目标是找到$f_{NF}$参数，以使得提取的特征$y\in Y$的可能性最大化
>
> 这些特征在潜在空间$Z$中是可量化的。通过$z=f_{NF}(y)$及变量变化公式，将问题转化为最大化对数似然性：
>
> 
> $$
> p_Y(y)=p_Z(z)\left|\operatorname{det} \frac{\partial z}{\partial y}\right|
> $$
> 本文使用负对数似然损失$\mathcal{L}(y)$转化为最小化问题：
>
> 
> $$
> \begin{array}{c}
> \log p_{Y}(y)=\log p_{Z}(z)+\log \left|\operatorname{det} \frac{\partial z}{\partial y}\right| \\
> \mathcal{L}(y)=\frac{\|z\|_{2}^{2}}{2}-\log \left|\operatorname{det} \frac{\partial z}{\partial y}\right|
> \end{array}
> $$
> 
>
> - 直观地讲，希望$f_{NF}$讲所有$y$映射到尽可能接近$z=0$的位置，同时用接近于$zero^2$的缩放系数来惩罚平凡解



Scoring Function

> 使用计算的似然值作为样本分类异常或正常的标准
>
> 为了获得鲁棒的异常分数$\tau(x)$，使用图像$x$的多次变换$T(i)$对负对数似然进行平均：
>
> 
> $$
> \tau(x)=\mathbb{E}_{T_{i} \in \mathcal{T}}\left[-\log p_{Z}\left(f_{\mathrm{NF}}\left(f_{\mathrm{ex}}\left(T_{i}(x)\right)\right)\right)\right]
> $$
>
> - $T$可以选择旋转和改变亮度及对比度
>
> 如果异常分数$\tau(x)$高于阈值$\theta$，则将图像分类为异常
>
> 
> $$
> \mathcal{A}(x)=\left\{\begin{array}{ll}
> 1 & \text { for } \tau(x) \geq \theta \\
> 0 & \text { for } \tau(x)<\theta
> \end{array}\right.
> $$
> 



Localization

> 可以将负对数似然度$\mathcal{L}$回传到输入图像$x$，每个输入通道$x_c$对应梯度$\nabla x_{c}$值，该值指示多少像素影响与异常有关的误差
>
> 为了获得更好的可视化效果，使用高斯核$G$对这些梯度进行模糊化处理，并根据以下等式得到梯度图$g_x$。将获得的map旋转回去后，对单个图像多次旋转的maps进行平均可以得到鲁棒的定位
>
> 
> $$
> g_x=\sum_{c\in C}\vert G\star \nabla x_{c}\vert
> $$
> 

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/30.png" alt="img" style="zoom:50%;" />





#### DeScarGAN

《DeScarGAN: Disease-Speciﬁc Anomaly Detection with Weak Supervision》

> 提出了一种弱监督和细节保留的方法，能够检测图像的结构变化
>
> 与标准的异常检测方法相比，本文的方法从一组相同患有病理的图像和一组健康图像中提取有关疾病特征的信息



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/33.png" alt="img" style="zoom:50%;" />

> - $\mathcal{F}=${$x\vert x:\mathbb{R}^2 \rightarrow \mathbb{R}$}是一组来自相同成像方式的医学图像，这些图像显示相同的解剖结构，其中$\mathcal{P} \subset \mathcal{F}$是受特定疾病影响的的患者图像集，$\mathcal{H} \subset \mathcal{F}$是健康对照组的图像集。对于给定的一个未知类的新图像，模型的目的是检测图像中与$\mathcal{P}$图像具有相同特征的区域并分配一个类别标签
>
> - 设$p$为$\mathcal{P}$中的图像类别，$h$为$\mathcal{H}$中的图像类别。$c,\bar{c}\in${$h,p$}且$c \neq \bar{c}$ (也就是说，$c$和$\bar{c}$一个为$p$，另一个为$h$)。本文的主要思想是转化一个标签类别为$c$的真实图像$r_c$为一个标签类别为$\bar{c}$的人造图像$a_{\bar{c}}$。**病理区域**定义为人造健康图像$a_h$和类别为$c$的真实图像$r_c$的差异$d:a_h-r_c$
> - 因此，在unpaired的集合$\mathcal{P}$和$\mathcal{H}$之间进行image-to-image的转换。对于任何图像$r_c$，生成器都能生成相同类别$c$的人造图像$a_c$和类别为$\bar{c}$的人造图像$a_{\bar{c}}$。为了保证$r_c$和$a_h$仅在病理区域不同，采用identity loss 和 reconstructtion loss 获得周期一致性(cycle consistency)

##### Details

> - 生成器包括两个分支
>
>   - 生成器 $G_p:\mathcal{F} \rightarrow \mathcal{P}$用于生成类为$p$的图像
>   - 生成器$G_h:\mathcal{F} \rightarrow \mathcal{H}$ 用于生成类为$h$的图像
>
> - 生成器采用U-net的结构，内部的skip connection可以确保人工图像保持输入图像的细节结构，从而使差异图$d$更加准确
>
>   > `注:`由于最上层的skip connection省略，使生成器能够执行结构的更改
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/35.png" alt="img" style="zoom:50%;" />
>
> - 判别器的任务包括：
>   - 分类图像是健康的还是有病理的，$D_{cls} :\mathcal{F}\rightarrow \mathbb{R}$用于二分类。
>   - 判别是真实图像还是人工图像，$D_p:\mathcal{P}\rightarrow \mathbb{R}$用于区分$p$类的图像是人造图还是真实图，$D_h:\mathcal{H}\rightarrow \mathbb{R}$用于区分$h$类的图像是人造图还是真实图
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/36.png" alt="img" style="zoom:50%;" />
>
> > `注：`$D_c$可以互换为$D_p$或$D_h$

##### Loss functions

> **Adversarial Loss:**
>
> 
> $$
> \mathcal{L}_{a d v, d}=-\mathbb{E}_{r_{c}, c}\left[\left(D_{c}\left(r_{c}\right)\right)\right]+\mathbb{E}_{r_{c}, \bar{c}}\left[D_{\bar{c}}\left(G_{\bar{c}}\left(r_{c}\right)\right]+\lambda_{g p} \mathbb{E}_{\hat{x}, c}\left[\left(\left\|\nabla_{\hat{x}} D_{c}\left(\hat{x}_{c}\right)\right\|_{2}-1\right)^{2}\right]\right.\\
> \mathcal{L}_{a d v, g}=-\mathbb{E}_{r_{c}, \bar{c}}\left[D_{\bar{c}}\left(G_{\bar{c}}\left(r_{c}\right)\right]\right.
> $$
> 
>
> **Identity Loss:**
>
> 
> $$
> \mathcal{L}_{i d}=\mathbb{E}_{r_{c}, c}\left[\left\|r_{c}-G_{c}\left(r_{c}\right)\right\|_{2}\right]
> $$
> 
>
> **Classification Loss:**
>
> 
> $$
> \mathcal{L}_{c l s, d}=\mathbb{E}_{r_{c}, c}\left[-\log D_{c l s}^{c}\left(r_{c}\right)\right]
> $$
> 
>
> **Reconstruction Loss:** 
>
> 
> $$
> \mathcal{L}_{r e c}=\mathbb{E}_{r_{c}, c}\left[\left\|r_{c}-G_{c}\left(G_{\bar{c}}\left(r_{c}\right)\right)\right\|_{2}\right]
> $$
> 
>
> **Total Loss Objective:**
>
> 
> $$
> \mathcal{L}_{g}=\lambda_{a d v, g} \mathcal{L}_{a d v, g}+\lambda_{r e c} \mathcal{L}_{r e c}+\lambda_{i d} \mathcal{L}_{i d}+\lambda_{c l s, g} \mathcal{L}_{c l s, g}\\
> \mathcal{L}_{d}=\lambda_{a d v, d} \mathcal{L}_{a d v, d}+\lambda_{c l s, d} \mathcal{L}_{c l s, d}
> $$
> 



#### Patch SVDD

详见[link](https://zju-cvs.github.io/2020/08/23/Patch-SVDD/)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/6.png" alt="img" style="zoom:50%;" />



#### FCDD

详见[link](https://zju-cvs.github.io/2020/08/23/Explainable-Deep-One-Class-Classification/)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/3.png" alt="img" style="zoom:50%;" />



#### Puzzle-AE

《Puzzle-AE: Novelty Detection in Images through Solving Puzzles》

> 引入自监督学习的思想：
>
> - 自监督学习方法能够提取语义上有意义和通用的特征
>
>   （感觉不太靠谱，破坏了结构信息）
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/37.png" alt="img" style="zoom:50%;" />
>
> 
>
> 

