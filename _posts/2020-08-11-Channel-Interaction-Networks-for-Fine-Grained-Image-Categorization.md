---
layout:     post
title:      Channel Interaction Networks for Fine-Grained Image Categorization
subtitle:   
date:       2020-08-11
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-grained
---



#### 1. Introduction

- Fine-grained 图像分类的目的是对属于同一基础类别的图像进行更加细致的子类划分

- 由于细微的**类间差异**和较大的**类间差异**，许多差异只能通过聚焦于**区分性区域块**(discriminative local parts)以实现有效区分。因此，与普通的图像分类任务相比，细粒度图像分类任务难度更大

  > 如下图，为了区分三种鸟类子类别，神经网络通常关注它们的翅膀和头部
  >
  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/10.png" alt="img" style="zoom:50%;" />

- 相关工作和方法

  > 对图像进行细粒度分类的方法，大多都是以深度卷积网络为基础的，大致可以分为以下四个方向：
  >
  > - 基于常规图像分类网络的微调方法
  >
  >   > 采用常见的深度卷积网络(VGG, ResNet, SENet...) + 对网络权值进行Finetune 来进行图像细粒度分类
  >
  > - 基于细粒度特征学习(fine-grained feature learning)的方法
  >
  >   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/6.png" alt="img" style="zoom:50%;" />
  >   >
  >   > 采用Bilinear CNNs (B-CNNs)模型，通过计算不同空间位置的外积，并对不同空间位置计算平均池化得到双线性特征。外积捕获了特征通道之间成对的相关关系，通过计算从而提高细粒度特征表示性能，可以表示为$\beta=(f_A,f_B,P,C)$的形式，$f_A$和$f_B$表示CNN的特征提取函数，$P$表示池化函数，$C$表示分类函数。
  >   >
  >   > 
  >   >
  >   > **Details:**   
  >   >
  >   >   $$
  >   >   f: \mathcal{L}\times \mathcal{I} \rightarrow \mathbb{R}^{K\times D}
  >   >   $$
  >   >
  >   >   > 特征函数获取图像$I\in \mathcal{I}$和位置$l\in \mathcal{L}$，并输出大小为$K\times D$的特征
  >   >
  >   > $$
  >   >   \text { bilinear }\left(l, I, f_{A}, f_{B}\right)=f_{A}(l, I)^{T} f_{B}(l, I) \in \mathbb{R}^{M\times N}\\
  >   >   \Phi(I)=\sum_{l \in \mathcal{L}} \text { bilinear }\left(l, I, f_{A}, f_{B}\right)=\sum_{l \in \mathcal{L}} f_{A}(l, I)^{T} f_{B}(l, I) \in \mathbb{R}^{M\times N}
  >   > $$
  >   >
  >   >   > 两个不同的特征提取函数$f_A$和$f_B$输出为$\mathbb{R}^{K\times M}$和$\mathbb{R}^{K\times N}$
  >   >   >
  >   >   > 通过矩阵外积得到每个位置$l$的特征输出
  >   >   >
  >   >   > 通过sum pooling对所有位置进行求和，在池化处理的过程中忽略了特征的位置，因此得到的bilinear特征是无序表示   
  >   > 
  >   > $$
  >   > \begin{array}{c}
  >   >   y=\operatorname{sign}(x) \sqrt{|x|} \\
  >   >   z=\frac{y}{\|y\|_{2}}
  >   >   \end{array}
  >   > $$
  >   > 
  >   >  > 对$x=\Phi(I)$进行归一化操作，输入分类函数中进行分类
  >   
  > - 基于目标块的检测(part detection)和对齐(alignment)的方法
  >
  >   > 先在图像中检测出目标所在的位置，然后再检测出目标中有区分性区域的位置，然后将目标图像（即前景）以及具有区分性的目标区域块同时送入深度卷积网络进行分类。
  >
  > - 基于视觉注意机制(visual attention)的方法
  >
  >   > 可以在不需要额外标注信息（比如目标位置标注框和重要部件的位置标注信息）的情况下，定位出图像中有区分性的区域
  >



#### 2. Method

> 提出了Channel Interaction Network (CIN)用于细粒度图像分类，基本架构如下图。
>
> - 给定一个图片对，首先用共享的backbone进行处理(如ResNet50)，生成一对卷积特征图
> - 通过一个self-channel interaction(SCI) 模块来对不同通道之间的相关性进行建模，计算的到channel-wise的补充信息，然后将原始特征图和补充信息进行聚合得到判别特征
> - 通过结合对比损失设计的一个contrastive channel interaction(CCI)模块，对两个图像之间的通道关系进行建模
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/9.png" alt="img" style="zoom:50%;" />

##### SCI Module

> 由于不同的特征通道中存在丰富的编码知识，因此设计SCI模块用于对图像中不同通道之间的相互作用进行建模，以获得每个通道的通道补充信息，从而增强每个通道学习的**判别特征(discriminate feature)**
>
> - 输入图像$I$，通过backbone提取特征得到$X'\in \mathbb{R}^{w\times h\times c}$，然后reshape成$X\in \mathbb{R}^{c\times l},l=w\times h$
>
> - SCI模块的输出是$Y=WX\in\mathbb{R}^{c\times l}$，其中$W\in \mathbb{R}^{c\times c}$表示SCI的权重矩阵，
>   $$
>   W_{i j}=\frac{\exp (-X X^{T} _{i j})}{\sum_{k=1}^{c} \exp (-X X^{T} _{i k})}
>   $$
>
>   
>   > 其中$\sum_{k=1}^{c} W_{i k}=1$，$Y_i$表示$X_i$和所有通道之间的交互，$Y_i=W_{i1}X_1+...+W_{ic}X_c$
>   >
> > 如下图所示，权重较大的通道在语义上倾向于与$X_i$互补，如$X_i$通道关注于头部，因此突出翅膀和脚的互补通道具有较大的权重，而突出头部的通道具有较小的权重
>   
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/11.png" alt="img" style="zoom:50%;" />
>   
> - 因为生成的特征$Y$会丢失原始特征的某些信息，因此将生成特征和原始特征聚合在一起得到判别特征$Z$ (discriminate features)
>
>   $Z=\phi(Y)+X$，其中$\phi$表示$3\times 3$卷积层



> SCI 模块可以形式化为Non-local： $Y=f(X,X)g(X)$
>
> - 其中$f(X,X)=softmax(-XX^{\top})\in\mathbb{R}^{c\times c}$，$g(X)=X\in \mathbb{R}^{c\times l}$
>
> - 与考虑空间维度相互作用的non-local模块不同，SCI更关注通道维度；此外，non-local更倾向于利用空间位置之间的正相关性，而SCI模块更关注负相关性，使模型能够发现语义上互补的通道信息



##### CCI Module

> 学习图像之间的通道关系，动态地从图像对中识别出**判别区域(discriminate region)**，以捕获细粒度分类中的细微差异
>
> - 利用图像$I_A$和$I_B$的SCI权重矩阵和生成特征($W_A, W_B, Y_A, Y_B$)可以得到CCI权重矩阵$W_{AB}$和$W_{BA}$     
>   $$
>   W_{AB}=\mid W_A-\eta W_{B}\mid, W_{B A}=\mid W_{B}-\gamma W_{A} \mid
>   $$
>
>   > 其中$\eta=\psi(\left[Y_{A}, Y_{B}\right]),\psi\left(\left[Y_{B}, Y_{A}\right]\right)$，$\psi$为全连接层，$\vert \vert$表示绝对值
>   >
>   > 使用减法能够抑制两张图片的共性，并突出显示独特的通道关系
>
> - 将CCI的权重矩阵$W_{AB}$和$W_{BA}$应用于特征$X_A$和$X_B$     
>   $$
>   Z'_A=\phi(Y'_A)+X_A=\phi(W_{AB}X_A)+X_A, Z'_B=\phi(Y'_B)+X_B=\phi(W_{BA}X_B)+X_B
>   $$
>
> - Loss Function
>
>   - 使用contrastive loss作为损失函数，假设每个batch有N个image pairs (2N images)，可得：       
>
>   $$
>   L_{cont}=\frac{1}{N} \sum_{A, B} \ell\left(Z_{A}^{\prime}, Z_{B}^{\prime}\right)
>   $$
>
>   - 其中$\ell$的定义如下：        
>     $$
>     \ell=\{\begin{array}{ll}
>     \Vert h(Z_{A}^{\prime})-h(Z_{B}^{\prime})\Vert^{2}, & \text { if } y_{A B}=1 \\
>     \max (0, \beta-\Vert h(Z_{A}^{\prime})-h(Z_{B}^{\prime})\Vert)^{2}, & \text { if } y_{A B}=0
>     \end{array}
>     $$
>     
>   
>     > $y_{AB}=1$表示图像$I_A$和$I_B$来自同一类别，$y_{AB}=0$表示negative pair，$h$表示将特征映射到$r$空间的全连接层。
>     >
>     > 对于正样本对，这个loss随着样本对生成表征之间的距离减小而减少，从而拉近正样本对；对于负样本对，loss只有在样本对生成表征的距离都大于$\beta$时为0。(设置阈值$\beta$的目的是**当某个负样本对中的表征足够好，体现在其距离足够远的时候，就没有必要在该负样本对中浪费时间去增大这个距离了，因此进一步的训练将会关注在其他更加难分别的样本对中**)
>   
>   - 最终的损失函数：     
>     $$
>     L_{total}=L_{soft}+\alpha \cdot L_{cont}
>     $$
>   
>     > 其中$L_{soft}$表示基于SCI生成的特征$Z$，使用softmax loss进行分类预测的损失



#### 3. Experiment

##### Datasets

> - **CUB-200-2011**: 11,788 images from 200 wild bird species
> - **Stanford Cars**: 16,185 images over 196 classes
> - **FGVC Aircraft**: 196 classes about 10,000 images.



##### Ablation Analysis

`以CUB-200-2011作为数据集，通过消融实验更好地了解每个模块对模型的效果的影响`

ResNet-50和ResNet-101作为backbone

- **SCI Module**的作用

  > - 与单独使用ResNet-50相比(84.9%)，仅加入SCI模块ResNet-50+SCI (87.1%)，性能提高了2.2%
  >
  > - 与常见的通道注意力模块SE相比，性能明显提高，因为SE仅关注最具区别性的特征而忽视了其他特征，SCI则利用通道互补来增强所有特征
  > - 与加入Non-local模块(关注positive space information)和Pos-SCI模块(关注positive channel-wise information)相比，SCI模块具有更好的性能

- **CCI Module**的作用

  > - 与没有对比损失的方法(ResNet-50+SCI)相比，引入CCI模块的方法(ResNet-50+SCI+CCI)的性能提升了0.4%
  > - ResNet-50+SCI+Cont将直接使用SCI权重矩阵得到特征进行对比学习，提升效果有限(87.1% ->87.2%)，而CCI模块能够突出明显不同的区域

- **Time Cost**

  > ResNet-50+CIN是一个1-Stage方法，与2-stage方法 ResNet-50+NTS相比，速度明显提升

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/13.png" alt="img" style="zoom:40%;" />

##### Plug-and-Play

> CIN方法中的SCI模块和CCI模块具有通用性和灵活性，能够很容易地集成到其他框架中，提高性能。（如将模块与NTS方法结合） 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/16.png" alt="img" style="zoom:50%;" />

##### Comparison with SOTA

> 在不同的数据集下，所提方法CIN均达到了最高精度
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/14.png" alt="img" style="zoom:30%;" />



##### Qualitative Visualization

> - 下图显示了三个不同数据集图像的SCI可视化，其中第一列表示SCI之前随机选择的通道$X_i$的激活，第2～4列表示其最互补的通道，最后一列表示经过SCI后的通道$Y_i$激活。表明SCI有效地进行了不同通道之间的交互，并结合了它们互补但有区别的部分以产生更多信息。
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/15.png" alt="img" style="zoom:30%;" />
>
> - 下图展示了CCI模块的特征激活，  “salty backed gull”和“Ivory Gull”的头部相似，因此CCI之后的特征在头部的激活较弱。 与“fish grow”相比，头部附近的激活作用更强。 对于其他两种鸟类，它们的外观差异很大，CCI模块可对整个身体部位提供强的响应。 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/12.png" alt="img" style="zoom:30%;" />
>
> 

#### 4. Conclusion

- 提出了一种用于细粒度图像分类的新通道交互网络
- SCI模块考虑通道之间的关系进行通道间信息的学习互补，CCI模块结合了对比学习方法，利用样本对之间的通道相关性，拉近正样本对并分离负样本对（pull positive pairs closer while pushing negative pairs away）
- CIN是一种端对端训练的网络，不需要边界框/部位的标注。

