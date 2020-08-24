---
layout:     post
title:      Patch SVDD
subtitle:   Patch-level SVDD for Anomaly Detection and Segmentation
date:       2020-08-23
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Paper
    - Anomaly Detection
---



#### 1. Introduction

> 异常检测和分割：
>
> - 异常检测包括对输入图像是否包含异常进行二元判断
> - 异常分割的目的是对异常进行像素级的定位



> `One-class support vector machine (OC-SVM)`和`Support vector data description (SVDD)`都是用于one-class classification的经典算法
>
> - 在给定核函数的情况下，OC-SVM从内核空间中从原点寻找一个最大边缘超平面
> - SVDD在内核空间中搜索一个data-enclosing hypersphere
>   - 基于深度学习提出了deep-SVDD，通过在核函数的位置部署一个深度神经网络，基于数据表示，无需手动选择合适的核函数



#### 2. Method

> 本文将deep-SVDD拓展到一种patch-wise的检测方法中，并结合自监督学习，实现了异常分割并提高异常检测的性能

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/5.png" alt="img" style="zoom:40%;" />



##### Patch-wise Deep SVDD

> - 训练一个编码器，该编码器将整个训练数据映射为位于特征空间中的small hypersphere内的特征
>
> - 使用以下损失函数训练编码器$f_{\theta}$以最小化特征与超球面中心之间的欧式距离：
>
> $$
> \mathcal{L}_{SVDD}=\sum_i \Vert f_{\theta}(x_i)-c \Vert_2
> $$
>
> - 在测试时，将输入的表示与**中心** $c$ 之间的距离作为异常分数
>
> $$
> \mathbf{c} \doteq \frac{1}{N} \sum_{i}^{N} f_{\theta}\left(\mathbf{x}_{i}\right)
> $$



> 本文将Deep SVDD这种方法拓展到**patch-wise**，即编码器对每个batch进行编码，而不是整张图
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/6.png" alt="img" style="zoom:40%;" />
>
> - Patch-wise inspection具有以下优点：
>   - 可以在每个位置获得检测结果，从而定位缺陷位置
>   
> - 这种细粒度的检测提高了整体检测的性能
>   
>   - 由下图可见
>   
>     > 对于相对简单的图像，使用$\mathcal{L}_{Patch \ SVDD}$训练的编码器
>     >
>     > 和使用$\mathcal{L}_{Patch}$训练的编码器都能很好地定位缺陷；       
>     >
>     > 然而对于较复杂的图像，$\mathcal{L}_{SVDD}$无法定位
>   
>     
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/7.png" alt="img" style="zoom:40%;" />
>
> - 对于高复杂度的图像，由于patch之间具有较高的类内差异(即一些patch对应于背景，而其他patch包含对象)，将不同patch的所有特征映射到单个中心会削弱表征(representation)和内容(content)之间的联系。因此，使用单个中心$c$是不合适的。
> - 为了解决上述问题，没有使用定义中心并分配patch的方法。而是训练编码器本身gather语义上相似的patch。其中语义相似的patch通过对空间相邻patch进行采样而获得的，并且使用以下损失函数对编码器进行训练，使特征之间的距离最小化
>
> $$
> \mathcal{L}_{SVDD'}=\sum_{i,i'} \Vert f_{\theta}(p_i)- f_{\theta}(p_{i'}) \Vert_2
> $$
>
> > 其中$p_{i'}$是$p_i$附近的patch



##### Self-supervised learning

`自监督编码器可以作为下游任务强大的特征提取器`

> 对于随机采样的patch $p_1$，在$3\times 3$网格中从其八个邻域之一采样另一个补丁$p_2$
>
> 
>
> 如果将真实的相对位置设置为$y\in$ {$0,1,\cdots,7$}，则训练分类器$C_{\phi}$以正确预测$y=C_{\phi}(f_{\theta}(p_1),f_{\theta}(p_2))$
>
> 自监督学习的损失函数为：
>
> 
> $$
> \mathcal{L}_{SSL}={Cross-entropy}(y,C_{\phi}(f_{\theta}(p_1),f_{\theta}(p_2)))
> $$
> 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/8.png" alt="img" style="zoom:40%;" />
>
> 
>
> 因此训练编码器的最终损失函数为：
>
> 
> $$
> \mathcal{L}_{Patch \ SVDD}=\lambda \mathcal{L}_{SVDD'}+\mathcal{L}_{SSL}
> $$
> 



##### Hierarchical encoding

> 由于异常的大小不同，因此部署具有不同接受范围的多个编码器有助于应对大小变化
>
> 因此采用一个hierarchical encoder，定义为：
>
> 
> $$
> f_{big}(p)=g_{big}(f_{small}(p))
> $$
>
> > 将输入patch划分为$2\times 2$，输入$f_{small}$得到局部特征
> >
> > 将局部特征聚合输入$g_{big}$得到一个全局特征特征
>
> 
>
> 本文实验中，编码器$f_{big}$的感受野/接受域为$K=64$，$f_{small}$的感受野/接受域为$K=32$，编码器都进行自监督训练
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/9.png" alt="img" style="zoom:40%;" />



##### Generating anomaly maps

`训练编码器后，编码器的表示将用于检测异常`

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/10.png" alt="img" style="zoom:40%;" />

> - 首先，计算并存储每个正常训练patch的表示{$f_{\theta}(p_{normal})\mid p_{normal}$}
>
> - 给定一个查询图像，根据$stride=S$划分得到大小为$K$的patch，并使用经过训练的编码器提取特征。计算特征空间中与其最接近的正常patch的$L_2$距离作为异常分数
>
> 
> $$
> \mathcal{A}_{\theta}^{\text {patch }}(\mathbf{p}) \doteq \min _{\mathbf{p}_{\text {normal }}}\left\|f_{\theta}(\mathbf{p})-f_{\theta}\left(\mathbf{p}_{\text {normal }}\right)\right\|_{2}
> $$
> 
>
> - 然后将逐块计算的异常分数分配给像素，生成异常图$\mathcal{M}$
>
> 
>
> - 考虑multiple编码器($f_{small},f_{big}$)构成了多个特征空间，从而产生多个异常图，
>
>   通过element-wise multiplication 聚合多个异常图，得到最终的$\mathcal{M}_{multi}$
>
> 
> $$
> \mathcal{M}_{multi}\doteq  \mathcal{M}_{small} \odot \mathcal{M}_{big}
> $$
>
> - $\mathcal{M}_{multi}$中异常分数高的像素被认为包含缺陷，结合来自多个编码器的多尺度结果能提高检测性能
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/11.png" alt="img" style="zoom:40%;" />
>
> - $\mathcal{M}_{multi}$像素中最大的异常分数即为该检测图像的**异常分数**$\mathcal{A}$
>
> 
> $$
> \mathcal{A}^{image}_{\theta}(x) \doteq \max_{i,j} \mathcal{M}_{multi}(x)_{ij}
> $$
> 



##### Training Process

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/12.png" alt="img" style="zoom:40%;" />

##### Test Process

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/13.png" alt="img" style="zoom:40%;" />



#### 3. Experiments

##### The eﬀect of the losses

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/14.png" alt="img" style="zoom:40%;" />

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/15.png" alt="img" style="zoom:40%;" />



##### The eﬀect of  the hierarchical encoders

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/16.png" alt="img" style="zoom:40%;" />