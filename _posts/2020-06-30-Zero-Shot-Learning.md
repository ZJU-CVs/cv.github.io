---
layout:     post
title:      Zero-shot Learning
subtitle:   零样本学习
date:       2020-06-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Overview
---



#### Introduction

##### ZSL介绍

> - zero-shot learning 得到的模型具有通过推理识别新类别的能力
>
> - zero-shot learning的一个重要理论基础就是利用高维语义特征代替样本的低维特征，使得训练的模型具有迁移性
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/1.jpg" alt="img" style="zoom:50%;" />



##### ZSL定义

(1) 零样本学习：准确预测unseen的类别

> 以图片分类为例，zsl模型训练及测试数据包括：
> - 训练集数据$X_{tr}$以及标签$Y_{tr}$
> - 测试集数据$X_{te}$以及标签$Y_{te}$
> - 训练集类别的描述$A_{tr}$以及测试集类别的描述$A_{te}$，每个类别$y_i \in Y$都可以表示成一个语义向量$a_i \in A$，这个语义向量的每一个维度都表示一种高级的属性
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/2.jpg" alt="img" style="zoom:50%;" />



(2) 广义零样本学习：要求准确预测**seen和unseen的类别**



##### ZSL研究分类

> - CIII (Class-Inductive Instance-Inductive setting): 只使用训练实例和seen标签集合来训练模型
> - CTII (Class-Transductive Instance-Inductive setting): 使用训练实例和seen标签集合，外加unseen标签集合来训练模型
> - CTIT(Class-Transductive Instance-Inductive setting): 使用训练实例和seen标签集合，外加unseen标签集合，对应未标注的测试集合来训练模型
>
> 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/10.png" alt="img" style="zoom:50%;" />



#### Related Works

ZSL算法框架主要分为三个部分：

> (1) 样本数据特征空间X的学习（如利用深度网络backbone提取图片特征）
>
> (2) 构建语义空间A中class的描述，即构建seen class和unseen class之间的潜在语义关系
>
> (3) 特征空间X和语义空间A之间的映射



- *其中，图像特征空间的表示学习可以利用大型数据集训练下的分类模型进行迁移作为特征提取器*

- *语义空间A的构建主要有以下几种方式：*

  > - attribute description: 数据集中的每个class都附加了一可描述的attributes
  >
  > - embedding表示：每个class可作为词获取语义向量
  > - Knowledge Graph/Knowledge Base：每个class可对应KG/KB中的一个实体

  

#### Problems

##### 领域漂移(domain shift problem)

> ref: [Transductive Multi-View Zero-Shot Learning](https://arxiv.org/pdf/1501.04560.pdf)
>
> **问题：**同一类别语义的具体实体之间差别可能很大。因为样本的特征维度比语义维度大很多，所以在映射到attribute space的过程中会丢失信息。
>
> 
>
> **解决方案**：提出transductive multi-view embedding，将映射到语义空间中的样本再重建回去，目标函数为：$$\min \left\|X_{t r}-W^{T} A_{t r}\right\|^{2}+\lambda\left\|W X_{t r}-A_{t r}\right\|^{2}$$ (类似于一个自编码器)
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/3.png" alt="img" style="zoom:50%;" />



##### 枢纽点问题(Hubness problem)

> ref: [Hubness and Pollution: Delving into Cross-Space Mapping for Zero-Shot Learning]()
>
> **问题：**在高维空间中，某些点会成为大多数点的最邻近点，而许多zsl方法在计算最终正确率时使用的是K-NN，因此会收到hubness problem的影响
>
> 
>
> **解决方法：**建立语义空间到特征空间的映射，$min⁡‖X_tr-W^T A_tr ‖^2+λ‖WX_tr-A_tr ‖^2$



##### 语义间隔问题(semantic gap)

> ref: [Zero-Shot Recognition using Dual Visual-Semantic Mapping Paths](https://arxiv.org/pdf/1703.05002.pdf)
>
> **问题：**样本的特征往往是视觉特征，而语义表示是非视觉的，导致样本在特征空间中所构成的流形与语义空间中类别构成的流形不一致
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/4.jpg" alt="img" style="zoom:80%;" />
>
> **解决方法：**将语义属性流形结构与输入样本的流形结构对齐（align），从而训练出一个更好的分类器
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/5.png" alt="img" style="zoom:80%;" />



#### Methods 

ref: [A survey of zero-shot learning: Settings, methods, and applications](https://dl.acm.org/doi/pdf/10.1145/3293318)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/9.png" alt="img" style="zoom:50%;" />



https://zhuanlan.zhihu.com/p/111602525

##### Based on Attribute Description (Inductive Setting)

(1) [Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer](https://ieeexplore.ieee.org/document/5206594)

>零样本问题的开创性文章，主要提出了直接属性预测(direct attribute prediction)和间接属性预测(indirect attribute prediction)
>
>> DAP: 在样本x和训练类别标签y之间加入一个属性表示层A，利用监督学习方式，学习到从x生成A的属性参数$\beta$。在测试阶段，可以利用属性表示层A来表示测试数据产生的类别Z，从而实现迁移学习
>>
>> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/6.png" alt="img" style="zoom:50%;" />
>>
>> IAP: 在训练阶段，在标签y上学习一层属性表示层A；在测试阶段，利用标签层y和属性层A推测出测试数据的类别z
>>
>> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/7.png" alt="img" style="zoom:50%;" />



(2) [Semantic Autoencoder for Zero-Shot Learning](https://arxiv.org/pdf/1704.08345.pdf)

> - 使用自编码器对原始样本进行编码，如下图所示，X为样本，S为自编码器的隐层，$\hat X$隐层重建的X的表示
>
> - 与普通自编码器的隐层不同，隐层S为属性层，是原样本X的抽象语义特征表示
>
> - 具体实现：
>
>   > - 设输入层到隐层的映射为W，隐层到输出层的映射为$W^*$，$W$和$W^*$是对称的，即$W^* = W^T$。期望输入和输出尽可能相似，目标函数为$$\min _{\mathbf{w}, \mathbf{w}^{*}}\left\|\mathbf{X}-\mathbf{W}^{*} \mathbf{W} \mathbf{X}\right\|_{F}^{2}$$
>   > - 期望中间隐层S具有抽象的语义，能表示样本属性或者类别标签，所以加入约束，使自编码器成为监督学习的问题。$$\min _{\mathbf{W}, \mathbf{W}^{*}}\left\|\mathbf{X}-\mathbf{W}^{*} \mathbf{W} \mathbf{X}\right\|_{F}^{2} \text { s.t. } \mathbf{W X}=\mathbf{S}$$
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/8.png" alt="img" style="zoom:50%;" />



##### Based on Embedding Resprentation (Inductive Setting)

(1) [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/11.png)

> - 数据集中每个class/label可作为一个词在语义空间进行embedding表示(如使用预训练skip-gram模型得到有关class的language feature vector，同时利用预训练的CNN-based模型提取图片的visual feature vector)
> - 将两个向量映射到同一维度的空间，进行相似度计算
> - 测试时，可根据语义之间的相似性进行图像分类



##### Based on Knowledge Graph (Inductive Setting)

(1) [Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs](https://arxiv.org/pdf/1803.08035.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/12.jpg" alt="img" style="zoom:80%;" />

> - 模型分为两个独立的部分
>
> - 首先使用CNN-based方法将输入图像抽取特征向量
>
> - 其次，将数据集中每个class作为graph中的一个节点，并对其做embedding表示并输入GCN(即输入为由N个k维节点组成的N*k特征矩阵（k是word-embedding vector的维数），通过GCN每一层之间信息的传递和计算，为每个节点输入一组D维的权重向量
>
> - 模型训练时，
>
>   - GCN的输入是n个类别(包括seen class和unseen class)的语义向量$$\mathcal{X}=\left\{x_{i}\right\}_{i=1}^{n}，\mathcal{X}\in n\times k$$，输出是每个类别的分类器参数$$\mathcal{W}=\left\{\hat{w_{i}}\right\}_{i=1}^{n}, \mathcal{W}\in n\times D$$，GCN每个节点的输出维度$D$和CNN输出的特征维度相等
>   - 训练时用seen类的CNN输出特征向量作为监督信号$w_i$（绿色节点）训练GCN模型的参数监督学习来更新整个GCN，$$\frac{1}{m} \sum_{i=1}^{m} L_{\mathrm{mse}}\left(\hat{w}_{i}, w_{i}\right)$$
>
> - 模型测试时，gcn中的unseen class节点输出对应的权重向量，同时，与CNN输出的图片特征向量做点乘，得到分类结果
>
>   `注：GCN的结构通过可表示ImageNet class之间结构的WordNet知识库得到`
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/13.png" alt="img" style="zoom:80%;" />

**模型优势：**

对每个实体的特征进行提取的同时，保留实体之间的语义结构关系



补充：

> <IJCAI-2018>Fine-grained Image Classification by Visual-Semantic Embedding
>
> <CVPR-2018>Multi-Label Zero-Shot Learning with Structured Knowledge Graphs
>
> <NIPS-2009>Zero-Shot Learning with Semantic Output Codes  



##### Based on Few-shot Learning (Transductive Setting)

Transductive setting 通过seen class和unseen class的少量样本，得到class之间的关系，常用few-shot learning（FSL）

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/14.jpg" alt="img" style="zoom:50%;" />



(1) [Learning to Compare: RelationNetwork for Few-Shot Learning](https://arxiv.org/pdf/1711.06025.pdf)

> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/15.png" alt="img" style="zoom:50%;" />
>
> 上述网络模型可应用于few-shot learning和zero-shot learning，整体架构由embedding module和relation module组成。embedding module用于产生query image和training image，relation module用于对query image和training image进行比较
>
> - **few-shot learning**
>
>   > - 存在三个数据集：training set，support set，testing set。如果support set中包含C个不同的类，并且每个类中包含K个标签的样本时，此时few-shot问题为C-way K-shot问题（k=1时为one-shot，k=0时为zero-shot）
>   >
>   > - **训练时：**
>   >   - 每一轮迭代时随机从训练集training set中挑出C个类，每个类包含K个样本，来构成sample set $S=\{(x_i,y_i)\}$，然后从剩下的部分中也挑出C个类，构成query set Q=$\{(x_j,y_j\}$
>   >   - $x_i$和$x_j$通过embedding module $f_{\phi}$产生两个feature map $f_{\phi}(x_i)$和$f_{\phi}(x_j)$，然后结合起来$C(f_{\phi}(x_i),f_{\phi}(x_j))$得到$F_c$
>   >   - $F_c$输入relation module $g_{\phi}$，并产生一个(0,1)之间的系数，**用来表征$x_i$和$x_j$之间的相似度**
>   > - **测试时：**
>   >   - support set作为对比学习的样例，拥有和测试集一样的标签
>   >   - 通过将support set与testing set做对比来实现对测试数据的识别
>
> - **zero-shot learning**
>
>   > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/16.png" alt="img" style="zoom:50%;" />



补充

> <ICLR-2018>Few-Shot Learning with Graph Neural Networks
>
> <BigData-2017>One-shot Learning for Fine-grained Relation Extraction via ConvolutionalSiamese Neural Network
>
> <NIPS-2016>Matching Networks for One Shot Learning
>
> <NIPS-2017>Prototypical Networks for Few-hot Learning
>
> <ICLR-2017>Optimization as a model for few-shot learning
>
> <ICML-2016>Meta-learningwith Memory-augmented Neural Networks



#### Datasets

> （1）**Animal with Attributes（AwA）**官网：[Animals with Attributes](https://link.zhihu.com/?target=https%3A//cvml.ist.ac.at/AwA/)
>
> > 提出ZSL定义的作者，给出的数据集，都是动物的图片，包括50个类别的图片，其中40个类别作为训练集，10个类别作为测试集，每个类别的语义为85维，总共有30475张图片。但是目前由于版权问题，已经无法获取这个数据集的图片了，作者便提出了AwA2，与前者类似，总共37322张图片。
>
> （2）**Caltech-UCSD-Birds-200-2011（CUB）**官网：[Caltech-UCSD Birds-200-2011](https://link.zhihu.com/?target=http%3A//www.vision.caltech.edu/visipedia/CUB-200-2011.html)
>
> > 全部都是鸟类的图片，总共200类，150类为训练集，50类为测试集，类别的语义为312维，有11788张图片。
>
> （3）**Sun database（SUN）**官网：[SUN Database](https://link.zhihu.com/?target=http%3A//groups.csail.mit.edu/vision/SUN/)
>
> > 总共有717个类别，每个类别20张图片，类别语义为102维。传统的分法是训练集707类，测试集10类。
>
> （4）**Attribute Pascal and Yahoo dataset（aPY）**官网：[Describing Objects by their Attributes](https://link.zhihu.com/?target=http%3A//vision.cs.uiuc.edu/attributes/)
>
> > 共有32个类，其中20个类作为训练集，12个类作为测试集，类别语义为64维，共有15339张图片。
>
> （5）**ILSVRC2012/ILSVRC2010（ImNet-2）**
>
> > 利用ImageNet做成的数据集，由ILSVRC2012的1000个类作为训练集，ILSVRC2010的360个类作为测试集，有254000张图片。它由 4.6M 的Wikipedia数据集训练而得到，共1000维。
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/dataset.jpg" alt="img" style="zoom:70%;" />
>
> 注：`上述数据集中（1）-（4）都是较小形（small-scale）的数据集，（5）是大形（large-scale）数据集`
>
> 
>
> - 提供一些已经用GoogleNet提取好的数据集图片特征
>
>   > [Zero-Shot Learing问题数据集分享（GoogleNet 提取)](https://zhuanlan.zhihu.com/p/29807635)