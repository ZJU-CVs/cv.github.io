---
layout:     post
title:      Destruction and Construction Learning for Fine-grained Image Recognition
subtitle:   
date:       2020-08-21
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Fine-grained

---



#### 1. Introduction

- 目标局部（part）的精细特征表示在精细识别（fine-grained recognition）中起着关键作用。现有的精细识别方法可以大致分为两类：

  - 一种是首先定位有判别性的目标局部区域，然后根据这些判别区域进行分类，这种两步法需要在目标或目标局部上添加额外的边界框标注，这些标注的成本往往都很高，如图(a)
  - 另一种以无监督的方式通过注意力机制自动定位判别区域，因此不需要额外的注视，然而这些方法通常需要额外的网络结构（如注意力机制），因此为训练和预测阶段引入了额外的计算开销，如图(b)

  

- 本文提出了一种DCL(Destruction and Construction Learning)方法来增加精细识别的精度。在处理标准分类骨干网络之外，引入了另一个“破坏和构建” DCL分支按规则“破坏”然后重建输入图像，用于学习具有判别性的区域特征。具体来说，对于“破坏”部分，首先将输入图像划分为局部区域，然后通过区域混淆机制(Region Confusion Mechanism, RCM)把它们打乱，为了补偿RCM引入的噪声，使用能区分原始图像和破坏图像的对抗损失，以抑制RCM引入的噪声分布；对于“构造”部分，使用一个区域对齐网络对打乱的局部区域之间的语义相关性建模，用于恢复局部区域的原始空间分布。通过参数共享的联合训练，DCL向分类模型提供更多有判别性的局部细节。一方面DCL自动定位判别区域，因此在训练时不需要额外的标注，另一方面，DCL结构仅在训练阶段采用，因此在预测时不引入计算开销，如图(c)

  - 改进点：

    > 通过使用RCM训练分类器，除目标的分类标签外不需要任何先验知识就可以自动检测判别性的区域；
    
    >不仅考虑了精细的局部区域特征表示，还考虑了整个图像中不同区域之间的语义相关性；计算高效，在预测时除了骨干分类网络外没有额外的计算开销。
  
  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/ Fine-grained.png)
  
  

#### 2. Method

`DCL框架由四个部分组成，分别是Region Confusion Mechanism，Classification Network，Region Alignment Network和Adversarial Learning Network。`

`当预测时，只需要classification network`

`DCL可以分为Destruction Learning 和 Construction learning`两个学习过程

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/DCL.png)



##### **Destruction Learning**

(1) 区域混乱机制(RCM)

- 将输入图像划分为局部图块然后随机打乱，旨在破坏图像局部区域的空间分布

  - 原因：在精细识别中局部细节比全局结构起着更重要的作用，因为来自不同精细类别的图像通常具有相同的全局结构或形状，仅在局部细节上不同，放弃全局结构保持局部细节可以迫使网络关注具有判别性的局部区域。RCM将局部区域打乱，就会忽略对精细识别不重要的无关区域，并将迫使网络基于判别性的局部细节对图像进行分类。

- 具体细节

  > 给定输入图像$I$，首先将图像均匀地划分为$N{\times}N$个子区域，每个子区域由$R_{i,j}$表示，
  >
  > 其中$i,j$分别是水平和垂直索引（$1 \leq i, j \leq N$）

  

  > RCM将这些分块的局部区域混合在它们的2D邻域。对于第$j$行的子区域$R$，会生成长度为$N$的随机向量$q_j$，其中第$i$个元素$q_{i,j}=i+r, r \sim U(-k,k)$是服从均匀分布的随机变量，$k$是定义领域范围的可调参数($1\leq k<N$)。这样通过对数列$q_j$重新排序得到第$j$行区域的新排列$\sigma_{j}^{r o w}$，同时可以验证：
  >
  > 
  > $$
  > \forall i \in 1, \ldots, N,\left|\sigma_{j}^{r o w}(i)-i\right|<2 k
  > $$
  > 

  

  > 同理，可以在列上运用$\sigma_{i}^{col}$来对区域重新排列，同时也可以验证：
  >
  > 
  > $$
  > \forall j \in 1, \ldots, N,\left|\sigma_{i}^{c o l}(j)-j\right|<2 k
  > $$

  

  > 这样就把原图中的区域坐标由$(i,j)$转换到了$\sigma(i,j)$:
  >
  > 
  > $$
  > \sigma(i, j)=\left(\sigma_{j}^{r o w}(i), \sigma_{i}^{c o l}(j)\right)
  > $$
  > 

  这种打乱方法破坏全局结构的同时能确保局部区域在其邻域内可调整的大小随机变动

  

- 原始图像$I$，其破坏版本$\phi(I)$及表明其真实精细类别的one-vs-all标签，这三个部分在训练时被组合到一起$<I,\phi(I),l>$。

  分类网络会将输入图像映射到概率分布向量$C(I,\theta_{cls})$，其中$\theta_{cls}$是分类网络中所有层的可学习参数。

  分类网络的损失函数$L_{cls}$可以写成：
  
  
  
  > $$
  > L_{c l s}=-\sum_{I \in \Gamma} l \cdot \log [C(I) C(\phi(I))]
  > $$
  >
  > 其中$\Gamma$是所有训练集，$l$是输入图片的标签值，1(图像属于该类)，0(不属于该类)
  
  

  

- 当属于该类时，需要$C(I)$和$C(\phi(I))$都接近于1时损失才会很小：当两者都接近1时，$C(I)$可以保证分类网络能学习到原输入图片正确的特征，但是$C(\phi(I))$不一定能保证从被破坏的图片学习到正确的特征，因为RCM引入了很多噪声，所以分类网络可能认为被破坏的图片中只要符合某些噪声分布就属于该类，这样就会导致对噪声的过拟合

  

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/RCM.png)




(2) 对抗学习 Adversarial Learning

- 为了防止过拟合的RCM引起的噪声模式进入特征空间，使用对抗性损失$L_{adv}$来区分原始图像和被破坏的图像，**可以最小化噪声的影响，以防止过拟合的RCM引起的噪声模式进入特征空间，仅保留有益的局部细节**。



- 具体细节：
  - 考虑把原始图像和被破坏的图像作为两个域，对抗性损失和分类损失以对抗方式工作：(1)保持域不变模式；(2)抑制$I$和$\phi(I)$之间的特定域模式。前者用于区分两种域，后者用于消除两种域之间的差异。
  
  - 给每张图片贴上 one-hot 标签向量$\mathbf{d} \in${$0,1$}$^{2}$，表示图像是否被破坏过(1表示破坏过，0表示未破坏过)。这样可以在DCL框架中添加**判别器**作为新分支，通过以下方式判断图像$I$是否被破坏过：
    
    > $$
    > D\left(I, \theta_{a d v}\right)=\operatorname{softmax}\left(\theta_{a d v} C\left(I, \theta_{c l s}^{[1, m]}\right)\right)
    > $$
  >
    > 其中，$C(I,\theta_{cls}^{[1,m]})$是从主干分类网络的第m层输出的特征向量，$\theta_{cls}^{[1,m]}$是分类网络的从第1层到第m层的可学习参数，$\theta_{a d v} \in \mathbb{R}^{d \times 2}$是一个线性映射。
  
  - 判别器网络的损失$L_{adv}$计算方式为：
  
    > $$
    > L_{a d v}=-\sum_{I \in \Gamma} \mathbf{d} \cdot \log [D(I)]+(1-\mathbf{d}) \cdot \log [D(\phi(I))]
    > $$
    >
    > 
    >
    > - 判别器是用来判别破坏了的图像和原始图像，然后计算损失。注意判别器的计算公式的输入是分类网络从图片（原始/破坏）学习到的特征向量。输入从原始输入图片学到的特征，判别器可以判断出这是原始输入图片；如果从被破坏的图片学习到很多噪声特征，那么从被破坏的图片学到的特征和从原始图片学到的特征肯定是不同的，判别器就会判断出这是被破坏的图片。
    >
    > - 让对抗性损失最小的话，就是让$D(I)$和$D(\theta(I))$都接近于1，这样就达到了去除特征域中噪声视觉模式的目的。对抗性损失和分类损失对抗的过程中，就会迫使分类网络既要学到判别性的特征同时又不能学习噪声特征。



- 为了更好地理解对抗性损失如何调整特征学习，文中进一步可对主干分类网络 ResNet-50 的特征可视化，包含使用和不使用对抗性损失两种情况。给定输入图像$I$，使用$F^k_m(I)$表示第m层的第k个特征图。对于ResNet-50，取最后一个全连接层前面的层的输出特征来进行对抗性学习，因此第m个卷积层的第k个卷积核对应真实类别c的响应为：

  > $$
  > r^{k}(I, c)=\overline{F}_{m}^{k}(I) \times \theta_{c l s}^{[m+1]}[k, c]
  > $$
  >
  > 
  >
  > - 其中$\theta_{c l s}^{[m+1]}[k, c]$是第k个特征图和对应的类别c之间的权重，即响应$r^k(I,c)$等于第k个卷积核对应的特征图乘	以全连接层对应的c的权重，以此来衡量卷积核是否能把输入图像映射到 c，响应越大表明映射的可信度越高。
  >
  > - $\delta_{k}$衡量第k个卷积核倾向于原图还是破坏图像中的视觉模式，值越大表明越倾向于原图。其中$\theta_{a d v}[k, 1]$是连接特征图$F_{m}^{k}(\cdot)$和表示原始图像标签的权重，$\theta_{a d v}[k, 2]$是连接特征图$F_{m}^{k}(\cdot)$和表示破坏图像标签的权重。
  >
  > $$
  > \delta_{k}={F}_{m}^{k}(I) \times \theta_{a d v}[k, 1]-{F}_{m}^{k}(\phi(I)) \times \theta_{a d v}[k, 2]
  > $$
  >
  > 
  >
  > 

(3) 小结

- $L_{cls}$和$L_{adv}$共同促进“破坏”学习，要想损失最小，既不能只学习总体轮廓这些粗略的特征，也不能学习边缘型的噪声模式，只能学习二者共有的特征。因此，增强了具有判别性的局部细节，并且过滤掉了不相关的特征。



##### **Construction Learning**

(1) 区域对齐网络(region alignment network)

- 用于恢复原始区域分布，作用与RCM相反，该网络需要理解每个区域的语义，包括那些有判别性的区域，通过“构造”，可以对不同局部区域之间的相关性进行建模

- 使用区域构造损失$L_{loc}$来衡量图像中不同区域的位置精度，引导主干分类网络通过端对端训练对区域间的语义相关性进行建模。

- 具体细节：

  - 给定图像$I$以及相应的破坏图像$\phi(I)$，位于图像$I$中位置$(i,j)$处的区域$R_{i,j}$与图像$\phi(I)$中的区域$R_{\sigma(i, j)}$一致。

  - 区域对齐网络是对分类网络第n个卷积层的输出特征图$C\left(\cdot, \theta_{c | s}^{[1, n]}\right)$经过$1\times1$的卷积处理得到只有两个通道的输出。然后这个输出经过ReLU和平均池化，得到一个大小为$2{\times}N{\times}N$的特征图。区域对其网络的输出可以写为：
    > $$
    > M(I)=h\left(C\left(I, \theta_{c l s}^{[1, n]}\right), \theta_{l o c}\right)
    > $$
    >
    > 
    >
    > - $M(I)$中这两个通道分别对应的是行和列的坐标，$h$是区域对齐网络，$\theta_{loc}$是区域对齐网络的参数。（即输出的二通道的特征图的每个空间位置点预测一个区域位置，每个空间位置点有两个值分别预测区域的横纵坐标，一共有$N{\times}N$个子区域）

  - 对于区域$R_{\sigma(i, j)}$预测位置为$M_{\sigma(i, j)}(\phi(I))$，对区域$R(i,j)$的预测位置是$M_{i,j}(I,i,j)$。这两个预测位置的真值都是$(i,j)$。对于区域对齐损失$L_{loc}$，定义为预测坐标与原始坐标的L1距离

$$
L_{l o c}=\sum_{I \in \Gamma} \sum_{i=1}^{N} \sum_{j=1}^{N}\left|M_{\sigma(i, j)}(\phi(I))-\left[\begin{array}{c}{i} \\ {j}\end{array}\right]\right|_{1}+\left|M_{i, j}(I)-\left[\begin{array}{l}{i} \\ {j}\end{array}\right]\right|_{1}
$$

(2) 小结

- 区域重建损失有助于定位图像中的主要目标，并且倾向于找到子区域之间的相关性。 通过端到端的训练，区域重建损失可以帮助主干分类网络建立对目标的深层理解，并对结构信息进行建模，如目标的形状和目标各部分之间的语义相关性。

  



##### **Destruction and Construction Learning**

- 在整个DCL框架中，分类、对抗性和区域对齐损失以端到端的方式进行训练，这样网络可以利用增强的局部细节和良好建模的目标局部间的相关性来进行精细识别。

- 训练目标：最小化$L$

$$
L=\alpha L_{c l s}+\beta L_{a d v}+\gamma L_{l o c}
$$

