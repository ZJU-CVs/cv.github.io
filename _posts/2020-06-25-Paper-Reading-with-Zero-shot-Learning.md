---
layout:     post
title:      Paper Reading with Zero-shot Learning 
subtitle:   零样本学习系列论文
date:       2020-06-25
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Record
---



#### Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer (对属性进行学习)

[Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer](https://ieeexplore.ieee.org/document/5206594)

> 零样本问题的**开创性文章**，主要提出了直接属性预测(direct attribute prediction)和间接属性预测(indirect attribute prediction)
>
> **算法：**
>
> > - 假设：$(x_1,l_1),...,(x_n,l_n)$为训练样本$x$及其相对应的类别标签$l$，一共有$n$对，属于$k$个类别，用$Y = \{y_1,y_2,...y_k\}$表示
> > - 目标：学习一个分类器$f:X \rightarrow Z$，其中$Z=\{z_1,...,z_L\}$为测试集包含的类别，与训练集$Y$中包含的类别没有交集，因此需要建立$Y$和$Z$之间的关系
> > - 流程：建立一个人工定义的属性层，该属性层能够较好的表现训练样本的类别信息（高维语义特征），即将基于图片的低维特征转化为高维语义特征，使训练出来的分类器分类能力更广，具有迁移学习的能力
>
> **模型：**
>
> > DAP: 在样本x和训练类别标签y之间加入一个属性表示层A，利用监督学习方式，学习到从x生成A的属性参数β。在测试阶段，可以利用属性表示层A来表示测试数据产生的类别Z，从而实现迁移学习。
> >
> > **模型架构：**
> >
> > > - DAP可以理解为一个三层模型：第一层是原始输入层，第二层是$p$维特征空间，每一维代表一个特征属性，第三层是输出层，输出样本的类别判断。
> > > - 在第一层和第二层中间，训练$p$个分类器
> > > - 在第二层和第三层，有一个语料知识库，用于保存$p$维特征空间和输出$y$的关系（这个语料知识库是人为设定的）
> >
> > **建模过程：**
> >
> > > - 首先，每个训练类别y都可以表示为长度为m的属性向量$a_y=(a_1,...,a_m)$，且该属性向量为二值属性
> > >
> > > - 然后通过监督学习，得到image-attribute层的概率表示$p(a|x)$，是样本$x$对于所有$a_m$的后验概率乘积
> > >
> > > - 在测试时，每个类别z可以用一个属性向量$a_z$表示，利用贝叶斯公式得到概率公式$p(z|x)$
> > >   $$
> > >   p(z \mid x)=\sum_{a \in\{0,1\}} p(z \mid a) p(a \mid x)=\frac{p(z)}{p\left(a^{z}\right)} \prod_{m=1}^{M} p\left(a_{m}^{z} \mid x\right)
> > >   $$
> > >   >  由于类别z是未知的，因此可以假设其的先验概率相同，对于先验概率$p(a_m)$,可以用训练时学习到的属性层值取平均值表示$$p\left(a_{m}\right)=\frac{1}{K} \sum_{k=1}^{K} a_{m}^{y_{k}}$$，则$$p(a)=\prod_{m=1}^{M} p\left(a_{m}\right)$$
> > >
> > > - 最终由$f:X \rightarrow Z$的推测可以使用MAP prediction得到：
> > >
> > > $$
> > > f(x)=\underset{l=1, \ldots, L}{\operatorname{argmax}} \prod_{m=1}^{M} \frac{p\left(a_{m}^{z_{l}} \mid x\right)}{p\left(a_{m}^{z_{l}}\right)}
> > > $$
> > >
> >
> > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/6.png" alt="img" style="zoom:50%;" />
> >
> > > IAP: 在训练阶段，在标签y上学习一层属性表示层A；在测试阶段，利用标签层y和属性层A推测出测试数据的类别z
> >
> > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/7.png" alt="img" style="zoom:50%;" />



```markdown
目前零样本学习方法直接对属性进行学习的不多，因为直接对属性进行学习会存在以下几个问题：
1. 对属性进行预测不是ZSL任务的直接目标，而是间接地解决问题，可能会导致：模型对属性预测是最优的，但对类别的预测未必是最优的
2. 无法利用unseen类的样本提供的先验知识，如使用seen和unseen节点之间的语义关系来利用先验知识
3. 无法利用新的样本(之前的训练样本中不存在的类别)逐步改善分类器的功能，即无法进行增量学习
4. 无法利用属性间的关系等额外信息，因为每个分类器只是针对一个属性进行学习的
5. 无法利用其他的辅助信息，例如词向量、语义等级层次等其他对类别的描述信息源
```



#### Label-Embedding for Attribute-Based Classification

[Label-Embedding for Attribute-Based Classification](https://ieeexplore.ieee.org/document/6618955)

> 从上述的DAP模型对属性学习的缺点出发，提出了标签嵌入框架，直接解决了对类别的预测问题，而不是简单地对属性进行预测。
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/24.png" alt="img" style="zoom:50%;" />
>
> > - 首先，通过网络提取出图像特征$\theta: \mathcal{X} \rightarrow \tilde{\mathcal{X}}$，标签$\mathcal{Y}$也嵌入一个属性空间$\varphi: \mathcal{Y} \rightarrow \tilde{\mathcal{Y}}$
> >
> > - 定义$f(x;w)$为预测函数，$f(x;w)=\arg \max _{y \in \mathcal{Y}} F(x, y ; w)$
> >
> > - 引入一个评分函数，来衡量视觉特征空间$x$嵌入语义空间的兼容度(compatibility)
> >   $$
> >   F(x, y ; W)=\theta(x)' W \varphi(y)
> >   $$
> >
> > - 当给定一个需要预测类别的数据$x$时，预测函数$f$所做的是从所有类别$y$中，找到一个类别$y$使得$F(x,y;w)$的值最大
> >
> > - 算法的核心思想是**让错误分类的得分尽可能比正确分类的得分小**
> >
> > - 损失函数
> >
> >   > 每个训练数据以$(x_n,y_n)$的形式存在，其中$x_n$是对象，$y_n$是对应的标签
> >   > $$
> >   > \frac{1}{N} \sum_{n=1}^{N} \max _{y \in \mathcal{Y}} \ell\left(x_{n}, y_{n}, y\right)+\frac{\mu}{2}\left\|\Phi-\Phi^{\Lambda}\right\|^{2}
> >   > $$
> >   >
> >   > $$
> >   > \ell\left(x_{n}, y_{n}, y\right)=\Delta\left(y_{n}, y\right)+F\left(x_{n}, y ; W\right)-F\left(x_{n}, y_{n} ; W\right)
> >   > $$
> >   >
> >   > - $\Delta: Y \times Y \rightarrow R$度量的是真实标签为$y_n$和预测值$y$之间的损失
> >   > - 对于每个样本，计算对应每个类别的得分，然后从其他所有不是正确类别的得分中找出最大的得分；逐样本累加后即得到损失函数的值，然后利用SGD等方法对参数进行更新
> >   > - 参数$\Phi$为一定维度随机初始化的参数，在使用SGD等方法进行参数更新的时候，为了使该损失函数的值尽可能小，需要$\Phi$尽可能接近$\Phi^{\Lambda}$，同时也利用了训练样本中存在的部分信息，从而使得模型达到可以逐步利用新的训练样本的信息来改善模型，实现增量学习



#### An embarrassingly simple approach to zero-shot learning

[An embarrassingly simple approach to zero-shot learning](http://proceedings.mlr.press/v37/romera-paredes15.pdf)

模型框架将**特征(features)**、**属性(attributes)**和**类(classes)**之间的关系建模为一个具有两个线性层的网络

> 第一层包含描述特征和属性之间关系的权重，并在训练阶段学习；
>
> 第二层对属性和类之间的关系建模，并使用类的指定属性signatures进行固定。

> - 在训练阶段，对于$z$类，每个有一个signature由$a$个属性组成，可以把这些signatures 表示为一个矩阵$S\in [0,1]^{a\times z}$，从而在属性和类之间提供soft link
>
> - 用$X\in R^{d\times m}$表示训练阶段的实例，其中$d$是数据的维度，$m$是实例数。使用$Y\in \{-1,1\}^{m\times z}$表示属于任一$z$类的每个训练实例的groud truth
>
> - 优化目标：
>   $$
>   \underset{W \in \mathbb{R}^{d \times z}}{\operatorname{minimise}} L\left(X^{\top} W, Y\right)+\Omega(W)
>   $$
>   
>
>   - $W=VS$，使用signatures 矩阵$S$和训练实例的学习矩阵$V$，从特征空间映射到属性空间
>
> - 在推理阶段，使用矩阵$V$，加上不可见类$S'\in [0,1]^{a\times z'}$的label，得到最终的线性模型$W'$，对于新的实例$x$，可以得到预测值$$\underset{i}{\operatorname{argmax}} x^{\top} V S_{i}^{\prime}$$
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/17.png" alt="img" style="zoom:50%;" />



#### Transductive Multi-View Zero-Shot Learning

[Transductive Multi-View Zero-Shot Learning](http://arxiv.org/abs/1501.04560)

> - 大多数现有的ZSL都是通过在带注释的训练集和不带注释测试集之间共享的中间层语义表示来转移学习。由于从低维特征空间到高维语义表示空间的映射是从辅助数据集(训练集)学习而来的，在进行应用时没有对目标数据集进行自适应，会产生**领域漂移**(domain shift)
>
> - 对于传统ZSL，一个类只有一个标签，然而同一类中不同个体之间的差异往往是巨大的，即在给定语义表示的情况下，通常只有一个原型可用于零样本学习，即**原型稀疏性**问题。每个类一个原型不足以表示类内部的可变性或帮助消除类间相重叠特征所带来的歧义，导致较大的类内差异和类间相似性。
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/18.png" alt="img" style="zoom:50%;" />
>
> - 对于**领域漂移**(domain shift) 提出了一个新的框架transductive multi-view embedding (直推式多视图嵌入)，将不同语义views中的目标实例与其low-level特征views相关联。可以减轻projection domain shift 问题，并提供了一个公共空间可以在其中直接比较异构视图，并利用其互补性。
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/19.png" alt="img" style="zoom:50%;" />
>
>   > 将每个目标类实例的三个视图$f^A(X_T),f^v(X_T)$和$X_T$投影到共享的嵌入空间
>
> - 对于**原型稀疏性**问题，提出novel heterogeneous multi-view hypergraph label propagation(异构多视图标签传播)，有效利用了不同语义表示提供互补信息，以一致的方式利用了多个表示空间的流形结构
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/20.png" alt="img" style="zoom:50%;" />
>
>   > - 黑色虚线连接表示多视图语义嵌入下同一节点的两个视图之间的相关性最大化
>   > - 红色/绿色虚线椭圆形表示异构超边，每个hyperedge由两个与该查询节点最相似的节点组成



#### Semantic Autoencoder for Zero-Shot Learning

[Semantic Autoencoder for Zero-Shot Learning](https://arxiv.org/pdf/1704.08345.pdf)

>> - 现有的零样本学习主要学习从特征空间到语义嵌入空间的映射函数，然而这种映射函数只关心被训练类的语义表示或分类。当应用于测试数据时，因为没有训练数据，往往会出现领域漂移问题。
>>
>> 
>>
>> **模型思想：**
>>
>> - 使用自编码器对原始样本进行编码，如下图所示，X为样本，S为自编码器的隐层，$\hat X$为隐层重建的$X$的表示
>>
>> - 与普通自编码器的隐层不同，隐层S为属性层，是原样本$X$的抽象语义特征表示
>>
>> - 具体实现：
>>
>>   > - 设输入层到隐层的映射为W，隐层到输出层的映射为$W^*$，$W$和$W^*$是对称的，即$W^* = W^T$。期望输入和输出尽可能相似，目标函数为$$\min _{\mathbf{w}, \mathbf{w}^{*}}\left\|\mathbf{X}-\mathbf{W}^{*} \mathbf{W} \mathbf{X}\right\|_{F}^{2}$$
>>   > - 期望中间隐层S具有抽象的语义，能表示样本属性或者类别标签，所以加入约束，使自编码器成为监督学习的问题。$$\min _{\mathbf{W}, \mathbf{W}^{*}}\left\|\mathbf{X}-\mathbf{W}^{*} \mathbf{W} \mathbf{X}\right\|_{F}^{2} \text { s.t. } \mathbf{W X}=\mathbf{S}$$
>>
>>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/8.png" alt="img" style="zoom:50%;" />
>



#### DeViSE: A Deep Visual-Semantic Embedding Model

[DeViSE: A Deep Visual-Semantic Embedding Model]((http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf))

> - 模型结合了传统视觉神经网络和词向量(word2vec)处理中的skip-gram，分别预训练一个视觉神经网络 (Visual Model Pre-training)和词向量网络 (Language Model Pre-training)，再结合两个网络进行训练
>
>   - 数据集中每个class/label可作为一个词在语义空间进行embedding表示(如使用预训练skip-gram模型得到有关class的language feature vector）
>   - 利用预训练的CNN-based模型提取图片的visual feature vector
>   - 将两个向量映射到同一维度的空间（两个向量维度一致），进行相似度计算
>
> - Loss function (采用hinge rank loss)
>   $$
>   loss(image,label)=\sum_{j \neq label} \max \left[0, margin-\text t_{label } M \vec{v}(image)+\vec{t}_{j} M \vec{v}(image)\right.
>   $$
>
>   > $t_{label}$表示label的vector，$v_{image}$表示image的vector，$M$表示linear transformation，margin为超参数
>
> - 测试时，可根据语义之间的相似性进行图像分类
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/11.png)



#### Zero-shot recognition using dual visual-semantic mapping paths

[Zero-shot recognition using dual visual-semantic mapping paths](https://arxiv.org/pdf/1703.05002.pdf)

> **直推式学习和纯半监督学习的区别**
>
> > **数据集：**
> >
> > > 假设有数据集，其中训练集为$X_L+X_U$，测试集为$X_{test}$，标记样本数据为$L$，为标记样本数目为$U$，$L<<U$
> > >
> > > - 标记样本$(x_{1:L},y_{1:L})$
> > > - 未标记样本$X_U=x_{(L+1:N)}$，训练时可用
> > > - 测试样本$X_{test}=x_{(N+1:)}$，只有在测试时可以看到
> >
> > 
> >
> > 纯半监督学习是一个归纳学习 (inductive learning)，在学习时并不知道最终的测试集，因此可以对测试样本$x_{test}$进行预测，具备泛化能力
> >
> > 直推式学习是transductive学习，仅仅可以对未标记样本$X_U$进行标记，不具备对测试样本$X_{test}$进行泛化的能力（即假设未标记的数据就是最终要用来测试的数据，学习的目的是在这些数据上取得最佳泛化能力）
>
> 
>
> **模型思想：**
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/5.png)
>
> > $\mathcal{K}$为p维语义嵌入空间，$D_s=\{x_i,k_i,y_i\}$为带标签的训练集，其中$x_i\in \mathcal{X}_{s}$是图片的特征表示。对于一个新的测试数据$x_j$，需要估计出它的语义嵌入$k_j$和标签$y_i$
> >
> > 对于**typical ZSR**问题，一般分为两步：(1) 通过visual-semantic $f_{s}: \mathcal{X}_{s} \rightarrow \mathcal{K}_{s}$映射预测嵌入$k_j$；(2) 通过比较$k_j$与default ZSR setting中的$\mathcal{K}_u$或default gZSR setting中的$\mathcal{K}_{s} \cup \mathcal{K}_{u}$
> >
> > 
> >
> > **存在问题：**如下图所示，圆和三角形分别表示可见类和不可见类，对于嵌入语义空间$\mathcal{K}$中的两个unseen classes ($v_u^i$和$v_u^j$)，如果它们在可见类子空间$\mathcal{S}=span(\mathcal{K_s})$中正交投影相同，则$\mathcal{K}$对于这两种未知类没有迁移能力
> >
> > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/21.png" alt="img" style="zoom:50%;" />
>
> **模型思想：**将$X_s$通过两条不同的visual-semantic映射函数$f_s$和$\tilde{f}_s$建立与不同空间$\mathcal{K}_s$和$\tilde{\mathcal{K}}_s$的关系，双映射主要步骤包括：
>
> - 学习两个异构空间$X_s$到$\mathcal{K}_s$的映射函数$f_s$
>   $$
>   \arg \min _{\mathbf{w}} l(\mathbf{W} \mathbf{X}, \mathbf{K})+\gamma g(\mathbf{W})
>   $$
>   
>- 提取$\mathcal{X}_s$的底层class-level流型，并生成和$\mathcal{X}_s$同构的$\tilde{\mathcal{K}}_s$
> 
>  > 对于每个类嵌入$k^i_s$，我们从$\mathcal{K}$中找到它的m个最近邻，然后将这些图像的平均值作为类级原型，即
>   >
>   > $\tilde{k}^i_s$。和$\mathcal{K}_s$相比，$\tilde{\mathcal{K}}_s=\{\tilde{k}^i_s\}^k_{i=1}$在语义上与$\mathcal{X}$更一致
> 
>- 将$\mathcal{X}_s$和$\tilde{K}_s$进行迭代对齐得到$\tilde{f}_s$，并优化$\tilde{K}_s$使其与$X_s$在语义上更加一致
> 
>
> 
>**测试阶段：**
> 
>> 将$f_s,\tilde{f}_s,\tilde{\mathcal{K}}_s,\mathcal{K}_s$和$\mathcal{K}_u$作为输入，对于$n_t$个测试实例$X_u \in R^{d\times n_t}$，首先预测它们的语义表示$f_s(X_u)$，然后直接构建$\tilde{\mathcal{K}}_s$，最后，对于每个测试实例$x_j$，得到对应的标签$y_j=\arg \max_c d\left(\tilde{f}_{s}\left(\mathbf{x}_{j}\right), \tilde{\mathbf{k}}_{c}\right)$，其中$\tilde{\mathbf{k}}_{c}\in\tilde{\mathcal{K}}_{u}$ in **ZSR**；$\tilde{\mathbf{k}}_{c}\in\{\tilde{\mathcal{K}}_s \cup\tilde{\mathcal{K}}_{u}\}$ in **gZSR**.



#### Feature Generating Networks for Zero-Shot Learning

[Feature Generating Networks for Zero-Shot Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Feature_Generating_Networks_CVPR_2018_paper.pdf)

> 提出了基于GAN框架，利用语义信息来生成unseen classes的CNN特征。文中利用WGAN和分类损失，生成判别性强的CNN特征，来训练分类器。
>
> - 由下图所示，CNN的特征可以从以下方面提取：
>
>   > 真实图像，但是在zero-shot学习中，无法获得任何不可见类的真实图像
>   >
>   > 合成图像，但是不够精确，无法提高图像分类性能
>
> - 对于以上两个问题，提出了新的基于属性条件特征生成对抗网络，即f-CLSWGAN，用于生成不可见类的cnn特征
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/22.png" alt="img" style="zoom:50%;" />
>
> **算法思想：**
>
> > 图像特征生成的优势：
> >
> > - 生成的特征数据是无限制的
> > - 生成的特征通过大规模数据集学习到的具有代表性的特征表示，在某种程度上泛化未知类别的特征
> > - 学习的特征具有很强的判别性
>
> 使用三种conditional GAN的变体来循序渐进的生成图像特征：embedding feature
>
> > - 通过下式表示可见类和不可见类
> >   $$
> >   S=\left\{(x, y, c(y)) \mid x \in X, y \in Y^{s}, c(y) \in C\right\} \\
> >   U=\left\{(u, c(u)) \mid u \in Y^{u}, c(u) \in C\right\}
> >   $$
> >
> >   > 其中$S$代表所有可见训练集，$x\in R^{d_x}$是CNN特征，$y^s = {y_1,...,y_k}$代表有$k$个相互独立的可见类，$y$代表类标签，$c(y)\in R^{d_c}$代表y类的embedding属性集，由语义向量构成。
> >   >
> >   > $U$代表不可见类，缺少了图像和CNN特征
> >   >
> >   > ZSL的任务：学习一个分类器$f_{zsl}:X \rightarrow Y^{u}$
> >   >
> >   > GZSL的任务：学习一个分类器$f_{gzsl}:X \rightarrow Y^{s}\cup Y^{u}$
> >
> > - 最终的损失函数：$$\min _{G} \max _{D} L_{W G A N}+\beta L_{C L S}$$
> >
> >   > $$\min _{G} \max _{D} L_{W G A N}=E[D(x, c(y))]-E[D(\tilde{x}, c(y))]-\lambda E\left[\left(\left\|\nabla_{\hat{x}} D(\hat{x}, c(y))\right\|_{2}-1\right)^{2}\right]$$
> >   >
> >   > $$L_{C L S}=-E_{\tilde{x} \sim p_{\tilde{x}}}[\log P(y \mid \tilde{x} ; \theta)]$$
> >   >
> >   > - 其中$$\tilde{x}=G(z, c(y))$$，$$\hat{x}=\alpha x+(1-\alpha) \tilde{x}$$，$$\alpha \sim U(0,1)$$，$\lambda$为处罚系数
> >   > - $P(y \mid \tilde{x} ; \theta)$表示$\tilde{x}$被类别标签$y$真实预测的概率，分类器的参数$\theta$是根据可见类的实际特征进行预训练
> >   > - 分类损失可以看作是正则化项，使生成器构造判别性强的特征
>
> **模型架构：**
>
> > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/23.png" alt="img" style="zoom:50%;" />
> >
> > > - 第一行是真实图像的特征提取过程，文中采用的预训练模型。将特征$x$与$x$所属类的属性描述$c(y)$拼接后输入Discriminator并判别为真
> > > - 第二行是生成数据的分支，用normal distribution的$z$与$c(y)$拼接后输入生成器，生成特征$\tilde{x}$；再次将与属性描述拼接后输入Discriminator并判别为假；同时输出$P(y \mid \tilde{x} ; \theta)$的值观察Generator构造的特征性的强弱



#### Learning a Deep Embedding Model for Zero-Shot Learning

[Learning a Deep Embedding Model for Zero-Shot Learning](https://arxiv.org/pdf/1611.05088)

> 提出一种端对端的深度模型来完成Zero-shot learning，将视觉特征空间作为嵌入空间(embedding space)要比语义空间作为嵌入空间的效果好的多。所提模型能够很好地解决hubness problem，如下图所示，$S$表示语义空间，$V$表示视觉特征空间，当将语义特征映射到视觉特征空间中时，hubness problem
>
> 
>
> **优势：**
>
> > - 能够更好地学习一个嵌入空间
> > - 为基于神经网络的联合嵌入模型提供了灵活性，能够解决多种迁移性问题
> > - 可以很自然地对多模态的数据进行fusing
>
> 
>
> **模型思想：**
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/27.png" alt="img" style="zoom:50%;" />
>
> > 语义表示单元(Semantic Representation Unit)可以利用三个结构进行代替，分别是单模态形式、多模态形式和RNN词嵌入形式
> >
> > - 输入图片通过深度CNN进行特征提取
> >
> > $$
> > \mathcal{L}\left(\mathbf{W}_{1}, \mathbf{W}_{2}\right)=\frac{1}{N} \sum_{i=1}^{N}\left\|\phi\left(\mathbf{I}_{i}\right)-f_{1}\left(\mathbf{W}_{2} f_{1}\left(\mathbf{W}_{1} \mathbf{y}_{i}^{u}\right)\right)\right\|^{2}+\lambda\left(\left\|\mathbf{W}_{1}\right\|^{2}+\left\|\mathbf{W}_{2}\right\|^{2}\right)
> > $$
> >
> > > 其中$y$表示语义输入，$I$表示输入图片，$W_1$和$W_2$分别为全连接层的权重，$f_1$表示ReLU
>
> 



#### Rethinking Knowledge Graph Propagation for Zero-Shot Learning

[Rethinking Knowledge Graph Propagation for Zero-Shot Learning](https://arxiv.org/pdf/1805.11724.pdf)

> 使用$\mathcal{C}$表示所有类的集合，$\mathcal{C}_{te}$和$\mathcal{C}_tr$分别为测试集和训练集中的类别，要求$$\mathcal{C}_{t e} \cap \mathcal{C}_{t r}=\emptyset$$。使用$S$维的语义表征向量$z\in R^S$表示所有的类别，$\mathcal{D}_{tr}=\{\left(\vec{X}_{i}, c_{i}\right), i=1, \ldots, N\}$表示训练集中的样本（图像及标签）
>
> **模型思想：**
>
> 使用标签的word embedding以及知识图谱来对未知的类进行预测
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/25.png" alt="img" style="zoom:50%;" />
>
> DGP中考虑了所有的seen和unseen特征，使用词嵌入向量的方式，通过预测一组新的参数来拓展CNN，使得这组参数能够适应unseen classes的分类
>
> > - 在训练过程中，DGP使用一种半监督的方法来监督CNN最后一层的参数，即能够利用知识图谱提供的类别的语义描述之间关系的信息来拓展原有的CNN分类器，使其适应unseen的类
> > - 具体来说，给定一个由$N$个节点的图，每个节点使用$S$维输入特征表示，则$X\in R^{N\times S}$就表示特征矩阵。每个节点表示一个不同的类，类之间的链接使用对称的邻接矩阵表示$A\in R^{N\times N}$，其中包括自环
> > - 图的传播法则表示为：$$H^{(l+1)}=\sigma\left(D^{-1} A H^{(l)} \Theta^{(l)}\right)$$，其中$H^{(l)}$表示第$l$层的激活结果，$\Theta^{(l)}\in R^{S\times F}$表示第$l$层的可学习参数
> > - 通过优化$$\mathcal{L}=\frac{1}{2 M} \sum_{i=1}^{M} \sum_{j=1}^{P}\left(W_{i, j}-\tilde{W}_{i, j}\right)^{2}$$来训练GCN预测参数的能力，其中$M$表示训练时的类别数目，$P$表示权重向量的维度，Ground turth的权重$w$通过抽取预训练的CNN分类器得到。
>
> 
>
> **距离加权框架**
>
> > 为了使得DGP能够更好地衡量不同邻节点之间的权重关系，提出了新的加权计算框架，通过节点之间距离来计算权重
> >
> > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-06-30-zsl/26.png" alt="img" style="zoom:50%;" />
> >
> > **带权重的传播公式：**
> > $$
> > H=\sigma\left(\sum_{k=0}^{K} \alpha_{k}^{a} D_{k}^{a^{-1}} A_{k}^{a} \sigma\left(\sum_{k=0}^{K} \alpha_{k}^{d} D_{k}^{d^{-1}} A_{k}^{d} X \Theta_{d}\right) \Theta_{a}\right)
> > $$
> > 
>
> **训练步骤：**
>
> > - 训练DGP来预测最后一层的预训练CNN参数
> > - 使用DGP预测的参数并固定，使用交叉熵损失调整特征提取部分的参数(对于seen classes)，使CNN特征适应于新得到的分类器

