---
layout:     post
title:      Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions
subtitle:   
date:       2020-08-02
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---



### 1. Introduction

- 目前许多少样本学习方法通过可见类学习一个实例嵌入函数，并将其应用于标签有限的不可见类的实例中。但是这种迁移方式的任务无关（Task-agnostic）的：相对于不可见类，嵌入函数的学习不是最佳的判别式，因此会影响模型在目标任务的性能。

- 如上所述，当前少样本学习方法缺少一种适应策略，即将从seen classes中提取的视觉知识调整为适合目标任务中的unseen classes。因此对于少样本学习模型，需要单独的嵌入空间，其中每个嵌入空间都是自定义的，因此对于给定的任务，使视觉特征最具有区分性。

- 本文提出了一种**few-shot model-based embedding adpatation**方法，该方法基于不同任务的不同seen classes调整实例嵌入模型。

  - 通过一个*set-to-set*函数使实例嵌入与目标分类任务相适应，产生具有任务特定性和区分性的嵌入。该函数映射从few-shot 支持集中获得所有实例，并输出适应后的support instance嵌入，集合汇总的元素相互配合

  - 然后将函数的输出嵌入作为每个视觉类别的原型，并用作最近邻的分类器。(如下图所示，FEAT的嵌入自适应步骤将支持嵌入从混乱的位置推向了自己的簇，从而可以更好地拟合其类别的测试数据)

    ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/36.png)

  

### 2. Method

#### Learning Embedding for Task-agnostic FSL

> - 在FSL中，一个任务表示为*M-shot N-way*分类问题      
>
> - 在只有少量训练实例的情况下，构造复杂的 $f(\cdot)$具有挑战性，因此常常通过元学习的方式       
>
> - 使用训练集$\mathcal{D}_{train}=\left\{ x_i,y_i \right\} ^{NM}_{i=1}$ ，学习$f(\cdot)$           
>   
>   > 在可见类的数据集进行采样来生成许多*M-shot N-way FSL tasks*，目标是得到一个函数$f(\cdot)$，通过$\hat{y}_{\text {test}}=f\left(x_{\text {test}} ; D _{\text {train}}\right) \in\{0,1\}^{N}$，对实例$ x_{test} $进行分类

$$
f^{*}=\underset{f}{\arg \min } \sum_{\left(\mathbf{x}_{\mathbf{test}}^{S}, \mathbf{y}_{\mathbf{tes t}}^{S}\right) \in \mathcal{D}_{\mathbf{tes t}}}^{\mathcal{L}} \ell\left(f\left(\mathbf{x}_{\mathbf{t} \mathbf{e s t}}^{\mathcal{S}} ; \mathcal{D}_{\mathbf{t r a i n}}^{\mathcal{S}}\right), \mathbf{y}_{\mathbf{t e s t}}^{\mathcal{S}}\right)
$$

- 分类器$f(\cdot)$包括两个部分，首先是一个嵌入函数$\phi_x=E(x)\in \mathbb{R}^d$将输入映射到表征空间；第二部分在表征空间中应用最近邻进行分类，**<u>可以看出嵌入函数的学习是任务无关的</u>**





#### Adapting Embedding for Task-speciﬁc FSL

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/37.png" alt="img" style="zoom:50%;" />

##### Adapting to Task-Speciﬁc Embeddings

$$
\begin{aligned}
\left\{\psi_{\mathbf{x}} ; \forall \mathbf{x} \in \mathcal{X}_{\text {train }}\right\} &=\mathbf{T}\left(\left\{\phi_{\mathbf{x}} ; \forall \mathbf{x} \in \mathcal{X}_{\text {train }}\right\}\right) \left.=\mathbf{T}\left(\pi\left\{\phi_{\mathbf{x}} ; \forall \mathbf{x} \in \mathcal{X}_{\text {train }}\right\}\right)\right)
\end{aligned}
$$



> - $\mathcal{X}_{train}$是针对目标任务的训练集合，$\pi(\cdot)$是 一个集合上的置换算子
> 
> - *set-to-set function*具有置换不变性
> 
>   ---
>
> $$
>\hat{\mathbf{y}}_{\text {test }}=f\left(\phi_{\mathbf{x}_{\text {test }}} ;\left\{\psi_{\mathbf{x}}, \forall(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{\text {train }}\right\}\right)
> $$
>
> 
>
> - 利用得到的嵌入$\psi_x$，通过计算最近邻对测试实例$x_{test}$进行分类

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/38.png" alt="img" style="zoom:30%;" />



##### Embedding Adaptation via Set-to-set Functions

`提出了不同的set-to-set functions`

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/42.png" alt="img" style="zoom:60%;" />

**(1) Bidirectional LSTM**



**(2) DeepSets**


$$
\psi_{x}=\phi_{x}+g\left(\left[\phi_{x} ; \sum_{x_{i}^{\prime} \in x^{0}} h\left(\phi_{x_{i}^{\prime}}\right)\right]\right)
$$

> 对于每个实例，首先将其互补集中的嵌入合并为一个集合向量作为上下文信息，然后将此向量与输入拼接在一起，获得自适应嵌入的残差部分



**(3) GCN**

- 首先构造度矩阵A来表示集合中实例的相似性，如果两个实例来自同一类，则将A中的对应元素设置为1，否则设置为0     

$$
S=D^{-\frac{1}{2}}(A+I) D^{-\frac{1}{2}}
$$



- 令$\Phi^0=\{\phi_x;\forall \mathbf{x} \in \mathcal{X}_{\text {train }}\}$，实例之间的关系基于$S$传播，即：

$$
\Phi^{t+1}=\mathbf{R} \mathbf{e} \mathbf{L} \mathbf{U}\left(S \Phi^{t} W\right), t=0,1, \ldots, T-1
$$





**(4) Transformer**      


$$
\mathcal{Q}=\mathcal{K}=\mathcal{V}=\mathcal{X}_{train}
$$

- 首先将输入$\mathcal{K}$映射到空间$K=W_K^T[\phi_{x_k};\forall x_k\in \mathcal{K}]\in \mathbb{R}^{d\times \mid{\mathcal{K}\mid}}$，对于$\mathcal{Q}$和$\mathcal{V}$同理分别映射到$W_Q$和$W_V$

- 利用自注意力公式得到注意力值，进行加权求和，更新输入得到$\psi_{x_q}$

$$
\alpha_{q k} \propto \exp \left(\frac{\phi_{\mathbf{x}_{q}}^{\top} W_{Q} \cdot K}{\sqrt{d}}\right)
$$

$$
\psi_{\mathbf{x}_{q}}=\phi_{\mathbf{x}_{q}}+\sum_{k} \alpha_{q k} V_{:, k}
$$



##### Contrastive Learning of Set-to-Set Functions

为了促进嵌入的适应性学习，还加入了对比目标，确保实例嵌入在适应后与同类相似而与不同类不相似

> - 将嵌入适应函数$T$应用于$N$类的不同实例，并得到转换后的嵌入$\psi^{'}_x$，类中心$\{c_n\}^N_{n=1}$
>- 采用对比目标确保训练实例更靠近自己的类中心，使set transformation提取相同类别实例的公共特征(preserve the category-wise similarity)
> 
> $$
>\begin{array}{l}
> \mathcal{L}\left(\hat{\mathbf{y}}_{\text {test }}, \mathbf{y}_{\text {test }}\right)=\ell\left(\hat{\mathbf{y}}_{\text {test }}, \mathbf{y}_{\text {test }}\right) +\lambda \cdot \ell\left(\text { softmax }\left(\operatorname{sim}\left(\psi_{\mathbf{x}_{\text {test }}}^{\prime}, \mathbf{c}_{n}\right)\right), \mathbf{y}_{\text {test }}\right)
> \end{array}
> $$
> 



### 3. Experiments

##### Datasets

> - MiniImageNet总共包括100个类，每个类600个示例。 分别使用64个类作为SEEN类别，将16和20类作为两组UNSEEN类别分别用于模型验证和评估。 
> -  TieredImageNet中包含351、97和160个类别，分别用于模型训练，验证和评估。
> - OfficeHome 数据集，以验证FEAT跨域的泛化能力。
>   - 在OfficeHome中有四个域，其中两个域“ Clipart”和“ Real World”）被选中，其中包含8722张图像。 在将所有类别随机划分后，将25个类别用作可见模型来训练模型，其余的15和25个类别将用作两个UNSEEN进行评估



##### Methods Comparison

- Comparison to previous State-of-the-arts

  (左图为MiniImageNet，右图为TiredImageNet)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/39.png" alt="img" style="zoom:50%;" />     <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/40.png" alt="img" style="zoom:50%;" />

- Comparison among the embedding adaptation models

  > Transformer作为set-to-set函数能够实现实例间的充分交互，从而提供较高的表征能力，可以对嵌入自适应过程进行建模

- Interpolation and extrapolation of classiﬁcation ways

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/44.png" alt="img" style="zoom:50%;" />

- Parameter efﬁciency

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/43.png" alt="img" style="zoom:40%;" />

  

##### Extended Few-Shot Learning Tasks

- FS Domain Generalization

  > unseen支持集和测试集中的实例来自不同的域

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/45.png" alt="img" style="zoom:50%;" />

- Transductive FSL

  > 预测取决于目标任务中unseen类别中的训练(支持)实例和所有可用的测试实例，具体来说，使用未标记的测试实例来扩充Transformer的key set和value set

- Generalized FSL

  > 同时考虑seen和unseen的测试实例

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/41.png" alt="img" style="zoom:50%;" />

