###  LEARN TO PAY ATTENTION
[code](https://github.com/SaoYan/LearnToPayAttention)

1、简介

- 提出了一种用于为图像分类的卷积神经网络（CNN）架构的端对端可训练注意力模块。该模块将2维特征矢量图作为输入，其形成CNN pipeline中不同阶段的输入图像的中间表示，并输出每个特征图的2维得分矩阵。
- 通过结合该模块来修改标准CNN架构，并在约束下训练中间2维特征向量的凸组合(由得分矩阵参数化)必须单独用于分类。通过激励方法相关性并抑制irrelevant或misleading，因此分数作为注意力值的作用。



2、方法介绍

- 利用CNN的中间的某一层提取的特征`local feature map:[l1,l2,l3,...,ln](n为空间分辨率)`和全连接层的`global feature:g`构建`attention map`

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/2.png)

- 具体细节：
  - 由$\mathcal{L}^{s}=\left\{\ell_{1}^{s}, \ell_{2}^{s}, \cdots, \ell_{n}^{s}\right\}$表示在给定卷积层$s \in\{1, \cdots, S\}$时，所提取的局部特征向量集。每个$\ell_{i}^{s}$是$n$个channel中第$i$个channel的激活矢量。（$S$表示获取局部特征的数量）
  - 全局特征向量$g$基于整个输入图像，经过网络的一系列卷积和非线性层输出，通过最终全连接层以产生该输入的原始架构的类得分
  - 由于local feature: $\ell_{i}^{s}$的维度一般大于global feature: g，所以本文先将$\ell_{n}^{s}$线性映射(降维)到$g$的相同维度大小，接下来计算每个$\ell_{i}^{s}$和$g$的兼容性compatibility score(兼容性函数C，将两个相等维度的向量作为参数并输出标量兼容性得分)，得到兼容性得分组$\mathcal{C}\left(\hat{\mathcal{L}}^{s}, \boldsymbol{g}\right)=\left\{c_{1}^{s}, c_{2}^{s}, \ldots, c_{n}^{s}\right\}$。计算方法有两种：

    - 有参法：先将两个向量做`element-wise`相加，然后通过线性回归得到一个标量值。$c_{i}^{s}=\left\langle{u},\ell_{i}^{s}+g\right\rangle, i \in\{1 \cdots n\}$
    - 无参法：直接将两个向量做点积操作,来衡量兼容性。$c_{i}^{s}=\left\langle\ell_{i}^{s}, g\right\rangle, i \in\{1 \cdots n\}$，在这种情况下，分数的相对大小将取决于高维特征空间中$g$和$\ell_{i}^{s}$之间的对齐以及$\ell_{i}^{s}$的激活强度
  - 计算完成后得到一张与feature map分辨率相同的score map，再将score map做通过softmax操作进行标准化：$a_{i}^{s}=\frac{\exp \left(c_{i}^{s}\right)}{\sum_{j}^{n} \exp \left(c_{j}^{s}\right)}, i \in\{1 \cdots n\}$
  - 通过简单的逐元素加权平均，使用归一化的兼容性分数$A^s=\{a_1^s,a_2^s,a_3^s,...a_n^s\}$为每层s生成单个矢量$g^s_a=\sum_{i=1}^{n} a_{i}^{s} \cdot \ell_{i}^{s}$。得到attention feature，作为用于最终分类的特征
  - 在单层(S=1)的情况下，如上所述计算得到的$g_a$，然后映射到$T$维向量，该向量通过softmax层以获得类预测概率$\left\{\hat{p}_{1}, \hat{p}_{2}, \cdots \hat{p}_{T}\right\}$，其中T是目标类的数量。在多层(S>1)的情况下，将全局向量连接成单个向量$g_a=[g_a^1,g_a^2,...g_a^S]$，并将其用作上述线性分类步骤的输入，或者使用S个不同的线性分类器并对输出类求平均值概率。
  - 在交叉熵损失函数下，在端到端训练中学习所有自由网络参数

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/3.png)