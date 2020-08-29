### Unsupervised Learning for Cell-Level Visual Representation in Histopathology Images With Generative Adversarial Networks

[code](https://github.com/tangye95/Unsupervised-Cell-level-Visual-Representation-learning-with-GAN)



1、简介

- 利用生成对抗网络对组织病理学图像中cell-level视觉表现进行无监督学习。
- 所提出的模型没有标签，易于训练，能够进行cell-level的无监督分类



2、方法介绍

- Cell-level Visual Representation Learning

  **Training Processing**

  > - 定义生成器网络G、鉴别器网络D和辅助网络Q
  >
  > - 通过对抗训练得到生成器G分布
  >
  >   - 定义一个随机噪声变量z，输入噪声z，由生成器G变换为样本$\tilde{x}=G(z),z\sim p(z)$
  >
  >   - $x \sim \mathbb{P}_{r}$,$\tilde{x} \sim \mathbb{P}_{g}$, 参考WGAN，定义: 
  > $$
  > W\left(\mathbb{P}_{r}, \mathbb{P}_{g}\right)=\sup _{\|f\|_{L \leq 1}} \mathbb{E}_{x \sim \mathbb{P}_{r}}[f(x)]-\mathbb{E}_{\tilde{x} \sim \mathbb{P}_{g}}[f(\tilde{x})]
  > $$
  >
  >   - $W\left(\mathbb{P}_{r}, \mathbb{P}_{g}\right)$是EM距离的有效近似，能够衡量生成器分布和实际数据分布的接近程度。
  >
  >   - 采用`WGAN-GP`思想训练G和D
  > $$
  > \min _{G} \max _{D \in \mathcal{D}} \mathbb{E}_{x \sim \mathbb{P}_{r}}[D(x)]-\mathbb{E}_{z \sim p(z)}[D(G(z))]​
  > $$
  >
  >   - `WGAN-GP`能够生成cell-level图像，但是无法利用cell的类别信息，因为噪声变量z不对应域任何可解释的特征。因此，模型的训练的第二个目标是使所选变量能够代表meaning and interpretable 的cell语义特征。
  >  - 受[`InfoGAN`](read/InfoGAN Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets.md)启发 ，将互信息引入模型
  > $$
  > I(X ; Y)=\mathrm{H}(X)-\mathrm{H}(X | Y)=\mathrm{H}(Y)-\mathrm{H}(Y | X)
  > $$
  > 
  >- $I(X;Y)$描述了两个独立变量X和Y之间的依赖关系，它测量两个随机变量之间不同方面的关联。如果所选择的随机变量对应于某些语义特征，则假设生成的样本和随机变量之间的互信息高。
  > 
  >-  文中定义了从固定噪声分布$p(c)$中采样的潜在变量c，然后随机噪声变量z和潜在变量c级联(concatenation)输入生成器得到$G(z,c)$。由于潜变量应与有意义的语义特征相对应，因此c和$G(z,c)$之间的互信息高，因此下一步是最大化互信息：
  > $$
  > I(c ; G(z, c))=H(c)-H(c | G(z, c))
  > $$
  > 
  >- 使用辅助网络Q最大化所选随机变量与所生成样本之间的互信息(mutual information)。下限$L_I$为：
  > $$
  > L_{I}(G, Q)=\mathbb{E}_{z \sim p(z), c \sim p(c)}[\log Q(c | G(z, c))]+H(c)
  > $$
  > 
  >​		其中$H(c)$是从固定噪声分布中采样变量的熵，通过最优化此下限可以最大化互信息$I(c,G(z,c))$
  > 
  >- 由于将潜在变量c引入模型，因此值函数$V(D,G)$被替换为：
  > $$
  > V(D, G) \leftarrow \mathbb{E}_{x \sim \mathbb{P}_{r}}[D(x)]-\mathbb{E}_{z \sim p(z), c \sim p(c)}[D(G(z, c))]
  > $$
  > 
  >- 将对抗过程和最大化互信息的过程相结合，增加超参数$\lambda_2$：
  > $$
  > \min _{G, Q} \max _{D \in \mathcal{D}} V(D, G)-\lambda_{2} L_{I}(G, Q)
  > $$
  > 
  >- $H(c)$可视为常数，因此辅助网络Q的损失可写为$Q(c|G(c,z))$与离散变量c之间的负对数似然，网络D，G，Q损失如下（其中$P_\hat x$被定义为沿着从数据分布$P_r$和生成器分布$P_g$采样的点对之间直线的均匀采样）：
  > 
  >$$
  > \begin{aligned} L_{D} \leftarrow & \mathbb{E}_{z \sim p(z), c \sim p(c)}[D(G(z, c))]-\mathbb{E}_{x \sim \mathbb{P}_{r}}[D(x)] +\lambda_{1} \mathbb{E}_{\hat{x} \sim \mathbb{P}_{x}}\left[\left\|\nabla_{\hat{x}} D(\hat{x})\right\|_{p}-1\right]^{2} \end{aligned}
  > $$
  > 
  >$$
  > L_{G}=-\mathbb{E}_{z \sim p(z), c \sim p(c)}[D(G(z, c))]
  > $$
  > 
  >$$
  > L_{Q}=-\lambda_{2} \mathbb{E}_{z \sim p(z), c \sim p(c)}[\log Q(c | G(z, c))]
  > $$
  
  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/6.png)
  
  **Testing Processing**
  
  >- 在训练过程中，学习生成器G分布以模仿真实数据分布。学习辅助网络Q分布以最大化下限。特别是如果从分类分布中采样，则应用softmax函数作为Q的最后一层，这是Q在测试过程中可以视为分类器，因为后验Q(c|x)是离散的。
  >- 假设c中的每个类别对应一种类型的cell，辅助网络Q可以将cell-level图像划分为不同的类别，而生成器G可以为每个类别的cell生成可解释的表示。
  
  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/7.png)
  
- Image-level Classification

  > - 文中提出了一种结合和分割和cell-level视觉表示的方法，以突出cell元素的多样性
  > - 使用计算的cell proportion进行图像级分类
  >
  > ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/8.png)
  >
  > - 具体步骤：
  >
  >   - 利用无监督分割方法，包括四个阶段(标准化、无监督颜色反卷积、强度阈值处理和后处理)，从背景中分割细胞核
  >
  >     ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/unsupervised segmentation.png)
  >
  >   - 利用无监督表征学习中训练的模型分布作为cell-level分类器。假设在训练过程中使用k维分类变量作为所选变量，则将实际数据(cell-level图像)分布分配到k维度。在测试过程中，cell-level图像被无监督分类为k个相应的类别。
  >
  >   - 将每个类别中的cell-level实例的数量计算为$\{X_1,X_2,X_3...X_k\}$。对于cell元素$i$，通过$P_{i}=\frac{X_{i}}{\sum_{i=1}^{k} X_{i}}$计算得到该图像中该cell元素的数量与cell构成的总数的比率，则$P_i$表示cell element的cell proportion
  >
  >   - 对于给定的{P_1, P_2, P_3,...P_k}作为图像的特征向量，利用k-means或SVM进行图像级的预测

