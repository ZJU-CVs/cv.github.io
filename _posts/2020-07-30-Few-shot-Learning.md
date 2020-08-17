---
layout:     post
title:      Few-shot Learning
subtitle:   少样本学习
date:       2020-07-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Overview
    - FSL
    - Update
---



`前言：对少样本学习的综述主要基于ECCV2020最新的一篇综述文章《Learning from Few Samples: A Survey》和之前看到的一篇《Generalizing from a few examples: A survey on few-shot learning》`

 [Learning from Few Samples: A Survey](https://export.arxiv.org/pdf/2007.15484): 对目前提出的少样本方法做了分类

[Generalizing from a few examples: A survey on few-shot learning](https://www.researchgate.net/profile/Quanming_Yao/publication/332342190_Generalizing_from_a_Few_Examples_A_Survey_on_Few-Shot_Learning/links/5e80d1d3a6fdcc139c13c69a/Generalizing-from-a-Few-Examples-A-Survey-on-Few-Shot-Learning.pdf): 从问题设置、技术、应用和理论方面进行了介绍（详见组会ppt: A Survey on Few-shot Learning）

下面主要对《Learning from Few Samples: A Survey》中不同的少样本方法进行介绍

### Introduction

`The gap between humans and machines`

- 从数据集角度考虑
  - 深度学习算法在大型(平衡标记)数据集和强大的计算机算力的支持下，能够实现图像识别、语音识别、自然语言处理和理解、视频分析等应用，且在某些情况下由于人类。
  - 人工智能的终极目标之一是能够对任何给定的任务有匹敌人类或优于人类的性能，为了实现这一目标，必须减少对**对大型平衡标记数据集**的依赖。然而，当标签数据稀少(仅少量样本的)时，当前基于大型数据集提出的算法模型在执行相应任务时性能显著下降。
  - 数据集的分布存在长尾现象，且标记数据集需要时间、人力等资源，成本昂贵
- 从学习范式考虑
  - 分析人类的学习方式可以发现，人类能够基于很少的数据，很快地学习到新的类别
  - 人类能够实时学习新的概念或新的类，而机器必须经历一个expensive offline process (即对整个模型反复的训练和再训练，以学习新的类)

- how to learn with small labeled data

  `ref: WSDM2020 tutorial`

  - Model-wise

    - transfer & reuse previous learned knowledge

      - transfer learning

      - multi-tasks learning

      - meta-learning

        

    - utilize the extra-knowledge (e.g. domain expert)

      - enrich representations using knowledge graph
      - domain-knowledge driven regularization

      

  - Data-wise

    - data augmentation from labeled/unlabeled data

    

`The potential solutions —— bridge the gap`

- **meta learning/few-shot learning/low-shot learning/zero-shot learning**等 ，目标是使模型更好地推广到由少量标记样本组成的新任务中

  > Few-shot learning and Meta-learning:
  >
  > - 在few-shot learning中，其基本思想是用大量含有多个类别的数据样本集合训练模型，并在测试过程中，为该模型提供新的类别的集合
  > - 在meta learning中，目标是概括或学习**学习的过程**，在这个过程中，模型根据特定任务进行训练，并在新集合汇总使用不同类别的函数。目标是找到最佳的超参数和模型权重，使模型能够轻松适应新任务
  >
  > Transfer Learning and Self-Supervised Learning:
  >
  > - transfer learning的总体目标是从一组任务中学习知识或经验，并将其转移到类似领域的任务中。用于训练模型以获取知识的任务具有大量的标记样本，而目标任务具有相对较少的标记数据，不足以训练模型并将其收敛到特定任务，需要通过先对源任务中的知识进行迁移。
  >   - 迁移学习的性能取决于两个任务之间的相关性
  >   - 对于每个新的迁移任务，如何迁移需要人工确定，而**元学习**技术能够自动适应新的任务
  > -  self-supervised learning技术的训练基于两个步骤：第一步，在预先定义的pretext task上训练模型，即利用大量未标记的数据样本集训练模型；第二步，学习的模型参数用于训练或微调下游任务的模型。
  >   - meta learning 和few-shot learning的理念域自监督学习非常相似，都是使用**先验知识**，完成一个新的任务
  >   - 研究表明，自监督学习可以与few-shot learning结合使用，以提高模型对新类别的性能



### Taxonomy and Organization

`meta learning, few-shot/low-shot/one-shot/zero-shot learning`等技术的主要目的都是通过基于**先验知识或经验**的迭代训练，使深度学习模型从少样本中更好地推广到新的类别。先验知识是通过在一个由大量样本组成的标记数据集上训练样本得到的，然后利用这些知识来完成仅有有限样本的新任务。

#### Data Augmentation Based

`基于数据扩充的技术在有监督学习领域非常popular。传统的增强技术(如缩放、裁剪、旋转等)常用来扩展训练数据集的大小，目标是使模型具有更好的泛化性(通用型)，避免过拟合/欠拟合。`

`在元学习空间，其思想是通过增加最小可用样本和生成更多不同样本来拓展先验知识以训练模型`

##### LaSO: Label-Set Operations networks

[paper](https://arxiv.org/pdf/1902.09811.pdf)

- 样本合成是解决小样本学习问题的方法之一。数据合成就是在给定少量训练样本的情况下，在特征空间利用训练样本合成新的样本，利用这些合成样本提升小样本学习任务的泛化能力

- 目前的合成方法仅处理的是**仅有一个类别标签的图像**的情况

- LaSO提出了一种多标签样本的合成方法

  > ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/25.png)
  >
  > - 系统的输入是两个不同的图像$x,y$和各自的多个标签$L(x),L(y)$
  >
  > - InceptionV3作为backbone $\mathcal{B}$来生成该特征空间，特征空间$F$中的特征向量用$F_x, F_y$表示
  >
  > - 将$F_x,F_y$拼接后输入到$M_{int},M_{uni},M_{sub}$三个子模块，分别表示“交”、“并”和“差”在特征空间通过样本对语义内容进行操作，生成特征向量$Z_{int},Z_{uni},Z_{sub}$
  >
  >   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/26.png" alt="img" style="zoom:30%;" />
  >
  >   - 用模型$M_{uni}$可以得到包含两张图像中出现的所有语义内容的特征向量$Z_{uni}$
  >
  >   - 用模型$M_{int}$可以得到包含两张图像共有语义内容的特征向量$Z_{int}$，如$M_{int}$接收两张被关在笼子里的动物的图像，可以得到一个特征共有特征向量“笼子”
  >
  >   - 用模型$M_{sub}$可以从另一个样本中移除某样本存在的语义内容，得到特征向量$Z_{sub}$。如将$M_{sub}$应用到笼中老虎图和表示“笼子”的特征向量上，可以得到表示“野外的老虎”的特征向量
  >
  >     ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/27.png)
  >
  > - $Z_{int},Z_{uni},Z_{sub},F_X,F_Y$被输入到分类器$C$中，使用BCE多标签分类损失来训练$C$和LaSO模型$M_{int},M_{uni},M_{sub}$
  >
  >   > $Z_{int}=M_{int}(F_X,F_Y)$
  >   >
  >   > $Z_{uni}=M_{uni}(F_X,F_Y)$
  >   >
  >   > $Z_{sub}=M_{sub}(F_X,F_Y)$
  >
  >   - 分类器$C$的训练
  >
  >     > $$
  >     > C_{l o s s}=B C E\left(C\left(F_{X}\right), L(X)\right)+B C E\left(C\left(F_{Y}\right), L(Y)\right)
  >     > $$
  >
  >   - LaSO模型的训练
  >
  >     > $$
  >     > \begin{aligned}
  >     > L a S O_{\text {loss}}=& B C E\left(C\left(Z_{\text {int}}\right), L(X) \cap L(Y)\right)+\\
  >     > & B C E\left(C\left(Z_{\text {uni}}\right), L(X) \cup L(Y)\right)+\\
  >     > & B C E\left(C\left(Z_{\text {sub}}\right), L(X) \backslash L(Y)\right)
  >     > \end{aligned}
  >     > $$
  >
  > - 此外，模型还包含一系列基于MSE的重构误差
  >
  >   - $R^{sym}_{loss}$用于增强“交”和“并”操作的对称性
  >
  >     > $$
  >     > \begin{aligned}
  >     > R_{\text {loss}}^{\text {sym}}=& \frac{1}{n}\left\|Z_{\text {int}}-M_{\text {int}}\left(F_{Y}, F_{X}\right)\right\|_{2}+ \frac{1}{n}\left\|Z_{\text {uni}}-M_{\text {uni}}\left(F_{Y}, F_{X}\right)\right\|_{2}
  >     > \end{aligned}
  >     > $$
  >   - 其中 $R^{mc}_{loss}$ 用于提高模型的稳定性，防止模型崩溃而导致每种可能的标签组合出现半固定的输出（如许多具有相同标签集的不同图像对，$M_{int}$ 可能会有非常相似的输出）
  >
  >     > $$
  >     > \begin{aligned}
  >     > R_{\text {loss}}^{m c}=& \frac{1}{n}\left\|F_{X}-M_{\text {uni}}\left(Z_{\text {sub}}, Z_{\text {int}}\right)\right\|_{2}^{2}+\frac{1}{n}\left\|F_{Y}-M_{\text {uni}}\left(M_{\text {sub}}\left(F_{Y}, F_{X}\right), Z_{\text {int}}\right)\right\|_{2}^{2}
  >     > \end{aligned}
  >     > $$

  

##### Recognition by Shrinking and Hallucinating Features

[paper](https://arxiv.org/pdf/1606.02819.pdf)



##### Learning via Saliency-guided Hallucination

[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Few-Shot_Learning_via_Saliency-Guided_Hallucination_of_Samples_CVPR_2019_paper.pdf)



##### Low-Shot Learning from Imaginary Data

[paper](https://arxiv.org/pdf/1801.05401.pdf)



##### A Maximum-Entropy Patch Sampler

[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf?source=post_page)



##### Image Deformation Meta-Networks

[paper](https://arxiv.org/pdf/1905.11641.pdf)

---

#### Embedding Based

##### Relation Network

[paper]()

##### Prototypical Network

[paper]()

##### Learning in localization of realistic settings

[paper]()

##### 

##### Learning for Semi-Supervised Classiﬁcation

[paper]()

##### 

##### Transferable Prototypical Networks

[paper]()

##### 

##### Matching Network

[paper]()

##### 

##### Task dependent adaptive metric learning

[paper]()

##### 

##### Representative-based metric learning

[paper]()

##### 

##### Task-Aware Feature Embedding

[paper]()

##### 



#### Optimization Based

##### LSTM-based Meta Learner

[paper]()

##### 

##### Memory Augmented Networks based Learning

[paper]()

##### 

##### Model Agnostic based Meta Learning

[paper]()

##### 

##### Task-Agnostic Meta-Learning

[paper]()

##### 

##### Meta-SGD

[paper]()

##### 

##### Learning to Learn in the Concept Space

[paper]()

##### 

##### ∆-encoder

[paper]()

##### 



#### Semantic Based

##### Learning with Multiple Semantics

[paper]()

##### 

##### Learning via Aligned Variational Autoencoders (VAE)

[paper]()

##### 

##### Learning by Knowledge Transfer With Class Hierarchy

[paper]()

##### 

### Future Direction



### Reference

> - B. Hariharan and R. Girshick, “Low-shot visual recognition by shrinking and hallucinating features,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 3018–3027.
>
> - Y.-X. Wang, R. Girshick, M. Hebert, and B. Hariharan, “Low-shot learning from imaginary data,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7278–7286.
>
> - A. Alfassy, L. Karlinsky, A. Aides, J. Shtok, S. Harary, R. Feris, R. Giryes, and A. M. Bronstein, “Laso: Label-set operations networks for multi-label few-shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 65486557.
>
> - H. Zhang, J. Zhang, and P. Koniusz, “Few-shot learning via saliencyguided hallucination of samples,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 27702779.
>
> - W.-H. Chu, Y.-J. Li, J.-C. Chang, and Y.-C. F. Wang, “Spot and learn: A maximum-entropy patch sampler for few-shot image classiﬁcation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 6251–6260.
>
> - Z. Chen, Y. Fu, Y.-X. Wang, L. Ma, W. Liu, and M. Hebert, “Image deformation meta-networks for one-shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 8680–8689.
>
> - O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al., “Matching networks for one shot learning,” in Advances in neural information processing systems, 2016, pp. 3630–3638.
>
> - S. Ravi and H. Larochelle, “Optimization as a model for few-shot learning,” Proceedings of the IEEE Conference on Learning Representations, 2016.
>
> - J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for few-shot learning,” in Advances in neural information processing systems, 2017, pp. 4077–4087.
>
> - F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. Torr, and T. M. Hospedales, “Learning to compare: Relation network for few-shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 1199–1208.
>
> - M. Ren, E. Triantaﬁllou, S. Ravi, J. Snell, K. Swersky, J. B. Tenenbaum, H. Larochelle, and R. S. Zemel, “Meta-learning for semi-supervised few-shot classiﬁcation,” arXiv preprint arXiv:1803.00676, 2018.
> - X. Wang, F. Yu, R. Wang, T. Darrell, and J. E. Gonzalez, “Tafe-net: Task-aware feature embeddings for low shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 1831–1840.
> - L. Karlinsky, J. Shtok, S. Harary, E. Schwartz, A. Aides, R. Feris, R. Giryes, and A. M. Bronstein, “Repmet: Representative-based metric learning for classiﬁcation and few-shot object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 5197–5206.
> - Y. Pan, T. Yao, Y. Li, Y. Wang, C.-W. Ngo, and T. Mei, “Transferrable prototypical networks for unsupervised domain adaptation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 2239–2247.
> - D. Wertheimer and B. Hariharan, “Few-shot learning with localization in realistic settings,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 6558–6567.
> - A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap, “One-shot learning with memory-augmented neural networks,” arXiv preprint arXiv:1605.06065, 2016.
> - C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017, pp. 1126–1135.
> - Z. Li, F. Zhou, F. Chen, and H. Li, “Meta-sgd: Learning to learn quickly for few-shot learning,” arXiv preprint arXiv:1707.09835, 2017.
> - F. Zhou, B. Wu, and Z. Li, “Deep meta-learning: Learning to learn in the concept space,” arXiv preprint arXiv:1802.03596, 2018.
> - B. Oreshkin, P. R. L´opez, and A. Lacoste, “Tadam: Task dependent adaptive metric for improved few-shot learning,” in Advances in Neural Information Processing Systems, 2018, pp. 721–731.
> - E. Schwartz, L. Karlinsky, J. Shtok, S. Harary, M. Marder, A. Kumar, R. Feris, R. Giryes, and A. Bronstein, “Delta-encoder: an effective sample synthesis method for few-shot object recognition,” in Advances in Neural Information Processing Systems, 2018, pp. 2845–2855.
> - M. A. Jamal and G.-J. Qi, “Task agnostic meta-learning for few-shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 11 719–11 727
> - Q. Sun, Y. Liu, T.-S. Chua, and B. Schiele, “Meta-transfer learning for few-shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 403–412.
> - E. Schwartz, L. Karlinsky, R. Feris, R. Giryes, and A. M. Bronstein, “Baby steps towards few-shot learning with multiple semantics,” arXiv preprint arXiv:1906.01905, 2019.
> - E. Schonfeld, S. Ebrahimi, S. Sinha, T. Darrell, and Z. Akata, “Generalized zero-and few-shot learning via aligned variational autoencoders,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 8247–8255.
> - A. Li, T. Luo, Z. Lu, T. Xiang, and L. Wang, “Large-scale few-shot learning: Knowledge transfer with class hierarchy,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 7212–7220.