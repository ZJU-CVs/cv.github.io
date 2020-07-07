---
layout:     post
title:      A Closer Look at Few-shot Classification
subtitle:   
date:       2020-07-04
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:

   - paper
---



### 1. Introduction

> 文章对现有的FSL方法进行了总结和一致性的对比实验，并指出三点问题：
>
> - 在base和novel classes之间存在有限域差异的情况下，减少类内差异的各种few-shot方法实际上并没有本质差别，通过单纯增加网络深度即可显著降低各个方法之间的性能差异
> - 对于减少类内差异这点，基于距离的分类器baseline方法的性能就可以达到当前sota的meta learning 算法
> - 设置了一个实际的评估设置，其中存在base classes和novel classes之间的域转移，即base classes和novel classes从不同领域取样（如从generic object categories中采样base classes，从fine-gained categories中采样新类）减少类内差异实际上会影响模型adaption的能力，各种few-shot方法在跨数据集时表现不佳



### 2. Related Works

#### Initialization based methods

##### Learning to fine-tune

`学习一个好的初始化模型，对于新的数据集进行fine-tune`

> > - Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the International Conference on Machine Learning (ICML), 2017
> >
> > - Alex Nichol and John Schulman. Reptile: a scalable metalearning algorithm. arXiv preprint arXiv:1803.02999, 2018
> >
> > - Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta learning with latent embedding optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2019
> >
> >   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/13.png" alt="img" style="zoom:43%;" />

##### Learning to optimizer

`学习一个好的optimizer`

> > - Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2017
> >
> >   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/14.png" alt="img" style="zoom:43%;" />
> >
> > - Tsendsuren Munkhdalai and Hong Yu. Meta networks. In Proceedings of the International Conference on Machine Learning (ICML), 2017

**虽然基于初始化的方法能够在有限数量的新类训练样本中实现快速自适应，但是本文实验表明这些方法在处理base classes和novel classes之间的域迁移方面存在问题**



#### Distance metric learning based methods

##### Learning to compare

`为了学习一个sophisticated comparison模型，基于元学习的方法在训练过程中对少量标记的实例基于distance或metric进行condition perdition`

> 通过孪生网络计算相似度
>
> - Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese neural networks for one-shot image recognition. In Proceedings of the International Conference on Machine Learning Workshops (ICML Workshops), 2015.
>
> 通过余弦相似度求距离
>
> - Oriol Vinyals, Charles Blundell, Tim Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in Neural Information Processing Systems (NIPS), 2016
>
> 通过欧式距离求每个类别的中心表示距离
>
> - Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In
>   Advances in Neural Information Processing Systems (NIPS), 2017.
>
>  基于relation network
>
> - Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/15.png" alt="img" style="zoom:43%;" />
>
> 基于ridge regression
>
> - Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. In Proceedings of the International Conference on Learning Representations (ICLR), 2019
>
> 基于GNN
>
> - Victor Garcia and Joan Bruna. Few-shot learning with graph neural networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2018‘

**本文比较了三种基于距离度量学习方法的性能，实验表明：与其他复杂算法相比，基于距离的分类器的简单baseline方法（不需要 像meta-learning那样对tasks/episodes进行训练）可以获得competitive performance**



#### Hallucination based methods

##### Learning to augment

`通过learning to augment，从base classes中的数据中学习一个生成器，并使用所学习的生成器来hallucinate新的类数据以进行数据扩充`

> **Generator aims at transferring appearance variations exhibited in the base classes**
>
> - Bharath Hariharan and Ross Girshick. Low-shot visual recognition by shrinking and hallucinating features. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
>
> - Antreas Antoniou, Amos Storkey, and Harrison Edwards. Data augmentation generative adversarial networks. In Proceedings of the International Conference on Learning Representations Workshops (ICLR Workshops), 2018.
>
>   
>
> **Directly integrate the generator into a meta-learning algorithm for improving the classification accuracy**
>
> - Yu-Xiong Wang, Ross Girshick, Martial Hebert, and Bharath Hariharan. Low-shot learning from
>   imaginary data. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/10.png" alt="img" style="zoom:43%;" />

**Hallucination based methods通常与其他few-shot methods一起使用**



#### Domain adaptation

`利用domain adaptation减少源域和目标域之间的域偏移`

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/11.png" alt="img" style="zoom:30%;" />

> -  Nanqing Dong and Eric P Xing. Domain adaption in one-shot learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2018.
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/12.png" alt="img" style="zoom:30%;" />

**现有的few-shot classification算法在处理域偏移方面存在局限性**



### 3. Methods

根据前人的工作设计了两个baseline网络的few-shot classification methods

> - Baseline：在分类时使用了线性分类器
> - Baseline++：在分类时使用基于cos距离的分类器，即在训练时使用weight vector来调整每个类别的特征向量，使其和同类的instance提出的特征向量的余弦距离更近，以保证训练时减少类内的特征差异(intra-class variations)
>
> > 两个网络的训练流程是一样的，在training stage采用base classes data训练一个特征提取器和分类器，在fine-tune stage，固定特征提取器的参数，采用新的类别样本对分类器进行fine-tune
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/2.png)



对比模型：meta learning few-shot classification(包括matching net、protonet、RelationNet和MAML)

> 该类算法流程为：首先在meta-train阶段，support set $S_b$和query set $Q_b$采用Episode的训练方法来训练一个meta-learning classifer $$M(.\mid S_b)$$。在meta-testing stage，novel support set $S_n$训练新的分类器
> $$M(.\mid S_n)$$来预测新类别中的目标。（不同的meta-learning算法，其主要区别在于分类器$$M(.\mid S)$$的设计）
>
> 
>
> 
>
> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/3.png)



#### 4. Experiments

**Datasets and scenarios**

> - 通用对象识别 Generic object recognition 
>
>   > 使用mini-ImageNet数据集（共100类，每类包含600幅图像，选择64类作为base classes，16类作为validation classes，20类作为novel classes）
>
> - 细粒度图像分类 Fine-grained image classification
>
>   > 使用CUB-200-2011数据集（共200类11788个图像，其中100类作为base classes，50类作为validation classes，50类作为novel classes）
>
> - 跨域适应 Cross-domain adaptation 
>
>   >  mini-ImageNet -> CUB-200-2011，用mini-ImageNet作为base classes，并使用CUB提供的50类作为validation classes，50类作为novel classes，通过评估跨域场景，分析域偏移对现有few-shot classification方法的影响



**Implementation details**

> - 在Baseline和Baseline++方法的训练阶段，训练了400个批次大小为16的epochs
>
> - 在meta-learning的meta-training阶段，训练1-shot 60000 episodes，5-shot 40000 episodes。使用validation set来选择best accuracy的训练training episode。对于每个episode，采样N个类构成N-way分类，对于每一类，选择k个标记的实例作为support set，并为k-shot任务选择16个实例作为query set。

##### Evaluation

> **Validating our re-implementation**
>
> - 方法复现，性能与official reported results相近
> - 对所有方法使用相同的优化器
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-07-fsl/4.png" alt="img" style="zoom: 33%;" />
>
> 
>
> **Few-shot classiﬁcation results for both the mini-ImageNet and CUB datasets**
>
> - Baseline++在很大程度上改进了baseline，与其他元学习相比也具有竞争力
> - 实验表明减少intra-class variation is an important factor in the current few-shot classification problem setting.
>
> <img src="/Users/jiangyu/Desktop/blog/img/2020-07-07-fsl/5.png" alt="7" style="zoom:33%;" />
>
> 
>
> **Few-shot classiﬁcation accuracy vs. backbone depth**
>
> - 通过提高特征提取器的提取能力(增加网络深度)，会减少所有方法的intra-class variaiton
> - 对比数据集CUB和mini-ImageNet的实验结果，可以发现：
>   - 加深网络深度在CUB数据集中效果显著，使acc增加；但是在mini-ImageNet数据集中，并不是所有方法中加深网络深度都会导致acc增加
>   - 分析原因：CUB和mini-ImageNet的区别在于它们在base class和novel class上的域差异，在word-net层次中，mini-ImageNet中的类比CUB有更大的分歧
>
> <img src="/Users/jiangyu/Desktop/blog/img/2020-07-07-fsl/6.png" alt="7" style="zoom:50%;" />
>
> **Domain differences between base and novel classes**
>
> - 设计了domain shifts场景，进一步深入研究域差异问题
> - 提出了一个新的跨域场景：mini-ImageNet -> CUB，认为在一个实际场景中，从一般类中收集图像相对容易，但是从细粒度类收集图像相对困难
> - 使用ResNet-18 feature backbone进行实验：Baseline outperforms all other meta-learning methods under this scenario. 因为当元学习方法在meta-training stage时，learn to learn from the support set，所有的base support sets都在同一个数据集中，无法适应差异太大的novel class。而baseline只是简单地替换和训练一个新的基于少数给定的novel class的分类器，因此能够快速适应一个新的novel class，并不受源域和目标域之间域偏移的影响
> - 此外，Baseline的性能比Baseline++方法好，可能是因为额外减少intra-class variation会损害适应性
>
> <img src="/Users/jiangyu/Desktop/blog/img/2020-07-07-fsl/7.png" alt="7" style="zoom:43%;" />
>
> - 下图可以发现，随着域差异越来越大，based on a few novel class instances 变得越来越重要
>
> <img src="/Users/jiangyu/Desktop/blog/img/2020-07-07-fsl/8.png" alt="7" style="zoom:50%;" />
>
> **Effect of further and adaptation**
>
> - 在MatchingNet和ProtoNet方法中应用简单的自适应方案(fix the features and train a new softmax classifier)
> - 对于MAML，由于是一种初始化方法，因此不能fix特征，可以更新尽可能多的迭代来further adaptation，以训练一个新的分类层
> - 对于RelationNet，特征是卷积映射而不是特征向量，因此不能用softmax代替，而是随机将novel class中的少量训练样本分为3个support 和2个query来finetune 关系模块
> - 分析结果表明：MatchingNet和MAML的性能在further adaptation后显著提高，因此可以认为：缺乏适应能力是meta-learning methods落后于baseline的原因
> - ProtoNet的结果表示在域差异较小的以后，性能会下降，因为
> - **Learning to learn adaptation in the meta-training stage would be an important direction for feature meta-learning research in few-shot classification.**
>
> <img src="/Users/jiangyu/Desktop/blog/img/2020-07-07-fsl/9.png" alt="7" style="zoom:40%;" />