---
layout:     post
title:      Adversarial Robustness-From Self-Supervised Pre-Training to Fine-Tuning
subtitle:   基于无监督学习的预训练方法
date:       2020-06-30
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
---



#### 1. Introduction

> 基于自监督学习的预训练模型通常用于对下游任务进行更快或更精确的fine-tune。然而，未考虑预训练的鲁棒性。

- 本文介绍的方法首次提出通用的鲁棒预训练模型

- robust的预训练模型对后续的下游任务fine-tune有以下好处

  > 提高最终模型的鲁棒性
  >
  > 在fine-tuning adversarial时节省计算成本



##### **背景介绍**

> 自监督技术：
>
> > - 自监督技术能够快速fine-tuning到多个下游任务，并有更好的泛化和校准性(generalization and calibration)
> > - 通过自监督的预训练证明可以实现高精度饿任务包括位置预测任务（Selfie，Jigsaw和旋转预测任务等）以及各种其他感知任务
>
> 
>
> 对抗训练：
>
> > - 对抗性攻击的脆弱性对深度学习中的标签和样本有效性是一种挑战，如即使是一个well-trained CNN，当输入受到imperceivable扰动好似，会出现高误分辨率。
> > - 对于对抗性攻击，对抗训练 (Adversarial training)是SOTA的模型防御方法，但是与standard training相比计算昂贵



##### 贡献点

> - 自监督最近与鲁棒性的研究相联系，如《Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty》中的工作，利用自监督学习提高模型的鲁棒性和不确定性
>
> - 本文提出了一个在adversarial robustness领域中进行自监督进行预训练和fine-tuning的框架，并思考与回答以下三个问题：
>
>   - 一个经过对抗预训练的模型是否能够有效地提高后续fine-tuning的鲁棒性？
>
>     > 首次证明了用于对抗性fine-tuning的鲁棒的预训练模型能够带来巨大的性能提升
>
>   - 对抗性预训练和对抗性fine-tuning，哪个提供了更好的准确性和效率？
>
>     > 对抗性fine-tuning是鲁棒性改善的主要部分，而鲁棒性预训练主要加速对抗性网络的调整
>
>   - 自监督的预训练任务类型如何影响最终模型的鲁棒性？
>
>     > 实验表明不同自监督任务产生的预训练模型具有不同的对抗性弱点，因此采用一组自监督任务进行预训练，以充分利用它们之间的互补优势



#### 2. Method

##### 自监督预训练

>- $\mathcal{T}_p$表示一个预训练任务，$\mathcal{D}_p$表示对应的(未标记)的预训练数据集。自监督预训练的目标是在没有明确的人工监督情况下，从$\mathcal{D}_p$本身学习一个模型。
>
>- 预训练损失$\mathcal{l}_p(\theta_p,\theta_{pc};\mathcal{D}_p)$，通过训练确定$\theta_p$使$l_p$最小化；$\theta_{pc}$表示基于$\mathcal{T}_p$得到的附加参数
>
>- 预训练任务主要为：
>
>  > **Selfie:** 通过masking out图像中选定的patches，selfie构造为一个分类问题，以确定要在被masked位置填充的正确patch
>  >
>  > 
>  >
>  > **Rotation:** 将图像随机旋转90度，rotation构造为一个分类问题，以确定应用于输入图像的旋转角度
>  >
>  > 
>  >
>  > **Jigsaw:** 通过将图像分成不同的快，Jigsaw训练一个分类器来预测这些patches正确的排列
>
>  
>
>  ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/ad1.png)



##### 有监督的Fine-tuning

> - 设$r(x;\theta_p)$表示自监督与训练任务$\mathcal{T}_p$从输入样本$x$到其嵌入空间的映射。给定一个带有标签数据$D_f$的目标fine-tuning任务（下游任务）$\mathcal{T}_f$，fine-tuning的目标是确定一个分类器，表示$r(x;\theta_p)$映射到标签空间
>
> - 为了学习分类器，可以使用固定或重新训练的模型$\theta_p$来最小化常见的监督训练损失$l_f(\theta_p,\theta_f;\mathcal{D}_f)$



##### 引入对抗训练

> 对抗训练是训练一个鲁棒性的分类器的强有力方法，通过将对抗训练引入自监督，来提供泛化能力更强的预训练模型



##### 多任务集成

> 不同预训练模型含有不同的对抗特性，因此把多个预训练模型集成，取得进一步的性能提升



#### 3. Experiments

##### 验证AT自监督预训练和fine-tuning对分类鲁棒性的提高

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/ad2.png)

> $\mathcal{P}_1$ (without pre-training)
>
> $\mathcal{P}_2$ (standard self-supervision pre-training),
>
> $\mathcal{P}_3$ (adversarial self-supervision pre-training)
>
> $\mathcal{F}_1$ (partial standard ﬁne-tuning) 
>
> $\mathcal{F}_2$ (partial adversarial ﬁne-tuning) 
>
> $\mathcal{F}_3$(full standard ﬁne-tuning)
>
> $\mathcal{F}_4$ (full adversarial ﬁne-tuning).

##### 验证AT的fine-tuning和预训练的可分离性

> 将预训练和fine-tuning分离，可以从一个warm start开始学习图像分类器，减轻了one-shot AT的计算消耗缺点，且性能优越。

baseline:《Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty》

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/ad4.png" alt="img" style="zoom:50%;" />

##### 验证Task Ensemble的有效性

> ![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/ad1.png)