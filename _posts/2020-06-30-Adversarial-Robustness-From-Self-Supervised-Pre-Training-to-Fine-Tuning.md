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

