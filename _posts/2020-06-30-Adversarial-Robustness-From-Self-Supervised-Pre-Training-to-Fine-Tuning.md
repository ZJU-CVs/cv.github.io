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



**背景介绍：**

> - 自监督技术能够快速fine-tuning到多个下游任务，并有更好的泛化和校准性(generalization and calibration)
>
> - 通过自监督的预训练证明可以实现高精度饿任务包括位置预测任务（Selfie，Jigsaw和旋转预测任务等）以及各种其他感知任务
> - 对抗性攻击的脆弱性对深度学习中的标签和样本有效性是一种挑战，如即使是一个well-trained CNN，当输入受到imperceivable扰动好似，会出现高误分辨率
> - 

