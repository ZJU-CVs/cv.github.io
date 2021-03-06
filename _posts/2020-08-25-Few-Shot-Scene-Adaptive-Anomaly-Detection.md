---
layout:     post
title:      Few-Shot Scene-Adaptive Anomaly Detection
subtitle:   
date:       2020-08-25
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Anomaly Detection
    - FSL
---



#### 1. Introduction

- 对于Anomaly Detection in Surveillance videos，前人工作主要是基于frame reconstruction的方法

  - 学习一种模型以重构正常样本，并使用重构误差来识别异常
  - 学习一种模型，将一系列的连续帧作为输入预测下一帧，根据预测帧和实际帧的差异进行异常检测

- 现有的异常检测方法具有以下局限性：

  - 训练和测试的视频需要来自同一场景（如需要由同一台摄像机捕获），如果利用一个场景下的视频训练得到的异常检测模型，应用到另一个不同的场景中直接测试，性能将会下降
  - 解决上述问题的一种方法是训练时使用各种场景收集的视频来训练异常模型，提高模型的泛化性。但这种模型的参数量大，不利用部署到计算能力有限的边缘设备 

  

- 本文提出了一种few-shot的场景自适应异常检测问题

  - 在训练过程中，输入多个场景收集的视频进行少样本学习，该模型训练后能够快速适应新场景
  - 在测试时，为模型提供几帧新目标场景的视频进行训练，生成适用于该目标场景的异常检测模型

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/17.png" alt="img" style="zoom:50%;" />



#### 2. Method

使用元学习框架**MAML**来学习少样本场景自适应的异常检测模型，包括一个元训练阶段和一个元测试阶段

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/18.png" alt="img" style="zoom:50%;" />



##### Tasks in Meta-Learning

以基于预测的异常检测模型为例，构建元训练任务如下：

- 利用$t$时刻前观测到的帧{$I_1,\cdots,I_{t}$}预测第$t+1$时刻的帧$\hat{I}_{t+1}$           

$$
f_{\theta}(I_{1:t})\rightarrow \hat{I}_{t+1}
$$



- 在元训练时有$M$个场景，表示为{$S_1,S_2,\cdots,S_M$}，对于给定的场景$S_i$，可以构造一个对应的任务$\mathcal{T}_i=(\mathcal{D_i}^{tr},\mathcal{D_i}^{val})$

  > - 首先将从$S_i$获得的视频拆分为许多长度为$t+1$的重叠连续段（overlapping consecutive segments）
  >
  > 
  >
  > - 对于每个分段 $(I_1,I_2,\cdots,I_t,I_{t+1})$，$x=(I_1,I_2,\cdots,I_t)$，$y=I_{t+1}$，因此可以得到一个输入/输出对$(x,y)$
  >
  > 
  >
  > - 在训练集$\mathcal{D}_{i}^{tr}$中，随机从$\mathcal{T_i}$采样$K$个输入/输出对，以学习未来帧的预测模型，即$\mathcal{D_i}^{tr}$={$(x_1,y_1),(x_2,y_2),\cdots,(x_K,y_K)$}；同时，随机采样$K$个输入/输出对 (不包括$\mathcal{D_i}^{tr}$中的对) 构成$\mathcal{D_i}^{val}$

  

##### Meta-Training

- 对于一个预训练参数为$\theta$的异常检测模型$f_{\theta}:x\rightarrow y$，

- 按照MAML，通过在该任务训练集$\mathcal{D}_i^{tr}$上

  定义的损失函数和一个梯度更新将参数从$\theta$调整为$\theta'_{i}$，以适应任务$\mathcal{T}_i$

$$
\begin{array}{l}
Eq.1: \\
\theta_{i}^{\prime}=\theta-\alpha \nabla_{\theta} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta} ; \mathcal{D}_{i}^{t r}\right),\\ where \ \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta} ; \mathcal{D}_{i}^{t r}\right)=\sum_{\left(x_{j}, y_{j}\right) \in \mathcal{D}_{i}^{t r}} L\left(f_{\theta}\left(x_{j}\right), y_{j}\right)
\end{array}
$$

> 其中$\alpha$为步长
>
> $L(f_\theta(x_j),y_j)$用于衡量预测帧$f_\theta(x_j)$与实际帧$y_j$之间的差异，$L(\cdot)$的定义为：
>
> 
> $$
> L(f_\theta(x_j),y_j)=\lambda_1 L_1(f_\theta(x_j),y_j)+\lambda_2 L_{ssm}(f_\theta(x_j),y_j)+\lambda_3 L_{gdl}(f_\theta(x_j),y_j)
> $$



- 更新后的参数$\theta'_i$适应任务$\mathcal{T}_i$，通过在$\mathcal{D}_i^{val}$进行性能验证。元训练的目的是学习初始化模型参数$\theta$，以通过任务自适应得到$\theta'$，使对应任务的损失函数最小：

$$
\begin{array}{l}Eq.3:\\
\mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta^{\prime}} ; \mathcal{D}_{i}^{v a l}\right)=\sum_{\left(x_{j}, y_{j}\right) \in \mathcal{D}_{i}^{v a l}} L\left(f_{\theta^{\prime}}\left(x_{j}\right), y_{j}\right)
\end{array}
$$

- 最终元学习的目标函数为所有任务损失的总和：

$$
\min_{\theta}\sum _{i=1}^M \mathcal{L}_{\mathcal{T}_{i}}(f_{\theta^{\prime}};\mathcal{D}^{val}_i)
$$

- 在实际训练时，每一次迭代从任务中采样mini-batch。算法步骤如下所示：

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/23.png" alt="img" style="zoom:50%;" />



##### Meta-Testing

- 对于一个新的目标场景，基于$S_{new}$中的$K$个实例，使用$Eq.1$获得自适应模型参数$\theta'$，然后用其余帧衡量模型性能

  即使用$S_{new}$中的一个视频的前几帧进行场景自适应，然后使用其余帧进行测试



##### Backbone Architecture

- 本文所提的场景自适应异常检测框架是通用的，从理论上讲，可以使用任何异常检测网络作为主干架构

- 本文参考`《Future frame prediction for anomaly detection — a new baseline》`中所提的异常检测模型和`《Convolutional lstm network: A machine learning approach for precipitation nowcasting》`中所提的ConvLSTM模块，提出了r-GAN作为backbone

- 具体而言，使用ConvLSTM和对抗训练来构建模型，以捕获视频的时空信息

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/24.png" alt="img" style="zoom:50%;" />

  <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/25.png" alt="img" style="zoom:50%;" />

  > 省略了光流约束的part
  >
  > 首先应用U-net来预测未来的帧，并将预测结果传递给ConvLSTM模块以保留先前步骤的信息
  >
  > 生成器和判别器进行对抗训练



注：ConvLSTM简单介绍 [link](https://zju-cvs.github.io/2020/08/25/Convolutional-LSTM-Network/)



#### 3. Experiment

##### Datasets

> Shanghai Tech
>
> UCF crime
>
> UCSD Pedestrian 1
>
> UR fall

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/19.png" alt="img" style="zoom:50%;" />

> 第一行显示了数据集正常帧的实例；第二行显示了异常帧的实例
>
> 训练的视频数据仅包含正常帧，异常帧仅用于测试



##### Baseline

> Pre-trained:
>
> Fine-tuned：



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/22.png" alt="img" style="zoom:50%;" />