### SeqGAN Sequence Generative Adversarial Nets with Policy Gradient

#### 1、Introduction

- GAN采用判别模型引导生成模型的训练在连续型数据上已经产生了很好的效果

- GAN有两个limitations：

  - 目标是离散数据时（如文本）很难将梯度更新(gradient update)从判别模型传递到生成模型。
  - 判别模型只能评估完整的序列，而对于部分生成的序列，很难权衡current score和生成了整个序列后的feature score。

  

#### 2、Inspiration

- SeqGAN模型采用强化学习(Reinforement Learning)的Reward思想，生成器直接实行梯度策略更新，不再进行判别器的区分，解决了<u>limitation1</u>。
- SeqGAN模型采用Monte Carlo seach(蒙特卡洛搜索)将不完整的序列补充完整，解决了<u>limitation2</u>
- RL的award信号来自GAN判别器对完整序列的评估



#### 3、Model 

- 给定$Y_{1: T}=\left(y_{1}, \dots, y_{t}, \dots, y_{T}\right), y_{t} \in \mathcal{Y}$



- 生成器模型$G_{\theta}\left(y_{t} | Y_{1: t-1}\right)$


$$
J(\theta)=\mathbb{E}\left[R_{T} | s_{0}, \theta\right]=\sum_{y_{1} \in \mathcal{Y}} G_{\theta}\left(y_{1} | s_{0}\right) \cdot Q_{D_{\phi}}^{G_{\theta}}\left(s_{0}, y_{1}\right)
$$

$$
Q_{D_{\phi}}^{G_{\theta}}\left(a=y_{T}, s=Y_{1: T-1}\right)=D_{\phi}\left(Y_{1: T}\right)
$$

![img](picture/SeqGAN.png)

#### 