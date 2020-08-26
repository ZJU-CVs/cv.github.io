### Self-Attention Generative Adversarial Networks

[Self-Attention Generative Adversarial Networks](ref/Self-Attention Generative Adversarial Networks.pdf)



1、简介

在gan生成中加入了attention机制，同时将[SNgan](#SNgan)的思想引入到生成器中。

#### SNgan (频谱归一化GAN) 

- 为了让正则化产生更明确的限制，提出了用频谱范数标准化神经网络的参数矩阵W，从而让神经网络的梯度被限制在一个范围内。

#### 加入attention机制 

- 传统GAN的问题：
  - 使用小的卷积核很难发现图像中的依赖关系
  - 使用大的卷积核就丧失了卷积网络参数与计算的效率
  - 卷积核的尺寸限制，只能捕获局部区域的关系，而在SAGAN中，能够利用所有位置的信息
  
- attention层详解：
  
  - 在前一层的feature maps上加入attention机制，使得GAN在生成时能够区别不同的feature maps
  
    ![img](picture\attention1.png)
    
    注：$1\times1$卷积的作用是减少图像中通道数量
    
    > 前面隐藏层输出的图像特点被转换成f，g两个特征空间，用来计算注意度（attention），$f(x)=W_f x, g(x)=W_g x$，$W_f, W_g$都是网络的参数
    >
    > 
    >
    > SoftMax来得到attention map，$\beta_{j,i}$表示在合成第j个区域时，模型注意到第$i$个位置的程度:
    > $$
    > \beta_{j, i}=\frac{\exp \left(s_{i j}\right)}{\sum_{i=1}^{N} \exp \left(s_{i j}\right)}, \text { where } s_{i j}=f\left(\boldsymbol{x}_{i}\right)^{T} \boldsymbol{g}\left(\boldsymbol{x}_{j}\right)
    > $$
    > 
    >
    > "注意层"(self-attention feature map)的输出为：
    > $$
    > \boldsymbol{o}_{j}=\sum_{i=1}^{N} \beta_{j, i} \boldsymbol{h}\left(\boldsymbol{x}_{i}\right), \text { where } \boldsymbol{h}\left(\boldsymbol{x}_{i}\right)=\boldsymbol{W}_{\boldsymbol{h}} \boldsymbol{x}_{i}
    > $$
    > 通过$y_i=\gamma o_i+x_i$，进行融合得到加入了attention机制的feature maps。$\gamma$的值初始化为0，再逐渐增大权重，因为一开始attention可能训练的不太好，用attention来指引效果不好，随着attention层训练的越来越好后，加大它的权重

![img](https://img-blog.csdn.net/20180603120914577)

​	
