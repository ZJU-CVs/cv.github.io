---
layout:     post
title:      Deep image prior
subtitle:   
date:       2020-06-21
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Backbone

---



#### 1. Introdution

- 通常实现denoising, superresolution, inpainting这些任务的时候，都是用大量数据集来训练某个神经网络，用训练后的网络来更正degraded image，从而还原目标图片。
- 本文中，使用generative model，用一些random vector来生成reconstructed image。这个generative model的parameters一开始是随机设定的，然后用单个degraded image为目标，计算loss function来optimize parameters。在optimization overfit之前停止，会发现生成的图片就是去噪/高分辨率/Inpainted的还原图像
- 大规模训练图像采用了训练集作为prior来获取图片等数据分布；**而本文的prior是network本身的structure**，认为structure本身有抓取image statics的功能
- 方法**优点**：根据输入图像本身的信息进行训练与预测



#### 2. Method



>  $x:$ real image
>
>  $\hat{x}:$ degraded image
>
>  $x^*:$ reconstruction image

  **以去噪为例：**

  - 初始化深度卷积生成(解码)网络$f$，网络权重随机初始化。此网络主要通过输入固定的随机编码向量$z$，生成出一个仿造的图像$f(z)$
  
  - 以使$x$与$f(z)$之间的差异尽可能的小为目标训练$f$的参数
  - 选择合适的损失函数(如MSE)
  - 在图像复原过程中，添加正则项$R(x)$来保持图像的光滑性质，此时损失函数变为$x^{*}=\min _{x} E\left(x ; x_{0}\right)+R(x)$
  - **在理论上，如果网络$f$足够大，训练时间足够久，可实现输出与$x$非常接近的图像，甚至一致(若不添加正则项)**
  - 但如果在训练到一半时终止训练，也就是我们给定迭代次数，会发现它会输出一幅“修复过的$x$”。可认为此时生成的图像还没来得及对原图非常细节部分的噪声进行学习。

  ```markdown
  这说明：深度卷积网络本身会先学会原图中“未被破坏的，符合自然规律的部分”，然后才会学会“被破坏的部分”。
  例如，它会先学会如何复制出一张没有噪点的x，然后才会学会复制出一张有噪点的x。
  这是因为卷积的不变性和深度网络逐层抽象的结构。
  ```

  **实验表明：生成网络有一种能力，能够先学会图像X中没有被破坏的部分，然后再学习被破坏部分的优化**

  

  

网络结构的细节，本质上是一个`decoder-encoder`

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/DIP.png)

