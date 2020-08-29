### U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation



1、简介

- 将attention机制引入生成器和判别器，能够引导生成器G关注那些区分源域与目标域的更重要的区域，从而使得G的性能能够更好发挥，并让G对于图像整体的改变与实体形变有更好的处理能力
- 在生成器的解码阶段采用自适应的ILN(Adaptive, adaptive instance-layer normalization)



2、模型介绍

**Attention机制**(采用CAM)

> - 通过网络的encode编码阶段得到特征图(encoder feature map)，如$C*H*W$
> - 然后将特征图**最大池化**成$C*1*1$，经过全连接层压缩到$B*1$维（这里的$B$是BatchSize，对于图像转换，通常取1）
> - 然后将全连接层参数$weight(w_s=w_s^1, w_s^2,...,w_s^n)$和特征图相乘，得到最大池化attention的特征图；
>
> - 同样对特征图**均值池化**然后同样的操作得到均值池化attention的特征图
>
> - 将两个特征图concat得到$C*2$通道的$H*W$的特征图，然后将新的特征图经过$1*1$卷积将通道改变成$C$送入decode解码阶段
>
> - 对于经过全连接得到的$B*1$维，做concat后输入到Auxiliary classification，用于源域和目标域的分类判断（二分类，是源域0还是目标域1）
>
> - 同样的attention操作作用于判别器。
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture\UGATIT2.png" alt="img" style="zoom:80%;" />



AdaLIN(自适应图层实例归一化)

`帮助注意力引导模型灵活控制形状和纹理的变化`

> - 通过将特征图flaten，经过全连接得到$C$通道的$\gamma$和$\beta$，作为AdaILN的参数，其中解码器的残差网络部分用AdaILN，upsample上采样部分用ILN，ILN中的所有超参数是可学习的。
>
>   
>
>   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture\UGATIT3.png" alt="img" style="zoom:70%;" />

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/U-GAT-IT.png)