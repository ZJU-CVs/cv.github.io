### Facial Expression Recognition by De-expression Residue Learning

1、简介

核心思想：一个人的人脸表情是由表情和中性脸组成。

![image-20190902210503226](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/image-20190902210503226.png)

方法：通过*De-expression Residue Learning*，提取面部表情组成部分的信息。

2、DeRL模型

- 首先通过cGAN训练一个生成模型，来学习用expression的图像生成中立图像

![page1image6056512.png](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/page1image6056512.png)

- 生成器中各个中间层保留了表情中的expressive component特征，因此可以用来训练分类器classifier，从而对人脸表情进行分类 

![image-20190902210251943](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/image-20190902210251943.png)

3、算法细节

- CGAN：

  输入图像对<$I_ {input}$ , $I_ {target}$>，训练cGAN。输入目标是显示任何表情的脸部图像，目标是同一主题的中性脸部图像。训练后的生成器为任何输入重建相应的中性面部图像，同时保持身份信息不变。从表情面部图像到中性面部图像，表达信息被记录在中间层中的表达成分

  -  鉴别器D目标表示：

  ![image-20190902212438145](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/image-20190902212438145.png)

  （其中Ｎ是训练图相对的总数）

  - 生成器G的结构设计采用Autoencoder的形式，从而保证了会存在相同尺寸的中间层，目标表示：

  ![1565656-20190326201342367-2136445850](https://img2018.cnblogs.com/blog/1565656/201903/1565656-20190326201342367-2136445850.png)

  	(其中使用L1损失来获得图像相似度而不是L2，因为L2损失倾向于过度模糊输出图像)
  -  最终目标

     ![image-20190902213125103](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/image-20190902213125103.png)

-  分类器

   -  用I代表查询图像，在输入生成模型G后，生成中性表达图像：

      ![img](https://img2018.cnblogs.com/blog/1565656/201903/1565656-20190326201552581-340108560.png)

   （其中，G是生成器，E属于六种基本原型面部表情中的任何一种）

   - 为了从发生器的中间层学习去表达残差，这些层的所有滤波器都是固定的，并且具有相同大小的所有层被连接并输入到本地CNN中，对于每个本地CNN模型，代价函数被标记为损失i，i∈[1,2,3,4]。

   - 每个本地CNN模型的最后全完连接的层被进一步连接并与用于面部表情分类的最后编码层组合。
     ```
      Total loss = λ1loss1 + λ2loss2 + λ3loss3 + λ4loss4 + λ5loss5
     ```

