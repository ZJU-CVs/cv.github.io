### Learning Transferable Architectures for Scalable Image Recognition

#### 1.  Introduction

> 在基于Neural Architecture Search (NAS)进行网络结构最优搜索的结构上进行改进，不搜索整个网络的结构，只搜索block（cell）的最优结构，因为目前最流行的网络都是cell堆叠起来的



#### 2. Method

cell分为两种：

- Normal Cell：返回的feature map和输入的dimension相同
- Reduction Cell：返回的feature map的size是输入的一半



搜索方式：通过使用RNN进行递归的搜索

> RNN产生一种网络结构，将该网络结构训练至收敛，其在验证集上的准确率用于更新RNN，以让其产生更好的网络架构

RNN用于生成Cell结构：

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Nas14.png)



> 在搜索空间中，每个单元cell接收作为输入的两个初始隐藏变量$h_i$和$h_{i-1}$，这两个状态是前两个较低层中的两个单元输出或最初输入图像。（Cell 初始时有两个特征图输入，其实是一个残差结构）
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Nas15.png" alt="img" style="zoom:50%;" />

> 考虑到这两个初始隐藏状态，控制器RNN递归地预测卷积单元的其余结构
>
> (1) 从隐藏状态集中（含有经过不同处理的特征图集合）选取一个hidden state
>
> (2) 重复第一步得到第二个hidden state
>
> (3) 对于第一步得到的hidden state选取一个operation
>
> (4) 对于第二步得到的hidden state选取一个operation
>
> (5) 将第三步和第四步得到的结果选取一个operation进行combine

采用以上五步用于生成一个cell结构，以上五步重复B次

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Nas13.png)

```
第(3)和(4)步的operation有：
  • identity
	• 1x7 then 7x1 convolution
	• 3x3 average pooling
	• 5x5 max pooling
  • 1x1 convolution
	• 3x3 depthwise-separable conv
	• 7x7 depthwise-separable conv
	• 1x3 then 3x1 convolution
	• 3x3 dilated convolution
	• 3x3 max pooling
  • 7x7 max pooling
  • 3x3 convolution
  • 5x5 depthwise-seperable conv

第(5)步的combine operation有：
	• element-wise addition
	• concatenation 
```



重复上述5个步骤B次后（B=5），所有未使用过的隐藏状态将在深度方向进行concat，作为cell的输出

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Nas12.png" alt="img" style="zoom:67%;" />

> Block -> Cell -> Architecture
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/Nas16.png" alt="img" style="zoom:50%;" />

