---
layout:     post
title:      Weakly Supervised Instance Segmentation using Class Peak Response
subtitle:   
date:       2020-08-27
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - paper
    - Segmentation

---



#### 1. Introduction

- 使用图像级别标签(label-level)，利用分类网络，实现弱监督实例分割

- 主要思路

  > 利用CNN网络生成class response maps，其指定每个图像位置处的分类置信度。
  >
  > 取局部极大值之后反向计算，得到与这个局部极大值相关的区域信息。
  >
  > 再结合类别信息等，从利用传统算法求出的segmentation mask里面进行打分排序，得到分割结果。 



#### 2. Method

- 网络训练只用到了分类信息，在正向传播时每个卷积层输入记作$U$，输出记作$V$，坐标$(i,j)$下的值就记作$V_{i,j}$

- 定义反传公式：

$$
P\left(U_{i j}\right)=\sum_{p=i-\frac{k H}{2}}^{i+\frac{k H}{2}} \sum_{q=j-\frac{k W}{2}}^{j+\frac{k W}{2}} P\left(U_{i j} | V_{p q}\right) \times P\left(V_{p q}\right)\\
P\left(U_{i j} | V_{p q}\right)=Z_{p q} \times \hat{U}_{i j} W_{(i-p)(j-q)}^{+}
$$



> 其中$\hat{U}_{ij}$是反向激化函数
>
> $W^+=RELU(W)$
>
> Z是归一化因子使得$\sum_{p,q}P(U_ij \mid Vpq)=1$   



![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/9.png)

- 反传得到相应图peak response map(PRM)，记作$R$，由形态梯度方法计算proposal区域对应的mask proposal，计算$Score$，选取最高得分作为Segmentation Mask结果


$$
\text {Score}=\underbrace{\alpha * R * S}_{\text {instance-aware }}+\underbrace{R * \hat{S}}_{\text {boundary-aware }}-\underbrace{\beta * Q * S}_{\text {class-aware }}
$$

> 其中$S$表示mask proposal，$\hat{S}$表示由形态计算学计算所得$S$的梯度，$Q$表示类响应图(Class Response Map)获得的背景掩模。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/PRM.png)

- 在测试阶段，出现的峰值被反向传播并有效地映射到每个对象实例的信息区域(如实例边界)



#### 3.  Conclusion

- 优点：利用图像级标签得到分类网络，并利用分类网络反传得到PRM图。利用PRM图得到mask proposal，计算Score，选取最高得分作为结果
- 缺点：当类别数逐渐增多的情况下，方法性能会下降

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/picture/PRM1.png)

