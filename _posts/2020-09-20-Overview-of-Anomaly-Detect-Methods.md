---
layout:     post
title:      Overview of Anomaly Detect Methods 
subtitle:   
date:       2020-09-20
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Anomaly Detection
---



#### P-Net

《Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images》

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/29.png" alt="img" style="zoom:50%;" />

- 异常检测方法主要分为SVDD类（学习判别超平面以将异常样本和正常样本区分来开）和Anomaly类（引入生成对抗网络来进行异常检测）

- 本文认为，纹理和结构信息有助于异常检测，因此提出了P-Net网络结构
  - 从原始图像中提取结构，然后将结构特征与图像特征进行融合以重构图像
  - 使用重构图像进一步提取特征，可以作为正则化器，有助于改善前一阶段的图像重构
  - 同时，通过衡量原始图像和重构图像间的内容误差和结构误差得到异常分数，用于判别正常/异常



##### Details

> 以医学图像为例，对于健康人群，视网膜的脉管系统分布和组织学是规则的，而对于患有疾病的受试者，病变将破坏脉管系统和组织学的规则性



> **模型细节：**
>
> - 网络架构包括三个模块：
>
>   1) 结构提取模块$G_s$，从原始图像$I$中提取结构$S$; 
>
>   2) 图像重构模块$G_r$，利用图像编码器最后一层输出的特征和结构来重构。通过最小化$I$和$\hat{I}$的差异，将纹理和结构之间的关系编码到网络中
>
>   3) 从重构图像模块中提取结构$\hat{S}$，通过最小化$S$和$\hat{S}$的差异。该模块使原始图像被$G_r$正确地重构



> **Structure extraction network with domain adaptation**
>
> 不同数据集间域适应
> $$
> \mathcal{L}_{seg}(I_{src})= - \sum S_{src} log(G_s(I_{src}))\\
> \mathcal{L}_{seg}(I_{tar})= \mathbb{E}[log(1-D(G_s(I_{tar})))]+\mathbb{E}[logD(G_s(I_{src}))]
> $$
> 
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/34.png" alt="img" style="zoom:80%;" />



> **Image Reconstruction Module**
>
> 分别用En1和En2对原始图像及其结构进行编码，然后将两个特征串联起来，输入解码器De以重建原始图像
>
> 在结构编码器En1和解码器之间为同一级别的特征引入了skip connection，从而避免了由于结构中的下采样池化而导致的信息丢失；
>
> 而在图像编码器En2和解码器之间没有skip connection，防止影响结构对于原始图像重构的作用
>
> 为了提高重建图像的质量，应用PatchGAN中的重构误差
> $$
> \mathcal{L}_{\mathrm{rec}}(\mathbf{I})=\Vert\mathbf{I}-\hat{\mathbf{I}}\Vert_{1}\\
> \mathcal{L}_{\mathrm{adv}}(\mathbf{I})=\mathbb{E}\left[\log \left(1-\mathbf{D}\left(\mathbf{G}_{r}(\mathbf{I}, \mathbf{S})\right)\right)\right]+\mathbb{E}[\log \mathbf{D}(\mathbf{I})]
> $$
> 



> **Structure Extraction From Reconstructed Image Module**
>
> 进一步将结构提取器$G_s$附加到重建图像上，有两个目的：
>
> 1）通过使从原始图像提取的结构与从重建图像提取的结构相同，可以更好地重建原始图像。 从这个意义上说，从重建的图像模块重建图像的行为像一个正则化器。  
>
> 2）一些病变在结构上更具区分性，分别从原始图像和重建图像中提取结构，并利用它们的差异进行异常检测
> $$
> \mathcal{L}_{\mathrm{str}}(\mathbf{I})=\Vert\mathbf{S}-\hat{\mathbf{S}}\Vert_{1}
> $$



> **Objective Function**
> $$
> \mathcal{L}=\lambda_1 \mathcal{L}_{adv}+\lambda_2 \mathcal{L}_{rec}+\lambda_s \mathcal{L}_{str}
> $$



> **Anomaly Detection for Testing Data**
> $$
> \mathcal{A}(I)=(1-\lambda_f)\Vert I-\hat{I}\Vert_1+\lambda_f\Vert S-\hat{S}\Vert_1
> $$
> 



#### DifferNet

《Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows》

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/30.png" alt="img" style="zoom:50%;" />



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/31.png" alt="img" style="zoom:50%;" />



<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/32.png" alt="img" style="zoom:50%;" />

#### DeScarGAN

《DeScarGAN: Disease-Speciﬁc Anomaly Detection with Weak Supervision》

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/33.png" alt="img" style="zoom:50%;" />



#### Patch SVDD

详见 https://zju-cvs.github.io/2020/08/23/Patch-SVDD/

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/6.png" alt="img" style="zoom:50%;" />



#### FCDD

详见 https://zju-cvs.github.io/2020/08/23/Explainable-Deep-One-Class-Classification/

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/Anomaly-Detection/3.png" alt="img" style="zoom:50%;" />