---
layout:     post
title:      Awesome Fine-grained Image Analysis (FGIA)
subtitle:   细粒度图像分析总结
date:       2020-08-05
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Overview
    - Fine-grained
    - Update
---


[TOC]



## Introduction

- 细粒度图像，相对于通用图像(general/generic images)的区别和难点在于其图像所属类别的粒度更为精细，是计算机视觉领域比较热门的一个方向，包括了分类、检索以及图像生成等

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/1.png" alt="img" style="zoom: 30%;" />

- 细粒度图像识别的难点和挑战主要在于：
  - **类间差异小** (small inter-class variance)：都属于同一个物种下的小类
  - **类内差异大** (large intra-class variance)：受姿态、尺度和旋转等因素影响



## Tutorials

- [Fine-Grained Image Analysis](http://www.weixiushen.com/tutorial/PRICAI18/FGIA.html).
  Xiu-Shen Wei, and Jianxin Wu. *Pacific Rim International Conference on Artificial Intelligence (PRICAI)*, 2018.

- [Fine-Grained Image Analysis](http://www.icme2019.org/conf_tutorials).
  Xiu-Shen Wei. *IEEE International Conference on Multimedia and Expo (ICME)*, 2019.



## Survey papers

- [Deep Learning for Fine-Grained Image Analysis: A Survey](https://arxiv.org/abs/1907.03069).
  Xiu-Shen Wei, Jianxin Wu, and Quan Cui. *arXiv: 1907.03069*, 2019.

- [A Survey on Deep Learning-based Fine-Grained Object Classification and Semantic Segmentation](https://link.springer.com/article/10.1007/s11633-017-1053-3).
  Bo Zhao, Jiashi Feng, Xiao Wu, and Shuicheng Yan. *International Journal of Automation and Computing*, 2017.

## Benchmark datasets

展示了 11 个数据集，如下图所示，其中 **BBox** 表示数据集提供物体的边界框信息，**Part anno** 则是数据集共了关键部位的位置信息，**HRCHY** 表示有分层次的标签，**ATR** 表示属性标签（比如翅膀颜色等），**Texts** 表示提供了图片的文本描述信息。

| Dataset name                                                 | Year | Meta-class       | images  | categories | BBox                                                         | Part annotation                                              | HRCHY                                                        | ATR                                                          | Texts                                                        |
| ------------------------------------------------------------ | ---- | ---------------- | ------- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [*Oxford flower*](https://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.pdf) | 2008 | Flowers          | 8,189   | 102        |                                                              |                                                              |                                                              |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |
| [*CUB200*](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) | 2011 | Birds            | 11,788  | 200        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |
| [*Stanford Dog*](http://vision.stanford.edu/aditya86/StanfordDogs/) | 2011 | Dogs             | 20,580  | 120        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |                                                              |                                                              |
| [*Stanford Car*](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | 2013 | Cars             | 16,185  | 196        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |                                                              |                                                              |
| [*FGVC Aircraft*](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | 2013 | Aircrafts        | 10,000  | 100        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |
| [*Birdsnap*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.450.500&rep=rep1&type=pdf) | 2014 | Birds            | 49,829  | 500        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |
| [*NABirds*](https://vision.cornell.edu/se3/wp-content/uploads/2015/05/Horn_Building_a_Bird_2015_CVPR_paper.pdf) | 2015 | Birds            | 48,562  | 555        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |                                                              |
| [*DeepFashion*](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | 2016 | Clothes          | 800,000 | 1,050      | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |
| [*Fru92*](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_VegFru_A_Domain-Specific_ICCV_2017_paper.pdf) | 2017 | Fruits           | 69,614  | 92         |                                                              |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |
| [*Veg200*](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_VegFru_A_Domain-Specific_ICCV_2017_paper.pdf) | 2017 | Vegetable        | 91,117  | 200        |                                                              |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |
| [*iNat2017*](http://openaccess.thecvf.com/content_cvpr_2018/papers/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.pdf) | 2017 | Plants & Animals | 859,000 | 5,089      | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |
| [*RPC*](https://rpc-dataset.github.io/)                      | 2019 | Retail products  | 83,739  | 200        | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                                                              |



## Fine-grained image recognition

### Fine-grained recognition by localization-classification subnetworks

`基于定位-分类网络`

#### Classical State-of-the-arts

- [Mask-CNN: Localizing Parts and Selecting Descriptors for Fine-Grained Im age Recognition](https://arxiv.org/pdf/1605.06878.pdf)

  > 主要包括两个模块：第一个是Part Localization，第二个全局和局部图像块的特征学习
  >
  > - 在Mask-CNN中，借助FCN学习一个三分类分割模型（一类为头部、一类为躯干、最后一类则是背景），GT mask是通过Part Annotation得到的头部和躯干部位的最小外接矩形。
  >
  >   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/3.png" alt="img" style="zoom: 50%;" />
  >
  > - FCN训练完毕后，可以对测试集中的细粒度图像进行较精确地part定位，得到part mask，合起来为object-mask，用于part localization和useful feature selection
  >
  >   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/4.png" alt="img" style="zoom: 40%;" />
  >
  > - 将不同部位输入到CNN子网络后输出feature map，利用前面得到的part-mask和object-mask作为权重，与对应像素点点乘。然后再分别进行max pooling和average pooling得到的特征级联作为子网络的最终feature vector。最后将三个子网特征再次级联作为整张图像的特征表示
  >
  >   <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/2.png" alt="img" style="zoom: 30%;" />
  >
  > 

- [Selective Sparse Sampling for Fine-Grained Image Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9008286)

  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/5.png" alt="img" style="zoom: 50%;" />
  >
  > 提出了一种捕捉细粒度级别特征同时不会丢失上下文信息的简单有效的框架，
  >
  > - 采用class peak responses，从class response map中定位局部最大值，从而形成sparse attention。sparse attention通常对应于精细的图像部分
  >
  > - 定义了两个平行的采样分支去重采样图片：
  >   - **判别性**(discriminative)分支：抽取判别性的特征
  >   - **互补性**(complementary)分支：抽取互补性的特征
  > - 将三个输出拼接后通过FC层实现最终的分类

  


#### Related Works 

- [Part-based R-CNNs for Fine-Grained Category Detection](https://arxiv.org/pdf/1407.3867.pdf).
  Ning Zhang, Jeff Donahue, Ross Girshick, and Trevor Darrell. *ECCV*, 2014. `[code]`

- [Fine-Grained Recognition without Part Annotations](http://vision.stanford.edu/pdf/joncvpr15.pdf).
  Jonathan Krause, Hailin Jin, Jianchao Yang, and Li Fei-Fei. *CVPR*, 2015. `[code]`

- [The Application of Two-level Attention Models in Deep Convolutional Neural Network for Fine-grained Image Classification](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_The_Application_of_2015_CVPR_paper.pdf).
  Tianjun Xiao, Yichong Xu, Kuiyuan Yang, Jiaxing Zhang, Yuxin Peng, and Zheng Zhang. *CVPR*, 2015.

- [Deep LAC: Deep Localization, Alignment and Classification for Fine-grained Recognition](http://jiaya.me/papers/deeplac_cvpr15.pdf).
  Di Lin, Xiaoyong Shen, Cewu Lu, and Jiaya Jia. *CVPR*, 2015.

- [Spatial Transformer Networks](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf).
  Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu. *NeurIPS*, 2015. `[code]`

- [Part-Stacked CNN for Fine-Grained Visual Categorization](https://arxiv.org/pdf/1512.08086.pdf).
  Shaoli Huang, Zhe Xu, Dacheng Tao, and Ya Zhang. *CVPR*, 2016.

- [Mining Discriminative Triplets of Patches for Fine-Grained Classification](http://users.umiacs.umd.edu/~morariu/publications/WangTripletsCVPR16.pdf).
  Yaming Wang, Jonghyun Choi, Vlad I. Morariu, and Larry S. Davis. *CVPR*, 2016.

- [SPDA-CNN: Unifying Semantic Part Detection and Abstraction for Fine-grained Recognition](https://zpascal.net/cvpr2016/Zhang_SPDA-CNN_Unifying_Semantic_CVPR_2016_paper.pdf).
  Han Zhang, Tao Xu, Mohamed Elhoseiny, Xiaolei Huang, Shaoting Zhang, Ahmed Elgammal, and Dimitris Metaxas. *CVPR*, 2016.

- [Picking Deep Filter Responses for Fine-grained Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Picking_Deep_Filter_CVPR_2016_paper.pdf).
  Xiaopeng Zhang, Hongkai, Xiong, Wengang Zhou, Weiyao Lin, and Qi Tian. *CVPR*, 2016.

- [Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf).
  Jianlong Fu, Heliang Zheng, and Tao Mei. *CVPR*, 2017.

- [Fine-Grained Recognition as HSnet Search for Informative Image Parts](http://web.engr.oregonstate.edu/~sinisa/talks/cvpr17_lstmsearch_oral.pdf).
  Michael Lam, Behrooz Mahasseni, and Sinisa Todorovic. *CVPR*, 2017.

- [Learning Multi-attention Convolutional Neural Network for Fine-Grained Image Recognition](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Learning_Multi-Attention_Convolutional_ICCV_2017_paper.pdf).
  Heliang Zheng, Jianlong Fu, Tao Mei, and Jiebo Luo. *ICCV*, 2017. `[code]`

- [Weakly Supervised Learning of Part Selection Model with Spatial Constraints for Fine-Grained Image Classification](https://pdfs.semanticscholar.org/bd2d/af3fc3566889b3fd1933e50589e12460cc97.pdf).
  Xiangteng He, and Yuxin Peng. *AAAI*, 2017.

- [Localizing by Describing: Attribute-Guided Attention Localization for Fine-Grained Recognition](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14323/14299).
  Xiao Liu, Jiang Wang, Shilei Wen, Errui Ding, and Yuanqing Lin. *AAAI*, 2017.

- [Learning to Navigate for Fine-grained Classification](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.pdf).
  Ze Yang, Tiange Luo, Dong Wang, Zhiqiang Hu, Jun Gao, and Liwei Wang. *ECCV*, 2018. `[code]`

- [Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Sun_Multi-Attention_Multi-Class_Constraint_ECCV_2018_paper.pdf).
  Ming Sun, Yuchen Yuan, Feng Zhou, and Errui Ding. *ECCV*, 2018. `[code]`

- [Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification From the Bottom Up](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_Weakly_Supervised_Complementary_Parts_Models_for_Fine-Grained_Image_Classification_From_CVPR_2019_paper.pdf).
  Weifeng Ge, Xiangru Lin, and Yizhou Yu. *CVPR*, 2019.

- [Selective Sparse Sampling for Fine-grained Image Recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf).
  Yao Ding, Yanzhao Zhou, Yi Zhu, Qixiang Ye, and Jianbin Jiao. *ICCV*, 2019. `[code]`

- [Interpretable and Accurate Fine-grained Recognition via Region Grouping](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Interpretable_and_Accurate_Fine-grained_Recognition_via_Region_Grouping_CVPR_2020_paper.pdf).
  Zixuan Huang, and Yin Li. *CVPR*, 2020.

- [Weakly Supervised Fine-grained Image Classification via Guassian Mixture Model Oriented Discriminative Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Weakly_Supervised_Fine-Grained_Image_Classification_via_Guassian_Mixture_Model_Oriented_CVPR_2020_paper.pdf).
  Zhihui Wang, Shijie Wang, Shuhui Yang, Haojie Li, Jianjun Li, and Zezhou Li. *CVPR*, 2020.

- [Graph-Propagation Based Correlation Learning for Weakly Supervised Fine-Grained Image Classification](https://aaai.org/ojs/index.php/AAAI/article/view/6912).
  Zhihui Wang, Shijie Wang, Haojie Li, Zhi Dou, and Jianjun Li. *AAAI*, 2020.

- [Filtration and Distillation: Enhancing Region Attention for Fine-Grained Visual Categorization](https://www.aiide.org/ojs/index.php/AAAI/article/view/6822).
  Chuanbin Liu, Hongtao Xie, Zheng-Jun Zha, Lingfeng Ma, Lingyun Yu, and Yongdong Zhang. *AAAI*, 2020.



### Fine-grained recognition by end-to-end feature encoding

`端对端特征编码`

#### Classical State-of-the-arts

- [Bilinear CNN Models for Fine-Grained Visual Recognition](https://arxiv.org/pdf/1504.07889v6.pdf)

  > 

#### Related Works 

- [Fine-Grained Visual Categorization via Multi-stage Metric Learning](http://qi-qian.com/papers/cvpr15.pdf).
  Qi Qian, Rong Jin, Shenghuo Zhu, and Yuanqing Lin. *CVPR*, 2015.

- [Hyper-Class Augmented and Regularized Deep Learning for Fine-Grained Image Classification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.717.6527&rep=rep1&type=pdf).
  Saining Xie, Tianbao Yang, Xiaoyu Wang, and Yuanqing Lin. *CVPR*, 2015.

- [Subset Feature Learning for Fine-Grained Category Classification](http://openaccess.thecvf.com/content_cvpr_workshops_2015/W03/papers/Ge_Subset_Feature_Learning_2015_CVPR_paper.pdf).
  ZongYuan Ge, Christopher McCool, Conrad Sanderson, and Peter Corke. *CVPR*, 2015.

- [Bilinear CNN Models for Fine-grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf).
  Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji. *ICCV*, 2015. `[code]`

- [Multiple Granularity Descriptors for Fine-Grained Categorization](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Multiple_Granularity_Descriptors_ICCV_2015_paper.pdf).
  Dequan Wang, Zhiqiang Shen, Jie Shao, Wei Zhang, Xiangyang Xue, and Zheng Zhang. *ICCV*, 2015.

- [Compact Bilinear Pooling](https://people.eecs.berkeley.edu/~yg/papers/compact_bilinear.pdf).
  Yang Gao, Oscar Beijbom, Ning Zhang, and Trevor Darrell. *CVPR*, 2016. `[code]`

- [Fine-Grained Image Classification by Exploring Bipartite-Graph Labels](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Fine-Grained_Image_Classification_CVPR_2016_paper.pdf).
  Feng Zhou, and Yuanqing Lin. *CVPR*, 2016. `[project page]`

- [Kernel Pooling for Convolutional Neural Networks](https://vision.cornell.edu/se3/wp-content/uploads/2017/04/cui2017cvpr.pdf).
  Yin Cui, Feng Zhou, Jiang Wang, Xiao Liu, Yuanqing Lin, and Serge Belongie. *CVPR*, 2017.

- [Low-rank Bilinear Pooling for Fine-Grained Classification](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kong_Low-Rank_Bilinear_Pooling_CVPR_2017_paper.pdf).
  Shu Kong, and Charless Fowlkes. *CVPR*, 2017. `[code]`

- [Higher-order Integration of Hierarchical Convolutional Activations for Fine-Grained Visual Categorization](http://azadproject.ir/wp-content/uploads/2014/07/2017-Higher-order-Integration-of-Hierarchical-Convolutional-Activations-for-Fine-grained.pdf).
  Sijia Cai, Wangmeng Zuo, and Lei Zhang. *ICCV*, 2017. `[code]`

- [Learning a Discriminative Filter Bank within a CNN for Fine-grained Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_a_Discriminative_CVPR_2018_paper.pdf).
  Yaming Wang, Vlad I. Morariu, and Larry S. Davis. *CVPR*, 2018. `[code]`

- [Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Towards_Faster_Training_CVPR_2018_paper.pdf).
  Peihua Li, Jiangtao Xie, Qilong Wang, and Zilin Gao. *CVPR*, 2018. `[code]`

- [Maximum-Entropy Fine Grained Classification](https://papers.nips.cc/paper/7344-maximum-entropy-fine-grained-classification).
  Abhimanyu Dubey, Otkrist Gupta, Ramesh Raskar, and Nikhil Naik. *NeurIPS*, 2018.

- [Pairwise Confusion for Fine-Grained Visual Classification](https://link.springer.com/chapter/10.1007/978-3-030-01258-8_5).
  Abhimanyu Dubey, Otkrist Gupta, Pei Guo, Ramesh Raskar, Ryan Farrell, and Nikhil Naik. *ECCV*, 2018. `[code]`

- [DeepKSPD: Learning Kernel-matrix-based SPD Representation for Fine-Grained Image Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Melih_Engin_DeepKSPD_Learning_Kernel-matrix-based_ECCV_2018_paper.pdf).
  Melih Engin, Lei Wang, Luping Zhou, and Xinwang Liu. *ECCV*, 2018.

- [Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.pdf).
  Chaojian Yu, Xinyi Zhao, Qi Zheng, Peng Zhang, and Xinge You. *ECCV*, 2018. `[code]`

- [Grassmann Pooling as Compact Homogeneous Bilinear Pooling for Fine-Grained Visual Classification](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xing_Wei_Grassmann_Pooling_for_ECCV_2018_paper.pdf).
  Xing Wei, Yue Zhang, Yihong Gong, Jiawei Zhang, and Nanning Zheng. *ECCV*, 2018.

- [Learning Deep Bilinear Transformation for Fine-grained Image Representation](http://papers.nips.cc/paper/8680-learning-deep-bilinear-transformation-for-fine-grained-image-representation.pdf).
  Heliang Zheng, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo. *NeurIPS*, 2019.

- [Looking for the Devil in the Details: Learning Trilinear Attention Sampling Network for Fine-Grained Image Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Looking_for_the_Devil_in_the_Details_Learning_Trilinear_Attention_CVPR_2019_paper.pdf).
  Heliang Zheng, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo. *CVPR*, 2019. `[code]`

- [Destruction and Construction Learning for Fine-grained Image Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf).
  Yue Chen, Yalong Bai, Wei Zhang, and Tao Mei. *CVPR*, 2019. `[code]`

- [Learning a Mixture of Granularity-Specific Experts for Fine-Grained Categorization](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_a_Mixture_of_Granularity-Specific_Experts_for_Fine-Grained_Categorization_ICCV_2019_paper.pdf).
  Lianbo Zhang, Shaoli Huang, Wei Liu, and Dacheng Tao. *ICCV*, 2019.

- [Cross-X Learning for Fine-Grained Visual Categorization](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf).
  Wei Luo, Xiong Yang, Xianjie Mo, Yuheng Lu, Larry S. Davis, Jun Li, Jian Yang, and Ser-Nam Lim. *ICCV*, 2019. `[code]`

- [Fine-grained Image-to-Image Transformation towards Visual Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiong_Fine-Grained_Image-to-Image_Transformation_Towards_Visual_Recognition_CVPR_2020_paper.pdf).
  Wei Xiong, Yutong He, Yixuan Zhang, Wenhan Luo, Lin Ma, and Jiebo Luo. *CVPR*, 2020.

- [Attention Convolutional Binary Neural Tree for Fine-Grained Visual Categorization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Attention_Convolutional_Binary_Neural_Tree_for_Fine-Grained_Visual_Categorization_CVPR_2020_paper.pdf).
  Ruyi Ji, Longyin Wen, Libo Zhang, Dawei Du, Yanjun Wu, Chen Zhao, Xianglong Liu, and Feiyue Huang. *CVPR*, 2020. `[code]`

- [Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches](https://arxiv.org/pdf/2003.03836.pdf).
  Ruoyi Du, Dongliang Chang, Ayan Kumar Bhunia, Jiyang Xie, Yi-Zhe Song, Zhanyu Ma, and Jun Guo. *ECCV*, 2020. `[code]`

- [Channel Interaction Networks for Fine-Grained Image Categorization](https://arxiv.org/pdf/2003.05235.pdf).
  Yu Gao, Xintong Han, Xun Wang, Weilin Huang, and Matthew R. Scott. *AAAI*, 2020.

- [Learning Attentive Pairwise Interaction for Fine-Grained Classification](https://arxiv.org/pdf/2002.10191.pdf).
  Peiqin Zhuang, Yali Wang, and Yu Qiao. *AAAI*, 2020.

- [Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes](https://arxiv.org/pdf/1912.06842v1.pdf).
  Guolei Sun, Hisham Cholakkal, Salman Khan, Fahad Shahbaz Khan, and Ling Shao. *AAAI*, 2020.



### Fine-grained by leveraging attention mechanisms

`利用注意力机制`

#### Classical State-of-the-arts

- [Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf)

  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/7.png" alt="img" style="zoom: 50%;" />



### Fine-grained by contrastive learning manners

`利用对比学习`

#### Classical State-of-the-arts

- [Learning Attentive Pairwise Interation for Fine-Grained Classification](https://arxiv.org/pdf/2002.10191.pdf)

  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/8.png" alt="img" style="zoom: 50%;" />

  

- [Channel Interaction Networks for Fine-Grained Image Categorization](https://arxiv.org/pdf/2003.05235.pdf)

  > <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/FGIA/9.png" alt="img" style="zoom: 50%;" />





### Fine-grained recognition with external information

`采用额外信息，减少标注成本`

**Fine-grained recognition with web data / auxiliary data**

`web data / auxiliary data需要利用模型进行数据降噪`

- [Augmenting Strong Supervision Using Web Data for Fine-Grained Categorization](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xu_Augmenting_Strong_Supervision_ICCV_2015_paper.pdf).
  Zhe Xu, Shaoli Huang, Ya Zhang, and Dacheng Tao. *ICCV*, 2015.

- [Webly Supervised Learning Meets Zero-shot Learning: A Hybrid Approach for Fine-grained Classification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Webly_Supervised_Learning_CVPR_2018_paper.pdf).
  Li Niu, Ashok Veeraraghavan, and Vshu Sabbarwal. *CVPR*, 2018.

- [Fine-Grained Visual Categorization using Meta-Learning Optimization with Sample Selection of Auxiliary Data](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yabin_Zhang_Fine-Grained_Visual_Categorization_ECCV_2018_paper.pdf).
  Yabin Zhang, Hui Tang, and Kai Jia. *ECCV*, 2018. `[code]`

- [Webly-Supervised Fine-Grained Visual Categorization via Deep Domain Adaptation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7778168).
  Zhe Xu, Shaoli Huang, Ya Zhang, and Dacheng Tao. *IEEE TPAMI*, 2018.

- [Learning from Web Data using Adversarial Discriminative Neural Networks for Fine-Grained Classification](https://github.com/sxzrt/Learning-from-web-data).
  Xiaoxiao Sun, Liyi Chen, and Jufeng Yang. *AAAI*, 2019.

- [Web-Supervised Network with Softly Update-Drop Training for Fine-Grained Visual Classification](https://www.aiide.org/ojs/index.php/AAAI/article/view/6973).
  Chuanyi Zhang, Yazhou Yao, Huafeng Liu, Guo-Sen Xie, Xiangbo Shu, Tianfei Zhou, Zheng Zhang, Fumin Shen, and Zhenmin Tang. *AAAI*, 2020.



**Fine-grained recognition with multi-modality data**

- [Fine-Grained Image Classification via Combining Vision and Language](http://openaccess.thecvf.com/content_cvpr_2017/papers/He_Fine-Grained_Image_Classification_CVPR_2017_paper.pdf).
  Xiangteng He, and Yuxin Peng. *CVPR*, 2017.

- [Audio Visual Attribute Discovery for Fine-Grained Object Recognition](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16740).
  Hua Zhang, Xiaochun Cao, and Rui Wang. *AAAI*, 2018.

- [Fine-grained Image Classification by Visual-Semantic Embedding](https://www.ijcai.org/proceedings/2018/0145.pdf).
  Huapeng Xu, Guilin Qi, Jingjing Li, Meng Wang, Kang Xu, and Huan Gao. *IJCAI*, 2018.

- [Knowledge-Embedded Representation Learning for Fine-Grained Image Recognition](https://www.ijcai.org/proceedings/2018/0087.pdf).
  Tianshui Chen, Liang Lin, Riquan Chen, Yang Wu, and Xiannan Luo. *IJCAI*, 2018.

- [Bi-Modal Progressive Mask Attention for Fine-Grained Recognition](https://ieeexplore.ieee.org/document/9103943).
  Kaitao Song, Xiu-Shen Wei, Xiangbo Shu, Ren-Jie Song, Jianfeng Lu. *IEEE TIP*, 2020.



**Fine-grained recognition with humans in the loop**

- [Fine-grained Categorization and Dataset Bootstrapping using Deep Metric Learning with Humans in the Loop](https://vision.cornell.edu/se3/wp-content/uploads/2016/04/1950.pdf).
  Yin Cui, Feng Zhou, Yuanqing Lin, and Serge Belongie. *CVPR*, 2016.

- [Leveraging the Wisdom of the Crowd for Fine-Grained Recognition](https://ieeexplore.ieee.org/document/7115172).
  Jia Deng, Jonathan Krause, Michael Stark, and Li Fei-Fei. *IEEE TPAMI*, 2016.



### Fine-grained image recognition with limited data

`少样本学习在细粒度识别的应用`

#### Classical State-of-the-arts

- [Piecewise Classifier Mappings: Learning Fine-Grained Learners for Novel Categories with Few Examples](https://arxiv.org/pdf/1805.04288.pdf)

  > 

- [Multi-Attention Meta Learning for Few-Shot Fine-Grained Image Recognition](http://vipl.ict.ac.cn/homepage/jsq/publication/2020-Zhu-IJCAI-PRICAI.pdf)

  > 

- [Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition](https://arxiv.org/pdf/2004.00705.pdf)

  > 



## Fine-grained image retrieval

### Unsupervised with pre-trained models

- [Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval](http://www.weixiushen.com/project/SCDA/SCDA.html).
  Xiu-Shen Wei, Jian-Hao Luo, Jianxin Wu, and Zhi-Hua Zhou. *IEEE TIP*, 2017. `[project page]`

### Supervised with metric learning

- [Centralized Ranking Loss with Weakly Supervised Localization for Fine-Grained Object Retrieval](https://www.ijcai.org/proceedings/2018/0171.pdf).
  Xiawu Zheng, Rongrong Ji, Xiaoshuai Sun, Yongjian Wu, Feiyue Huang, and Yanhua Yang. *IJCAI*, 2018.

- [Towards Optimal Fine Grained Retrieval via Decorrelated Centralized Loss with Normalize-Scale layer](http://mac.xmu.edu.cn/rrji/papers/AAAI2019_zxw.pdf).
  Xiawu Zheng, Rongrong Ji, Xiaoshuai Sun, Baochang Zhang, Yongjian Wu, and Feiyue Huang. *AAAI*, 2019.



## Fine-grained image generation

### Generating from fine-grained image distributions

- [CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bao_CVAE-GAN_Fine-Grained_Image_ICCV_2017_paper.pdf).
  Jianmin Bao, Dong Chen, Fang Wen, Houqiang Li, and Gang Hua. *ICCV*, 2017. `[code]`

- [FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery](http://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_FineGAN_Unsupervised_Hierarchical_Disentanglement_for_Fine-Grained_Object_Generation_and_Discovery_CVPR_2019_paper.pdf).
  Krishna Kumar Singh, Utkarsh Ojha, and Yong Jae Lee. *CVPR*, 2019. `[code]`

### Generating from text descriptions

- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf).
  Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. *CVPR*, 2018. `[code]`



## Future directions of FGIA

### Fine-grained few shot learning

- [Piecewise classifier mappings: Learning fine-grained learners for novel categories with few examples](http://www.weixiushen.com/publication/tip19PCM.pdf).
  Xiu-Shen Wei, Peng Wang, Lingqiao Liu, Chunhua Shen, and Jianxin Wu. *IEEE TIP*, 2019.

- [Meta-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition](http://papers.nips.cc/paper/8570-meta-reinforced-synthetic-data-for-one-shot-fine-grained-visual-recognition.pdf).
  Satoshi Tsutsui, Yanwei Fu, and David Crandall. *NeurIPS*, 2019.

- [Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Revisiting_Pose-Normalization_for_Fine-Grained_Few-Shot_Recognition_CVPR_2020_paper.pdf).
  Luming Tang, Davis Wertheimer, and Bharath Hariharan. *CVPR*, 2020. `[code]`

- [Multi-attention Meta Learning for Few-shot Fine-grained Image Recognition](https://www.ijcai.org/Proceedings/2020/0152.pdf).
  Yaohui Zhu, Chenlong Liu, and Shuqiang Jiang. *IJCAI*, 2020.

### Fine-Grained hashing

- [ExchNet: A Unified Hashing Network for Large-Scale Fine-Grained Image Retrieval](http://www.weixiushen.com/publication/eccv20_ExchNet.pdf).
  Quan Cui, Qing-Yuan Jiang, Xiu-Shen Wei, Wu-Jun Li, and Osamu Yoshie. *ECCV*, 2020.

### Fine-grained domain adaptation

- [Fine-grained Recognition in the Wild: A Multi-Task Domain Adaptation Approach](https://ai.stanford.edu/~tgebru/papers/iccv.pdf).
  Timnit Geru, Judy Hoffman, and Li Fei-Fei. *ICCV*, 2017.

- [Progressive Adversarial Networks for Fine-Grained Domain Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/progressive-adversarial-networks-cvpr20.pdf).
  Sinan Wang, Xinyang Chen, Yunbo Wang, Mingsheng Long, and Jianmin Wang. *CVPR*, 2020.

- [An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition](http://www.weixiushen.com/publication/wacv20.pdf).
  Yimu Wang, Ren-Jie Song, Xiu-Shen Wei, and Lijun Zhang. *WACV*, 2020.

### FGIA within more realistic settings

- [The iNaturalist Species Classification and Detection Dataset](http://openaccess.thecvf.com/content_cvpr_2018/papers/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.pdf).
  Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. *CVPR*, 2018.

- [RPC: A Large-Scale Retail Product Checkout Dataset](https://arxiv.org/abs/1901.07249).
  Xiu-Shen Wei, Quan Cui, Lei Yang, Peng Wang, and Lingqiao Liu. *arXiv: 1901.07249*, 2019. `[project page]`

- [Presence-Only Geographical Priors for Fine-Grained Image Classification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Aodha_Presence-Only_Geographical_Priors_for_Fine-Grained_Image_Classification_ICCV_2019_paper.pdf).
  Oisin Mac Aodha, Elijah Cole, and Pietro Perona. *ICCV*, 2019.

## Recognition leaderboard

在数据集 **CUB200-2011** 数据集上的测试准确率，列举出目前最好的方法和其是否采用标准信息、额外的数据、采用的网络结构、输入图片的大小设置以及分类准确率

| Method                                                       | Publication   | BBox?                                                        | Part?                                                        | External information? | Base model   | Image resolution       | Accuracy |
| ------------------------------------------------------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------- | ------------ | ---------------------- | -------- |
| [PB R-CNN](https://people.eecs.berkeley.edu/~nzhang/papers/eccv14_part.pdf) | ECCV 2014     |                                                              |                                                              |                       | Alex-Net     | 224x224                | 73.9%    |
| [MaxEnt](https://papers.nips.cc/paper/7344-maximum-entropy-fine-grained-classification) | NeurIPS 2018  |                                                              |                                                              |                       | GoogLeNet    | TBD                    | 74.4%    |
| [PB R-CNN](https://people.eecs.berkeley.edu/~nzhang/papers/eccv14_part.pdf) | ECCV 2014     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                       | Alex-Net     | 224x224                | 76.4%    |
| [PS-CNN](https://zpascal.net/cvpr2016/Huang_Part-Stacked_CNN_for_CVPR_2016_paper.pdf) | CVPR 2016     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                       | CaffeNet     | 454x454                | 76.6%    |
| [MaxEnt](https://papers.nips.cc/paper/7344-maximum-entropy-fine-grained-classification) | NeurIPS 2018  |                                                              |                                                              |                       | VGG-16       | TBD                    | 77.0%    |
| [Mask-CNN](http://www.weixiushen.com/publication/pr17.pdf)   | PR 2018       |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                       | Alex-Net     | 448x448                | 78.6%    |
| [PC](https://link.springer.com/chapter/10.1007/978-3-030-01258-8_5) | ECCV 2018     |                                                              |                                                              |                       | ResNet-50    | TBD                    | 80.2%    |
| [DeepLAC](http://jiaya.me/papers/deeplac_cvpr15.pdf)         | CVPR 2015     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                       | Alex-Net     | 227x227                | 80.3%    |
| [MaxEnt](https://papers.nips.cc/paper/7344-maximum-entropy-fine-grained-classification) | NeurIPS 2018  |                                                              |                                                              |                       | ResNet-50    | TBD                    | 80.4%    |
| [Triplet-A](https://vision.cornell.edu/se3/wp-content/uploads/2016/04/1950.pdf) | CVPR 2016     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | Manual labour         | GoogLeNet    | TBD                    | 80.7%    |
| [Multi-grained](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Multiple_Granularity_Descriptors_ICCV_2015_paper.pdf) | ICCV 2015     |                                                              |                                                              | WordNet etc.          | VGG-19       | 224x224                | 81.7%    |
| [Krause *et al.*](http://vision.stanford.edu/pdf/joncvpr15.pdf) | CVPR 2015     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              |                       | CaffeNet     | TBD                    | 82.0%    |
| [Multi-grained](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Multiple_Granularity_Descriptors_ICCV_2015_paper.pdf) | ICCV 2015     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                                                              | WordNet etc.          | VGG-19       | 224x224                | 83.0%    |
| [TS](https://people.eecs.berkeley.edu/~yg/papers/compact_bilinear.pdf) | CVPR 2016     |                                                              |                                                              |                       | VGGD+VGGM    | 448x448                | 84.0%    |
| [Bilinear CNN](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf) | ICCV 2015     |                                                              |                                                              |                       | VGGD+VGGM    | 448x448                | 84.1%    |
| [STN](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf) | NeurIPS 2015  |                                                              |                                                              |                       | GoogLeNet+BN | 448x448                | 84.1%    |
| [LRBP](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kong_Low-Rank_Bilinear_Pooling_CVPR_2017_paper.pdf) | CVPR 2017     |                                                              |                                                              |                       | VGG-16       | 224x224                | 84.2%    |
| [PDFS](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Picking_Deep_Filter_CVPR_2016_paper.pdf) | CVPR 2016     |                                                              |                                                              |                       | VGG-16       | TBD                    | 84.5%    |
| [Xu *et al.*](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xu_Augmenting_Strong_Supervision_ICCV_2015_paper.pdf) | ICCV 2015     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | Web data              | CaffeNet     | 224x224                | 84.6%    |
| [Cai *et al.*](http://azadproject.ir/wp-content/uploads/2014/07/2017-Higher-order-Integration-of-Hierarchical-Convolutional-Activations-for-Fine-grained.pdf) | ICCV 2017     |                                                              |                                                              |                       | VGG-16       | 448x448                | 85.3%    |
| [RA-CNN](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf) | CVPR 2017     |                                                              |                                                              |                       | VGG-19       | 448x448                | 85.3%    |
| [MaxEnt](https://papers.nips.cc/paper/7344-maximum-entropy-fine-grained-classification) | NeurIPS 2018  |                                                              |                                                              |                       | Bilinear CNN | TBD                    | 85.3%    |
| [PC](https://link.springer.com/chapter/10.1007/978-3-030-01258-8_5) | ECCV 2018     |                                                              |                                                              |                       | Bilinear CNN | TBD                    | 85.6%    |
| [CVL](http://openaccess.thecvf.com/content_cvpr_2017/papers/He_Fine-Grained_Image_Classification_CVPR_2017_paper.pdf) | CVPR 2017     |                                                              |                                                              | Texts                 | VGG          | TBD                    | 85.6%    |
| [Mask-CNN](http://www.weixiushen.com/publication/pr17.pdf)   | PR 2018       |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                       | VGG-16       | 448x448                | 85.7%    |
| [GP-256](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xing_Wei_Grassmann_Pooling_for_ECCV_2018_paper.pdf) | ECCV 2018     |                                                              |                                                              |                       | VGG-16       | 448x448                | 85.8%    |
| [KP](https://vision.cornell.edu/se3/wp-content/uploads/2017/04/cui2017cvpr.pdf) | CVPR 2017     |                                                              |                                                              |                       | VGG-16       | 224x224                | 86.2%    |
| [T-CNN](https://www.ijcai.org/proceedings/2018/0145.pdf)     | IJCAI 2018    |                                                              |                                                              |                       | ResNet       | 224x224                | 86.2%    |
| [MA-CNN](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Learning_Multi-Attention_Convolutional_ICCV_2017_paper.pdf) | ICCV 2017     |                                                              |                                                              |                       | VGG-19       | 448x448                | 86.5%    |
| [MaxEnt](https://papers.nips.cc/paper/7344-maximum-entropy-fine-grained-classification) | NeurIPS 2018  |                                                              |                                                              |                       | DenseNet-161 | TBD                    | 86.5%    |
| [DeepKSPD](http://openaccess.thecvf.com/content_ECCV_2018/papers/Melih_Engin_DeepKSPD_Learning_Kernel-matrix-based_ECCV_2018_paper.pdf) | ECCV 2018     |                                                              |                                                              |                       | VGG-19       | 448x448                | 86.5%    |
| [OSME+MAMC](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Sun_Multi-Attention_Multi-Class_Constraint_ECCV_2018_paper.pdf) | ECCV 2018     |                                                              |                                                              |                       | ResNet-101   | 448x448                | 86.5%    |
| [StackDRL](https://www.ijcai.org/proceedings/2018/0103.pdf)  | IJCAI 2018    |                                                              |                                                              |                       | VGG-19       | 224x224                | 86.6%    |
| [DFL-CNN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_a_Discriminative_CVPR_2018_paper.pdf) | CVPR 2018     |                                                              |                                                              |                       | VGG-16       | 448x448                | 86.7%    |
| [Bi-Modal PMA](https://ieeexplore.ieee.org/document/9103943) | IEEE TIP 2020 |                                                              |                                                              |                       | VGG-16       | 448x448                | 86.8%    |
| [PC](https://link.springer.com/chapter/10.1007/978-3-030-01258-8_5) | ECCV 2018     |                                                              |                                                              |                       | DenseNet-161 | TBD                    | 86.9%    |
| [KERL](https://www.ijcai.org/proceedings/2018/0087.pdf)      | IJCAI 2018    |                                                              |                                                              | Attributes            | VGG-16       | 224x224                | 87.0%    |
| [HBP](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.pdf) | ECCV 2018     |                                                              |                                                              |                       | VGG-16       | 448x448                | 87.1%    |
| [Mask-CNN](http://www.weixiushen.com/publication/pr17.pdf)   | PR 2018       |                                                              | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                       | ResNet-50    | 448x448                | 87.3%    |
| [DFL-CNN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_a_Discriminative_CVPR_2018_paper.pdf) | CVPR 2018     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.4%    |
| [NTS-Net](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.pdf) | ECCV 2018     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.5%    |
| [HSnet](http://web.engr.oregonstate.edu/~sinisa/talks/cvpr17_lstmsearch_oral.pdf) | CVPR 2017     | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) | ![surd](http://www.weixiushen.com/project/Awesome_FGIA/eqs/779537073705367497-130.png) |                       | GoogLeNet+BN | TBD                    | 87.5%    |
| [Bi-Modal PMA](https://ieeexplore.ieee.org/document/9103943) | IEEE TIP 2020 |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.5%    |
| [CIN](https://arxiv.org/pdf/2003.05235.pdf)                  | AAAI 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.5%    |
| [MetaFGNet](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yabin_Zhang_Fine-Grained_Visual_Categorization_ECCV_2018_paper.pdf) | ECCV 2018     |                                                              |                                                              | Auxiliary data        | ResNet-34    | TBD                    | 87.6%    |
| [Cross-X](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf) | CVPR 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.7%    |
| [DCL](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf) | CVPR 2019     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.8%    |
| [ACNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Attention_Convolutional_Binary_Neural_Tree_for_Fine-Grained_Visual_Categorization_CVPR_2020_paper.pdf) | CVPR 2020     |                                                              |                                                              |                       | VGG-16       | 448x448                | 87.8%    |
| [TASN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Looking_for_the_Devil_in_the_Details_Learning_Trilinear_Attention_CVPR_2019_paper.pdf) | CVPR 2019     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 87.9%    |
| [ACNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Attention_Convolutional_Binary_Neural_Tree_for_Fine-Grained_Visual_Categorization_CVPR_2020_paper.pdf) | CVPR 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 88.1%    |
| [CIN](https://arxiv.org/pdf/2003.05235.pdf)                  | AAAI 2020     |                                                              |                                                              |                       | ResNet-101   | 448x448                | 88.1%    |
| [DBTNet-101](http://papers.nips.cc/paper/8680-learning-deep-bilinear-transformation-for-fine-grained-image-representation.pdf) | NeurIPS 2019  |                                                              |                                                              |                       | ResNet-101   | 448x448                | 88.1%    |
| [Bi-Modal PMA](https://ieeexplore.ieee.org/document/9103943) | IEEE TIP 2020 |                                                              |                                                              | Texts                 | VGG-16       | 448x448                | 88.2%    |
| [GCL](https://aaai.org/ojs/index.php/AAAI/article/view/6912) | AAAI 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 88.3%    |
| [S3N](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf) | CVPR 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 88.5%    |
| [Sun *et al*.](https://arxiv.org/pdf/1912.06842v1.pdf)       | AAAI 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 88.6%    |
| [FDL](https://www.aiide.org/ojs/index.php/AAAI/article/view/6822) | AAAI 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 88.6%    |
| [Bi-Modal PMA](https://ieeexplore.ieee.org/document/9103943) | IEEE TIP 2020 |                                                              |                                                              | Texts                 | ResNet-50    | 448x448                | 88.7%    |
| [DF-GMM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Weakly_Supervised_Fine-Grained_Image_Classification_via_Guassian_Mixture_Model_Oriented_CVPR_2020_paper.pdf) | CVPR 2020     |                                                              |                                                              |                       | ResNet-50    | 448x448                | 88.8%    |
| [PMG](https://arxiv.org/pdf/2003.03836.pdf)                  | ECCV 2020     |                                                              |                                                              |                       | VGG-16       | 550x550                | 88.8%    |
| [FDL](https://www.aiide.org/ojs/index.php/AAAI/article/view/6822) | AAAI 2020     |                                                              |                                                              |                       | DenseNet-161 | 448x448                | 89.1%    |
| [PMG](https://arxiv.org/pdf/2003.03836.pdf)                  | ECCV 2020     |                                                              |                                                              |                       | ResNet-50    | 550x550                | 89.6%    |
| [API-Net](https://arxiv.org/pdf/2002.10191.pdf)              | AAAI 2020     |                                                              |                                                              |                       | DenseNet-161 | 512x512                | 90.0%    |
| [Ge *et al.*](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_Weakly_Supervised_Complementary_Parts_Models_for_Fine-Grained_Image_Classification_From_CVPR_2019_paper.pdf) | CVPR 2019     |                                                              |                                                              |                       | GoogLeNet+BN | Shorter side is 800 px | 90.4%    |