---
layout:     post
title:      Imbalance Problems in Object Detection
subtitle:   目标检测中不均衡问题总结
date:       2020-07-04
author:     WangPeng
header-img: img/post-bg.jpg
catalog: true
tags:
    - Overview
---

K. Oksuz, B. C. Cam, S. Kalkan, E. Akbas, "Imbalance Problems in Object Detection: A Review", (under review), 2019.[[preprint]](https://arxiv.org/abs/1909.00169)



### Table of Contents 

1. [Class Imbalance](#1)  
    1.1 [Foreground-Background Class Imbalance](#1.1)  
    1.2 [Foreground-Foreground Class Imbalance](#1.2)    
2. [Scale Imbalance](#2)  
    2.1 [Object/box-level Scale Imbalance](#2.1)  
    2.2 [Feature-level Imbalance](#2.2)    
3. [Spatial Imbalance](#3)  
    3.1 [Imbalance in Regression Loss](#3.1)  
    3.2 [IoU Distribution Imbalance](#3.2)  
    3.3 [Object Location Imbalance](#3.3)  
4. [Objective Imbalance](#4)


![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/imbalance-review.png)
#### 1. Class Imbalance （类别不均衡）
##### 1.1. Foreground-Background Class Imbalance （前景-背景类别不均衡）
> 定义：前景-背景类别不均衡是目标检测中研究最广泛，程度最深的一类不平衡。这种不平衡并不是由于数据集引起，而是由于现有目标检测架构引起（要生成大量的box，在此基础上实现分类和回归任务），因此background boxes 远远多于 foreground boxes。前景-背景不平衡问题发生在训练过程中，不依赖于数据集中各个类的数量。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/class-imbalance1.png)

- Hard Sampling Methods（硬采样）
- 硬采样从给定的标记box集合中选择一个子集的正反示例（具有所需数量）,忽略未选择的示例。因此，选中的每个样本对损失的贡献相等，而未选择的样本对当前迭代的训练没有贡献。
   - Random Sampling（随机抽样） 
   - 随机抽样，用于R-CNN系列检测器，其中，为了训练RPN，随机（在所有正示例中）均匀地采样128个正样本，并以类似的方式对128个负锚定点进行采样；再从各自的集合中随机抽取，用于训练检测网络。
   - Hard Example Mining(硬示例挖掘方法)
     使用硬示例（即具有高损耗的示例）训练检测器以获得更好的性能，利用一个子集的负样本训练初始模型，然后利用分类器失效的负样本（即硬样本），训练一个新的分类器。通过迭代应用同一过程得到多个分类器。

     - Bootstrapping, NeurIPS 1996, [[paper]](https://papers.nips.cc/paper/1168-human-face-detection-in-visual-scenes.pdf) 

     - SSD, ECCV 2016, [[paper]](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)
     第一个在训练中使用硬示例的深度目标探测器是SSD，只选择产生最大损失值的负示例。

     - Online Hard Example Mining, CVPR 2016, [[paper]](https://zpascal.net/cvpr2016/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)
     提出了一种更系统的方法来考虑正样本和负样本的损失值，然而，OHEM需要额外的记忆，导致训练速度下降

     - IoU-based Sampling, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.pdf)
     提出将示例的硬度与其IoU相关联，并再次对负示例使用抽样方法，而不是计算整个集合的损失函数，将负样本的IoU区间划分为K个区间，在每个区间内随机抽取相等数量的负样本，以提升IoU较高的样本，期望这些样本具有更高的损失值。
     - Overlap Sampler, WACV 2020, [[paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Chen_Overlap_Sampler_for_Region-Based_Object_Detection_WACV_2020_paper.pdf)
   - 
   Limit Search Space（限制搜索空间）
   限制搜索空间，以使难挖掘的示例更易于挖掘 
     - Two-stage Object Detectors 
     找到给定锚定的最可能的边界框（即ROI），然后选择目标得分最高的前N个ROI，并对其应用额外的抽样方法。
     - IoU-lower Bound, ICCV 2015, [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
     设置负ROI的IoU下限为0.1而不是0，以提高硬性负样本，然后应用随机抽样
- Objectness Prior, CVPR 2017, [[paper]](http://zpascal.net/cvpr2017/Kong_RON_Reverse_Connection_CVPR_2017_paper.pdf)
     提出了一种在端到端设置中学习对象优先性的方法，以指导在哪里搜索对象。在训练过程中，使用了目标先验大于阈值的所有正例子，而选择了部分负例子使得正类和负类之间保持了期望的平衡（即1:3）
     - Negative Anchor Filtering, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf)
提出在一阶段检测器中用锚精化模块确定锚的置信度，并再次采用阈值来消除容易出现的负锚进行过滤
     - Objectness Module, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nie_Enriched_Feature_Guided_Refinement_Network_for_Object_Detection_ICCV_2019_paper.pdf)
     在SSD算法模型中使用了级联检测模块（在每个预测模块之前有一个目标模块）。这些目标模块是二元分类器，用来过滤掉容易出现的锚。

     


- Soft Sampling Methods（软采样法）
 软采样根据每个样本对训练过程的相对重要性来调整其贡献。这种方法与硬采样不同，不丢弃样本，而是利用整个数据集更新参数。

   - Focal Loss, ICCV 2017, [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
   Focal Loss动态地为硬样本分配更多的权重，增加了两个超参数，改进了交叉熵损失
- Gradient Harmonizing Mechanism, AAAI 2019, [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/4877)
   梯度协调机制（GHM）抑制来自简单正样片和负样片的梯度。作者首先观察到小梯度范数的样本太多，中等梯度范数的样本数量有限，而大梯度范数的样本数量却相当多。GHM是一种基于计数的方法，它计算具有类似梯度范数的样本数，并在存在许多具有类似梯度的样本时对样本损失进行惩罚
   - Prime Sample Attention, arXiv 2019, [[paper]](https://arxiv.org/pdf/1904.04821.pdf)
PrIme-Sample-Attention（PISA）根据不同的标准为正、负样本分配权重。当iou值较高的阳性样本被看好时，前景分类得分较大的负片被提升。
   
   

- Sampling-Free Methods（免取样法）
 为了避免前面提到的手工抽样方法，减少训练过程中的超参数数量，出现了一些替代方法
   - Is Sampling Heuristics Necessary in Training Deep Object Detectors?, arXiv 2019, [[paper]](https://arxiv.org/pdf/1909.04868.pdf)
- Residual Objectness for Imbalance Reduction, arXiv 2019, [[paper]](https://arxiv.org/pdf/1908.09075.pdf)   
   在检测网络中添加了一个目标分支，以预测剩余目标得分。虽然这个新的分支处理前景-背景不平衡，但分类分支只处理正类。在推理过程中，分类分数由分类结果和目标分支输出相乘得到提高了性能。
   - AP Loss, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers
   Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.pdf)
 直接建立最终性能指标的模型，并在此基础上进行实例加权。将损耗的分类部分作为一个排序任务并使用平均精度（AP）作为该任务的损失函数。
   - DR Loss, arXiv 2019, [[paper]](https://arxiv.org/pdf/1907.10156.pdf)
  DR Loss，该方法使用基于Hinge loss的分类损耗定义方法
   
   

- Generative Methods（生成方法）
 直接生成人工样本并将其注入训练数据集来解决不平衡问题。

   - Adversarial Faster-RCNN, CVPR 2017, [[paper]](http://zpascal.net/cvpr2017/Wang_A-Fast-RCNN_Hard_Positive_CVPR_2017_paper.pdf) 
   一种方法是使用生成性对抗网络（generative atterial networks，GANs）。GANs的一个优点是，在训练过程中，由于这些网络的损失值直接取决于最终检测中生成的样本的分类精度，因此它们能够在训练过程中自动生成较难的样本。对抗性的快速RCNN模型，该模型生成具有遮挡和各种变形的硬示例。提出了两种网络：（i）用于生成遮挡特征图的对抗性空间丢失网络，以及（ii）用于变形（变换）特征图生成的对抗性空间变换器网络。这两个网络在网络设计中按顺序排列，以提供更难的示例，并以端到端的方式集成到传统的对象训练网络中。

   - Task Aware Data Synthesis, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tripathi_Learning_to_Generate_Synthetic_Data_via_Compositing_CVPR_2019_paper.pdf)
   使用GANs生成图像，而不是复制现有对象，使用三个相互竞争的网络生成硬示例：合成器、鉴别器和目标网络，其中合成器通过产生高质量的合成硬图像。给定一个图像和一个前景对象遮罩，合成器的目标是将前景对象遮罩放置在图像上，以产生逼真的硬示例。为了增强合成器对真实合成图像的处理能力，采用了鉴别器。

   - PSIS, arXiv 2019, [[paper]](https://arxiv.org/pdf/1906.00358.pdf) 
   在一对图像之间交换属于同一类的单个对象，同时考虑候选实例的比例和形状。通过交换低性能类的对象来生成图像可以提高检测质量。因此，在确定要交换的对象和要生成的图像的数量时，它们使用类的性能排名。

   - pRoI Generator, WACV 2020, [[paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Oksuz_Generating_Positive_Bounding_Boxes_for_Balanced_Training_of_Object_Detectors_WACV_2020_paper.pdf)
   与生成图像不同，正RoI（pRoI）生成器生成一组具有给定IoU、BB相对空间和前景类分布的正RoI。该方法依赖于一个边界盒生成器，该生成器能够使用给定的边界框（即基本真实）生成具有所需IoU的边界框（即正例）。注意到输入BB的IoU与其硬度有关，pRoI生成器是模拟并分析硬采样方法的基础。
![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/class-imbalance2.png)
##### 1.2. Foreground-Foreground Class Imbalance（前景-前景类别不均衡） 
> 定义：在前景类不平衡中，过度表示类和欠表示类都是前景类。前景类之间的这种不平衡并没有像背景-背景不平衡那样引起人们的兴趣。数据集中对象类之间自然存在不平衡，训练的每个批次也存在不均衡，过度拟合有利于过度表示的类可能是不可避免的。

   - Fine-tuning Long Tail Distribution for Obj.Det., CVPR 2016, [[paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.pdf)

   使用基于视觉相似性的聚类，确定了两个影响训练的因素：（i）预测的准确性；（ii）实例的数量。基于这一观察结果，他们根据预先训练的主干网（即GoogLe Net[101]）最后一层特征的内积，手工构建了类之间的相似性度量，并对类进行了分层分组，以补偿数据集级前景类的不平衡。

   - PSIS, arXiv 2019, [[paper]](https://arxiv.org/pdf/1906.00358.pdf)

   - OFB Sampling, WACV 2020, [[paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Oksuz_Generating_Positive_Bounding_Boxes_for_Balanced_Training_of_Object_Detectors_WACV_2020_paper.pdf)
      提出OFB抽样法，通过为每个待采样的边界框分配概率，可以在批处理级别上缓解前景类不平衡问题，从而使批内不同类的分布均匀。换言之，该方法的目的是在抽样过程中，促进样本数量较少的类。虽然该方法是有效的，但性能改进并不显著。



#### 2. Scale Imbalance（尺度不平衡）

##### 2.1. Object/box-level Scale Imbalance（对象/边界框等级不平衡）
> 定义：当数据集中的对象或输入边界框的某些大小被过度表示时，就会出现尺度不平衡，会影响估计ROI的尺度和整体检测性能。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/scale-imbalance1.png)

- Methods Predicting from the Feature Hierarchy of Backbone Features（基于主干特征层次的预测方法）
根据主干网不同层次的特征进行独立预测。考虑了多个尺度上的目标检测，因为不同的层次在不同的尺度上有不同的编码信息

  - Scale-dependent Pooling, CVPR 2016, [[paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Exploit_All_the_CVPR_2016_paper.pdf)
  - SSD, ECCV 2016, [[paper]](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)
  根据不同层的特征进行预测
- Multi Scale CNN, ECCV 2016, [[paper]](https://arxiv.org/pdf/1607.07155.pdf)
  在第一阶段估计区域时使用不同层次的主干网
  - Scale Aware Fast R-CNN, IEEE Transactions on Multimedia, 2018 [[paper]](https://ieeexplore.ieee.org/document/8060595)
根据估计的RoI的规模选择一个合适的层进行池化；称为规模相关池（SDP），如果RoI的高度很小，则汇集来自较早层的特征。
  
  

- Methods Based on Feature Pyramids（基于特征金字塔的方法）
  独立地使用不同层次的特征，而不需要整合低层和高层特征。

  - FPN, CVPR 2017, [[paper]](https://zpascal.net/cvpr2017/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)
  在进行预测之前将不同尺度的特征结合起来,利用了一个额外的自上而下的途径，通过横向连接，来自较高层次的特征由较低层次的特征支持，从而使这些特征得到均衡的混合。自上而下的路径包括上采样，以确保尺寸兼容，横向连接基本上是1×1卷积。与特性层次结构类似，RoI池化步骤考虑RoI的规模，以选择从哪个级别合并。这些改进使得预测器网络可以应用于各个层次，从而提高了性能，特别是对于中小型对象。
  - See feature-level imbalance methods

  

- Methods Based on Image Pyramids（基于图像金字塔的方法）

  - SNIP, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Singh_An_Analysis_of_CVPR_2018_paper.pdf)
  SNIP用不同大小的图像来训练多个建议和检测器网络，但是对于每个网络，只有适当的输入边界盒尺度被标记为有效的，从而保证了多尺度训练而不丢失数据。
  - SNIPER, NeurIPS 2018, [[paper]](https://papers.nips.cc/paper/8143-sniper-efficient-multi-scale-training.pdf)
  GPU内存的限制也是一项重要的挑战，提出了一种新的图像裁剪方法是克服了图像裁剪SNIPER

  

- Methods Combining Image and Feature Pyramids（结合图像金字塔与特征金字塔的方法）
基于图像金字塔的方法在计算时间和内存方面通常不如基于特征金字塔的方法。然而，基于图像金字塔的方法被期望表现得更好，因为基于特征金字塔的方法是这些方法的有效近似。所以，为了从这两种方法的优点中获益，可以将它们组合在一个模型中。
  - Efficient Featurized Image Pyramids, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Efficient_Featurized_Image_Pyramid_Network_for_Single_Shot_Detector_CVPR_2019_paper.pdf)
  使用五幅不同比例的图像，其中四幅被提供给轻量级的特征图像金字塔网络模块（代替额外的主干网），原始输入被输入到主干网。这种轻量级网络由4个连续的卷积层组成，专门为每个输入设计。从该模块中收集到的特征根据其大小被集成到适当级别的主干网特征中，从而使用特征注意模块将图像特征与从主干网提取的特征相结合。此外，在注意模块之后，通过前向融合模块将收集到的特征与高层进行融合，得到最终的预测结果。

  - Enriched Feature Guided Refinement Network, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nie_Enriched_Feature_Guided_Refinement_Network_for_Object_Detection_ICCV_2019_paper.pdf) 
  同样建立在SSD上并使用下采样图像的类似方法是丰富的特征引导优化网络，其中单个降采样图像是作为多尺度上下文功能（MSCF）模块的输入。MSCF由两个连续的卷积层和三个平行的扩张卷积层组成，这也与三阶段网络中的思想相似。使用1×1卷积再次组合扩张卷积的输出。作者设置了降采样图像大小以满足常规SSD架构中的第一预测层（例如，对于320×320输入，下采样图像的大小为40×40）。

  - Super-Resolution for Small Objects, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Noh_Better_to_Follow_Follow_to_Be_Better_Towards_Precise_Supervision_ICCV_2019_paper.pdf)
  它们的体系结构在基础网络检测器上增加了四个模块：（i）给定原始图像，目标提取器通过扩张卷积将目标输出给鉴别器。该网络还与主干网共享参数。（ii）给定使用原始图像和0.5倍下采样图像从主干网获得的特征，生成器网络为小roi生成超分辨率特征映射。（iii）给定（i）和（ii）的输出，在常规GAN设置中训练鉴别器。（iv）最后，如果RoI是一个阈值范围内的小对象，则由小预测器或大预测器进行预测。

  - Scale Aware Trident Network, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.pdf)
  结合了基于特征金字塔和图像金字塔的方法的优点，不使用多个下采样图像，而只使用扩张卷积。作者使用扩张卷积，在并行分支中使用扩张率为1、2和3的卷积来生成特定比例的特征映射，与基于特征金字塔的方法相比，该方法更加精确。为了确保每个分支都针对特定的比例专门化，将根据分支的大小为相应的分支提供一个输入边界框。他们分析了感受野大小对不同尺度物体的影响，结果表明，更大的扩张率更适合于尺度更大的物体。此外，由于使用多个分支会由于操作数量的增加而降低效率，因此他们提出了一种用单个参数共享分支来近似这些分支的方法，并且性能损失最小（微不足道）。



##### 2.2. Feature-level Imbalance 

> 定义：来自主干网的特性的集成预计将在低级和高级特性方面达到平衡，以便能够进行一致的预测，高层次和低层次特征的影响是不同的。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/scale-imbalance2.png)
- Methods Using Pyramidal Features as a Basis（基于金字塔特征的方法）
这些方法的目的是通过使用额外的操作或步骤来改进FPN收集的金字塔特征
  
  - PANet, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Path_Aggregation_Network_CVPR_2018_paper.pdf)
FPN收集的特征可以进一步增强，RoI可以映射到金字塔的每一层，而不是将其与单个层相关联。作者认为低层次的特征，如边缘、角点，对于定位对象是有用的，但是FPN体系结构并没有充分利用这些特征。自下而上的路径扩充扩展了特征金字塔，以允许低层特征以较短的步骤到达预测发生的层。因此，在某种程度上为初始图层中的要素创建了快捷方式。这一点很重要，因为这些特性由于边缘或实例部分而具有丰富的本地化信息。2） 而在FPN中，每个RoI根据其大小与单个级别的特征相关联，PANet将每个RoI关联到每个级别，应用RoI池，使用元素级max或sum运算进行融合，得到的固定大小的特征网格被传播到检测器网络。这个过程称为自适应特性池。
  - Libra FPN, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.pdf)
  通过一次使用所有FPN层的所有特征来学习剩余特征，步骤如下：1）集成：通过重新缩放和平均化，将来自不同层的所有特征映射简化为一个单一的特征映射。因此，此步骤没有任何可学习参数。2） 细化：通过卷积层或非局部神经网络对综合特征图进行细化。
  
  


- Methods Using Backbone Features as a Basis（基于主干特征的方法）
  建立在主干特征上的体系结构，通过采用不同的特征集成机制忽略了FPN自顶向下的路径

  - STDN, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1376.pdf)
  利用DenseNet的最后一个块构建了由六层组成的金字塔特征。为了将这些层映射到较低的大小，该方法使用不同感受野大小的平均池。对于第四个特征映射，使用标识映射。对于DenseNet特征映射到更高维的最后两层，作者提出了一种尺度转换层方法。该层没有任何可学习的参数，并且给定r，通过减少特征图的总数（即信道），特征图的宽度和高度被r放大。STDN在DenseNet块的帮助下结合了高层和底层特性，不容易适应其他主干网。此外，在DenseNet的最后一个块中，没有采用任何方法来平衡低层和高层特征。
- Parallel-FPN, ECCV 2018, [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seung-Wook_Kim_Parallel_Feature_Pyramid_ECCV_2018_paper.pdf)
  使用主干网的最后一层，并通过利用空间金字塔池（SPP）生成多尺度特征。不同的是，它通过将主干网的最后D个特征映射以不同的大小进行多次合并，从而得到不同比例尺的特征图，从而增加了网络的宽度。将其合并三次且D=2时的情况。采用1×1卷积将特征映射数减少到1个。然后，这些特征映射被输入多尺度上下文聚合（MSCA）模块，该模块集成来自其他尺度的对应层的上下文信息。因此，基于比例尺的MSCA具有以下输入：空间金字塔集合的D特征地图和来自其他尺度的简化特征图。
  - Deep Feature Pyramid Reconfiguration, ECCV 2018, [[paper]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Tao_Kong_Deep_Feature_Pyramid_ECCV_2018_paper.pdf)
将不同层次的主干特征转化为单个张量，然后从该张量中学习一组剩余特征。将两个模块的序列应用于张量X，以学习添加到主干网每层的剩余特征映射。这些模块是：1）全局注意模块的目的是学习张量X的不同特征映射之间的相互依赖关系，作者采用压缩和激励块，其中每个特征图的信息最初被压缩到低维特征（即压缩步骤），然后基于包含非线性（即激励步长）的可学习函数为每个特征映射学习一个权值。2）局部配置模块通过卷积层来改善全局注意模块后的特征。该模块的输出显示了从主干网为特征层添加的剩余特征。
  - Zoom Out-and-In, IJCV 2019, [[paper]](https://arxiv.org/pdf/1709.04347.pdf)
  结合了主干网的低层和高层特征。此外，它还包括基于反卷积的放大阶段，在该阶段中，学习中间步骤金字塔特征。注意，与FPN不同，放大阶段没有与主干网的横向连接，放大阶段基本上是一系列反卷积层（放大阶段见红色箭头）。通过在放大阶段的缩小和放大阶段叠加相同尺寸的特征图，实现高、低层次特征的集成。另一方面，提出的地图注意力决策模块学习各层的权重分布。
- Multi-level FPN, AAAI 2019, [[paper]](https://arxiv.org/pdf/1811.04533.pdf)
  堆叠一个最高和一个较低级别的特征层，递归地输出一组金字塔特征，这些特征最终以比例方式组合成单个特征金字塔。
  - NAS-FPN, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghiasi_NAS-FPN_Learning_Scalable_Feature_Pyramid_Architecture_for_Object_Detection_CVPR_2019_paper.pdf)
不是使用手工构建的体系结构，而是通过使用神经体系结构搜索方法来搜索生成金字塔特征的最佳体系结构。这一思想以前也被应用到图像分类任务中，并且表现出良好的性能
  - Auto-FPN, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Auto-FPN_Automatic_Network_Architecture_Adaptation_for_Object_Detection_Beyond_Classification_ICCV_2019_paper.pdf)
  Auto-FPN也是另一个使用NAS的例子，同时学习从主干特性到金字塔特性以及其他方面的连接。虽然NAS-FPN实现了更高的性能，但Auto-FPN更高效，内存占用更少，这一思想也被应用到目标检测的主干网设计中。

  


#### 3. Spatial Imbalance （空间不均衡）

##### 3.1. Imbalance in Regression Loss（回归损失不均衡） 
> 定义：这种不平衡问题与不同个体对回归损失的贡献不均有关。如L1和L2，iou loss等。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/spatial-imbalance1.png)

- Lp norm based
  - Smooth L1, ICCV 2015, [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
  Smooth L1损失是第一个专门为深部目标探测器设计的损失函数，由于它减少了异常值的影响（与L2损失相比），并且对于小误差（与L1损失相比）更稳定，因此被广泛采用.

  - Balanced L1, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/
  Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.html)
  由于异常值的梯度对平滑L1损失中梯度较小的样本的学习仍有负面影响，Balanced L1损失增加了样本在总梯度中的贡献损失价值。

  - KL Loss, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.pdf)
  在某些情况下，由于遮挡、物体形状或不准确的标记，地面真值盒可能是模糊的。因此，作者的目的是预测每个BB坐标的概率分布，而不是直接BB预测。这种思想类似于除了分类和回归之外，还具有额外的局部化置信度相关预测分支的网络，以便在推理过程中使用预测的置信度。不同的是，KL-Loss，即使没有提出的NMS，与基线相比在定位上也有改进。该方法假设每个盒坐标是独立的，并且遵循平均值为ˆx和标准差σ的高斯分布。因此，除了传统的盒子之外，网络中还添加了一个分支来预测标准偏差，即σ，并且使用预测值和地面真实值之间的KL散度来反向传播损失，从而使真实框由以方框坐标为中心建模。

  - Gradient Harmonizing Mechanism, AAAI 2019, [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/4877/4750)

- IoU based
  - IoU Loss, ACM IMM 2016, [[paper]](https://arxiv.org/pdf/1608.01471.pdf)
    利用IoU的可微性。一个早期的例子是IoU损失
  
    > L_{iou}=-In(IoU)$
  
  - Bounded IoU Loss, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0794.pdf)
    利用1-IoU度量性质的另一种方法是有界IoU损失。该损失将1-IoU的修改版本扭曲为平滑L1函数。修改包括通过固定除要计算的参数以外的所有参数来限定IoU，这意味着计算一个参数的最大可达IoU
  
    > $L_{iou}=1-IoU$
  
- Generalized IoU Loss, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf)
    基于最佳损失函数是性能度量本身的思想，在广义交大于并（GIoU）中，表明IoU可以直接优化，IoU和提出的GIoU可以用作损失函数。在修正IoU的主要缺点（即IoU=0时的瓶颈）时，GIoU被提议作为性能度量和损失函数。以这种方式，即使两个框不重叠，一个GIoU值可以分配给它们，这使得函数在整个输入域中具有非零梯度，而不是被限制在IoU>0。与IoU不同，GIoU∈[−1,1]。GIoU是IoU的下界，GIoU保留了IoU的优点，并使其在IoU=0时可微。另一方面，由于正标记BBs的IoU大于0.5，因此这部分函数在实际中从未访问过，但GIoU Loss比直接使用IoU作为损失函数要好。

    > $L_{Giou}(B,\overline{B})=IoU(B,\overline{B})-\frac{A(E)-A(B,\overline{B})}{A(E)}$
  
E为包含$B$和$\overline{B}$区域的最小框
  
  - Distance IoU Loss, AAAI 2020, [[paper]](https://arxiv.org/pdf/1911.08287.pdf)
  在传统的IoU误差（即1−IoU（B，−B））上添加惩罚项，以确保更快、更准确地收敛。为了达到这个目的，在距离IoU（DIoU）损失中，一个与B的中心距离有关的惩罚项
  - Complete IoU Loss, AAAI 2020, [[paper]](https://arxiv.org/pdf/1911.08287.pdf)
    在DIoU loss基础上扩展了一个额外的惩罚项，因为两个盒子的宽高比不一致。由此产生的损失函数，称为Complete IoU（CIoU）
  
  

##### 3.2. IoU Distribution Imbalance （IoU分布不均衡）
> 定义： bounding boxes 在 IoU 段的分布上呈现出明显不均匀的分布。

- Cascade R-CNN, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)
 第一个解决IoU不平衡的方法。基于（i）单个检测器对单个IoU阈值是最优的，以及（ii）IoU分布的倾斜使回归器超过单个阈值的论点，他们表明正样本的IoU分布对回归分支有影响。为了缓解这个问题，作者训练了三个检测器，对于阳性样本，IoU阈值分别为0.5、0.6和0.7。级联中的每个检测器使用前一级的盒子，而不是重新取样。通过这种方式，他们证明了分布的偏斜可以从左偏移到近似均匀甚至右偏，从而使模型有足够的样本来训练它的最优IoU阈值。作者指出，这种级联方案比先前的工作（如多区域CNN和AttractioNet更好地工作，后者迭代地将同一网络应用于边界框。

- Hierarchical Shot Detector, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Hierarchical_Shot_Detector_ICCV_2019_paper.pdf)
不是在不同的级联级别上使用分类器和回归器，而是在框回归后运行其分类器，从而获得更均衡的分布。

- IoU-uniform R-CNN, arXiv 2019, [[paper]](https://arxiv.org/pdf/1912.05190.pdf)
随机生成的边界框被用来为Faster R-CNN的第二阶段提供一组具有均衡IoU分布的正输入边界框。IoU uniform R-CNN增加了可控的变化，并且以这种方式仅向回归器提供近似一致的正输入边界框（即分类分支仍然使用RPN roi）。

- pRoI Generator, WACV 2020, [[paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Oksuz_Generating_Positive_Bounding_Boxes_for_Balanced_Training_of_Object_Detectors_WACV_2020_paper.pdf)
不同的是，pRoI-Generator使用生成的roi训练两个分支，但是性能的提高并不是那么显著，这可能是因为训练集覆盖的空间比测试集大得多。使用提议的边界框生成器系统地生成边界框。利用这个pRoI生成器，他们对不同的IoU分布进行了一系列实验，结果表明：（i）输入边界框的IoU分布不仅影响回归，而且影响分类性能。（ii）IoU与其硬度有关，表明OHEM的效果取决于正输入BBs的IoU分布。当一个右倾斜的IoU分布与OHEM一起使用时，可以观察到显著的性能改进。（iii）当IoU分布均匀时，表现最佳。


##### 3.3. Object Location Imbalance (物体位置不平衡)
> 定义： 目标在图像中的分布很重要，因为当前的深部目标探测器使用密集采样的锚作为滑动窗口分类器。对于大多数方法，锚定点均匀分布在图像中，因此图像中的每个部分都被认为具有相同的重要性级别。另一方面，图像中的对象不遵循均匀分布，即对象位置存在不平衡。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/spatial-imbalance2.png)
- Guided Anchoring, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Region_Proposal_by_Guided_Anchoring_CVPR_2019_paper.html)
学习锚定的位置、尺度和纵横比属性，以减少锚的数量，同时提高召回率。具体地说，在给定主干特征图的情况下，为每一个任务设计一个预测分支来生成锚：（i）锚位置预测分支预测一个每一个位置判断该位置是否包含目标，并根据输出概率采用硬阈值方法确定锚定点，（ii）锚定形状预测分支生成每个位置的锚形状。由于锚因图像而异，不同于在特征映射上使用完全卷积分类器的传统方法（即单级生成器和RPN），作者提出了基于可变形卷积的锚引导特征自适应，以便根据锚定尺寸获得平衡的表示。

- FreeAnchor, NeurIPS 2019, [[paper]](https://papers.nips.cc/paper/8309-freeanchor-learning-to-match-anchors-for-visual-object-detection.pdf)
没有学习锚定点，而是放松了匹配策略的硬约束（即，如果IoU>0.5，样本为正），这样，每个锚都被视为每个真实框的匹配候选者。为了做到这一点，作者使用与真实框的锚排序的iou为每个真实框挑选一系列候选锚。在这些候选样本中，所提出的损失函数旨在通过考虑回归和分类任务与地面真实情况相匹配来实施最合适的锚定。

#### 4. Objective Imbalance （多任务损失优化之间的不平衡）
> 定义： 目标不平衡是指在训练过程中最小化的目标（损失）功能。根据定义，目标检测需要一个多任务丢失，以便同时解决分类和回归任务。但是，不同的任务会导致不平衡，这是由于以下原因：（i）任务的梯度规范可能不同，一个任务可以主导训练。（二）不同任务的损失函数取值范围可能不同，这会影响任务的一致性和均衡性优化。（iii）任务的难度可能不同，这会影响学习任务的速度，从而阻碍培训过程。

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/2020-07-04-imbalance-problems-in-object-detection/object-imbalance1.png)

- Task Weighting
通过一个额外的超参数作为权重因子来平衡损失项。超参数是使用验证集选择的。当然，增加任务的数量，就像两级检测器的情况一样，会增加加权因子的数量和搜索空间的维数（请注意，两级检测器中有四个任务，一级检测器中有两个任务）。

- Classification Aware Regression Loss, arXiv 2019, [[paper]](https://arxiv.org/pdf/1904.04821.pdf)
将分类和回归任务结合起来的一个更突出的方法是分类感知回归损失（CARL），它假设分类和回归任务是相关的。为了合并损失项，回归损失通过边界框的（分类）置信分数确定的系数进行缩放。

- Guided Loss, arXiv 2019, [[paper]](https://arxiv.org/pdf/1909.04868.pdf)
交叉熵引起的累积损失需要动态加权，因为当使用交叉熵损失时，每个时期单个损失分量的贡献率可能不同。为了防止这种不平衡，作者提出了Guided Loss，该方法通过考虑损失的总大小来加权分类分量
