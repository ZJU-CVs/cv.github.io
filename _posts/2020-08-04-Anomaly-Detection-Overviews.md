---
layout:     post
title:      Deep Learning for Anomaly Detection 
subtitle:   A Review
date:       2020-08-04
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Overview
    - Anomaly Detection
    - Upgrade
---



### Anomaly Detection

#### Problem

- Unknownness
- Heterogeneous anomaly classes

- Rartiy and class imbalance
- Diverse types of anomaly
  - Point anomalies
  - Conditional anomalies
  - Group anomalies



#### Challenges

- Low anomaly detection recall rate
  - Many normal instances are wrongly reported as anomalies while true yet sophisticated anomalies are missed.
  - The sota methods, especially unsupervised methods often incur high false positives on real-world datasets.

- Anomaly detection in high-dimensional and/or not independent data
  - Anomalies often exhibit evident abnormal characteristics in a low-dimensional space **yet become hidden and unnoticeable in a high-dimensional space**
  - However, identifying intricate feature interactions and couplings may be essential in high-dimensional data.
  - Due to the unknowns and heterogeneities of anomalies, it is challenging to guarantee the new feature space preserved proper information for specific detection methods is critical to downstream accurate anomaly detection.
  - It is challenging to detect anomalies from instances that may be dependent on each other.

- Data-efficient learning of normality/abnormality
  - Two major challenges are **how to learn expressive normality/abnormality representations with a small amount of labeled anomaly data** and **how to learning detection models that are generalized to novel anomalies uncovered by the given labeled anomaly data**.
- Noise-resilient anomaly detection
  - The main challenge is that the amount of noises can differ significantly from datasets and noisy instances may be irregularly distributed in the data space.
- Detection of complex anomalies
  - Current methods mainly focus on detect anomalies from single data source, while many applications require the detection of anomalies with multiple heterogeneous data sources.
  - Some anomalies can be detected only when considering two or more data sources.

- Anomaly explanation
  - In many critical domains, there may be some major risks if anomaly detection models are directly used as black-box models.
  - Most existing anomaly detection studies focus on devising accurate detection models only, ignoring the capability of providing explanation of the identified anomalies.
  - A main challenge to well balance the model's interpretability and effectiveness.



#### Methods

![img](https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/AD.png)

##### Deep learning for feature extraction

> It aims at leveraging deep learning to extract low-dimensional feature representations from high-dimensional and/or non-linearly separable data for downstream anomaly detection.
>
> - The feature extraction and the anomaly scoring are fully disjointed and independent from each other.
> - The deep learning components work purely as dimensionality reduction only.
>
> **Advantages:**
>
> - The SOTA deep models offer more powerful dimensionality reduction.
>
> **Disadvantages:**
>
> - Pre-trained deep models are typically limited to specific types of data.
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/AD1.png" alt="img" style="zoom:50%;" />

##### Learning feature representations of normality 

> - Generic Normality Feature Learning
>   - AutoEncoders
>   - Generative Adversarial Networks
>   - Predictability Modeling
>   - Self-supervised Classification
> - Anomaly Measure-dependent Feature Learning
>   - Distance-based Measure
>   - One-class Classification-based Measure
>   - Clustering-based Measure
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/AD2.png" alt="img" style="zoom:50%;" />

##### End-to-end anomaly score learning

> - Ranking Models
> - Prior-driven Models
> - Softmax Models
> - End-to-end One-class Classification
>
> <img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/notes/AD3.png" alt="img" style="zoom:50%;" />