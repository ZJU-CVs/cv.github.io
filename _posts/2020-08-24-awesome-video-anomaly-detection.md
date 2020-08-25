---
layout:     post
title:      Awesome Video Anomaly Detection
subtitle:   
date:       2020-08-24
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Record
    - Anomaly Detection
    - Video Analysis

---



## Datasets

1. UMN [`Download link`](http://mha.cs.umn.edu/)
2. UCSD [`Download link`](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
3. Subway Entrance/Exit [`Download link`](http://vision.eecs.yorku.ca/research/anomalous-behaviour-data/)
4. CUHK Avenue [`Download link`](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
   - HD-Avenue <span id = "05">[Skeleton-based](#01902)</span>
5. ShanghaiTech [`Download link`](https://svip-lab.github.io/dataset/campus_dataset.html)
   - HD-ShanghaiTech <span id = "00">[Skeleton-based](#01902)</span>
6. UCF-Crime (Weakly Supervised)
   - UCFCrime2Local (subset of UCF-Crime but with spatial annotations.) [`Download_link`](http://imagelab.ing.unimore.it/UCFCrime2Local), <span id = "01">[Ano-Locality](#21902)</span>
   - Spatial Temporal Annotations [`Download_link`](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation) <span id = "02">[Background-Bias](#21901)</span>
7. Traffic-Train
8. Belleview
9. Street Scene (WACV 2020) <span id = "03">[Street Scenes](#02001)</span>, [`Download link`](https://www.merl.com/demos/video-anomaly-detection)
10. IITB-Corridor (WACV 2020) <span id = "04">[Rodrigurs.etl](#02002)</span>



__The Datasets belowed are about Traffic Accidents Anticipating in Dashcam videos or Surveillance videos__

11. CADP [(CarCrash Accidents Detection and Prediction)](https://github.com/ankitshah009/CarCrash_forecasting_and_detection)
12. DAD  [paper](https://yuxng.github.io/chan_accv16.pdf), [`Download link`](https://aliensunmin.github.io/project/dashcam/)
13. A3D  [paper](https://arxiv.org/abs/1903.00618?), [`Download link`](https://github.com/MoonBlvd/tad-IROS2019)
14. DADA  [`Download link`](https://github.com/JWFangit/LOTVS-DADA)
15. DoTA   [`Download_link`](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
16. Iowa DOT [`Download_link`](https://www.aicitychallenge.org/2018-ai-city-challenge/)

-----



## Method

### Unsupervised

#### 2016

1. <span id = "01601">[Conv-AE]</span> [Learning Temporal Regularity in Video Sequences](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.pdf), `CVPR 16`. [Code](https://github.com/iwyoo/TemporalRegularityDetector-tensorflow/blob/master/model.py)

#### 2017

1. <span id = "01701">[Hinami.etl]</span> [Joint Detection and Recounting of Abnormal Events by Learning Deep Generic Knowledge](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hinami_Joint_Detection_and_ICCV_2017_paper.pdf), `ICCV 2017`. (Explainable VAD)
2. <span id = "01702">[Stacked-RNN]</span> [A revisit of sparse coding based anomaly detection in stacked rnn framework](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf), `ICCV 2017`. [code](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection)
3. <span id = "01703">[ConvLSTM-AE]</span> [Remembering history with convolutional LSTM for anomaly detection](https://ieeexplore.ieee.org/abstract/document/8019325), `ICME 2017`.[Code](https://github.com/zachluo/convlstm_anomaly_detection)
4. <span id = "01704">[Conv3D-AE]</span> [Spatio-Temporal AutoEncoder for Video Anomaly Detection](https://dl.acm.org/doi/abs/10.1145/3123266.3123451),`ACM MM 17`.
5. <span id = "01705">[Unmasking]</span> [Unmasking the abnormal events in video](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ionescu_Unmasking_the_Abnormal_ICCV_2017_paper.pdf), `ICCV 17`.
6. <span id = "01706">[DeepAppearance]</span> [Deep appearance features for abnormal behavior detection in video](https://www.researchgate.net/profile/Radu_Tudor_Ionescu/publication/320361315_Deep_Appearance_Features_for_Abnormal_Behavior_Detection_in_Video/links/5a469e9fa6fdcce1971b7258/Deep-Appearance-Features-for-Abnormal-Behavior-Detection-in-Video.pdf)

#### 2018

1. <span id = "01801">[FramePred]</span> [Future Frame Prediction for Anomaly Detection -- A New Baseline](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf), `CVPR 2018`. [code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
2. <span id = "01802">[ALOOC]</span> [Adversarially Learned One-Class Classifier for Novelty Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf), `CVPR 2018`. [code](https://github.com/khalooei/ALOCC-CVPR2018)
3. [Detecting Abnormality Without Knowing Normality: A Two-stage Approach for Unsupervised Video Abnormal Event Detection](https://dl.acm.org/doi/10.1145/3240508.3240615), `ACM MM 18`.

#### 2019

1. <span id = "01901">[Mem-AE]</span> [Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf), `ICCV 2019`.[code](https://github.com/donggong1/memae-anomaly-detection)
2. <span id = "01902">[Skeleton-based]</span> [Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos](http://openaccess.thecvf.com/content_CVPR_2019/papers/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.pdf), `CVPR 2019`.[code](https://github.com/RomeroBarata/skeleton_based_anomaly_detection)
3. <span id = "01903">[Object-Centric]</span> [Object-Centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf), `CVPR 2019`.
4. <span id = "01904">[Appearance-Motion Correspondence]</span> [Anomaly Detection in Video Sequence with Appearance-Motion Correspondence](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.pdf), `ICCV 2019`.[code](https://github.com/nguyetn89/Anomaly_detection_ICCV2019)
5. <span id = "01905">[AnoPCN]</span>[AnoPCN: Video Anomaly Detection via Deep Predictive Coding Network](https://people.cs.clemson.edu/~jzwang/20018630/mm2019/p1805-ye.pdf), ACM MM 2019.

#### 2020

1. <span id = "02001">[Street-Scene]</span> [Street Scene: A new dataset and evaluation protocol for video anomaly detection](http://openaccess.thecvf.com/content_WACV_2020/papers/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.pdf), `WACV 2020`.
2. <span id = "02002">[Rodrigurs.etl])</span> [Multi-timescale Trajectory Prediction for Abnormal Human Activity Detection](http://openaccess.thecvf.com/content_WACV_2020/papers/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.pdf), `WACV 2020`.
3. <span id = "02003">[GEPC]</span> [Graph Embedded Pose Clustering for Anomaly Detection](https://arxiv.org/pdf/1912.11850.pdf), `CVPR 2020`.[code](https://github.com/amirmk89/gepc)
4. <span id = "02004">[Self-trained]</span> [Self-trained Deep Ordinal Regression for End-to-End Video Anomaly Detection](https://arxiv.org/pdf/2003.06780.pdf), `CVPR 2020`. 
5. <span id = "02005">[MNAD]</span> [Learning Memory-guided Normality for Anomaly Detection](https://arxiv.org/pdf/2003.13228.pdf), `CVPR 2020`. [code](https://cvlab.yonsei.ac.kr/projects/MNAD)
6. <span id = "02006">[Continual-AD]]</span> [Continual Learning for Anomaly Detection in Surveillance Videos](https://arxiv.org/pdf/2004.07941),`CVPR 2020 Worksop.`
7. <span id = "02007">[OGNet]</span> [Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.pdf), `CVPR 2020`. [code](https://github.com/xaggi/OGNet)
8. <span id = "02008">[Any-Shot]</span> [Any-Shot Sequential Anomaly Detection in Surveillance Videos](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Doshi_Any-Shot_Sequential_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.pdf),`CVPR 2020 workshop`.
9. <span id = "02009">[Few-Shot]</span>[Few-Shot Scene-Adaptive Anomaly Detection](https://arxiv.org/pdf/2007.07843.pdf)`ECCV 2020 Spotlight` [code](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection)
10. <span id = "02010">[CDAE]</span>[Clustering-driven Deep Autoencoder for Video Anomaly Detection]()`ECCV 2020`



### Weakly-Supervised

#### 2018

1. <span id = "11801">[Sultani.etl]</span> [Real-world Anomaly Detection in Surveillance Videos](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf), `CVPR 2018` [code](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)

#### 2019

1. <span id = "11901">[GCN-Anomaly]</span> [Graph Convolutional Label Noise Cleaner:Train a Plug-and-play Action Classifier for Anomaly Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf),` CVPR 2019`, 
   [code](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection)
2. <span id = "11902">[MLEP]</span> [Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies](https://pdfs.semanticscholar.org/e878/6acbfabaf4938c9c8e2d3a15e0f110a1ec7f.pdf), `IJCAI 2019`.
3. <span id = "11903">[IBL]</span> [Temporal Convolutional Network with Complementary Inner Bag Loss For Weakly Supervised Anomaly Detection](https://ieeexplore.ieee.org/abstract/document/8803657/). `ICIP 19`.
4. <span id = "11904">[Motion-Aware]</span> [Motion-Aware Feature for Improved Video Anomaly Detection](https://arxiv.org/pdf/1907.10211). `BMVC 19`.

#### 2020

1. <span id = "12001">[Siamese]</span> [Learning a distance function with a Siamese network to localize anomalies in videos](https://arxiv.org/abs/2001.09189), `WACV 2020`.
2. <span id = "12002">[AR-Net]</span> [Weakly Supervised Video Anomaly Detection via Center-Guided Discrimative Learning](https://ieeexplore.ieee.org/document/9102722),` ICME 2020`.[code](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)
3. [Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision](https://arxiv.org/pdf/2007.04687.pdf) `ECCV 2020`



### Supervised

#### 2019

1. <span id = "21901">[Background-Bias]</span>[Exploring Background-bias for Anomaly Detection in Surveillance Videos](https://dl.acm.org/doi/abs/10.1145/3343031.3350998), `ACM MM 19`.
2. <span id = "21902">[Ano-Locality]</span>[Anomaly locality in video suveillance](https://arxiv.org/pdf/1901.10364), `ICIP 19`.

------





## Reviews / Surveys

1. An Overview of Deep Learning Based Methods for Unsupervised and Semi-Supervised Anomaly Detection in Videos, J. Image, 2018.[page](https://beedotkiran.github.io/VideoAnomaly.html)
2. DEEP LEARNING FOR ANOMALY DETECTION: A SURVEY, [paper](https://arxiv.org/pdf/1901.03407.pdf)
3. Video Anomaly Detection for Smart Surveillance [paper](https://arxiv.org/pdf/2004.00222.pdf)



## Books

1. Outlier Analysis. Charu C. Aggarwal

------

Generally, anomaly detection in recent researchs are based on the datasets get from pedestrian (likes UCSD, Avenue, ShanghaiTech, etc.)， or UCF-Crime (real-wrold anomaly).
However some focus on specefic scene as follows.



## Specific Scene

### Traffic

CVPR 2018 workshop, CVPR 2019 workshop, AICity Challenge series.

#### First-Person Traffic

1. Unsupervised Traffic Accident Detection in First-Person Videos, IROS 2019.

#### Driving

When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos. [github](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)

### Old-man Fall Down

### Fighting/Violence

1. Localization Guided Fight Action Detection in Survellance Videos. ICME 2019.
2. 

### Social/ Group Anomaly

1. Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks, Neurips 2019.



## Related Topics

1. Video Representation (Unsupervised Video Representation, reconstruction, prediction etc.)
2. Object Detection
3. Pedestrian Detection
4. Skeleton Detection
5. Graph Neural Networks
6. GAN
7. Action Recongnition / Temporal Action Localization
8. Metric Learning
9. Label Noise Learning
10. Cross-Modal/ Multi-Modal
11. Dictionary Learning
12. One-Class Classification / Novelty Detection / Out-of-Disturibution Detection
13. Action Recognition.
    - Human in Events: A Large-Scale Benchmark for Human-centric Video Analysis in Complex Events. ACM MM 2020 workshop.





## Performance

### Performance Evaluation Methods

1. AUC
2. PR-AUC
3. Score Gap
4. False Alarm Rate on Normal with 0.5 as threshold (Weakly supervised, proposed in CVPR 18)



### Performance Comparision on UCF-Crime 

| Model                                               | Reported on Convference/Journal | Supervised | Feature  | End2End | 32 Segments | AUC (%) | FAR@0.5 on Normal (%) |
| --------------------------------------------------- | ------------------------------- | ---------- | -------- | ------- | ----------- | ------- | --------------------- |
| <span id = "31801">[Sultani.etl](#11801)</span>     | CVPR 18                         | Weakly     | C3D RGB  | X       | √           | 75.41   | 1.9                   |
| <span id = "31903">[IBL](#11903)</span>             | ICIP 19                         | Weakly     | C3D RGB  | X       | √           | 78.66   | -                     |
| <span id = "31904">[Motion-Aware](#11904)</span>    | BMVC 19                         | Weakly     | PWC Flow | X       | √           | 79.0    | -                     |
| <span id = "31901">[GCN-Anomaly](#11901)</span>     | CVPR 19                         | Weakly     | TSN RGB  | √       | X           | 82.12   | 0.1                   |
| <span id = "31902">[Background-Bias](#21901)</span> | ACM MM 19                       | Fully      | NLN RGB  | √       | X           | 82.0    | -                     |



### Perfromace Comparision on ShanghaiTech

| Model                                             | Reported on Conference/Journal | Supervision                   | Feature            | End2Emd | AUC(%) | FAR@0.5 (%) |
| ------------------------------------------------- | ------------------------------ | ----------------------------- | ------------------ | ------- | ------ | ----------- |
| <span id = "41601">[Conv-AE](#01601)</span>       | CVPR 16                        | Un                            | -                  | √       | 60.85  | -           |
| <span id = "41702">[stacked-RNN](#01702)</span>   | ICCV 17                        | Un                            | -                  | √       | 68.0   | -           |
| <span id = "41801">[FramePred](#01801)</span>     | CVPR 18                        | Un                            | -                  | √       | 72.8   | -           |
| <span id = "41902">[FramePred*](#11902)</span>    | IJCAI 19                       | Un                            | -                  | √       | 73.4   | -           |
| <span id = "41901-1">[Mem-AE](#01901)</span>      | ICCV 19                        | Un                            | -                  | √       | 71.2   | -           |
| <span id = "42005">[MNAD](#02005)</span>          | CVPR 20                        | Un                            | -                  | √       | 70.5   | -           |
| <span id = "41902-1">[MLEP](#11902)</span>        | IJCAI 19                       | 10% test vids with Video Anno | -                  | √       | 75.6   | -           |
| <span id = "41902-2">[MLEP](#11902)</span>        | IJCAI 19                       | 10% test vids with Frame Anno | -                  | √       | 76.8   | -           |
| <span id = "42002-1">[Sultani.etl](#12002)</span> | ICME 2020                      | Weakly (Re-Organized Dataset) | C3D-RGB            | X       | 86.3   | 0.15        |
| <span id = "42002-2">[IBL](#12002)</span>         | ICME 2020                      | Weakly (Re-Organized Dataset) | I3D-RGB            | X       | 82.5   | 0.10        |
| <span id = "41901-2">[GCN-Anomaly](#11901)</span> | CVPR 19                        | Weakly (Re-Organized Dataset) | C3D-RGB            | √       | 76.44  | -           |
| <span id = "41901-3">[GCN-Anomaly](#11901)</span> | CVPR 19                        | Weakly (Re-Organized Dataset) | TSN-Flow           | √       | 84.13  | -           |
| <span id = "41901-4">[GCN-Anomaly](#11901)</span> | CVPR 19                        | Weakly (Re-Organized Dataset) | TSN-RGB            | √       | 84.44  | -           |
| <span id = "42002">[AR-Net](#12002)</span>        | ICME 20                        | Weakly (Re-Organized Dataset) | I3D-RGB & I3D Flow | X       | 91.24  | 0.10        |



### Performance Comparision on Avenue 

| Model                                                        | Reported on Conference/Journal | Supervision                   | Feature                | End2End | AUC(%) |
| ------------------------------------------------------------ | ------------------------------ | ----------------------------- | ---------------------- | ------- | ------ |
| <span id = "51601">[Conv-AE](#01601)</span>                  | CVPR 16                        | Un                            | -                      | √       | 70.2   |
| <span id = "51601-2">[Conv-AE*](#01801)</span>               | CVPR 18                        | Un                            | -                      | √       | 80.0   |
| <span id = "51703">[ConvLSTM-AE](#01703)</span>              | ICME 17                        | Un                            | -                      | √       | 77.0   |
| <span id = "51706">[DeepAppearance](#01706)</span>           | ICAIP 17                       | Un                            | -                      | √       | 84.6   |
| <span id = "51705">[Unmasking](#01705)</span>                | ICCV 17                        | Un                            | 3D gradients+VGG conv5 | X       | 80.6   |
| <span id = "51702">[stacked-RNN](#01702)</span>              | ICCV 17                        | Un                            | -                      | √       | 81.7   |
| <span id = "51801">[FramePred](#01801)</span>                | CVPR 18                        | Un                            | -                      | √       | 85.1   |
| <span id = "51901-1">[Mem-AE](#01901)</span>                 | ICCV 19                        | Un                            | -                      | √       | 83.3   |
| <span id = "51904">[Appearance-Motion Correspondence](#01904) </span> | ICCV 19                        | Un                            | -                      | √       | 86.9   |
| <span id = "51902">[FramePred*](#11902)</span>               | IJCAI 19                       | Un                            | -                      | √       | 89.2   |
| <span id = "52005">[MNAD](#02005)</span>                     | CVPR 20                        | Un                            | -                      | √       | 88.5   |
| <span id = "51801-1">[MLEP](#11902)</span>                   | IJCAI 19                       | 10% test vids with Video Anno | -                      | √       | 91.3   |
| <span id = "51801-2">[MLEP](#11902)</span>                   | IJCAI 19                       | 10% test vids with Frame Anno | -                      | √       | 92.8   |

