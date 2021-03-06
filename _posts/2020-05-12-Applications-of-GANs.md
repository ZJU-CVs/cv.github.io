---
layout:     post
title:      Applications of GANs
subtitle:   
date:       2020-05-12
author:     JY
header-img: img/post-bg.jpg
catalog: true
tags:
    - Record
    - GAN
---

ref: https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900



### **1. Create Anime characters**

Game development and animation production are expensive and hire many production artists for relatively routine tasks. GAN can auto-generate and colorize Anime characters.

[Towards the automatic Anime characters creation with Generative Adversarial Networks](https://arxiv.org/pdf/1708.05509.pdf)

 

### **2. Pose Guided Person Image Generation**

https://arxiv.org/pdf/1705.09368.pdf

With an additional input of the pose, we can transform an image into different poses. For example, the top right image is the ground truth while the bottom right is the generated image.

 

[Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/7.png" alt="img" style="zoom:50%;" />



### **3. Cross-domain transfer**

Cross-domain transfer GANs will be likely the first batch of commercial applications. These GANs transform images from one domain (say real scenery) to another domain (Monet paintings or Van Gogh).



[CycleGAN](https://github.com/junyanz/CycleGAN)

CycleGAN builds 2 networks **G** and **F** to construct images from one domain to another and in the reverse direction. It uses discriminators *D* to critic how well the generated images are. For example, **G** converts real images to Van Gogh style painting and *Dy* is used to distinguish whether the image is real or generated.

Domain A ➝ Domain B:

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/8.png" alt="img" style="zoom:50%;" />

We repeat the process in the reverse direction Domain B➝ Domain A

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/9.png" alt="img" style="zoom:50%;" />



[StarGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf)

StarGAN is an image-to-image translation for one domain to another. 

In (b), the generator generates a fake image based on an input image and a target domain label (say angry). In (c), given this fake image and the original domain of the image (say happy), it reconstructs the image using the generator. In (d), we feed real and fake images to the discriminator to label it as real or not as well as its domain classification. The cost function will involve reconstruction errors as well as the discriminator cost in identifying the images and their labels.

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/10.png" alt="img" style="zoom:50%;" />





### **4. PixelDTGAN**

Suggesting merchandise based on celebrity pictures has been popular for fashion blogger and e-commerce. PixelDTGAN creates clothing images and styles from an image.

[PixelDTGAN](https://arxiv.org/pdf/1603.07442.pdf)

[code](https://github.com/fxia22/PixelDTGAN)



### **5. Super resolution**

Create super-resolution images from the lower resolution. This is one area where GAN shows very impressive result with immediate commercial possibility.

[SRGAN](https://arxiv.org/pdf/1609.04802.pdf)

Similar to many GAN designs, it composes of many layers of convolutional layer, batch normalization, advanced ReLU and skip connections.



### **6. Progressive growing of GANs**

Progressive GAN is probably one of the first GAN showing commercial-like image quality.  

[Progressive growing of GANs](https://arxiv.org/pdf/1710.10196.pdf)

It applies the strategy of divide-and-conquer to make training much feasible. Layers of convolution layers are trained once at a time to build images of 2× resolution.

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/11.png" alt="img" style="zoom:40%;" />

In 9 phases, a 1024 × 1024 image is generated.

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/12.png" alt="img" style="zoom:40%;" />



### **7. High-resolution image synthesis**

Generating images from a semantic map. 

Collecting samples are very expensive. We have trying to supplement training dataset with generated data to lower development cost. It will be handy to generate videos in training autonomous cars rather than see them cruising in your neighborhood.

[pix2pixHD](https://tcwang0509.github.io/pix2pixHD/)

[GauGAN](http://nvidia-research-mingyuliu.com/gaugan) synthesizes photorealistic images given an input semantic layout.



### **8. Text to image**

Text to image is one of the earlier application of domain-transfer GAN. We input a sentence and generate multiple images fitting the description.

[StackGAN](https://arxiv.org/pdf/1612.03242v1.pdf)

[Code](https://github.com/hanzhanggit/StackGAN)



### **9. Text to Image Synthesis**

Another popular implementation:

[Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)



### **10. Face synthesis**

Synthesis faces in different poses: With a single input image, we create faces in different viewing angles. For example, we can use this to transform images that will be easier for face recognition.

[TP-GAN](https://arxiv.org/pdf/1704.04086.pdf)



### **11. Image inpainting**

Repair images have been an important subject decades ago. GAN is used to repair images and fill the missing part with created “content”.

[Context encoder](https://github.com/pathak22/context-encoder)



### **12. Learn Joint Distribution**

It is expensive to create GANs with different combinations of facial characters *P(blond, female, smiling, with glasses)*, *P(brown, male, smiling, no glasses)* etc…The curse of dimensionality makes the number of GANs to grow exponentially. Instead, we can learn individual data distribution and combine them to form different distributions. i.e. different attribute combinations.

[CoGAN](https://arxiv.org/pdf/1606.07536.pdf)



[DiscoGAN](https://github.com/carpedm20/DiscoGAN-pytorch)

DiscoGAN provides matching style: many potential applications. DiscoGAN learns cross domain relationship without labels or pairing. For example, it successfully transfers style (or patterns) from one domain (handbag) to another (shoe).



### **13. Pix2Pix**

[Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)

[code](https://github.com/phillipi/pix2pix)

Pix2Pix is an image-to-image translation that get quoted in cross-domain GAN’s paper frequently. 



[DTN](https://arxiv.org/pdf/1611.02200.pdf)

Creating Emoji from pictures.



### **14. Texture synthesis**

[MGAN](https://arxiv.org/pdf/1604.04382.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/14.png" alt="img" style="zoom:50%;" />



### **15. Image editing**

Reconstruct or edit images with specific attributes.

[IcGAN](https://github.com/Guim3/IcGAN)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/15.png" alt="img" style="zoom:50%;" />



### **16. Face aging **

[Age-cGAN](https://arxiv.org/pdf/1702.01983.pdf)



### **17. Neural Photo Editor**

[Neural Photo Editor](https://github.com/ajbrock/Neural-Photo-Editor): Content based image editing.

[DeblurGAN](https://arxiv.org/pdf/1711.07064.pdf) performs motion deblurring.



### **18. Refine image**

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/16.png" alt="img" style="zoom:50%;" />





### **19.Object detection**

This is one application in enhancing an existing solution with GAN.

[Perceptual GAN](https://arxiv.org/pdf/1706.05274v2.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/17.png" alt="img" style="zoom:50%;" />



### **20. Image blending**

Blending images together.

[GP-GAN](https://github.com/wuhuikai/GP-GAN)



### **21. Video generation**

https://arxiv.org/pdf/1609.02612.pdf

Create new video sequence. It recognizes what is background and create new time sequence for the foreground action.



### **22. Generate 3D objects**

This is one often quoted paper in creating 3D objects with GAN.

[3DGAN](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/19.png" alt="img" style="zoom:40%;" />



### **23. Music generation**

GAN can be applied to non-image domain, like composing music.

[MidiNet](https://arxiv.org/pdf/1703.10847.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/20.png" alt="img" style="zoom:50%;" />



### **24. Medical (Anomaly Detection)**

GAN can also extend to other industry, for example medical in tumor detection.

[AnoGAN](https://arxiv.org/pdf/1703.05921.pdf)

<img src="https://github.com/ZJU-CVs/zju-cvs.github.io/raw/master/img/gan/21.png" alt="img" style="zoom:50%;" />

