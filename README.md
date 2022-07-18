# Learning Depth from Focus in the Wild 
[ECCV2022] Official pytorch implementation of "Learning Depth from Focus in the Wild"
[paper] (Not yet)
Please feel free to contatct Changyeon Won (cywon1997@gm.gist.ac.kr) if you have any questions.
## Requirements
* Python == 3.7.7
* imageio==2.16.1
* h5py==3.6.0
* numpy==1.21.5
* matplotlib==3.5.1
* opencv-python==4.5.5.62
* OpenEXR==1.3.7
* Pillow==7.2.0
* scikit-image==0.19.2
* scipy==1.7.3
* torch==1.6.0
* torchvision==0.7.0
* tensorboard==2.8.0
* tqdm==4.46.0
* mat73==0.58
* typing-extensions==4.1.1

## Depth estimation Network
* Our depth estimation network is implemented based on the codes released by PSMNet [1] and CFNet [2].
### 1. Download Datasets
* DDFF-12-Scene Dataset [5]
  : Download offical link (https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-test.h5)
* DefocusNet Dataset [6], 4D Light Field Dataset [9], Middlebury Dataset [8]
  : Follow the procedure written on the official github of AiFDepthNet [7] (https://github.com/albert100121/AiFDepthNet)
* Smartphone dataset [10]
  : Download dataet from the offical website of Learning to Autofocus (https://learntoautofocus-google.github.io/)
* Put the datasets in folders in 'Depth_Estimation_Test/Datasets/'
### 2. Use pretrained model
###  Or train the model by using train codes.
  * Put the datasets in folders in 'train_codes/Datasets/' .
  * Run train codes in 'train_codes'
###  
    python train_code_[Dataset].py --lr [learning rate]
### 3. Run test.py
    python test.py --dataset [Dataset]
## Simulator
### 1. Download NYU v2 dataset [4]
* Official_link : http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
* Put 'nyu_depth_v2_labeled.mat' in 'Simulator' folder.
* This simulator is implemented based on the codes released by [3].
### 2. Run the code.
    python synthetic_blur_movement.py
## End-To-End Network
### 1. Download the real dataset and pretrained model.
*  Put the dataset in 'End_to_End/Datasets/' folder.
### 2. Run test.py
    python test_real_scenes.py
### *Limitation
 Our network shows poor performance on extreme motions because it only handles 3 basis motions which can not cover all motions.
## Sources
>[1] Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.[code](https://github.com/JiaRenChang/PSMNet) [paper](https://arxiv.org/abs/1803.08669)

>[2] Shen, Zhelun, Yuchao Dai, and Zhibo Rao. "Cfnet: Cascade and fused cost volume for robust stereo matching." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.[code](https://github.com/gallenszl/CFNet) [paper](https://arxiv.org/abs/2104.04314)

>[3] Abuolaim, Abdullah, et al. "Learning to reduce defocus blur by realistically modeling dual-pixel data." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.[code](https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel) [paper](https://arxiv.org/pdf/2012.03255.pdf)

 
>[4]Silberman, Nathan, et al. "Indoor segmentation and support inference from rgbd images." European conference on computer vision. Springer, Berlin, Heidelberg, 2012. [page](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

>[5]Hazirbas, Caner, et al. "Deep depth from focus." Asian Conference on Computer Vision. Springer, Cham, 2018.[code](https://github.com/soyers/ddff-pytorch) [paper](https://arxiv.org/pdf/1704.01085.pdf)

>[6]Maximov, Maxim, Kevin Galim, and Laura Leal-Taixé. "Focus on defocus: bridging the synthetic to real domain gap for depth estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.[code](https://github.com/dvl-tum/defocus-net) [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Maximov_Focus_on_Defocus_Bridging_the_Synthetic_to_Real_Domain_Gap_CVPR_2020_paper.html)

>[7]Wang, Ning-Hsu, et al. "Bridging Unsupervised and Supervised Depth from Focus via All-in-Focus Supervision." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.[code](https://github.com/albert100121/AiFDepthNet) [paper](https://arxiv.org/abs/2108.10843)

>[8] Scharstein, Daniel, et al. "High-resolution stereo datasets with subpixel-accurate ground truth." German conference on pattern recognition. Springer, Cham, 2014. [page](https://vision.middlebury.edu/stereo/data/)

>[9] Honauer, Katrin, et al. "A dataset and evaluation methodology for depth estimation on 4D light fields." Asian Conference on Computer Vision. Springer, Cham, 2016. [page](https://lightfield-analysis.uni-konstanz.de/)

>[10] Herrmann, Charles, et al. "Learning to autofocus." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020. [page](https://learntoautofocus-google.github.io/)


