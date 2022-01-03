# DCNGAN
DCNGAN: A Deformable Convolution-Based GAN with QP Adaptation for Perceptual Quality Enhancement of Compressed Video

## 1. Environment

- Pytorch 1.6.0, CUDA 10.1, Python 3.7
- Build DCNv2

```bash
cd ops/dcn/
bash build.sh
```
- Check if DCNv2 works successfully

```bash
python simple_check.py
```
 
## 2. Dataset

### 2.1. Open-source dataset. 

We use the open-source dataset used in [MFQE 2.0](https://github.com/RyanXingQL/MFQEv2.0). 

Please download raw videos [here](https://github.com/RyanXingQL/MFQEv2.0/wiki/MFQEv2-Dataset). (108 video sequences for training and 18 video sequences for testing) 

Please compress all video sequences by H.265/HEVC reference software HM16.5 under Low Delay P (LDP) configuration with QPs set to be 22, 27, 32 and 37.
  
Please establish two folders named "gt" (ground truth) and "lq" (low quality), respectively. Rename all videos for training, like "QP**_video**_gt" and "QP**_video**_lq". Note that "QP22_video**_gt", "QP27_video**_gt", "QP32_video**_gt" and "QP37_video**_gt" are actually the same. We just repeatly rename the same raw video** four times. While "QP22_video**_lq", "QP27_video**_lq", "QP32_video**_lq" and "QP37_video**_lq" are four low quality versions of the raw video**. Make sure the raw video and the compressed video (in low quality) are one-to-one corresponding. Put all raw videos in the "gt" folder, and put all compressed videos in the "lq" folder. For example,
 
 ```tex
gt/
├── QP22_video_name1_gt
├── QP22_video_name2_gt
...
├── QP22_video_name108_gt
├── QP27_video_name1_gt
...
├── QP37_video_name108_gt

lq/
├── QP22_video_name1_lq
├── QP22_video_name2_lq
...
├── QP22_video_name108_lq
├── QP27_video_name1_lq
...
├── QP37_video_name108_lq
```

Now we have 432 videos in the gt folder and 432 videos in the lq folder. Videos are one-to-one corresponding.

### 2.2. Generate "LMDB" files.

We refer to the way to "generate LMDB" described in [Pytorch implementation of STDF](https://github.com/RyanXingQL/STDF-PyTorch) to generate our own training dataset. Use LMDB files can speed up IO during training.

```
python dataset/create_lmdb_mfqev2.py
```
Note that you need to set your own path in "create_lmdb_mfqev2.py". 

There are four paths need to be set.

(1) "gt_dir": the path of folder "gt" mentioned above.

(2) "lq_dir": the path of folder "lq" mentioned above.

(3) "lmdb_gt_path": the path where you want to store gt LMDB file.

(4) "lmdb_lq_path": the path where you want to store lq LMDB file.

Make sure you have enough space to store LMDB files. gt LMDB file needs about 141 GB. lq LMDB file needs about 762 GB. 

After successfully generate training dataset, set 

(1)"gt_root" and "lq_root" in dataset/mfqev2.py (class MFQEv2Dataset). "gt_root" is "lmdb_gt_path". "lq_root" is "lmdb_lq_path" (mentioned above).

(2)"gt_root" and "lq_root" in dataset/mfqev2.py (class VideoTestMFQEv2Dataset). "gt_root" is the path where raw video for validation is stored. "lq_root" is the path where the corresponding compressed video for validation is stored. You can choose only one video to validate during the training (just for checking if training goes normally), or you can choose to remove all things about validation to save training time. 


It should be noted that our training dataset includes compressed sequences at all of the four QPs to give the proposed DCNGAN the ability of QP-conditional adaptation. 

## 3. Training your own model

```
python train.py
```

Parameters are set in 3frames_mfqev2_1G.yml.

Every time you run train.py, a folder named "exp" will be generated. Delete it or rename it if you want to re-run train.py.


## 4. Testing your own model

```
python test.py
```

Please set your own path in test.py

(1) ckp_path: pre-trained model path

(2) rec_yuv_save_path: enhanced video path (output path)

(3) cmp_yuv_path: compressed video path (input path)

(4) raw_yuv_base_path: raw video (video before compression) path

## 5. Quantitative performance evaluation

Two perceptual quality metrics, i.e., [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [DISTS](https://github.com/dingkeyan93/DISTS), are used to evaluate the performance. To calcluate LPIPS and DISTS, you need to run "matlab/main.m" which calls "matlab/YUVtoRGB.m" to convert a video sequence to sequential RGB images firstly. Note that you need convert both raw videos (the orignal videos before compression) and enhanced videos (the outputs of the network). Corresponding raw image and enhanced image compose a image pair. Then you can calculate LPIPS (lpips.LPIPS(net='alex',version=0.1)) and DISTS for each pair of images. The average LPIPS or average DISTS of all pairs of images of a video is regarded as the LPIPS or DISTS of the video.

## 6. License

We adopt Apache License v2.0.

We would like to thank the authors of [MFQE 2.0](https://github.com/RyanXingQL/MFQEv2.0) for their open-source dataset and the author of [Pytorch implementation of STDF](https://github.com/RyanXingQL/STDF-PyTorch) for his open-source implementation and the way to generate training dataset in LMDB format.

If you also find them useful, please cite their works

```
@article{2019xing,
    doi = {10.1109/tpami.2019.2944806},
    url = {https://doi.org/10.1109%2Ftpami.2019.2944806},
    year = 2021,
    month = {mar},
    publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
    volume = {43},
    number = {3},
    pages = {949--963},
    author = {Zhenyu Guan and Qunliang Xing and Mai Xu and Ren Yang and Tie Liu and Zulin Wang},
    title = {{MFQE} 2.0: A New Approach for Multi-Frame Quality Enhancement on Compressed Video},
    journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence}
}

@misc{2020xing2,
  author = {Qunliang Xing},
  title = {PyTorch implementation of STDF},
  howpublished = "\url{https://github.com/RyanXingQL/STDF-PyTorch}",
  year = {2020}, 
  note = "[Online; accessed 11-April-2021]"
}
```
