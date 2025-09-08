# 视频帧插值的实时中间流估计
## [YouTube](https://www.youtube.com/results?search_query=rife+interpolation&sp=CAM%253D) | [BiliBili](https://search.bilibili.com/all?keyword=SVFI&order=stow&duration=0&tids_1=0) | [Colab](https://colab.research.google.com/github/hzwer/ECCV2022-RIFE/blob/main/Colab_demo.ipynb) | [教程](https://www.youtube.com/watch?v=gf_on-dbwyU&feature=emb_title) | [DeepWiki](https://deepwiki.com/hzwer/ECCV2022-RIFE)

## 简介
本项目是 [视频帧插值的实时中间流估计](https://arxiv.org/abs/2011.06294) 的实现。目前，我们的模型在2080Ti GPU上可以以30+FPS的速度进行2倍720p插值。它支持一对图像之间的任意时间步插值。

**2024.08 - 我们发现 [4.22.lite](https://github.com/hzwer/Practical-RIFE/tree/main?tab=readme-ov-file#model-list) 非常适合 [一些扩散模型生成的视频](https://drive.google.com/drive/folders/1hSzUn10Era3JCaVz0Z5Eg4wT9R6eJ9U9?usp=sharing) 的后处理。**

2023.11 - 我们最近发布了针对动漫场景优化的新版本 [v4.7-4.10](https://github.com/hzwer/Practical-RIFE/tree/main#model-list)！我们借鉴了 [SAFA](https://github.com/megvii-research/WACV2024-SAFA/tree/main) 的研究。

2022.7.4 - 我们的论文被 ECCV2022 接受。感谢所有相关作者、贡献者和用户！

从2020年到2022年，我们提交了RIFE五次（被CVPR21 ICCV21 AAAI22 CVPR22拒绝）。感谢所有匿名审稿人，您的建议极大地改进了论文！

[ECCV 海报](https://drive.google.com/file/d/1xCXuLUCSwhN61kvIF8jxDvQiUGtLK0kN/view?usp=sharing) | [ECCV 5分钟演示](https://youtu.be/qdp-NYqWQpA) | [论文中文介绍](https://zhuanlan.zhihu.com/p/568553080) | [反驳 (2WA1WR->3WA)](https://drive.google.com/file/d/16IVjwRpwbTuJbYyTn4PizKX8I257QxY-/view?usp=sharing) 

**推荐软件：[RIFE-App](https://grisk.itch.io/rife-app) | [FlowFrames](https://nmkd.itch.io/flowframes) | [SVFI (中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation)**

两张输入图像的16倍插值结果：

![Demo](./demo/I2_slomo_clipped.gif)
![Demo](./demo/D2_slomo_clipped.gif)

## 软件
[Flowframes](https://nmkd.itch.io/flowframes) | [SVFI(中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) | [Autodesk Flame](https://vimeo.com/505942142) | [SVP](https://www.svp-team.com/wiki/RIFE_AI_interpolation) | [MPV_lazy](https://github.com/hooke007/MPV_lazy) | [enhancr](https://github.com/mafiosnik777/enhancr)

[RIFE-App(付费)](https://grisk.itch.io/rife-app) | [Steam-VFI(付费)](https://store.steampowered.com/app/1692080/SVFI/) 

我们不对上述软件的开发负责，也不参与其中。根据开源许可证，我们尊重其他开发者的商业行为。

[VapourSynth-RIFE](https://github.com/HolyWu/vs-rife) | [RIFE-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | [VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan) | [vs-mlrt](https://github.com/AmusementClub/vs-mlrt)

<img src="https://api.star-history.com/svg?repos=hzwer/ECCV2022-RIFE,Justin62628/Squirrel-RIFE,n00mkrad/flowframes,nihui/rife-ncnn-vulkan,hzwer/Practical-RIFE&type=Date" height="320" width="480" />

如果您是开发者，欢迎关注 [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)，它旨在通过添加各种功能和设计速度更快的新模型，使 RIFE 对用户更实用。

您可以查看 [此拉取请求](https://github.com/megvii-research/ECCV2022-RIFE/pull/300) 以支持 macOS。
## 命令行界面使用

### 安装

```
git clone git@github.com:megvii-research/ECCV2022-RIFE.git
cd ECCV2022-RIFE
pip3 install -r requirements.txt
```

* 从 [这里](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing) 下载预训练的 **HD** 模型。（百度网盘链接：https://pan.baidu.com/share/init?surl=u6Q7-i4Hu4Vx9_5BJibPPA 密码:hfk3，将压缩包解压后放在 train_log/\*）

* 解压并将预训练参数移动到 train_log/\*

* 本模型未在我们的论文中提及，有关我们论文中的模型，请参阅 [评估](https://github.com/hzwer/ECCV2022-RIFE#evaluation)。

### 运行

**视频帧插值**

您可以使用我们的 [演示视频](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) 或您自己的视频。
```
python3 inference_video.py --exp=1 --video=video.mp4 
```
（生成 video_2X_xxfps.mp4）
```
python3 inference_video.py --exp=2 --video=video.mp4
```
（用于4倍插值）
```
python3 inference_video.py --exp=1 --video=video.mp4 --scale=0.5
```
（如果您的视频分辨率很高，例如4K，我们建议设置 --scale=0.5（默认1.0）。如果您在视频上生成了无序图案，请尝试设置 --scale=2.0。此参数控制光流模型的处理分辨率。）
```
python3 inference_video.py --exp=2 --img=input/
```
（从png读取视频，例如 input/0.png ... input/612.png，确保png名称是数字）
```
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60
```
（添加慢动作效果，音频将被移除）
```
python3 inference_video.py --video=video.mp4 --montage --png
```
（如果您想蒙太奇原始视频并保存png格式输出）

**扩展应用**

您可以参考 [#278](https://github.com/megvii-research/ECCV2022-RIFE/issues/278#event-7199085190) 进行 **光流估计**，并参考 [#291](https://github.com/hzwer/ECCV2022-RIFE/issues/291#issuecomment-1328685348) 进行 **视频拼接**。

**图像插值**

```
python3 inference_img.py --img img0.png img1.png --exp=4
```
（2^4=16倍插值结果）
之后，您可以使用png生成mp4：
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p output/slomo.mp4 -q:v 0 -q:a 0
```
您也可以使用png生成gif：
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

### 在docker中运行
将预训练模型放置在 `train_log/\*.pkl` 中（如上所述）

构建容器：
```
docker build -t rife -f docker/Dockerfile .
```

运行容器：
```
docker run --rm -it -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
```
```
docker run --rm -it -v $PWD:/host rife:latest inference_img --img img0.png img1.png --exp=4
```

使用GPU加速（需要docker的正确GPU驱动）：
```
docker run --rm -it --gpus all -v /dev/dri:/dev/dri -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
```

## 评估
下载我们的论文中提到的 [RIFE 模型](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view?usp=sharing) 或 [RIFE_m 模型](https://drive.google.com/file/d/147XVsDXBfJPlyct2jfo9kpbL944mNeZr/view?usp=sharing)。

**UCF101**：在 ./UCF101/ucf101_interp_ours/ 下载 [UCF101 数据集](https://liuziwei7.github.io/projects/VoxelFlow)

**Vimeo90K**：在 ./vimeo_interp_test 下载 [Vimeo90K 数据集](http://toflow.csail.mit.edu/)

**MiddleBury**：在 ./other-data 和 ./other-gt-interp 下载 [MiddleBury OTHER 数据集](https://vision.middlebury.edu/flow/data/)

**HD**：在 [这里](https://github.com/baowenbo/MEMC-Net) 下载 [HD 数据集](https://github.com/baowenbo/MEMC-Net)。我们还提供了一个 [google drive 下载链接](https://drive.google.com/file/d/1iHaLoR2g1-FLgr9MEv51NH_KQYMYz-FA/view?usp=sharing)。
```
# RIFE
python3 benchmark/UCF101.py
# "PSNR: 35.282 SSIM: 0.9688"
python3 benchmark/Vimeo90K.py
# "PSNR: 35.615 SSIM: 0.9779"
python3 benchmark/MiddleBury_Other.py
# "IE: 1.956"
python3 benchmark/HD.py
# "PSNR: 32.14"

# RIFE_m
python3 benchmark/HD_multi_4X.py
# "PSNR: 22.96(544*1280), 31.87(720p), 34.25(1080p)"
```

## 训练与复现
下载 [Vimeo90K 数据集](http://toflow.csail.mit.edu/)。

我们使用16个CPU、4个GPU和20G内存进行训练：
```
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --world_size=4
```

## 修订历史

2021.3.18 [arXiv](https://arxiv.org/pdf/2011.06294v5.pdf)：修改了主要的实验数据，特别是与运行时相关的问题。

2021.8.12 [arXiv](https://arxiv.org/pdf/2011.06294v6.pdf)：移除了预训练模型依赖，并提出了用于帧插值的特权蒸馏方案。移除了 [census loss](https://github.com/hzwer/arXiv2021-RIFE/blob/0e241367847a0895748e64c6e1604c94db54d395/model/loss.py#L20) 监督。

2021.11.17 [arXiv](https://arxiv.org/pdf/2011.06294v11.pdf)：支持任意时间帧插值，即RIFEm，并增加了更多实验。

## 推荐
我们真诚推荐一些相关论文：

CVPR22 - [通过视频帧插值优化视频预测](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Optimizing_Video_Prediction_via_Video_Frame_Interpolation_CVPR_2022_paper.html)

CVPR22 - [使用Transformer进行视频帧插值](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_Video_Frame_Interpolation_With_Transformer_CVPR_2022_paper.html)

CVPR22 - [IFRNet: 用于高效帧插值的中间特征细化网络](https://openaccess.thecvf.com/content/CVPR2022/html/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.html)

CVPR23 - [用于视频预测的动态多尺度体素流网络](https://huxiaotaostasy.github.io/DMVFN/)

CVPR23 - [通过帧间注意力提取运动和外观以实现高效视频帧插值](https://arxiv.org/abs/2303.00440)

## 引用
如果您认为本项目有帮助，请随意点赞或引用我们的论文：

```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## 参考

光流：
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

视频插值：
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)  [EQVI](https://github.com/lyh-18/EQVI)