# SmartEraser: A Smart Object Removal Model

[**SmartEraser: Remove Anything from Images using Masked-Region Guidance**](https://arxiv.org/abs/2501.08279)

[Longtao Jiang*](https://github.com/longtaojiang), [Zhendong Wang*](https://github.com/ZhendongWang6), [Jianmin Bao*†](), [Wengang Zhou†](), [Dongdong Chen](), [Lei Shi](), [Dong Chen](), [Houqiang Li](),

(*Equal contribution, †corresponding author)

Project Page: https://longtaojiang.github.io/smarteraser.github.io/
Arxiv: https://arxiv.org/abs/2501.08279

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.03594-b31b1b.svg)](https://arxiv.org/abs/2312.03594)
[![Project Page](https://img.shields.io/badge/PowerPaint-Website-green)](https://powerpaint.github.io/)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/JunhaoZhuang/PowerPaint-v1) -->

**Your star means a lot for us to develop this project!** :star:

SmartEraser is a object removal model built on the novel Masked-Region Guidance paradigm. SmartEraser outperforms existing methods by smartly identifying the target object to remove while effectively preserving the surrounding context. To facilitate research on this paradigm, we propose Syn4Removal, a large-scale, high-quality dataset containing over a million image triplets, specifically designed for object removal tasks. Through extensive experiments, we demonstrate that SmartEraser achieves superior performance in both quality and robustness compared to previous object removal methods.

### TODO

- [ ] Release dataset synthesis pipeline.
- [ ] Release Training and inference codes.
- [ ] Release Syn4Removal 1M dataset.
- [ ] Release SmartEraser pretrain models.
- [ ] Release SmartEraser online demo for use.

<!-- <img src='https://github.com/open-mmlab/mmagic/assets/12782558/acd01391-c73f-4997-aafd-0869aebcc915'/> -->


<!-- ## 🚀 News

**May 22, 2024**:fire:

- We have open-sourced the model weights for PowerPaint v2-1, rectifying some existing issues that were present during the training process of version 2. [![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1)

**April 7, 2024**:fire:

- We open source the model weights and code for PowerPaint v2. [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/zhuangjunhao/PowerPaint_v2) [![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/JunhaoZhuang/PowerPaint_v2)

**April 6, 2024**:

- We have retrained a new PowerPaint, taking inspiration from Brushnet. The [Online Demo](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint) has been updated accordingly. **We plan to release the model weights and code as open source in the next few days**.
- Tips: We preserve the cross-attention layer that was deleted by BrushNet for the task prompts input.

|  | Object insertion | Object Removal|Shape-guided Object Insertion|Outpainting|
|-----------------|-----------------|-----------------|-----------------|-----------------|
| Original Image| ![cropinput](https://github.com/Sanster/IOPaint/assets/108931120/bf91a1e8-8eaf-4be6-b47d-b8e43c9d182a)|![cropinput](https://github.com/Sanster/IOPaint/assets/108931120/c7e56119-aa57-4761-b6aa-56f8a0b72456)|![image](https://github.com/Sanster/IOPaint/assets/108931120/cbbfe84e-2bf1-425b-8349-f7874f2e978c)|![cropinput](https://github.com/Sanster/IOPaint/assets/108931120/134bb707-0fe5-4d22-a0ca-d440fa521365)|
| Output| ![image](https://github.com/Sanster/IOPaint/assets/108931120/ee777506-d336-4275-94f6-31abf9521866)| ![image](https://github.com/Sanster/IOPaint/assets/108931120/e9d8cf6c-13b8-443c-b327-6f27da54cda6)|![image](https://github.com/Sanster/IOPaint/assets/108931120/cc3008c9-37dd-4d98-ad43-58f67be872dc)|![image](https://github.com/Sanster/IOPaint/assets/108931120/18d8ca23-e6d7-4680-977f-e66341312476)|

**December 22, 2023**:wrench:

- The logical error in loading ControlNet has been rectified. The `gradio_PowerPaint.py` file and [Online Demo](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint) have also been updated.

**December 18, 2023**

*Enhanced PowerPaint Model*

- We are delighted to announce the release of more stable model weights. These refined weights can now be accessed on [Hugging Face](https://huggingface.co/JunhaoZhuang/PowerPaint-v1/tree/main). The `gradio_PowerPaint.py` file and [Online Demo](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint) have also been updated as part of this release. -->

<!-- ## Get Started

**Recommend Environment:** `cuda 11.8` + `python 3.9`

```bash
# Clone the Repository
git clone git@github.com:open-mmlab/PowerPaint.git

# Create Virtual Environment with Conda
conda create --name ppt python=3.9
conda activate ppt

# Install Dependencies
pip install -r requirements/requirements.txt
```

Or you can construct a conda environment from scratch by running the following command:

```bash
conda env create -f requirements/ppt.yaml
conda activate ppt
```

## Inference

You can launch the Gradio interface for PowerPaint by running the following command:

```bash
# Set up Git LFS
conda install git-lfs
git lfs install

# Clone PowerPaint Model
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1/ ./checkpoints/ppt-v1

python app.py --share
```

We suggest PowerPaint-V2 that is built upon BrushNet with RealisticVision as the base model, which exhibits higher visual quality. You can run the following command:
```bash
# Clone PowerPaint Model
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint_v2/ ./checkpoints/ppt-v2

python app.py --share --version ppt-v2 --checkpoint_dir checkpoints/ppt-v2
```
Specifically, if you have downloaded the weights and want to skip the step of cloning the model, you can skip that step by enabling `--local_files_only`. -->

<!-- ### Object Removal

For object removal, you need to select the tab of `Object removal inpainting` and you don't need to input any prompts. PowerPaint is able to fill in the masked region according to context background.

We remain the text box for inputing prompt, allowing users to further suppress object generation by using negative prompts.
Specifically, we recommend to use 10 or higher value for Guidance Scale. If undesired objects appear in the masked area, you can address this by specifically increasing the Guidance Scale.

|Input|Output|
|---------------|-----------------|
| <img src="assets/gradio_objremoval.jpg"> | <img src="assets/gradio_objremoval_result.jpg"> -->

<!-- ## Training

1. Prepare training data. You may need to rewrite [`Datasets`](./powerpaint/datasets/__init__.py）per your need (e.g., data and storage formats). Here, we use petreloss to read training dataset from cloud storages. Besides, the recipe of datasets for training a versatile model can be tricky but intuitive.

2. Start training. We suggest using PowerPaint-V2 version, which is built upon BrushNet and requires smaller batch size for training. You can train it with the following command,
```shell
# running on a single node
accelerate launch --config_file configs/acc.yaml train_ppt2_bn.py --config configs/ppt2_bn.yaml --output_dir runs/ppt1_sd15

# running on one node by slurm, e.g., 1 nodes with 8 gpus in total
python submit.py --job-name ppt2_bn --gpus 8 train_ppt2_bn.py --config configs/ppt2_bn.yaml --output_dir runs/ppt2_bn
```
where `configs/acc.yaml` is the configuration file for using accelerate, and `configs/ppt2_bn.yaml` is the configuration file for training PowerPaint-V2.

PowerPaint-V1 version often requires much larger training batch size to converge (e.g., 1024). You can train it with the following command,

```shell
# running on a single node
accelerate launch --config_file configs/acc.yaml train_ppt1_sd15.py --config configs/ppt1_sd15.yaml --output_dir runs/ppt1_sd15 --gradient_accumulation_steps 2 --train_batch_size 64

# running on two nodes by slurm, e.g., 2 nodes with 8 gpus in total
python submit.py --job-name ppt1_sd15 --gpus 16 train_ppt1_sd15.py --config configs/ppt1_sd15.yaml --output_dir runs/ppt1_sd15 --train_batch_size 64
```
where `configs/acc.yaml` is the configuration file for using accelerate, and `configs/ppt1_sd15.yaml` is the configuration file for training PowerPaint-V1. -->

## Contact Us

**Longtao Jiang**: taotao707@mail.ustc.edu.cn

**Zhendong Wang**: zhendongwang6@outlook.com

<!-- ## BibTeX

```
@misc{zhuang2023task,
      title={A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting},
      author={Junhao Zhuang and Yanhong Zeng and Wenran Liu and Chun Yuan and Kai Chen},
      year={2023},
      eprint={2312.03594},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->
