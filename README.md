# PERSO WIP - CUDA accelerated preprocessing module already implemented, but code cleanup, screenshots, updated instructions and benchmark to come.

From the forked repo (https://github.com/ruotianluo/self-critical.pytorch)


# Partial readme

# An Image Captioning codebase

This is a codebase for image captioning research.

It supports:
- Self critical training from [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)
- Bottom up feature from [ref](https://arxiv.org/abs/1707.07998).
- Test time ensemble
- Multi-GPU training. (DistributedDataParallel is now supported with the help of pytorch-lightning, see [ADVANCED.md](ADVANCED.md) for details)
- Transformer captioning model.

A simple demo colab notebook is available [here](https://colab.research.google.com/github/ruotianluo/ImageCaptioning.pytorch/blob/colab/notebooks/captioning_demo.ipynb)

## Requirements
- Python 3
- PyTorch 1.3+ (along with torchvision)
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule) (**Remember to follow initialization steps in coco-caption/README.md**)
- yacs
- lmdbdict
