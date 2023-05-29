# SpiralMLP

The groundbreaking advancement of MLP-based architecture has demonstrated a promising approach in the field of computer vision, allowing researchers to enjoy the computationally efficient models. In this paper, we introduce SpiralMLP, a powerful MLP architecture, that incorporates an expansive receptive field while maintaining linear computational complexity with respect to the input size and accommodating diverse input resolutions. Extensive experiments have been done in comparison with other SOTAs to demonstrate SpiralMLP outperforms on various downstream vision tasks, including image classification and object detection.

# Updates
(25/05/2023) release of code

# Usage
## Data Preparation
```
|root/
|- train/
|    |-n01484850
|    |    |-n01484850_254.JPEG
|    |    |-.......
|    |-......
|- val/
|    |-n01531178
|    |     |-ILSVRC2012_val_00013407.JPEG
|    |     |-........
|    |-.........
```

## Training
Training on ImageNet with 8 gpus
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model SpiralMLP_B5 --batch-size 4 --data-path /path/to/imagenet_root
```
**Note** the Batch size in distributed is num_of_gpus * batch_size

Training on ImageNet with single gpus
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model SpiralMLP_B5 --batch-size 4 --data-path /path/to/imagenet_root
```

# Acknowledgement

This code is based on [CyleMLP](https://github.com/ShoufaChen/CycleMLP) and [DeiT](https://github.com/facebookresearch/deit).

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/mmcandlee/spiralmlp/blob/main/LICENSE) file.
