# M2FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation

This repository is the official **PyTorch** implementation of our work: [M2FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation](https://doi.org/10.1109/JBHI.2023.3326151), presented at IEEE-JBHI 2023.

## Setup

### Environment

All our experiments are implemented based on the PyTorch framework with two 24G NVIDIA Geforce RTX 3090 GPUs, and we recommend installing the following package versions:

- python=3.8
- pytorch=1.12.1
- torchvision=0.13.1

Dependency packages can be installed using following command:

```bash
conda create --name m2ftrans python=3.8
conda activate m2ftrans

pip install -r requirements.txt
```

### Data preparation

We provide two different versions of training framework\, based on previous work [RFNet](https://github.com/dyh127/RFNet) and [SMU-Net](https://github.com/rezazad68/smunet), corresponding to [M2FTrans_v1](https://github.com/Jun-Jie-Shi/M2FTrans/tree/main/M2FTrans_v1) and [M2FTrans_v2](https://github.com/Jun-Jie-Shi/M2FTrans/tree/main/M2FTrans_v2).

**M2FTrans_v1**

- Download the preprocessed dataset (BraTS2020 or BraTS2018) from [RFNet](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A?usp=sharing) and unzip them in the `BraTS` folder .

  ```bash
  tar -xzf BRATS2020_Training_none_npy.tar.gz
  tar -xzf BRATS2018_Training_none_npy.tar.gz
  ```
- If you want to preprocess by yourself, the preprocessing code ``preprocess.py`` is also provided, see [RFNet](https://github.com/dyh127/RFNet) for more details.
- For BraTS2021, download the train dataset from [this](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) link and extract it inside the `BraTS` folder, change the the path of src_path and tar_path in `preprocess_brats2021.py`, then run:
- ```bash
  python preprocess_brats2021.py
  ```
- The train.txt, val.txt and test.txt of different datasets should be added in `BraTS20xx_Training_none_npy` folders, we also provide in `BraTS/BraTS20xx_Training_none_npy` folders.

**M2FTrans_v2**

- Download the BraTS2018 train dataset from [this](https://www.kaggle.com/sanglequang/brats2018) link and extract it inside the `BraTS` folder.

The folder structure is assumed to be:

```
M2FTrans/
├── BraTS
│   ├── BRATS2018_Training_none_npy
│   │   ├── seg
│   │   ├── vol
│   │   ├── ...
│   ├── BRATS2020_Training_none_npy
│   │   ├── seg
│   │   ├── vol
│   │   ├── ...
│   ├── BRATS2021_Training_none_npy
│   │   ├── seg
│   │   ├── vol
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── val.txt
│   ├── BRATS2021_Training_Data
│   │   ├── ...
│   ├── MICCAI_BraTS_2018_Data_Training
│   │   ├── HGG
│   │   ├── LGG
│   │   ├── ...
├── M2FTrans_v1
│   ├── ...
├── M2FTrans_v2
│   ├── ...
└── ...
```

## Training

**M2FTrans_v1**

- Changing the paths and hyperparameters in ``train.sh``, ``train.py`` and ``predict.py``.
- Set different splits for BraTS20xx in ``train.py``.
- Then run:

  ```bash
  bash train.sh
  ```
- Noting that you may need more training epochs to get a better performance, you can also choose to load the pretrained model you trained or we [provided](https://drive.google.com/drive/folders/10lBPIO_gjuJvVMHJzQqADdi4WhptwdLI?usp=sharing) by setting the resume path in ``train.sh``.

**M2FTrans_v2**

- Changing the paths and hyperparameters in ``config.yml``, ``train.py``, and ``predict.py``.
- Then run:

  ```bash
  python train.py
  ```

## Evaluation

Checking the relevant paths in path in ``eval.sh`` or ``eval.py``.

**M2FTrans_v1**

```bash
bash eval.sh
```

**M2FTrans_v2**

```bash
python eval.py
```

- The pretrained models are also available in [Google Drive](https://drive.google.com/drive/folders/10lBPIO_gjuJvVMHJzQqADdi4WhptwdLI?usp=sharing).

## Acknowledgement

The implementation is based on the repos: [RFNet](https://github.com/dyh127/RFNet), [mmFormer](https://github.com/YaoZhang93/mmFormer) and [SMU-Net](https://github.com/rezazad68/smunet), we'd like to express our gratitude to these open-source works.

## Citations

Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package:

```
@ARTICLE{10288381,
  author={Shi, Junjie and Yu, Li and Cheng, Qimin and Yang, Xin and Cheng, Kwang-Ting and Yan, Zengqiang},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={M $^{2}$ FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation},
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/JBHI.2023.3326151}}
```
