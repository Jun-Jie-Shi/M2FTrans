# M2FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation

This repository is the official **PyTorch** implementation of our work:

M2FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation

## Setup

### Environment

```bash
conda create --name m2ftrans python=3.8
conda activate m2ftrans

pip install -r requirements.txt
```

### Data preparation

We provide two different versions of the implementation, based on previous work [RFNet](https://github.com/dyh127/RFNet) and [SMU-Net](https://github.com/rezazad68/smunet), corresponding to [M2FTrans_v1](https://github.com/Jun-Jie-Shi/M2FTrans/tree/main/M2FTrans_v1) and [M2FTrans_v2](https://github.com/Jun-Jie-Shi/M2FTrans/tree/main/M2FTrans_v2).

**M2FTrans_v1**

- Download the preprocessed dataset (BraTS2020 or BraTS2018) from [RFNet](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A?usp=sharing) and unzip them in the `BraTS` folder . 

  ```bash
  tar -xzf BRATS2020_Training_none_npy.tar.gz
  tar -xzf BRATS2018_Training_none_npy.tar.gz
  ```

- The preprocessing code ```prepeocess.py``` is also provided, see [RFNet](https://github.com/dyh127/RFNet) for more details.

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
│   ├── MICCAI_BraTS_2018_Data_Training
│   │   ├── HGG
│   │   ├── LGG
│   │   ├── ...
├── M2FTrans_v1
│   ├── ...
└── M2FTrans_v2
    └── ...
```

## Training

**M2FTrans_v1**

- Changing the paths and hyperparameters in ```train.sh```, ```train.py``` and ```predict.py```.

- Set different splits for BraTS2018 in ```train.py```.

- Then run:

  ```bash
  bash train.sh
  ```

**M2FTrans_v2**

- Changing the paths and hyperparameters in ```config.yml```, ```train.py```, and ```predict.py```.

- Then run:

  ```bash
  python train.py
  ```

## Evaluation

Checking the relevant paths in path in ```eval.sh``` or ```eval.py```.

**M2FTrans_v1**

```bash
bash eval.sh
```

**M2FTrans_v2**

```bash
python eval.py
```

## Acknowledgement

The implementation is based on the repos: [RFNet](https://github.com/dyh127/RFNet), [mmFormer](https://github.com/YaoZhang93/mmFormer) and [SMU-Net](https://github.com/rezazad68/smunet).
