# BKAI IGH NeoPolyp Challenge

## Overview

The BKAI-IGH NeoPolyp-Small dataset is a publicly available dataset curated by the BKAI Research Center in collaboration with the Institute of Gastroenterology and Hepatology (IGH) at Vietnam's Hanoi University of Science and Technology. This dataset is designed for research in medical imaging analysis, particularly focused on polyp detection in endoscopic images.

### Dataset

- **Total Images**: 1200 images (1000 WLI and 200 FICE images)
- **Training Set**: 1000 images
- **Test Set**: 200 images

### Image Classes

- **Neoplastic**: Polyps denoted by red color
- **Non-Neoplastic**: Polyps denoted by green color

### Data Collection and Annotation

- All images were collected at IGH (Institute of Gastroenterology and Hepatology).
- Annotations (segmentation and classification) were added and verified by two experienced endoscopists from IGH.

## Result

The result evaluated on the public dataset is currently **0.76958**

## Configuration

This project utilizes the following libraries (all are mentioned in requirements.txt):

- **torchsummary**
- **wandb** (version 0.16.0)
- **torchgeometry** (version 0.1.2)
- **torch** (version 2.0.0)
- **tqdm** (version 4.65.0)
- **timm** (version 0.9.10)
- **opencv-python** (version 4.8.1.78)
- **numpy** (version 1.22.3)
- **pandas** (version 1.4.2)
- **torchvision** (version 0.15.1)
- **Pillow** (version 9.5.0)
- **albumentations** (version 1.3.1)
- **gdown**

## Pretrained weight of UNet using backbone as Resnet152
```bash
https://drive.google.com/file/d/17IEmiObweFG1a7U8-l5zsl_-6aychTrK/view?usp=share_link
```

## Usage Instructions

### Setup Process

#### Clone this repository on Kaggle

1. Open a Kaggle Notebook.
2. Use the following commands in a code cell:

```bash
!git clone https://github.com/namhainguyen2803/BKAI-IGH-NeoPolyp-Challenge.git
%cd /kaggle/working/BKAI-IGH-NeoPolyp-Challenge
```

#### Clone this repository on Google Colab

1. Open a Google Colab Notebook.
2. Use the following commands in a code cell:

```bash
!git clone https://github.com/namhainguyen2803/BKAI-IGH-NeoPolyp-Challenge.git
%cd /content/BKAI-IGH-NeoPolyp-Challenge
```

### Required dependencies installation

```bash
!pip install -r requirements.txt
```

### Training script

```bash
!python3 train.py --epochs 300
```


### Evaluation script

```bash
!python3 infer.py
```













