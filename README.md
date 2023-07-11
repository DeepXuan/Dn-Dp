# Dn-Dp

This repository contains the official implementation of the paper "Diffusion Probabilistic Priors for Zero-Shot Low-Dose CT Image Denoising" by [Xuan Liu et al.] https://arxiv.org/abs/2305.15887. The paper introduces a novel approach for denoising low-dose CT images using diffusion priors.

## Test Environment
- OS: Windows 11
- GPU: NVIDIA RTX 3090
- Python (=3.9)
- Pytorch (=1.13.0)
- Torchvision (=0.14.0)
## Requirements
```
pip install -r requirements.txt
```
## Pre-trained Models
We share our pre-trained cascaded diffusion models. Please download from https://drive.google.com/drive/folders/1sHWtDlUCO-4cb-v_ijR1i3c2PYqB44xs?usp=sharing.

## How to run
### 1. Run for Test data
We prepare 2 abdomen CT images and 2 Chest CT images in folder ./test_data. Run the following commands for testing.

```python
python denoising.py -c ./config/Dn_liver_128.yaml # Denoise low-resolution abdomen CT images 
```
```python
python denoising.py -c ./config/Dn_liver_128_512.yaml # Denoise high-resolution abdomen CT images with denoised low-resolution images as conditions  
```
```python
python denoising.py -c ./config/Dn_Chest_256.yaml # Denoise low-resolution chest CT images 
```
```python
python denoising.py -c ./config/Dn_Chest_256_512.yaml # Denoise low-resolution chest CT images with denoised low-resolution images as conditions  
```
### 2. Run for your own data
Place your test data in sub-folders of a root-folder.
```
root 
├── input_folder # Put your test images here.  
├── output_folder # Results will be saved here.
├── target_folder # Required if you need to calculate metrics else None.
└── cond_folder # Required if use a conditional diffusion model else None.
```
Then modify the corresponding parameters in the YAML file.
``` yaml
root: root
  input_folder: input_folder
  cond_folder: cond_folder
  output_folder: output_folder
  target_folder: target_folder
```
Specify the diffusion models used in the YAML file.
```yaml
model:
  cfg_path: ./config/sample_sr3_128_Liver.json # unconditional diffusion for abdomen CT images of 128x128
  pretrained_path: ./Pretrained/Liver_128.pth # path of pre-trained model

  cfg_path: ./config/sr_sr3_128_512_Liver.json # conditional diffusion for abdomen CT images of 128x128-->512x512
  pretrained_path: ./Pretrained/Liver_128_512.pth

  cfg_path: ./config/sample_sr3_256_Chest.json # unconditional diffusion for chest CT images of 256x256
  pretrained_path: ./Pretrained/Chest_256.pth

  cfg_path: ./config/sr_sr3_256_512_Chest.json # conditional diffusion for chest CT images of 256x256-->512x512
  pretrained_path: ./Pretrained/Chest_256_512.pth
            
```
You can also change other denoising parameters in the YAML file. 
```yaml
data:
  res: 128 # the input images will be first resized to (res, res) befor denoising
  len: -1 # -1 for testing all images in the input folder
diffusion:
  ddim: True
  ddim_eta: 1.0
dn:
  # mode==0: ConsLam
  # mode==1: AdaLamI 
  # mode==2: AdaLamII
  # mode==3: AdaLamI&II
  mode: 0
  lam0: 0.002 # ConsLam: $\lambda_t=\lambda_0\cdot\sqrt{\bar{\alpha}_t}$
  a: 0 
  b: 0 # AdaLamI: $\lambda_0^{ada} = a \cdot std(\widehat{n}) + b$
  c: 0 # AdaLamII: $\Lambda_0^{ada} = c \cdot \widehat{n}$
  resume: 3
  mean_num: 10 # multiple denoising and take average, reduce mean_num if CUDA_OUT_OF_MEMORY
  bs: 10 # batch size
```
If your image files are not .png format of uint16, please change the two functions in denosing.py to read and save your own data.
```python
import imageio.v2 as imageio
def read_a_img(path): # function of read one image file from $path$ --> np.ndarray of range [0.0, 1.0]
    return imageio.imread(path) / 65535. 

def save_a_img(img, path): # function of save $img$ (np.ndarray of range [0.0, 1.0]) to $path$
    return imageio.imwrite(path, (img*65535).astype(np.uint16))
```

At last, run denoising!
```python
python denoising.py -c YourYAML.yaml
```
## Acknowledgment
Big thanks to the repository https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement for providing the codes that facilitated the training of the cascaded diffusion models.
