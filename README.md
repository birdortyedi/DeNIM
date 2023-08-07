# DeNIM: Deterministic Neural Illuminant Mapping for Efficient Auto-White Balance Correction

![][results]

> **Deterministic Neural Illuminant Mapping for Efficient Auto-White Balance Correction**<br>
> Furkan Kınlı, Doğa Yılmaz, Barış Özcan, Furkan Kıraç <br>
> *Accepted to RCV2023 at ICCV2023* <br>
>
>**Abstract:** Auto-white balance (AWB) correction is a critical operation in image signal processors (ISPs) for accurate and consistent color correction across various illumination scenarios. This paper presents a novel and efficient AWB correction method that achieves at least 35 times faster processing with equivalent or superior performance on high-resolution images for the current state-of-the-art methods. Inspired by deterministic color style transfer, our approach introduces deterministic illumination color mapping, leveraging learnable projection matrices for both canonical illumination form and AWB-corrected output. It involves feeding high-resolution images and corresponding latent representations into a mapping module to derive a canonical form, followed by another mapping module that maps the pixel values to those for the corrected version. This strategy is designed as resolution-agnostic and also enables seamless integration of any pre-trained AWB network as the backbone. Experimental results confirm the effectiveness of our approach, revealing significant performance improvements and reduced time complexity compared to state-of-the-art methods. Our method provides an efficient deep learning-based AWB correction solution, promising real-time, high- quality color correction for digital imaging applications.

<!-- [Paper][paper] | [arXiv][arxiv] -->


## Description
The official implementation of the paper titled "Deterministic Neural Illuminant Mapping for Efficient Auto-White Balance Correction".
We propose a novel and efficient strategy for AWB correction, which learns deterministic color mappings for both canonical illumination and AWB-corrected forms with the help of learnable projection matrices.


## Requirements
To install requirements:

```
pip install -r requirements.txt
```


## Architecture
![][model]


## Updates

**8/8/2023:** Release of the code

**6/8/2023** Accepted to RCV2023 in conjunction with ICCV2023

**26/7/2023:** Submission of the paper to RCV2023 at ICCV2023


## Training

To train DeNIM from the scratch in the paper, run this command:

```
python main.py --cfg configs/train_<backbone_type>_<patch_size>_<num_channels>.yaml
```

* **Available backbones:** "style_wb", "mixed_wb"
* **Available patch sizes:** 64, 128
* **Available number of channels for different WB settings:** 9, 15

## Evaluation

To evaluate DeNIM on Cube+ dataset, run:

```
python main.py --cfg configs/test_<backbone_type>_<patch_size>_<num_channels>.yaml --is_train False
```


## Citation
```
TDB
```

## Contacts
Please feel free to open an issue or to send an e-mail to ```furkan.kinli@ozyegin.edu.tr```


[results]: images/paper/results.png
[model]: images/paper/denim.png
<!-- [paper]:  
[arxiv]:   -->