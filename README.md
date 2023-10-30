#  Pre-Trained CNN Architecture Analysis for Transformer-Based Image Captioning

We conduct a comprehensive analysis of cutting-edge image captioning techniques, where our experimentation takes place within the domain of the Flickr dataset using Transformer model as the decoder.

The paramount finding that emerges from these investigations is the remarkable performance improvement achieved by fine-tuning the encoder.


## Preparation

1. Download the Dataset:

Image: https://drive.google.com/drive/folders/10JDnoTQK-ZnE93lgp8D6iJajIPRLL5gq?usp=sharing

Annotation: https://drive.google.com/drive/folders/1QgSdzriDrvpKcUcooebmKl603qubkBbe?usp=share_link

2. Commands needed for preparation

```
!pip install -r requirements.txt
!pip install "git+https://github.com/salaniz/pycocoevalcap.git"

```

3. Vocab file generation:
```
!python vocab_builder.py
```

## Running the Experiments
Please run this code inside your Google Colab Cells to reproduce the results like in the paper, after vocabulary is generated (replace x with parameters value needed).

```
!python train.py --encoder-type encodername --decoder-type transformer --num-heads x --num-tf-layers x --beam-width x --batch-size x --batch-size-val x --num-epochs x --experiment-name experimentname
```

For example (using VGG16):

```
!python train.py --encoder-type vgg16 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --beam-width 3 --batch-size 64 --batch-size-val 16 --num-epochs 50 --experiment-name nv_vgg16_tf_h1_l3_bs64_16_ep50
```

## Results Notebooks

https://colab.research.google.com/drive/11jZAiZuLmOEq1VusyvAB0_7-XIgxQ-dg?usp=sharing

## Python Framework

* [PyTorch](https://pytorch.org/)