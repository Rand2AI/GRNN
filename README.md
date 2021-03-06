# GRNN: Generative Regression Neural Network - A Data Leakage Attack for Federated Learning

## Introduction

This is the implementation of the paper "GRNN: Generative Regression Neural Network - A Data Leakage Attack for Federated Learning". In this paper, we show that, in Federated Learning (FL) system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimise two branches of the generative model by minimising the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model.

<div align=center><img src="https://github.com/Rand2AI/GRNN/blob/main/images/GRNN.Details.png"/></div>

## Requirements

python==3.6.9

torch==1.4.0

torchvision==0.5.0

numpy==1.18.2

tqdm==4.45.0

...

## Examples

<div align=center><img src="https://github.com/Rand2AI/GRNN/blob/main/images/Examples.png"/></div>

## Performance

<div align=center><img src="https://github.com/Rand2AI/GRNN/blob/main/images/Results.png"/></div>

## How to use

### Prepare your data:

 * Download LFW, VGGFace or ILSVRC datasets online respectively and extract them to ./Data/.
    
 * MNIST and CIFAR-100 can be downloaded automatically when you first run the script.

### Train GRNN and recover data, run:

    python GRNN.py

### Notes:

* If only one GPU is available, please set:

      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
      device0=0
      device1=0

* Recovered images are all saved in:

      ./Results/

* No model is saved to local.

## Citation

If you find this work helpful for your research, please cite the following paper:

    @article{10.1145/3510032,
             author = {Ren, Hanchi and Deng, Jingjing and Xie, Xianghua},
             title = {GRNN: Generative Regression Neural Network - A Data Leakage Attack for Federated Learning},
             year = {2022},
             publisher = {Association for Computing Machinery},
             address = {New York, NY, USA},
             issn = {2157-6904},
             url = {https://doi.org/10.1145/3510032},
             journal = {ACM Trans. Intell. Syst. Technol.}
            }

## Acknowledgement

We used the code part from DLG (https://github.com/mit-han-lab/dlg). Thanks for their excellent work very much.
