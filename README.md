# PINTO  
This is a Pytorch implementation for our ICLR 2023 paper: 
Faithful Language Reasoning Using Prompt-Generated Rationales [[arxiv](https://arxiv.org/abs/2211.01562)].

## Prepare data

We have provided the datasets augmented with prompt-generated rationales. Just untar the file `data.tar.gz`. Since our data is based on existing benchmarks, please cite their works if you use the data.

Alternaltively, you can use the code for generating rationales `rationalization_prompting.py` to prepare the rationales for your own datasets. 

## Training
Run the script `run.sh`. Change the `dataset` argument to specify the dataset for experiment. After training, the evaluation result is saved to `./checkpoint`.

## Citation
```
@inproceedings{
wang2023pinto,
title={{PINTO}: Faithful Language Reasoning Using Prompted-Generated Rationales},
author={PeiFeng Wang and Aaron Chan and Filip Ilievski and Muhao Chen and Xiang Ren},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=WBXbRs63oVu}
}
```
