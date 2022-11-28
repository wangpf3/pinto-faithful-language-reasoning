# PINTO  
This is a Pytorch implementation for our recent work under review: 
Faithful Language Reasoning Using Prompt-Generated Rationales [[arxiv](https://arxiv.org/abs/2211.01562)].

## Prepare data

We have provided the datasets augmented with prompt-generated rationales. Just untar the file `data.tar.gz`. Since our data is based on existing benchmarks, please cite their works if you use the data.

Alternaltively, you can use the code for generating rationales `rationalization_prompting.py` to prepare the rationales for your own datasets. 

## Training
Run the script `run.sh`. Change the `dataset` argument to specify the dataset for experiment. After training, the evaluation result is saved to `./checkpoint`.

## Citation
```
@article{wang2022pinto,
  title={PINTO: Faithful Language Reasoning Using Prompt-Generated Rationales},
  author={Wang, Peifeng and Chan, Aaron and Ilievski, Filip and Chen, Muhao and Ren, Xiang},
  journal={arXiv preprint arXiv:2211.01562},
  year={2022}
}
```
