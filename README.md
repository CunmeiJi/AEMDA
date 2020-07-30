# AEMDA
Implementation of AEMDA for inferring potential disease-miRNA associations. Our method contains three sub-models: a disease model for learning representation of diseases, a miRNA model for learning representation of miRNA, and an autoencoder based model for predicting. 

# Requirements
  * PyTorch 1.0 or higher
  * GPU (default).

# Usage
  * download code and data
  * execute ```python main.py``` to train a predictor
  

Note: If you wanna infer a disease-specific miRNAs, you should use the concatenated vector $[d, m_i]$ as the input of the predictor and get the reconstruction error, this process repeat $nm$ times, while $nm$ is the number of miRNAs. Then, sort all candidates and you can analyze for further biological experiments.

# Cite
Please cite our paper if you use this code in your own work:
```
@article{Ji2020,
author = {Ji, Cunmei and Gao, Zhen and Ma, Xu and Wu, Qingwen and Ni, Jiancheng and Zheng, Chunhou},
doi = {10.1093/bioinformatics/btaa670},
issn = {1367-4803},
journal = {Bioinformatics},
month = {jul},
title = {{AEMDA: Inferring miRNA-disease associations based on deep autoencoder}},
url = {https://doi.org/10.1093/bioinformatics/btaa670},
year = {2020}
}
```
