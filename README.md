
---

<div align="center">    

# PROST: Physical Reasoning about Objects through Space and Time
<!-- TODO: Add Arxiv and ACL Findings links 
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)]()
[![ACLFindings](http://img.shields.io/badge/ACLFindings-2021-4b44ce.svg)]()
-->
</div>

## Description 
Code and dataset for "PROST: Physical Reasoning about Objects through Space and Time". This repository is not maintained, but is here for posterity.

## How to Use

### Using PROST
If you'd like to use PROST outside the scope of this repo, we highly recommend using the version hosted on the [Huggingface Hub](https://huggingface.co/datasets/corypaik/prost) as it requires no additional dependencies.

```python
from datasets import load_dataset

dataset = load_dataset('corypaik/prost', split='test')
```

You can find more details about how to use Huggingface Datasets [here](https://github.com/huggingface/datasets).

**Note:** PROST is also implemented as a Huggingface Dataset in [`/src/prost`](/src/prost), which will generate the same data, but builds the dataset directly from yaml templates and requires some extra dependencies. 

### Reproducing experiments

First, clone the project 
```bash
# clone project
git clone https://github.com/nala-cub/prost

# goto project
cd prost
```
This project optionally uses [Bazel](https://docs.bazel.build/versions/4.0.0/install.html). Please see the provided links for install instructions relevant to your OS if you wish to use Bazel. 

Run with Bazel:
```bash
bazel run //src:run
```

If you don't have Bazel installed, you can install dependencies via pip.
```bash
virtualenv venv
. venv/bin/activate
# install all pip requirements
pip install -r tools/no_bzl_requirements.txt
pip install -r tools/requirements.txt
```

Run experiments
```bash
cd src 
python run.py 
```

### Results
The location of results is dependent on how you set up and run this repository. If ran with Bazel, all results will be under [`_bazel/out/results`](/_bazel/out/results). Otherwise, the results should be under `/tmp/prost-results`.

## Citation 
If this code was useful, please cite the paper:

```
@inproceedings{aroca-ouellette-etal-2021-prost,
    title = "{PROST}: {P}hysical Reasoning about Objects through Space and Time",
    author = "Aroca-Ouellette, St{\'e}phane  and
      Paik, Cory  and
      Roncone, Alessandro  and
      Kann, Katharina",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.404",
    pages = "4597--4608",
}
```

## License
PROST is licensed under the Apache 2.0 license. The text of the license can be found [here](LICENSE).