# Are Language Models Performing Reasoning?

## Introduction

This is the code repository associated with Wenting and Seth's final project for CS6741. In this project, we explore whether language models are actually performing reasoning when being applied to reasoning tasks. We are particularly interested in multi-hop reasoning; some specific questions we asked are if training on lower-hop questions can be generalized to answering higher-hop examples, how are different layers of the model contribute to understanding a reasoning question, etc. We note that we assume access to GPUs in order to run all the following code.

### What's written by us & what's the part we took from other resources

- From others:
    - Training and testing ROBERTa-large
    - BPE encoding of a dataset
- Written by us (any file we talked about usage below is by us):
    - Processing JSON datasets
    - All experiments presented in our report including training a probe on different datasets

## Fine-tuning on RACE

We follow the instruction from [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.race.md). To fine-tune race:
```
bash fine_tune_race.sh
```
To evaluate after fine-tuning:
```
bash evaluate_race.sh
```

## Preparing Datasets

To download: [RuleTaker](https://rule-reasoning.apps.allenai.org/about) and [ProofWriter](https://allenai.org/data/proofwriter).

To process these datasets:
```
cd proofwriter-dataset-V2020.12.2/OWA/
python process_proof.py {in_dir} {out_dir} # to get D<={x} datasets, where x=0,1,2,3,5
python process_multi_proof.py {in_dir0} {in_dir1} .. {out_dir} # to get a D={x} dataset where the examples could be from all the D<={x} datasets
python extract_intermediate.py {in_dir} {out_dir} # to construct a dataset where they are all the intermediate steps from one of the D={x} datasets
python process_proof_depth.py {in_dir} {out_dir} # to construct a dataset where inputs are questions and outputs are the steps required to answer the questions.
```

To prepare these datasets to be fed to the model:
```
bash encode_rule.sh      # need to make sure the directories in the file are set correctly 
bash preprocess_rule.sh  # need to make sure the directories in the file are set correctly
```

## Fine-tune and Evaluate ROBERTa-large

To fine-tune ROBERTa-large on one of the prepared datasets:
```
bash train_ruletaker.sh # need to make sure the directories and head name in the file are set correctly
```

To evaluate ROBERTa-large on one of the prepared datasets:
```
bash evaluate_rule.sh # need to make sure the directories in the file are set correctly
```

## Extract CLS embedding from each layer and Train a Probe

To extract CLS token embedding from all layers and train probes
```
python probe.py {dataset_trained_on} {dataset_to_be_evaluated} {model_checkpoint_path} {model_trained_data_path} {dataset_to_run_probe} {model_head_name} {number_of_layers_to_probe_on} {out_dim}
```

## Others

We have also developed scripts to draw figures we included in our final report. Simply run
```
python draw_exps.py
python draw_exps2.py
```
