# Are Language Models Performing Reasoning?

## Introduction

This is the code repository associated with Wenting and Seth's final project for CS6741. In this project, we explore whether language models are actually performing reasoning when being applied to reasoning tasks. We are particularly interested in multi-hop reasoning; some specific questions we asked are if training on lower-hop questions can be generalized to answering higher-hop examples, how are different layers of the model contribute to understanding a reasoning question, etc.

### What's written by us & what's the part we took from other resources

- From others:
    - Training and testing ROBERTa-large
    - BPE encoding of a dataset
- Written by us:
    - Processing JSON datasets
    - All experiments presented in our report including training a probe on different datasets

## Preparing Datasets

To download: [RuleTaker](https://rule-reasoning.apps.allenai.org/about) and [ProofWriter](https://allenai.org/data/proofwriter).

