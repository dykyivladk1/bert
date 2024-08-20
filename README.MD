# BERT


<img src="assets/bert.png" style="width: 100%; height: 20%;" alt="Image">


#### BERT (Bidirectional Encoder Representations from Transformers) 


This repository contains a simplified implementation of the original BERT model, it demonstrated the core concepts of the BERT architecture, including Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)


## Introduction 

BERT is a powerful model designed to understand the context of words in a sentence by considering both the left and the right context, that is why it is called Bidirectional. This repository provides a hands-on approach to learning how BERT works by implementing a simplified version that retains the essential functionalities. The only differences are: the number of layers used, different optimizer, and the lack of tokenizer.


## Features

- **Masked Language Modeling (MLM):** Predict masked tokens within a sentence.
- **Next Sentence Prediction (NSP):** Determine whether one sentence logically follows another.


## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm


## Installation 

Clone the repository and install the dependecies:

```bash
git clone 
cd bert
pip install -r requirements.txt
```

## Usage 

You can use the provided scripts to train and test the BERT model on the included small dataset.

## Training

Run the training script to train the BERT model:

```bash
python src/train.py
```

In order to change the hyperparameters, please edit `config/config.yaml`.

## Testing 

Run the testing script to evaluate the BERT model on new data:

```bash
python test.py
```


