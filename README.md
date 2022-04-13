# ✅ Fact-checking ⁉️

This repository contains a project realized for an assignment of the *Natural Language Processing* course of the [Master's degree in Artificial Intelligence](https://corsi.unibo.it/2cycle/artificial-intelligence), University of Bologna.

## Description

*Fact checking* is a popular NLP task which consists in verify the reliability of some statement, by comparing it with a given knowledge base. In this experiment, we’ll use the FEVER dataset to train a model able to understand whether a fact is verifiable. The model consists in a neural network, structured in different ways, working on embeddings obtained with GloVE. A voting mechanism is then required to make predictions.

## Dataset
The [FEVER dataset](https://fever.ai) is about facts taken from Wikipedia documents that have to be verified. In particular, facts could face manual modifications in order to define fake information or to give different formulations of the same concept.

The dataset consists of 185,445 claims manually verified against the introductory sections of Wikipedia pages and classified as ```Supported```, ```Refuted``` or ```NotEnoughInfo```. For the first two classes, systems and annotators need to also return the combination of sentences forming the necessary evidence supporting or refuting the claim.

An already pre-precessed version of the dataset is been used, in order to concentrate on the classification pipeline (pre-processing, model definition, evaluation and training).

## Request and solution proposed
The task to comply with is described in the [assignment description](./Assignment_2.ipynb).
In order to have a better understanging of our proposed solution, take a look to the [notebook](./FactChecking.ipynb) and the [report](./report.pdf).

## Model
What the neural network does is to encode two different inputs (claim and evidence), merge them in some way, and output a single value, representing the probability that the claim is correct. The following simplified schema shows the model architecture:


![](https://drive.google.com/uc?export=view&id=1Wm_YBnFwgJtxcWEBpPbTBEVkpKaL08Jp)

Two different models are been trained: 
- a base one that use the last state of a LSTM layer for the sentence embedding and combine the two sentences through an Add layer
- the other one is just an extension using the same configuration of the previous and adding the cosine similarity between claim and evidence to the input of the network.

The trained models are available at the following [link](https://drive.google.com/drive/folders/1Ec8iRzONGmZ7Z9J8dCs4x-mVxei5wOgO?usp=sharing).

## Results
Two evaluation strategies have been used: multi-input evaluation (standard approach in classification) and claim evaluation (majority voting).
As seen in the table below, performances were better for the extended model.

|       Dataset split      | Accuracy base model | Accuracy extension model |
|:------------------------:|:----------:|:---------------:|
|      Validation set      |   0.7609   |     0.7611      |
|    Test set (normal)     |   0.735    |      0.743      |
| Test set (majority vote) |   0.710    |      0.718      |


This is an example of the prediction made by the extended model on a test set of pairs claim-evidence:
```
CLAIM:  Scream has some level of success.
EVIDENCES:
	1. (SUPPORTS) The first series entry , Scream , was released on December 20 , 1996 and is currently the highest-grossing slasher film in the United States 
	2. (SUPPORTS) It received several awards and award nominations 
	3. (SUPPORTS) The film went on to financial and critical acclaim , earning $ 173 million worldwide , and became the highest-grossing slasher film in the US in unadjusted dollars 
PREDICTION:
	1. SUPPORTED (Confidence 98%)
	2. SUPPORTED (Confidence 99%)
	3. SUPPORTED (Confidence 98%)
```

## Resources & Libraries

* NLTK
* Tensorflow + Keras



## Versioning

We use Git for versioning.



## Group members

| Reg No. |   Name    |  Surname  |                 Email                  |                       Username                        |
| :-----: | :-------: | :-------: | :------------------------------------: | :---------------------------------------------------: |
| 1005271 | Giuseppe  |   Boezio  | `giuseppe.boezio@studio.unibo.it`      | [_giuseppeboezio_](https://github.com/giuseppeboezio) |
|  983806 | Simone    |  Montali  |    `simone.montali@studio.unibo.it`    |         [_montali_](https://github.com/montali)         |
|  997317 | Giuseppe  |    Murro  |    `giuseppe.murro@studio.unibo.it`    |         [_gmurro_](https://github.com/gmurro)         |



## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details
