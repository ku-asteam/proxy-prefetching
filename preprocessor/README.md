# Preprocessor

Preprocessor for web prefetching

## Introduction

Preprocessor reads the data set which is logs of client's requests and prepares for encoding. It collects basic [LINK, ID, TIME] informations from the raw data, and builds link sequences data_set.

## Requirements and Dependencies

* *An UNIX-like Operating System* 
* Python3
* Predicting parts are supported by `Numpy`, `sklearn`, `pytorch` or ` tensorflow`; 
* The visualizing part are supported by `matplotlib`

## Instruction

* How to run
    1. `python preprocessor.py`