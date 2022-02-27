# Relation-Extract-CNN-Mem-Net


### This repo contains the code for our project in CMPUT 656.
Relation Extraction using Convolution and Memory Networks.
The inputs to the models are the entities obtained from the table. The output is an embedding which is fed to a softmax layer for multi-class classification.
<<Provide description for the project. E.g., inputs, ouputs, methods, etc.>>


### Install the following packages.
1. Tensorflow >=2.5.0
2. sentencepiece
3. TensorflowHub

The packages are in addition to the generic python data science stack (numpy, scipy, pandas etc.).

Follow pip installation guidelines. If using Anaconda package manager, use conda to install packages, but generally pip should work.

### Instructions to run the code.

Run main.py as follows.
```
python main.py
```

### Results

| CNN Filters | LSTM Units | BERT       | Test Accuracy |
|:-----------:|:----------:|:----------:|:-------------:|
| 16          | 4          | Pretrained |               |
| 8           | 2          | Pretrained |               |
|             |            |            |               |




TODO: 
- [ ] Refractor code to have the preprocessing steps in a separate python file and the keras model in the main.py only.
- [ ] Add command line arguments to run different cases (for example: load pretrained proposed model to predict relations, retrain proposed model from scratch with validation etc., also provide options for providing different architecture hyperparamters - different number of conv filters, different LSTM units (but also have a default option)).
- [ ] Calculate F1 score on the test data.

### Questions
- [ ] Fine-tuning BERT provides slight acc benefit on val data but this may be important if the dataset is huge - translates to many more relations.
- Try without fine-tuning as well.