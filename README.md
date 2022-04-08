# Relation-Extract-CNN-Mem-Net


### This repo contains the code for our project in CMPUT 656.
Relation Extraction using Convolution and Memory Networks.
The inputs to the models are the entities obtained from the table. The output is an embedding which is fed to a softmax layer for multi-class classification.



### Install the following packages.
1. Tensorflow >=2.5.0
2. sentencepiece
3. TensorflowHub

The packages above are in addition to the generic python data science stack (numpy, scipy, pandas etc.).

Follow pip installation guidelines. If using Anaconda package manager, use conda to install packages, but generally pip should work.

### Instructions to run the code.

Run main.py as follows.
```
python main.py
```

### Hyperparameter comparison

| Hyperparameter              | CoMemNet | Macdonald and Barbosa (2020) |
|:---------------------------:|:--------:|:----------------------------:|
| CNN Filters                 | 8        | None                         |
| LSTM/BiLSTM units           | 8        | 1 (only LSTM)                |
| Batch Size                  | 16       | 16                           |
| Optimizer                   | Adam     | RMSProp                      |
| Max Token Length            | 80       | 50                           |
| Learning Rate               | 2e-5     | 0.001                        |
| Dropout (for LSTM / BiLSTM) | 0.2      | None                         |



### Results
Results shown for 5 seeds.

| CNN Filters                 | Accuracy | F1     | #Relations (#Tables) | Epochs |
|:---------------------------:|:--------:|:------:|:--------------------:|:------:|
| Macdonald and Barbosa 2020  | 92%      | 95%    | 29 (All)             | 50     |
| Macdonald and Barbosa 2020* | 91.05%   | 84.80% | 29 (Max 500)         | 6      |
| CoMemNet-LSTM               | 97.51%   | 95.41% | 29 (Max 500)         | 6      |
| CoMemNet-BiLSTM             | 97.89%   | 96.03% | 29 (Max 500)         | 6      |
| BiLSTM (8 units)            | 98.97%   | 97.97% | 29 (Max 500)         | 6      |
*Reimpleted for fair comparison.


~~TODO:
- [ ]Refractor code to have the preprocessing steps in a separate python file and the keras model in the main.py only.
- [ ] Add command line arguments to run different cases (for example: load pretrained proposed model to predict relations, retrain proposed model from scratch with validation etc., also provide options for providing different architecture hyperparamters - different number of conv filters, different LSTM units (but also have a default option)).
- [ ] Calculate F1 score on the test data.~~

~~### Questions
- [ ] Fine-tuning BERT provides slight acc benefit on val data but this may be important if the dataset is huge - translates to many more relations.
- Try without fine-tuning as well.~~