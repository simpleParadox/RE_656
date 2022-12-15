# Relation-Extraction CoMemNet 👋

**Future Updates**: ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) -> containerization for easy distribution of code. 


### Tabular Relation Extraction using Convolution and Memory Networks.

The inputs to the models are the entities and contextual information obtained from the tables and its surroundings. The embeddings go into a CNN (for extracting features) and then into LSTM/BiLSTM which is then fed to a softmax layer for multi-class classification.

Dataset used: https://doi.org/10.7939/DVN/SHL1SL


### Package installation.

First create a new conda environment using the following (make sure you have [Anaconda](https://www.anaconda.com/) installed.)
```
conda create --name comemnet python=3.8
```
```
conda activate comemnet
```
Now inside the 'comemnet' conda environment, install the following packages. Follow pip installation guidelines. If using Anaconda package manager, use conda to install packages, but generally pip should work.

Python data science stack.
```
pip install pandas
pip install numpy
python -m pip install -U matplotlib
pip install seaborn
pip install -U scikit-learn
```

1. Tensorflow >=2.5.0
```
pip install tensorflow
```
2. sentencepiece
```
pip install sentencepiece
`````````
3. TensorflowHub
```
pip install "tensorflow>=2.0.0"
pip install --upgrade tensorflow-hub
```

NOTE: If you want to create a demo for the trained model, you need gradio. No need to install if demo is not required. This is **optional**.
```
!pip install -q gradio
```


### Instructions to run the code.
Before running the code, make sure you have the pretrained model checkpoint files (if using the pretrained model).
Download [this](https://drive.google.com/drive/folders/1I_pwygMoS7xofFVMSXwRgsrAMiUUh8T9?usp=sharing) folder from Google Drive. It is a big folder (~few gigabytes). As the default behavior uses the pretrained model, you will need the checkpoint files.

```
python cmput656_full_data.py #for CNN-BiLSTM
python cnn_plus_lstm.py #for CNN-LSTM
python bilstm_only.py #for BiLSTM-only
```

### Hyperparameter comparison

| Hyperparameter              | Ours     | Macdonald and Barbosa (2020) |
|:---------------------------:|:--------:|:----------------------------:|
| CNN Filters                 | 8        | None                         |
| LSTM/BiLSTM units           | 8        | 1 (only LSTM)                |
| Batch Size                  | 16       | 16                           |
| Optimizer                   | Adam     | RMSProp                      |
| Max Token Length            | 80       | 50                           |
| Learning Rate               | 2e-5     | 0.001                        |
| Dropout (for LSTM / BiLSTM) | 0.2      | None                         |
 
 
### Trainable parameters comparison

| Model                                | Parameters |
|:------------------------------------:|:----------:|
| Macdonald and Barbosa* (1 LSTM only) | 4,559      |
| CNN + LSTM (CoMemNet - LSTM)         | 40,581     |
| CNN + BiLSTM  (CoMemNet - BiLSTM)    | 50,405     |
| BiLSTM only                          | 86,877     |



### Results
Results shown for 5 seeds.

| CNN Filters                 | Accuracy | F1     | #Relations (#Tables) | Epochs |
|:---------------------------:|:--------:|:------:|:--------------------:|:------:|
| Macdonald and Barbosa 2020  | 92%      | 95%    | 29 (All)             | 50     |
| CNN-LSTM                    | 97.57%   | 91.44% | 29 (All)             | 40     |
| CNN-BiLSTM                  | 97.80%   | 92.46% | 29 (All)             | 40     |
| BiLSTM (8 units)            | 98.19%   | 94.35% | 29 (All)             | 40     |
