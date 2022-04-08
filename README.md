# Relation-Extract-CNN-Mem-Net


### This repo contains the code for our project in CMPUT 656.
Relation Extraction using Convolution and Memory Networks.
The inputs to the models are the entities obtained from the table. The output is an embedding which is fed to a softmax layer for multi-class classification.



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
Download [this](https://drive.google.com/drive/folders/1I_pwygMoS7xofFVMSXwRgsrAMiUUh8T9?usp=sharing) folder from Google Drive. It is a big folder (~few gigabytes). As the default behavior uses the pretrained model, you will need the checkpoints files.

Run python main.py -h to see the list of arguments.
```
python main.py -h
usage: main.py [-h] [-m MODEL] [-d DEMO] [-p PRETRAINED]

Train the proposed model. An example on how to run the script is as follows:
python main.py --model=comemnet-bilstm

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Can be either 'comemnet-lstm', 'comemnet-bilstm' or
                        'erin'. Example: model=comemnet-bilstm
                        (default=comemnet-bilstm)
  -d DEMO, --demo DEMO  Boolean. Whether to create a gradio demo
                        (default=False).
  -p PRETRAINED, --pretrained PRETRAINED
                        Boolean. Whether to use pretrained model.
                        (default=False). If pretrained=True, please download
                        the google drive files mentioned in the README.
```
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
 
 
### Trainable parameters comparison

| Model                                | Parameters |
|:------------------------------------:|:----------:|
| Macdonald and Barbosa* (1 LSTM only) | 4,559      |
| CNN + LSTM (CoMemNet - LSTM)         | 40,581     |
| CNN + BiLSTM  (CoMemNet - BiLSTM)    | 50,405     |
| BiLSTM only                          | 86,877     |


\* Reimpleted for fair comparison.



### Results
Results shown for 5 seeds.

| CNN Filters                 | Accuracy | F1     | #Relations (#Tables) | Epochs |
|:---------------------------:|:--------:|:------:|:--------------------:|:------:|
| Macdonald and Barbosa 2020  | 92%      | 95%    | 29 (All)             | 50     |
| Macdonald and Barbosa 2020* | 91.05%   | 84.80% | 29 (Max 500)         | 6      |
| CoMemNet-LSTM               | 97.51%   | 95.41% | 29 (Max 500)         | 6      |
| CoMemNet-BiLSTM             | 97.89%   | 96.03% | 29 (Max 500)         | 6      |
| BiLSTM (8 units)            | 98.97%   | 97.97% | 29 (Max 500)         | 6      |

\* Reimpleted for fair comparison.