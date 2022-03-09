#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# In[2]:


import tokenization

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


# In[3]:

# Parse arguments here.
parser = argparse.ArgumentParser(description="Relation Extraction model that takes in wikipedia tables entity and finds relations between them.\n" \
                                             "The model uses a CNN to encode input entities and surrounding table information followed by an" \ 
                                             "LSTM/BiLSTM to learn dependencies among words in the input.")
parser.add_argument('model_lstm', metavar='lstm', type=str)
parser.add_argument("")


# In[4]:


device_name = tf.test.gpu_device_name()
print(device_name)
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[5]:


train_data = pd.read_csv('Processed_Input 2.tsv', encoding='utf-8', sep = '\t')


# In[6]:


train_data.fillna("", inplace = True)


# In[7]:


# Shuffle data so that there is a higher chance of the train and test data being from the same distribution.
train_data = shuffle(train_data, random_state = 1)


# In[8]:


# Now read the rows, convert them into strings and then only keep the unique ones.
sentences_and_lables = np.array([[' '.join(map(str, row[:-1].tolist())).strip(), row[-1]] for row in train_data.iloc[:,:].values])
print(sentences_and_lables.shape)
sentences = sentences_and_lables[:, 0]
#labels = sentences_and_lables[:, 1]


# In[9]:


label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['relation'])
# y = to_categorical(y) # doing this later.


# In[10]:


print(y[:5])


# In[11]:


print(sentences[:5])


# In[12]:


m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=False)


# In[13]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[14]:

def build_model_bilstm(bert_layer, max_len=512):
    """
    The
    """


def build_model(bert_layer, max_len=512, bidirectional=False):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    print(tf.shape(sequence_output))
    clf_output = sequence_output[:, :, :]
    print(tf.shape(clf_output))
    
    lay = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding="same", activation="relu")(clf_output)
    lay = tf.keras.layers.MaxPooling1D(2, 2)(lay)
    if bidirectional:
        forward_lstm = tf.keras.layers.LSTM(2, return_sequences=True, dropout=0.2)(lay)
        backward_lstm = tf.keras.layers.LSTM(2, return_sequences=True, dropout=0.2, go_backwards=True)(lay)

        # NOTE: The backward_layer is an optional argument. If not supplied, the forward_lstm layer will be automatically used as a backward layer with go_backwards=True
        lay = tf.keras.layers.Bidirectional(forward_lstm, backward_layer=backward_lstm)
    else:
        lay = tf.keras.layers.LSTM(2, return_sequences=True, dropout=0.2)(lay)
    lay = tf.keras.layers.Flatten()(lay)
    out = tf.keras.layers.Dense(6, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    #model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# ### Obtaining Train, test splits.
# ###### In the train splits, we will have a separate validation split.

# In[15]:


checkpoint_path = "training_relations/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# In[16]:


def get_labels(y_pred):
    y_pred_label = np.zeros((len(y_pred),1))
    print(y_pred_label.shape)
    for index in range(len(y_pred)):
        y_pred_label[index] = np.argmax(y_pred[index])
    return y_pred_label


# In[ ]:


with tf.device(device_name):
    print(f"Training on {device_name}.")
    splits = 5 # For five fold cross-validation.
#     seeds = [i for i in range(splits)]  # Fix the seed value for reproducibility.
    seeds = [2]

    val_dict = {}
    test_dict = {}

    # First get random train-test splits. Doesn't include validation, which will be obtained from the train set.
    for seed in seeds:
        x_t, x_test, y_t, y_test = train_test_split(sentences, y, random_state=seed, test_size=0.2)   # Global training and test sets.

        # Now get validation sets from each training set.
        kf = KFold(n_splits=5, shuffle=False) # Setting shuffle=False because shuffled dataset already before.
        fold_count = 0

        for train_index, val_index in kf.split(x_t):
            #print(x_t.shape)
            #print(y_t.shape)
            x_train, x_val = x_t[train_index], x_t[val_index]   # Training and validation features.
            y_train, y_val = y_t[train_index], y_t[val_index]   # Training and validation labels.

            #encode train data
            max_len = 80
            train_input = bert_encode(x_train, tokenizer, max_len=max_len)
            train_labels = y_train
            x_val = bert_encode(x_val, tokenizer, max_len=max_len)

            # Change the LSTM to BiLSTM by setting bidirectional=True.
            model = build_model(bert_layer, max_len=max_len, bidirectional=True)
            model.summary()
            #checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
            train_sh = model.fit(
            train_input, train_labels,
            #validation_split=0.2,
            validation_data=(x_val, y_val),
            epochs=2,
            callbacks=[checkpoint, earlystopping],
            batch_size=4,
            verbose=1)


            # Validation sets can be used for hyperparamter tuning.  
            val_dict[str(seed) + str(fold_count)] = train_sh.history

        #encode whole train data
        max_len = 80
        train_input = bert_encode(x_t, tokenizer, max_len=max_len)
        train_labels = y_t
        
        #train model on whole train data
        model = build_model(bert_layer, max_len=max_len)
        model.summary()
        #checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
        train_sh = model.fit(
        train_input, train_labels,
        epochs=2,
        callbacks=[checkpoint, earlystopping],
        batch_size=4,
        verbose=1)

        #encode test data 
        test_input = bert_encode(x_test, tokenizer, max_len=max_len)

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data for ", seed)
        results = model.evaluate(test_input, y_test, batch_size=4)

        #calculate F1-score
        y_pred = model.predict(test_input, verbose=1)
        y_pred_label = get_labels(y_pred)
        f1_val = f1_score(y_test, y_pred_label, average='macro')
        print(f1_val)
        results.append(f1_val)
        print("test loss, test acc, F1-score:", results)

        test_dict[seed] = results


# In[ ]:


# print(f1_val)


# In[19]:


with open('results.txt','w') as data: 
    data.write(str(val_dict))
    data.write(str(test_dict))

