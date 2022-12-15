import tokenization

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


reduced_label_mappings = {
    0: 'None',
    1: 'award-nominee',
    2: 'author-works_written',
    3: 'book-genre',
    4: 'company-industry',
    5: 'person-graduate',
    6: 'actor-character',
    7: 'director-film',
    8: 'film-country',
    9: 'film-genre',
    10: 'film-language',
    11: 'film-music',
    12: 'film-production_company',
    13: 'actor-film',
    14: 'producer-film',
    15: 'writer-film',
    16: 'political_party-politician',
    17: 'location-contains',
    18: 'musician-album',
    19: 'musician-origin',
    20: 'person-place_of_death',
    21: 'person-nationality',
    22: 'person-parents',
    23: 'person-place_of_birth',
    24: 'person-profession',
    25: 'person-religion',
    26: 'person-spouse',
    27: 'football_position-player',
    28: 'sports_team-player'
}

relations_path = "Processed Data/Input_all_29_relation.tsv"

train_data = pd.read_csv(relations_path, encoding='utf-8', sep = '\t')
train_data.fillna("", inplace = True)
train_data = shuffle(train_data, random_state = 1)

sentences = train_data.iloc[:,:-1].values.tolist()

sentences = [' '.join(sent).strip() for sent in sentences]

label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['relation'])
label_mappings = integer_mapping = {i: l for i, l in enumerate(label.classes_)}



m_url = "/home/rsaha/scratch/re_656_data/bert_en_uncased_L-12_H-768_A-12_4"
bert_layer = hub.KerasLayer(m_url, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def get_labels(y_pred):
    y_pred_label = np.zeros((len(y_pred),1))
    print(y_pred_label.shape)
    for index in range(len(y_pred)):
        y_pred_label[index] = np.argmax(y_pred[index])
    return y_pred_label

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text_num in tqdm(range(len(texts))):
        text = texts[text_num]
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

def build_model(bert_layer, max_len=512, seed=0):
    
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    mlm_inputs = dict(
    input_word_ids=input_word_ids,
    input_mask=input_mask,
    input_type_ids=input_type_ids,
)

    
    # pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    bert_layer_outputs = bert_layer(mlm_inputs)
    # print("Bert_layer outputs: ", bert_layer_outputs)
    pooled_output = bert_layer_outputs['pooled_output']
    sequence_output = bert_layer_outputs['sequence_output']
    print("Sequence output: ", sequence_output)
    #np.savez_compressed(f"bert_sequence_op_seed_{seed}.npz", sequence_output)
    
    print(tf.shape(sequence_output))
    clf_output = sequence_output[:, :, :]
    print(tf.shape(clf_output))
    input_shape = tf.shape(clf_output)
    lay = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=input_shape[1:])(clf_output)
    lay = tf.keras.layers.MaxPooling1D(2, 2)(lay)
    #lay = tf.keras.layers.LSTM(2, return_sequences=True, dropout=0.2)(lay)
    lay = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, dropout=0.2))(lay)
    lay = tf.keras.layers.Flatten()(lay)
    out = tf.keras.layers.Dense(29, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=out)
    #model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model




X_train, X_val, y_train, y_val = train_test_split(sentences, y, stratify=y, random_state=0, test_size=0.4)   # Global training and test sets.
cms = []
#code for loading weights from checkpoint
with tf.device('/device:GPU:0'):
    results = []
    splits = 5 # For five fold cross-validation.
    #seeds = [i for i in range(splits)]  # Fix the seed value for reproducibility.
    seeds = [1]#0, 1, 2, 3, 4]
    
    val_dict = {}
    test_dict = {}
    
    # First get random train-test splits. Doesn't include validation, which will be obtained from the train set.
    for seed in seeds:
        x_t, x_test, y_t, y_test = train_test_split(X_train, y_train, random_state=seed, test_size=0.33)   # Global training and test sets.

        # Evaluate the new model
        max_len = 80
        new_model = build_model(bert_layer, max_len=max_len)
        new_model.summary()
        # Loads the weights for new model
        checkpoint_path = f"best_model_weight/cnn_bilstm/seed 1/cp.ckpt"
        training_relations_path = f"best_model_weight/cnn_bilstm/seed 1/"
        

        new_model.load_weights(checkpoint_path)

        #encode test data
        test_input = bert_encode(x_test, tokenizer, max_len=max_len)

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data for ", seed, flush=True)
        # results = new_model.evaluate(test_input, y_test, batch_size=16)

        #calculate F1-score
        y_pred = new_model.predict(test_input, batch_size=16, verbose=1)
        y_pred_label = get_labels(y_pred)
        f1_value = f1_score(y_test, y_pred_label, average='macro')
        np.savez_compressed(training_relations_path +  "predictions.npz", y_pred)
        cm = confusion_matrix(y_pred_label, y_test, labels=[i for i in label_mappings.keys()])
        cms.append(cm)

        results.append(f1_value)
        print("Test loss, Test acc, F1-score:", results)



seeds = [1]
for seed in seeds:
    training_relations_path = f"best_model_weight/cnn_bilstm/seed {seed}/"
    # x_t, x_test, y_t, y_test = train_test_split(X_train, y, random_state=seed, test_size=0.2)
    # del x_t
    # del x_test
    # del y_t
    predictions = np.load(training_relations_path + "predictions.npz", allow_pickle=True)['arr_0'].tolist()
    predictions = get_labels(predictions)
    cms.append(confusion_matrix(predictions, y_test, labels=[i for i in label_mappings.keys()]))



plt.clf()
averaged_cms = np.mean(cms, axis=0)
averaged_cms = averaged_cms.astype('float') / averaged_cms.sum(axis=1)[:, np.newaxis]
cms_df = pd.DataFrame(averaged_cms, index = [value for value in reduced_label_mappings.values()],
                      columns=[value for value in reduced_label_mappings.values()])
fig, ax = plt.subplots(figsize=(15, 10))
# pallete = sns.color_palette("Reds", as_cmap=True)
# pallete = sns.light_palette("seagreen", as_cmap=True)
pallete = sns.color_palette("blend:#EBE7E0,#000000", as_cmap=True)
heat = sns.heatmap(cms_df, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Accuracy'}, cmap=pallete)

cbar = heat.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
heat.figure.axes[-1].yaxis.label.set_size(20)
yticks = [i.upper() for i in cms_df.index]
xticks = [i.upper() for i in cms_df.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
plt.xticks(plt.xticks()[0], labels=xticks, rotation=270)
plt.title("Confusion matrix for all 29 relations - CNN + BiLSTM", fontsize=20)
plt.tight_layout()
# plt.savefig(f"cms/confusion_cnn_bilstm_seed_1_custom_test.png", dpi=300)
plt.savefig(f"cms/confusion_cnn_bilstm_seed_1_custom_final.pdf")