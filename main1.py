# -*- coding: utf-8 -*-


"""
@Author: Rohan Saha and Arif Shahriar
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import tokenization

import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


# Importing argparse for command line arguments.
import argparse

# Parsing command line arguments.
parser = argparse.ArgumentParser(description='Train the proposed model. An example on how to run the script is as follows: \n python main.py --model=comemnet-bilstm')
parser.add_argument('-m','--model',default='comemnet-bilstm',  help="Can be either 'comemnet-lstm', 'comemnet-bilstm' or 'erin'.\n Example: model=comemnet-bilstm (default=comemnet-bilstm)")
parser.add_argument('-d','--demo', default=False, help="Boolean. Whether to create a gradio demo (default=False).")
parser.add_argument('-p', '--pretrained', default=False, help='Boolean. Whether to use pretrained model. (default=False). If pretrained=True, please download the google drive files mentioned in the README.')

args = vars(parser.parse_args())
print("You selected the following parameters for the script to run.")
print(args)

if args['demo']:
    import gradio as gr

# print("Using TensorFlow version: ",tf.__version__)

device_name = tf.test.gpu_device_name()
print(device_name)
if device_name != '/device:GPU:0':
  print('GPU device not found, falling back to CPU. Warning, code will run slowly on CPU.')
print('Found GPU at: {}'.format(device_name))

print("Loading and preprocessing data")
data_path = 'Processed Data/Input_500_29_relation.tsv'

train_data = pd.read_csv(data_path, encoding='utf-8', sep = '\t')

train_data.fillna("", inplace = True)

# Shuffle data so that there is a higher chance of the train and test data being from the same distribution.
train_data = shuffle(train_data, random_state = 1)

# # Now read the rows, convert them into strings and then only keep the unique ones.
sentences_and_labels =  np.array([[' '.join(map(str, row[:-1].tolist())).strip(), row[-1]] for row in train_data.iloc[:,:].values])


sentences = sentences_and_labels[:, 0]

label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['relation'])
label_mappings = integer_mapping = {i: l for i, l in enumerate(label.classes_)}


def get_labels(y_pred):
    y_pred_label = np.zeros((len(y_pred), 1))
    print(y_pred_label.shape)
    for index in range(len(y_pred)):
        y_pred_label[index] = np.argmax(y_pred[index])
    return y_pred_label


if args['pretrained'] == False:
    print("Retraining model from scratch.")
    print("Loading bert from TensorFlow-Hub, this may take a while... Please be patient.")
    m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    bert_layer = hub.KerasLayer(m_url, trainable=False)

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


    def build_model_erin(bert_layer, max_len=512):
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        print(tf.shape(sequence_output))
        clf_output = sequence_output[:, :, :]
        print(tf.shape(clf_output))

        #     lay = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding="same", activation="relu")(clf_output)
        #     lay = tf.keras.layers.MaxPooling1D(2, 2)(lay)
        lay = tf.keras.layers.LSTM(1, return_sequences=True)(clf_output)
        lay = tf.keras.layers.Flatten()(lay)
        out = tf.keras.layers.Dense(29, activation='softmax')(lay)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def build_model_comemnet_lstm(bert_layer, max_len=512):
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        # np.savez_compressed(f"bert_sequence_op_seed_{seed}.npz", sequence_output)

        print(tf.shape(sequence_output))
        clf_output = sequence_output[:, :, :]
        print(tf.shape(clf_output))

        lay = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding="same", activation="relu")(clf_output)
        lay = tf.keras.layers.MaxPooling1D(2, 2)(lay)
        lay = tf.keras.layers.LSTM(8, return_sequences=True, dropout=0.2)(lay)
        # lay = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, dropout=0.2))(lay)
        lay = tf.keras.layers.Flatten()(lay)
        out = tf.keras.layers.Dense(29, activation='softmax')(lay)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        # model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # model.compile(tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model


    def build_model_comemnet_bilstm(bert_layer, max_len=512):
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        # np.savez_compressed(f"bert_sequence_op_seed_{seed}.npz", sequence_output)

        # print(tf.shape(sequence_output))
        clf_output = sequence_output[:, :, :]
        # print(tf.shape(clf_output))

        lay = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding="same", activation="relu")(clf_output)
        lay = tf.keras.layers.MaxPooling1D(2, 2)(lay)
        #lay = tf.keras.layers.LSTM(2, return_sequences=True, dropout=0.2)(lay)
        lay = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, dropout=0.2))(lay)
        lay = tf.keras.layers.Flatten()(lay)
        out = tf.keras.layers.Dense(29, activation='softmax')(lay)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        #model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.compile(tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    """### Obtaining Train, test splits.
    ###### In the train splits, we will have a separate validation split.
    """

    print("Checkpoint will only be saved for the last epoch.")
    checkpoint_path = "training_relations/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    """# Do not run the following cell if using checkpointed files."""

    with tf.device(device_name):
        splits = 5 # For five fold cross-validation.
        seeds = [i for i in range(splits)]  # Fix the seed value for reproducibility.
        # seeds = [0]

        val_dict = {}
        test_dict = {}
        test_accs = []
        test_f1s = []

        # First get random train-test splits. Doesn't include validation, which will be obtained from the train set.
        for seed in seeds:
            print(f"Training model for seed {seed}.")
            x_t, x_test, y_t, y_test = train_test_split(sentences, y, random_state=seed, test_size=0.2)   # Global training and test sets.

            # Now get validation sets from each training set.
            # kf = KFold(n_splits=5, shuffle=False) # Setting shuffle=False because shuffled dataset already before.
            # fold_count = 0

            # for train_index, val_index in kf.split(x_t):
            #     #print(x_t.shape)
            #     #print(y_t.shape)
            #     x_train, x_val = x_t[train_index], x_t[val_index]   # Training and validation features.
            #     y_train, y_val = y_t[train_index], y_t[val_index]   # Training and validation labels.

            #     #encode train data
            #     max_len = 80
            #     train_input = bert_encode(x_train, tokenizer, max_len=max_len)
            #     train_labels = y_train
            #     x_val = bert_encode(x_val, tokenizer, max_len=max_len)


            #     model = build_model(bert_layer, max_len=max_len)
            #     model.summary()
            #     #checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
            #     checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
            #     earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
            #     train_sh = model.fit(
            #     train_input, train_labels,
            #     #validation_split=0.2,
            #     validation_data=(x_val, y_val),
            #     epochs=2,
            #     callbacks=[checkpoint, earlystopping],
            #     batch_size=16,
            #     verbose=1)

            #     # Validation sets can be used for hyperparamter tuning.
            #     val_dict[str(seed) + str(fold_count)] = train_sh.history
            #     fold_count += 1

            #encode whole train data
            max_len = 80
            print("Encoding input through BERT encoder.")
            train_input = bert_encode(x_t, tokenizer, max_len=max_len)
            train_labels = y_t

            #train model on whole train data
            if args['model'] == "comemnet-bilstm":
                model = build_model_comemnet_bilstm(bert_layer, max_len=max_len)
            elif args['model'] =='comemnet-lstm':
                model = build_model_comemnet_lstm(bert_layer, max_len=max_len)
            elif args['model'] == 'erin':
                model = build_model_erin(bert_layer, max_len=max_len)
            model.summary()
            #checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
            train_sh = model.fit(
            train_input, train_labels,
            validation_split=0,
            epochs=6,
            callbacks=[checkpoint, earlystopping],
            batch_size=16,
            verbose=1)

            #encode test data
            test_input = bert_encode(x_test, tokenizer, max_len=max_len)

            # Evaluate the model on the test data using `evaluate`
            print("Evaluating on test data for ", seed)
            results = model.evaluate(test_input, y_test, batch_size=16)

            #calculate F1-score
            y_pred = model.predict(test_input, verbose=1)
            y_pred_label = get_labels(y_pred)
            f1_value = f1_score(y_test, y_pred_label, average='macro')
            results.append(f1_value)
            test_f1s.append(f1_value)
            test_accs.append(results[1])
            print(f"Test loss, Test acc, F1-score for seed {seed}: ", results)

            test_dict[seed] = results
        print("Average accuracy: ", np.mean(test_accs))
        print("Average F1 score: ", np.mean(test_f1s))
        print("Results for all seed values: ", test_dict)

"""## The following section is for loading the checkpoint weights and then obtaining the confusion matrix."""

if args['pretrained']:
    print("Currently confusion matrix is only supported for CoMeMNet-BiLSTM")
    cms = []
    # #code for loading weights from checkpoint
    # with tf.device(device_name):
    #     results = []
    #     splits = 5 # For five fold cross-validation.
    #     #seeds = [i for i in range(splits)]  # Fix the seed value for reproducibility.
    #     seeds = [4]#0, 1, 2, 3, 4]
    #
    #     val_dict = {}
    #     test_dict = {}
    #
    #     # First get random train-test splits. Doesn't include validation, which will be obtained from the train set.
    #     for seed in seeds:
    #         x_t, x_test, y_t, y_test = train_test_split(sentences, y, random_state=seed, test_size=0.2)   # Global training and test sets.
    #
    #         # Evaluate the new model
    #         max_len = 80
    #         if args['model'] == "comemnet-bilstm":
    #             model = build_model_comemnet_bilstm(bert_layer, max_len=max_len)
    #         elif args['model'] == 'comemnet-lstm':
    #             model = build_model_comemnet_lstm(bert_layer, max_len=max_len)
    #         elif args['model'] == 'erin':
    #             model = build_model_erin(bert_layer, max_len=max_len)
    #         model.summary()
    #         # Loads the weights for new model
    #         checkpoint_path = f"/content/drive/Shareddrives/CMPUT 656 Data and Results/Result 29 Relations/Seed {seed}/training_relations/cp.ckpt"
    #         training_relations_path = f"/content/drive/Shareddrives/CMPUT 656 Data and Results/Result 29 Relations/Seed {seed}/training_relations/"
    #         # /content/drive/Shareddrives/CMPUT 656 Data and Results/Result 29 Relations/Seed 0/training_relations/cp.ckpt.data-00000-of-00001
    #
    #
    #         model.load_weights(checkpoint_path)
    #
    #         #encode test data
    #         test_input = bert_encode(x_test, tokenizer, max_len=max_len)
    #
    #         # Evaluate the model on the test data using `evaluate`
    #         print("Evaluate on test data for ", seed)
    #         # results = new_model.evaluate(test_input, y_test, batch_size=16)
    #
    #         #calculate F1-score
    #         y_pred = model.predict(test_input, batch_size=16, verbose=1)
    #         y_pred_label = get_labels(y_pred)
    #         f1_value = f1_score(y_test, y_pred_label, average='macro')
    #         np.savez_compressed(training_relations_path +  "predictions.npz", y_pred_label)
    #         cm = confusion_matrix(y_pred_label, y_test, labels=[i for i in label_mappings.keys()])
    #         cms.append(cm)
    #
    #         results.append(f1_value)
    #         print("Test loss, Test acc, F1-score:", results)



    for k, v in label_mappings.items():
        print(k, v)

    reduced_label_mappings = {
        0: 'None',
        1: 'award=nominee',
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

    """# Read in all the predictions for different seeds and make a confusion matrix by averaging them together. 
    
    NOTE: The confusion matrix is normalized.
    """

    cms = []
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        training_relations_path = f"Result 29 Relations/Seed {seed}/training_relations/"
        x_t, x_test, y_t, y_test = train_test_split(sentences, y, random_state=seed, test_size=0.2)
        del x_t
        del x_test
        del y_t
        predictions = np.load(training_relations_path + "predictions.npz", allow_pickle=True)['arr_0'].tolist()
        cms.append(confusion_matrix(predictions, y_test, labels=[i for i in label_mappings.keys()]))

    plt.clf()
    averaged_cms = np.mean(cms, axis=0)
    averaged_cms = averaged_cms.astype('float') / averaged_cms.sum(axis=1)[:, np.newaxis]
    cms_df = pd.DataFrame(averaged_cms, index = [value for value in reduced_label_mappings.values()],
                          columns=[value for value in reduced_label_mappings.values()])
    fig, ax = plt.subplots(figsize=(15, 10))
    heat = sns.heatmap(cms_df, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Normalized value'})
    heat.figure.axes[-1].yaxis.label.set_size(20)
    yticks = [i.upper() for i in cms_df.index]
    xticks = [i.upper() for i in cms_df.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks, rotation=270)
    plt.title("Confusion matrix for all 29 relations - CoMemNet-BiLSTM", fontsize=20)
    plt.show()



"""# Gradio app to demo the model."""


if args['demo']:
    print("Loading BERT encoder, this might take some time... Please be patient.")
    m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    bert_layer = hub.KerasLayer(m_url, trainable=False)


    def get_labels(y_pred):
        y_pred_label = np.zeros((len(y_pred),1))
        print(y_pred_label.shape)
        for index in range(len(y_pred)):
            y_pred_label[index] = np.argmax(y_pred[index])
        return y_pred_label

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

    def build_model(bert_layer, max_len=512):
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        # print(tf.shape(sequence_output))
        clf_output = sequence_output[:, :, :]
        # print(tf.shape(clf_output))

        lay = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding="same", activation="relu")(clf_output)
        lay = tf.keras.layers.MaxPooling1D(2, 2)(lay)
        lay = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, dropout=0.2))(lay)
        lay = tf.keras.layers.Flatten()(lay)
        out = tf.keras.layers.Dense(29, activation='softmax')(lay)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        #model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.compile(tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

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

    seed = 0
    print(f"Using model checkpoint file for seed {seed}")
    checkpoint_path = f"Result 29 Relations/Seed {seed}/training_relations/cp.ckpt"
    max_len = 80

    new_model = build_model(bert_layer, max_len=max_len)
    new_model.load_weights(checkpoint_path)

    def predict_relation(sentence):
        print("Encoding input through BERT encoder.")
        test_input = bert_encode(np.array([sentence]), tokenizer, max_len=max_len)
        y_pred = new_model.predict(test_input, batch_size=16, verbose=1)
        y_pred_label = get_labels(y_pred)
        relations = [rel for rel in reduced_label_mappings.values()]
        probabilities = y_pred.tolist()[0]
        result_dict = {}
        for k, v in zip(relations, probabilities):
            result_dict[k] = v
        return result_dict

    iface = gr.Interface(
        predict_relation,
        inputs="text",
        outputs="label",
        interpretation="default",
        title="CoMemNet - Relation Extractor", description="NOTE: Model is trained on Wikipedia table data and not continuous text."
    )

    iface.launch(debug=False)