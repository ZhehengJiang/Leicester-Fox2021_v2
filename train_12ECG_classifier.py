#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import load as load
import network as network
import json
import keras
MAX_EPOCHS = 1
# MAX_LEN = 71936
MAX_LEN = 16384
STEP = 256
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def train_12ECG_classifier(input_directory, filename):

    df = pd.read_csv('dx_mapping_scored.csv', sep=',')
    codes = df.values[:,1].astype(np.str)
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    ### delete equivalent_classes
    index=[]
    for i in range(len(codes)):
        for j in range(len(equivalent_classes)):
            if codes[i] == equivalent_classes[j][1]:
                index.append(i)
    codes = np.delete(codes, index)
    codes = np.append(codes, 'O')
    # Load data.
    print('Loading data...')

    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    classes = get_classes(input_directory, header_files)
    num_classes = len(classes)
    num_files = len(header_files)
    recordings = list()
    headers = list()
    labels = list()

    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        recordings.append(recording.T)
        headers.append(header)
        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(' ')
                num_labels = int(recording.shape[1] / STEP)
                labels_act = [arrs[1]]*num_labels
        labels.append(labels_act)

    array_len = np.array([i.shape[0] for i in recordings])
    index = np.where(array_len>MAX_LEN)
    for i in index[0]:
        recordings[i] = recordings[i][0:MAX_LEN,:]
        labels[i] = labels[i][0:int(MAX_LEN/STEP)]
    remove_index = []
    for i in range(len(labels)):
        label_e = labels[i][0]
        split_label = label_e.split(',')
        new_labels = []
        for label in split_label:
            if label in codes:
                new_labels.append(label)
        if new_labels == []:
            remove_index.append(i)
        else:
            for j in range(len(labels[i])):
                labels[i][j] = ','.join(new_labels)
    for index in sorted(remove_index, reverse=True):
        del recordings[index]
        del labels[index]

    # Train model.
    print('Training model...')
    print("Building preprocessor...")
    preproc = load.Preproc(recordings,codes)
    config_file = 'config.json'
    params = json.load(open(config_file, 'r'))
    CPT = np.zeros((len(codes), len(codes)))
    diag_matrix = np.identity(len(codes))
    gts = labels
    for i in range(len(codes)-1):
        num_all = 0
        for n in range(len(gts)):
            gt_split = gts[n][0].split(',')
            if codes[i] in gt_split:
                num_all = num_all + 1
                for j in range(len(codes)-1):
                    if i != j:
                        if codes[j] in gt_split:
                            CPT[i, j] = CPT[i, j] + 1
        CPT[i, :] = CPT[i, :] / num_all
    params.update({
        "input_shape": [MAX_LEN, 12],
        "num_categories": len(preproc.classes),
        "CPT": CPT,
        "diag_matrix": diag_matrix
    })

    model = network.build_network(**params)
    # model.summary()
    stopping = keras.callbacks.EarlyStopping(patience=10)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)


    batch_size = params.get("batch_size", 32)
    train_gen = load.data_generator(batch_size, preproc, recordings,labels)

    model.fit_generator(
        train_gen,
        steps_per_epoch=int(len(recordings) / batch_size),
        epochs=MAX_EPOCHS,
        callbacks=[reduce_lr, stopping]
)
    # model.save_weights('net_weights.h5')
    # Save model.
    print('Saving model...')

    final_model={'model':model, 'classes':preproc.classes}

    # filename = os.path.join(output_directory, 'finalized_model.sav')
    # joblib.dump(final_model, filename, protocol=0)
    model.save_weights(filename)

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)
