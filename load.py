from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
STEP = 256
# MAX_LEN =71936
MAX_LEN = 16384

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

class Preproc:

    def __init__(self, ecg, codes):
        self.mean, self.std = compute_mean_std(ecg)
        # self.classes = sorted(set(l for label in labels for l in label))
        self.classes = codes
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad_x(x)
        x = (x - self.mean) / self.std
        x = x[:, :, :]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc

        y = pad_y(y, val=self.classes[-1], dtype=np.dtype((str, 100)) )
        multi_labels = [s[0] for s in y]
        multi_labels = [s.strip().split(",") for s in multi_labels]
        y_new = []
        n=0
        for labels in multi_labels:
            targets=np.zeros((1,len(self.classes)))
            for i in range(len(labels)):
                l = keras.utils.np_utils.to_categorical(
                        self.class_to_int[labels[i]], num_classes=len(self.classes))
                targets = targets + l
            y_new.append(np.repeat(targets,len(y[n]),axis=0))
            n=n+1
        return np.array(y_new)

def pad_x(x, val=0, dtype=np.float32):
    # max_len = max(i.shape[0] for i in x)
    max_len = MAX_LEN
    padded = np.full((len(x), max_len,x[0].shape[1]), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i),:i.shape[1]] = i
    return padded

def pad_y(y, val=0, dtype=np.float32):
    # max_len = max(len(i) for i in y)
    max_len = int(MAX_LEN/STEP)
    padded = np.full((len(y), max_len), val, dtype=dtype)
    for e, i in enumerate(y):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.vstack(x)
    return (np.mean(x,axis=0).astype(np.float32),
           np.std(x,axis=0).astype(np.float32))

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze().transpose()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)
    trunc_samp = STEP * int(ecg.shape[0] / STEP)
    return ecg[:trunc_samp,:]

