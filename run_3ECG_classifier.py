#!/usr/bin/env python

import numpy as np
import joblib
import os
import network
import json
from sklearn.preprocessing import normalize
import pandas as pd
# MAX_LEN = 119808
# MAX_LEN = 71936
MAX_LEN = 16384
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def run_3ECG_classifier(data,header_data,model):

    # net_classes = ['AF', 'AF,LBBB', 'AF,LBBB,STD', 'AF,PAC', 'AF,PVC', 'AF,RBBB', 'AF,STD', 'AF,STE', 'I-AVB', 'I-AVB,LBBB',
    #  'I-AVB,PAC', 'I-AVB,PVC', 'I-AVB,RBBB', 'I-AVB,STD', 'I-AVB,STE', 'LBBB', 'LBBB,PAC', 'LBBB,PVC', 'LBBB,STE',
    #  'Normal', 'PAC', 'PAC,PVC', 'PAC,STD', 'PAC,STE', 'PVC', 'PVC,STD', 'PVC,STE', 'RBBB', 'RBBB,PAC', 'RBBB,PAC,STE',
    #  'RBBB,PVC', 'RBBB,STD', 'RBBB,STE', 'STD', 'STD,STE', 'STE']
    df = pd.read_csv('dx_mapping_scored.csv', sep=',')
    codes = df.values[:,1].astype(np.str)
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    index=[]
    for i in range(len(codes)):
        for j in range(len(equivalent_classes)):
            if codes[i] == equivalent_classes[j][1]:
                index.append(i)
    classes = np.delete(codes, index)
    classes = np.append(classes, 'O')
    num_classes = len(classes)
    class_to_int = dict(zip(classes, range(num_classes)))
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class.
    input_length = MAX_LEN
    if data.T.shape[0]>input_length:
        # input_length = int(data.T.shape[0]/256+1)*256
        data = data[:,0:input_length]
    input_data = np.zeros([1,input_length,data.T.shape[1]])
    input_data[0,0:data.T.shape[0],:] = data.T
    score = model.predict(input_data)
    pred_scores = np.sum(score,axis=1)
    pred_scores = pred_scores[:, 0:-1]
    pred_scores = normalize(pred_scores)
    pred_label = pred_scores.argmax(axis=1)
    pred_c = classes[pred_label[0]]
    pred_c_split = pred_c.split(',');
    for i in range(len(pred_c_split)):
        current_label[class_to_int[pred_c_split[i]]] = 1
        current_score[class_to_int[pred_c_split[i]]] = np.max(pred_scores)
    pred_l = np.argwhere(pred_scores[0,:]>0.5)
    if pred_l.size != 0:
        for ii in range(len(pred_l)):
            pred_c = classes[pred_l[ii][0]]
            pred_c_split = pred_c.split(',')
            for j in range(len(pred_c_split)):
                current_label[class_to_int[pred_c_split[j]]] = 1
                current_score[class_to_int[pred_c_split[j]]] = pred_scores[0,pred_l[ii][0]]

    # for i in range(num_classes):
    #     current_score[class_to_int[pred_c_split[i]]] = np.max(pred_scores)

    return current_label, current_score, classes

def load_3ECG_model(weight_filename,feature_indices):
    # load the model from disk
    config_file = 'config.json'
    params = json.load(open(config_file, 'r'))
    df = pd.read_csv('dx_mapping_scored.csv', sep=',')
    codes = df.values[:,1].astype(np.str)
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    index=[]
    for i in range(len(codes)):
        for j in range(len(equivalent_classes)):
            if codes[i] == equivalent_classes[j][1]:
                index.append(i)
    codes = np.delete(codes, index)
    codes = np.append(codes, 'O')
    diag_matrix = np.identity(len(codes))
    CPT = params['CPT']
    params.update({
        "input_shape": [MAX_LEN, 3],
        "CPT": np.array(CPT),
        "diag_matrix": diag_matrix
    })
    params['mean'] = np.array(params['mean'])[feature_indices].tolist()
    params['std'] = np.array(params['std'])[feature_indices].tolist()
    loaded_model = network.build_network(**params)
    loaded_model.load_weights(weight_filename)

    return loaded_model
