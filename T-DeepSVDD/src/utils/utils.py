import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def draw_data(data):
    plt.plot(data)
    plt.show()


def load_model(state_path, model):
    model_dict = torch.load(state_path)
    state_dict = deepcopy(model_dict['state_dict'])
    center = None
    for key, val in list(state_dict.items()):
        print(key)
        if key.startswith('decoder') or key.startswith('output'):
            state_dict.pop(key)
    model.load_state_dict(state_dict)

    if model_dict.get('c') is not None:
        center = model_dict['c']
    return model, center

def save_model(model, c, save_path):
    torch.save({
        'c': c,
        'state_dict': model.state_dict()
    }, save_path)

def prepare_data(data_path, test=False, norm = None):
    data = pd.read_csv(data_path)
    label = []
    if(test == True):
        if 'label' in data.columns:
            label = data['label']
            data = data.drop(columns=['label'])
    if norm is not None:
        data = normaolize(data, norm)
    data = torch.from_numpy(data)
    print('Data Shape: ' + str(data.shape))
    masks = torch.ones_like(data, dtype=bool)
    data = torch.unsqueeze(data, -1)
    data = torch.tensor(data, dtype=torch.float32)
    return data, masks, label

def normaolize(df, norm_type):
    if norm_type == "standardization":
        mean = df.mean()
        std = df.std()
        return (df - mean) / (std + np.finfo(float).eps)

    elif norm_type == "minmax":
        max_val = df.max()
        min_val = df.min()
        return (df - min_val) / (max_val - min_val + np.finfo(float).eps)

    elif norm_type == "per_sample_std":
        grouped = df.groupby(by=df.index)
        return (df - grouped.transform('mean')) / grouped.transform('std')

    elif norm_type == "per_sample_minmax":
        return MinMaxScaler().fit_transform(df.T).T