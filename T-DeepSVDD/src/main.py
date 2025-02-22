import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'
import pandas as pd
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from utils import utils
from options import Options
from models.ts_transformer import model_factory
from deepSVDD_trainer import *
from utils import utils

def main(config):
    data_path = 'datasets/synthetic_train.csv'
    test_data_path = 'datasets/labeled_data_864.csv'
    logger.info("Creating model ...")
    Test = True
    OutFeature = True

    #Creating Model
    center = None
    pretraion_model = model_factory(config)
    svdd_model = model_factory(config)

    pretrain_path = 'model_house/pretrained_neurIPS-TS.pth'
    svdd_path = 'detection_model/detect_model_864_1.pth'

    #Prepare Data
    data, masks, _ = utils.prepare_data(data_path, test=False, norm=config['normalization'])
    dataset = TensorDataset(data, masks)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    pass

    if OutFeature == True:
        test_data, masks, label = utils.prepare_data(test_data_path, Test, config['normalization'])
        svdd_model, center = utils.load_model(svdd_path, svdd_model)
        outputs = svdd_model(test_data, masks)
        pass
        return

    #Test
    if Test == True:
        test_data, masks, label = utils.prepare_data(test_data_path, Test, config['normalization'])
        test_dataset = TensorDataset(test_data, masks)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
        svdd_model, center = utils.load_model(svdd_path, svdd_model)
        trainer = DeepSVDDTrainer('one-class', R=0.0, c=center, nu=0.1)
        score = trainer.test(test_loader, svdd_model)
        plt.plot(score)
        # plt.xticks(np.arange(len(score)))
        plt.show()
        np.savetxt('results/test_score_UCR_ECG1_011.csv', score)
        pass
        return

    #Load pretrained model
    model, center = utils.load_model(pretrain_path, pretraion_model)

    #Train
    trainer = DeepSVDDTrainer('one-class', R=0.0, c=center, nu=0.1)
    model = trainer.train(train_loader, model)
    trainer.test(train_loader, model)
    save_path = 'detection_model/detect_model_neurIPS-TS.pth'
    utils.save_model(model, trainer.c, save_path)
    pass


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = args.__dict__  # configuration dictionary
    main(config)
