# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:39:51 2022

@author: User
"""

from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
import pywt
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import sys

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(0)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32) 
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (16 x 94 x 94)
        x = self.pooling1(x)  # (16 x 18 x 18)
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        x = self.pooling2(x)  # (32 x 5 x 5)
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        x = self.pooling3(x)  # (64 x 1 x 1)
        x = x.view((-1, 64))  # (64,)
        #x = torch.cat((x1, x2), dim=1)  # (68,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x


def worker(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before, after = 90, 110

    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    
    x, y, groups = [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        if categories[i] == 4:  # remove AAMI Q class
            continue

        # cv2.resize is used to sampling the scalogram to (100 x100)
        x.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        #faut modifier la ligne siuvante
       
        y.append(categories[i])
        groups.append(data["record"])

    return x, y, groups


def load_data(wavelet, scales, sampling_rate, filename="./dataset/mitdb.pkl"):
    import pickle

    with open(filename, "rb") as f:
        train_data, test_data = pickle.load(f)

    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1  # for multi-process

    # for training
    x_train, y_train, groups_train = [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data):
            x_train.append(x)
            
            y_train.append(y)
            groups_train.append(groups)
            
            
    print(sys.getsizeof(x_train)/(1024*1024))
    x_train = np.expand_dims(np.concatenate(x_train, axis=0), axis=1).astype(np.float32)
    
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # for test
    x_test, y_test, groups_test = [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data):
            x_test.append(x)
            y_test.append(y)
            groups_test.append(groups)

    x_test = np.expand_dims(np.concatenate(x_test, axis=0), axis=1).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)

    # normalization
 
    return (x_train, y_train, groups_train), (x_test, y_test, groups_test)


def main():
    sampling_rate = 360

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    (x_train, y_train, groups_train), (x_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
    print("Data loaded successfully!")

    log_dir = "./logs/{}".format(wavelet)
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
        TensorBoard(SummaryWriter(log_dir))
    ]
    net = NeuralNetClassifier(  # skorch is extensive package of pytorch for compatible with scikit-learn
        MyModule,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=30,
        batch_size=1024,
        train_split=predefined_split(Dataset({"x": x_test}, y_test)),
        verbose=1,
        device="cpu",
        callbacks=callbacks,
        iterator_train__shuffle=True,
        optimizer__weight_decay=0,
    )
    net.fit({"x": x_train}, y_train)
    y_true, y_pred = y_test, net.predict({"x": x_test})

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    net.save_params(f_params="./models/model_{}.pkl".format(wavelet))


if __name__ == "__main__":
    main()