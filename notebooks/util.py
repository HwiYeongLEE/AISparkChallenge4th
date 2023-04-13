import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def build_container(train_df, test_df, type_num = 8):
    data_container = [(train_df[train_df.type == i].iloc[:,:-1], test_df[test_df.type == i].iloc[:,:-1])  for i in range(type_num)]
    type_container = [(train_df[train_df.type == i].iloc[:,-1].to_frame(), test_df[test_df.type == i].iloc[:,-1].to_frame())  for i in range(type_num)]
    return data_container, type_container

def scaler(train, test, method='standard'):
    scaler = StandardScaler() if method=='standard' else MinMaxScaler()
    train_index = train.index
    test_index = test.index
    train_scaled = pd.DataFrame(index=train_index, data=scaler.fit_transform(train))
    test_scaled = pd.DataFrame(index=test_index, data=scaler.transform(test))
    train_scaled.columns = train_scaled.columns.astype('str')
    test_scaled.columns = test_scaled.columns.astype('str')
    return train_scaled, test_scaled

def pca(train, test, n_components=2):
    pca = PCA(n_components=n_components)
    train_index = train.index
    test_index = test.index
    train_pca = pd.DataFrame(index=train_index, data=pca.fit_transform(train))
    test_pca = pd.DataFrame(index=test_index, data=pca.transform(test))
    train_pca.columns = train_pca.columns.astype('str')
    test_pca.columns = test_pca.columns.astype('str')
    return train_pca, test_pca

def ols_detect_anomaly(train_df, test_df, by='max', ep=0.01):
    lr = LinearRegression()
    lr.fit(train_df.iloc[:,:-1], train_df.iloc[:,-1])
    pred = lr.predict(test_df.iloc[:,:-1])
    residual = abs(test_df.iloc[:,-1] - pred)
    if by == 'max':
        threshold = np.max(abs(train_df.iloc[:,-1] - lr.predict(train_df.iloc[:,:-1])))
    elif by == 'mae':
        threshold = mean_absolute_error(train_df.iloc[:,-1], lr.predict(train_df.iloc[:,:-1]))
    elif by == 'whisker':
        error = abs(train_df.iloc[:,-1] - lr.predict(train_df.iloc[:,:-1]))
        q1 = np.percentile(error, 25)
        q3 = np.percentile(error, 75)
        iqr = q3-q1
        threshold = q3 + 1.5*iqr
    # elif by == 'norm_max':
    #     train_error = train_df.iloc[:,-1] - lr.predict(train_df.iloc[:,:-1])
    #     threshold = np.max(abs((train_error - np.mean(train_error)) / np.std(train_error)))
    
    is_anomaly = (residual > threshold).astype(int)
    return is_anomaly

def svm_detect_anomaly(train_df, test_df, model):
    svm = model
    svm.fit(train_df)
    is_anomaly = np.where(model.predict(test_df)==1, 0, 1)
    return is_anomaly



