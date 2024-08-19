import numpy as np
import pandas as pd
import time
from numba import jit
import datetime
import os
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
# import statsmodels.api as sm 
from IPython.display import display, HTML
# import plotly
import tensorflow as tf
import random

def get_data(Transformation, lags):
    # lags=1 : month over month
    # lags=12: year over year

    price_var = ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL', 
             'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 
             'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA']
    
    Data = pd.read_csv(os.path.join('Data', '2024-06.csv'))
    p = Data.shape[1]

    temp_list = []
    for i in range(1,p):
        tcode = Data.iloc[0,i].copy()
        data = Data.iloc[1:,i].copy()
        if data.name == 'CPIAUCSL':
            data_transform = np.log(data).diff(periods=lags)           # Year over year growth
        elif np.isin(data.name, list(set(price_var)-set(['CPIAUCSL']))):
            if Transformation == 'Transform':
                data_transform = np.log(data).diff(periods=lags)       # Year over year growth
            else:
                data_transform = data
        else:
            if np.isin(Transformation, ['Transform']):
                if tcode == 1:
                    data_transform = data
                elif tcode == 2:  # First difference
                    data_transform = data.diff()
                elif tcode == 3: # Second difference
                    data_transform = data.diff().diff()
                elif tcode == 4: # Log
                    data_transform = np.log(data)
                elif tcode == 5: #First difference of log
                    data_transform = np.log(data).diff()
                elif tcode == 6: #Second difference of natural log
                    data_transform = np.log(data).diff().diff()
                elif tcode == 7: # First difference of percent change
                    data_transform = data.pct_change().diff()
            elif Transformation == 'No Transform':
                data_transform = data
        temp_list.append(data_transform.copy())
    Data_transform = pd.DataFrame(temp_list).T

    Date = Data.iloc[1:,0]
    Y = Data_transform['CPIAUCSL'] # Inflation

    num_lags = 12
    X = Data_transform
    for p in range(1,num_lags):
        X['CPIAUCSL_lag%i' % p] = Y.shift(p)

    h = 1 # One step ahead forecast
    X = X.shift(h)

    X_used = X.iloc[12+num_lags:,:].reset_index(drop=True)
    Y_used = Y.iloc[12+num_lags:].reset_index(drop=True)
    Date_used = Date.iloc[12+num_lags:].reset_index(drop=True)
    Date_used = pd.to_datetime(Date_used)
    
    return X_used, Y_used, Date_used

@jit(nopython=True)
def get_Gram(X,n,gamma):
    Gram_rbf = np.zeros((n,n))
    for t in range(n):
        Gram_rbf[t,:] = np.exp(-gamma*np.sum((X[t,:]-X)**2,1))
    return Gram_rbf

@jit(nopython=True)
def get_Gram_test(X_train,X_test,n_train,n_test,gamma):
    Gram_rbf = np.zeros((n_test,n_train))
    for t in range(n_test):
        Gram_rbf[t,:] = np.exp(-gamma*np.sum((X_test[t,:]-X_train)**2,1))
    return Gram_rbf

