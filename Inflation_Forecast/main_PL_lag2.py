## Inflation Forecast
## Target variable: Year over Year inflation
## Partially Linear Model with 2 lags as linear part
import os
import time
import pickle
import sqlite3
import numba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from functions import get_data
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import sqlite3
import random
import re
import xgboost as xgb
import tensorflow as tf
# from functions import *

# Transform:
price_var = ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL', 
             'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 
             'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA']

# Transform:
# Transformation = 'No Transform'
Transformation = 'Transform' 
# :Transforms according to the recommendations given by McCracken and Ng (2015) for all but Group 7 (Prices),
#  which are transformed as year over year growth

Target = 'Inflation'

lags = 12
X_used, Y_used, Date_used = get_data(Transformation, lags)

num_lags = 2
V_used = X_used[np.append(['CPIAUCSL'],['CPIAUCSL_lag%i' % p for p in range(1,num_lags)])]
W_used = X_used.drop(np.append(['CPIAUCSL'],['CPIAUCSL_lag%i' % p for p in range(1,12)]),axis=1)

n = X_used.shape[0]

forecast_period = pd.to_datetime('2015-01-01')<=Date_used
forecast_idx = np.where(forecast_period)[0]
n_test = np.sum(forecast_period)

validation_period = (pd.to_datetime('2005-08-01')<=Date_used) & (pd.to_datetime('2015-01-01') > Date_used)
validation_idx = np.where(validation_period)[0]
n_val = np.sum(validation_period)

training_period = pd.to_datetime('2005-08-01')>Date_used
training_idx = np.where(training_period)[0]
n_train = np.sum(training_period)


V_train = V_used.loc[training_period]
W_train = W_used.loc[training_period,:]
Y_train = Y_used.loc[training_period]

V_val = V_used.loc[validation_period]
W_val = W_used.loc[validation_period,:]
Y_val = Y_used.loc[validation_period]

V_test = V_used.loc[forecast_period]
W_test = W_used.loc[forecast_period,:]
Y_test = Y_used.loc[forecast_period]

nnan_idx = np.sum(np.isnan(W_used),axis=0)==0
W_used_nnan = W_used.loc[:,nnan_idx]

W_train_nnan = W_used_nnan.loc[training_period,:]
W_val_nnan = W_used_nnan.loc[validation_period,:]
W_test_nnan = W_used_nnan.loc[forecast_period,:]

Validation_Err = {}
beta_tot = {}

# Seed number used
seed_list = [42, 43, 44, 45, 46]

database_name = 'database_predict_inflation.db'
for seed in seed_list:
    # Make Database
    con = sqlite3.connect(os.path.join('Results', database_name))
    cur = con.cursor()

    res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
    table_names = res.fetchall()
    if ~np.isin('Results', table_names):
        print("CREATE NEW DATABASE")
        cur.execute("""CREATE TABLE IF NOT EXISTS Results(
                    Date TEXT NOT NULL,
                    Target TEXT NOT NULL,
                    Value REAL NOT NULL,
                    Prediction REAL NOT NULL,
                    Model TEXT NOT NULL,
                    Seed INTEGER NOT NULL,
                    Parameter TEXT,
                    Window_size INTEGER NOT NULL,
                    Validation_size INTEGER NOT NULL,
                    Transformation TEXT NOT NULL,
                    PRIMARY KEY (Date, Target, Model, Seed, Window_size, Validation_size, Transformation))""")
        con.commit()
    else:
        print("DATABASE ALREADY EXISTS")
        con.commit()

    #######################################################################################
    #######################   Partially Linear Random Forest  #############################
    #######################################################################################
    max_depth_list = np.append(np.arange(1,50,3),None)
    val_err = np.zeros((n_val, len(max_depth_list)))
    RFmodel_dict = {}
    beta_dict_RF = {}
    for cv_i, max_depth in enumerate(max_depth_list):
        model1 = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                        max_depth=max_depth,random_state=seed)
        model2 = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                        max_depth=max_depth,random_state=seed)
        RFmodel_dict[cv_i] = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                        max_depth=max_depth,random_state=seed)    
        model1.fit(W_train_nnan, V_train)
        
        resi_v = V_train.values - model1.predict(W_train_nnan)
        model2.fit(W_train_nnan, Y_train)
        resi_y = Y_train.values - model2.predict(W_train_nnan)
        beta = np.linalg.inv(resi_v.T @ resi_v) @ resi_v.T @ resi_y
        
        resi_train = Y_train.values - V_train.values @ beta
        
        RFmodel_dict[cv_i].fit(W_train_nnan, resi_train)
        resi_hat = RFmodel_dict[cv_i].predict(W_val_nnan)
        
        Y_hat = V_val.values @ beta + resi_hat
        beta_dict_RF[cv_i] = beta
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_RF = np.mean(np.array(val_err)**2, axis=0)

    Validation_Err['RF',seed] = pd.DataFrame()
    Validation_Err['RF',seed]['max_depth_list'] = max_depth_list
    Validation_Err['RF',seed]['val_err'] = val_err_RF
    beta_tot['RF',seed] = beta_dict_RF[min_idx]
    
    # temp_grid = ['None' if x==None else x for x in max_depth_list]
    # plt.plot(temp_grid, val_err_RF)
    # plt.xlabel('max_depth')
    # plt.title('Validation Error, Random Forest, Minimum=%s'%str(temp_grid[min_idx]))
    # plt.savefig("Figures/RF_validation_seed%i.png"%seed)
    # plt.close()
    # plt.show()


    Y_hat = V_test.values @ beta_dict_RF[min_idx] + RFmodel_dict[min_idx].predict(W_test_nnan)
    test_err_RF = Y_test.values - Y_hat
    RMSE_RF = np.sqrt(np.sum(test_err_RF**2)/len(test_err_RF))

    RF_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'Random Forest-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(max_depth_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    RF_out = pd.DataFrame.from_dict(RF_out)
        
    #######################################################################################
    ###################################   XGBoost  ########################################
    #######################################################################################
    n_estimators_list = np.arange(1,50,3)
    val_err = np.zeros((n_val,len(n_estimators_list)))
    XGBmodel_dict = {}
    beta_dict_XGB = {}
    for cv_i, n_estimators in enumerate(n_estimators_list):

        model1 = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed)
        model2 = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed)
        XGBmodel_dict[cv_i] = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed)

        model1.fit(W_train_nnan, V_train)
        resi_v = V_train.values - model1.predict(W_train_nnan)
        model2.fit(W_train_nnan, Y_train)
        resi_y = Y_train.values - model2.predict(W_train_nnan)
        
        beta = np.linalg.inv(resi_v.T @ resi_v) @ (resi_v.T @ resi_y)
        resi_train = Y_train.values - V_train.values @ beta
        
        XGBmodel_dict[cv_i].fit(W_train_nnan, resi_train)
        resi_hat = XGBmodel_dict[cv_i].predict(W_val_nnan)
        Y_hat = V_val.values @ beta + resi_hat
        
        beta_dict_XGB[cv_i] = beta
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_XGB = np.mean(np.array(val_err)**2, axis=0)

    Validation_Err['XGB',seed] = pd.DataFrame()
    Validation_Err['XGB',seed]['n_estimators_list'] = n_estimators_list
    Validation_Err['XGB',seed]['val_err'] = val_err_XGB
    beta_tot['XGB',seed] = beta_dict_XGB[min_idx]

    # plt.plot(n_estimators_list,val_err_XGB)
    # plt.xlabel('n_estimators')
    # plt.title('Validation Error, XGBoost, Minimum=%i'%n_estimators_list[min_idx])
    # plt.savefig("Figures/XGB_validation_seed%i.png"%seed)
    # plt.close()
    # plt.show()

    Y_hat = V_test.values @ beta_dict_XGB[min_idx] + XGBmodel_dict[min_idx].predict(W_test_nnan)
    test_err_XGB = Y_test.values - Y_hat
    RMSE_XGB = np.sqrt(np.sum(test_err_XGB**2)/len(test_err_XGB))


    XGB_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'XGBoost-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(n_estimators_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    XGB_out = pd.DataFrame.from_dict(XGB_out)

    #######################################################################################
    ##############################   XGBoost with subsampling #############################
    #######################################################################################

    num_parallel_tree = 100
    subsample = np.sqrt(W_train_nnan.shape[0])/W_train_nnan.shape[0]
    n_estimators_list = np.arange(1,50,3)
    val_err = np.zeros((n_val,len(n_estimators_list)))
    XGBmodel_dict = {}
    beta_dict_XGBs = {}
    for cv_i, n_estimators in enumerate(n_estimators_list):

        model1 = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed,
                                num_parallel_tree = num_parallel_tree, subsample = subsample)
        model2 = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed,
                                num_parallel_tree = num_parallel_tree, subsample = subsample)
        XGBmodel_dict[cv_i] = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed,
                                                num_parallel_tree = num_parallel_tree, subsample = subsample)

        model1.fit(W_train_nnan, V_train)
        resi_v = V_train.values - model1.predict(W_train_nnan)
        model2.fit(W_train_nnan, Y_train)
        resi_y = Y_train.values - model2.predict(W_train_nnan)
        beta = np.linalg.inv(resi_v.T @ resi_v) @ (resi_v.T @ resi_y)
        resi_train = Y_train.values - V_train.values @ beta
        
        XGBmodel_dict[cv_i].fit(W_train_nnan, resi_train)
        resi_hat = XGBmodel_dict[cv_i].predict(W_val_nnan)
        Y_hat = V_val.values @ beta + resi_hat
        
        beta_dict_XGBs[cv_i] = beta
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_XGBs = np.mean(np.array(val_err)**2, axis=0)

    Validation_Err['XGBs',seed] = pd.DataFrame()
    Validation_Err['XGBs',seed]['n_estimators_list'] = n_estimators_list
    Validation_Err['XGBs',seed]['val_err'] = val_err_XGBs
    beta_tot['XGBs',seed] = beta_dict_XGBs[min_idx]

    # plt.plot(n_estimators_list, val_err_XGBs)
    # plt.xlabel('n_estimators')
    # plt.title('Validation Error, XGBoost with subsample, Minimum=%i'%n_estimators_list[min_idx])
    # # plt.savefig("Figures/XGBs_validation_seed%i.png"%seed)
    # # plt.close()
    # plt.show()

    Y_hat = V_test.values @ beta_dict_XGBs[min_idx] + XGBmodel_dict[min_idx].predict(W_test_nnan)
    test_err_XGBs = Y_test.values - Y_hat
    RMSE_XGBs = np.sqrt(np.sum(test_err_XGBs**2)/len(test_err_XGBs))

    XGBs_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'XGBoost-subsample-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(n_estimators_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    XGBs_out = pd.DataFrame.from_dict(XGBs_out)
    
    #######################################################################################
    ####################################   PCR    #########################################
    #######################################################################################
    W_train_stzd = (W_train_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    W_val_stzd = (W_val_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    W_test_stzd = (W_test_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)    

    Sigma_hat = W_train_stzd.T@W_train_stzd/n_train
    eigval, eigvec = np.linalg.eig(Sigma_hat)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    idx = eigval.argsort()[::-1]
    eigval_sorted = eigval[idx]
    eigvec_sorted = eigvec[:, idx]
    F_train = W_train_stzd @ eigvec_sorted
    F_val = W_val_stzd @ eigvec_sorted
    F_val.columns = F_train.columns
    F_test = W_test_stzd @ eigvec_sorted
    F_test.columns = F_test.columns

    # fig, ax = plt.subplots()
    # ax.plot(W_train_stzd.columns,eigvec_sorted[:,0], label='First basis')
    # ax.plot(W_train_stzd.columns,eigvec_sorted[:,1], label='Second basis')
    # ax.plot(W_train_stzd.columns,eigvec_sorted[:,2], label='Third basis')
    # plt.xticks(rotation=-45)
    # plt.legend()
    # plt.show()


    nfactors_list = np.arange(1,20)
    val_err = np.zeros((n_val, len(nfactors_list)))
    nfactors = 2
    PCR_dict = {}
    beta_dict_PCR = {}
    for cv_i, nfactors in enumerate(nfactors_list):

        PCR_dict[cv_i] = LinearRegression(fit_intercept=True)
        PCR_dict[cv_i].fit(np.concatenate((V_train.values,F_train.iloc[:,:nfactors]),axis=1), Y_train)
        
        
        Y_hat = PCR_dict[cv_i].predict(np.concatenate((V_val.values,F_val.iloc[:,:nfactors]),axis=1))
        
        beta_dict_PCR[cv_i] = PCR_dict[cv_i].coef_[:2]
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_PCR = np.mean(np.array(val_err)**2, axis=0)

    Validation_Err['PCR',seed] = pd.DataFrame()
    Validation_Err['PCR',seed]['nfactors_list'] = nfactors_list
    Validation_Err['PCR',seed]['val_err'] = val_err_PCR
    beta_tot['PCR',seed] = beta_dict_PCR[min_idx]

    # plt.plot(nfactors_list, val_err_PCR)
    # plt.xlabel('Number of principal components')
    # plt.title('Validation Error, PCR,Minimum=%i'%nfactors_list[min_idx])
    # plt.show()
    # plt.savefig("Figures/PCR_validation_seed%i.png"%seed)
    # plt.close()
    # # plt.show()
    

    Y_hat = PCR_dict[min_idx].predict(np.concatenate((V_test.values,F_test.iloc[:,:nfactors_list[min_idx]]),axis=1))
    test_err_PCR = Y_test.values - Y_hat
    RMSE_PCR = np.sqrt(np.sum(test_err_PCR**2)/len(test_err_PCR))
    
    PCR_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'PCR-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(nfactors_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    PCR_out = pd.DataFrame.from_dict(PCR_out)


    #######################################################################################
    ####################################   PCRp    #########################################
    #######################################################################################
    W_train_stzd2 = (W_train_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    W_val_stzd2 = (W_val_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    W_test_stzd2 = (W_test_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)

    W_train_stzd = W_train_stzd2[list(set(W_train_stzd2.columns).intersection(set(price_var)))]
    W_val_stzd = W_val_stzd2[list(set(W_train_stzd2.columns).intersection(set(price_var)))]
    W_test_stzd = W_test_stzd2[list(set(W_train_stzd2.columns).intersection(set(price_var)))]

    Sigma_hat = W_train_stzd.T@W_train_stzd/n_train
    eigval, eigvec = np.linalg.eig(Sigma_hat)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    idx = eigval.argsort()[::-1]
    eigval_sorted = eigval[idx]
    eigvec_sorted = eigvec[:, idx]
    F_train = W_train_stzd @ eigvec_sorted
    F_val = W_val_stzd @ eigvec_sorted
    F_val.columns = F_train.columns
    F_test = W_test_stzd @ eigvec_sorted
    F_test.columns = F_test.columns

    # fig, ax = plt.subplots()
    # ax.plot(W_train_stzd.columns,eigvec_sorted[:,0], label='First basis')
    # ax.plot(W_train_stzd.columns,eigvec_sorted[:,1], label='Second basis')
    # ax.plot(W_train_stzd.columns,eigvec_sorted[:,2], label='Third basis')
    # plt.xticks(rotation=-45)
    # plt.legend()
    # plt.show()


    nfactors_list = np.arange(1,20)
    val_err = np.zeros((n_val, len(nfactors_list)))
    nfactors = 2
    PCRp_dict = {}
    beta_dict_PCRp = {}
    for cv_i, nfactors in enumerate(nfactors_list):

        PCRp_dict[cv_i] = LinearRegression(fit_intercept=True)
        PCRp_dict[cv_i].fit(np.concatenate((V_train.values,F_train.iloc[:,:nfactors]),axis=1), Y_train)
        
        Y_hat = PCRp_dict[cv_i].predict(np.concatenate((V_val.values,F_val.iloc[:,:nfactors]),axis=1))
        
        beta_dict_PCRp[cv_i] = PCRp_dict[cv_i].coef_[:2]
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_PCRp = np.mean(np.array(val_err)**2, axis=0)

    Validation_Err['PCRp',seed] = pd.DataFrame()
    Validation_Err['PCRp',seed]['nfactors_list'] = nfactors_list
    Validation_Err['PCRp',seed]['val_err'] = val_err_PCRp
    beta_tot['PCRp',seed] = beta_dict_PCRp[min_idx]

    # plt.plot(nfactors_list, val_err_PCRp)
    # plt.xlabel('Number of principal components')
    # plt.title('Validation Error, PCR,Minimum=%i'%nfactors_list[min_idx])
    # plt.show()
    # plt.savefig("Figures/PCR_validation_seed%i.png"%seed)
    # plt.close()
    # # plt.show()
    

    Y_hat = PCRp_dict[min_idx].predict(np.concatenate((V_test.values,F_test.iloc[:,:nfactors_list[min_idx]]),axis=1))
    test_err_PCRp = Y_test.values - Y_hat
    RMSE_PCRp = np.sqrt(np.sum(test_err_PCRp**2)/len(test_err_PCRp))
    
    PCRp_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'PCRp-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(nfactors_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    PCRp_out = pd.DataFrame.from_dict(PCRp_out)

    #######################################################################################
    ################################# RKHS with l2 penalty ###############################
    #######################################################################################
    W_train_stzd = (W_train_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    W_val_stzd = (W_val_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    W_test_stzd = (W_test_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)

    Sigma_hat = W_train_stzd.T@W_train_stzd/n_train
    eigval, eigvec = np.linalg.eig(Sigma_hat)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    idx = eigval.argsort()[::-1]
    eigval_sorted = eigval[idx]
    eigvec_sorted = eigvec[:, idx]

    @numba.njit
    def get_Gram_rbf(X_train, X_test, n_train, n_test, gamma):
        Gram_rbf = np.zeros((n_test,n_train))
        for t in range(n_test):
            Gram_rbf[t,:] = np.exp(-gamma*np.sum((X_test[t,:]-X_train)**2,1))
        return Gram_rbf

    gamma = (1/W_test_stzd.shape[1])
    Kernel_Gram = get_Gram_rbf(W_train_stzd.values,W_train_stzd.values, n_train,n_train,gamma)

    K_val = get_Gram_rbf(W_train_stzd.values, W_val_stzd.values, n_train, n_val, gamma)
    K_test = get_Gram_rbf(W_train_stzd.values, W_test_stzd.values, n_train, n_test, gamma)

    alpha_hat_dict = {}
    beta_hat_dict = {}

    V_train_one = np.concatenate((np.ones((n_train,1)),V_train.values,),axis=1)
    M_v = (np.identity(n_train)-V_train_one@np.linalg.inv(V_train_one.T@V_train_one)@V_train_one.T)
    lambda_list = np.linspace(0,500,100)
    val_err = np.zeros((n_val,len(lambda_list)))
    
    for cv_j, lam in enumerate(lambda_list):
        
        alpha_hat = np.linalg.inv(M_v@Kernel_Gram + lam*np.eye(n_train))@(M_v@Y_train)
        beta_hat = np.linalg.inv(V_train_one.T@V_train_one)@V_train_one.T@(Y_train-Kernel_Gram@alpha_hat)
        
        Y_hat = np.concatenate((np.ones((n_val,1)),V_val.values,),axis=1)@beta_hat + K_val@alpha_hat
        alpha_hat_dict[cv_j] = alpha_hat
        beta_hat_dict[cv_j] = beta_hat
        val_err[:,cv_j] = Y_val.values-Y_hat


    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_RKHS = np.mean(np.array(val_err)**2, axis=0)
    
    Validation_Err['RKHS',seed] = {}
    Validation_Err['RKHS',seed]['nfactors_list'] = nfactors_list
    Validation_Err['RKHS',seed]['lambda_list'] = lambda_list
    Validation_Err['RKHS',seed]['val_err'] = val_err_RKHS
    beta_tot['RKHS',seed] = beta_hat_dict[min_idx]

    # plt.plot(lambda_list, val_err_RKHS)
    # plt.xlabel('alpha')
    # plt.title('Validation Error, RKHS, argmin=%0.2f'%(lambda_list[min_idx]))
    # plt.show()
    
    Y_hat = np.concatenate((np.ones((n_test,1)),V_test.values,),axis=1)@beta_hat_dict[min_idx] + K_test@alpha_hat_dict[min_idx]
    test_err_RKHS = Y_test.values - Y_hat
    RMSE_RKHS = np.sqrt(np.sum(test_err_RKHS**2)/len(test_err_RKHS))


    RKHS_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'RKHS-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(lambda_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    RKHS_out = pd.DataFrame.from_dict(RKHS_out)
    # #######################################################################################
    # ################################# RKHS with l2 penalty ###############################
    # #######################################################################################
    # W_train_stzd = (W_train_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    # W_val_stzd = (W_val_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)
    # W_test_stzd = (W_test_nnan - np.mean(W_train_nnan, axis=0))/np.std(W_train_nnan, axis = 0)

    # Sigma_hat = W_train_stzd.T@W_train_stzd/n_train
    # eigval, eigvec = np.linalg.eig(Sigma_hat)
    # eigval = np.real(eigval)
    # eigvec = np.real(eigvec)
    # idx = eigval.argsort()[::-1]
    # eigval_sorted = eigval[idx]
    # eigvec_sorted = eigvec[:, idx]

    # @numba.njit
    # def get_Gram_rbf(X_train, X_test, n_train, n_test, gamma):
    #     Gram_rbf = np.zeros((n_test,n_train))
    #     for t in range(n_test):
    #         Gram_rbf[t,:] = np.exp(-gamma*np.sum((X_test[t,:]-X_train)**2,1))
    #     return Gram_rbf
    

    # gamma = (1/W_test_stzd.shape[1])
    # Kernel_Gram = get_Gram_rbf(W_train_stzd.values,W_train_stzd.values, n_train,n_train,gamma)
    # eigenvalues, eigenvectors = np.linalg.eigh(Kernel_Gram) 

    # idx = eigenvalues.argsort()[::-1]
    # eigval_sorted = eigenvalues[idx]
    # eigvec_sorted = eigenvectors[:, idx]

    # F_train = pd.DataFrame(eigvec_sorted*eigval_sorted)

    # K_val = get_Gram_rbf(W_train_stzd.values, W_val_stzd.values, n_train, n_val, gamma)
    # K_test = get_Gram_rbf(W_train_stzd.values, W_test_stzd.values, n_train, n_test, gamma)
    # F_val = pd.DataFrame(K_val@eigvec_sorted)
    # F_test = pd.DataFrame(K_test@eigvec_sorted)


    # nfactors_list = np.arange(1,20)
    # delta_hat_dict = {}
    # beta_hat_dict = {}
    # nfactors_price = 8

    # V_train_one = np.concatenate((np.ones((n_train,1)),V_train.values,),axis=1)
    # F_train_proj = (np.identity(n_train)-V_train_one@np.linalg.inv(V_train_one.T@V_train_one)@V_train_one.T)@F_train
    # lambda_list = np.linspace(0,50,100)
    # val_err = np.zeros((n_val, len(nfactors_list), len(lambda_list)))
    # for cv_i, nfactors in enumerate(nfactors_list):
    #     for cv_j, lam in enumerate(lambda_list):
    #         F_train_used = F_train.iloc[:,:nfactors]
    #         F_train_proj_used = F_train_proj.iloc[:,:nfactors]
            
    #         delta_hat = np.linalg.inv(F_train_used.T@F_train_proj_used + lam*np.diag(eigval_sorted[:nfactors]))@(F_train_proj_used.T@Y_train)
    #         # alpha_hat = eigvec_sorted[:,:nfactors]@delta_hat
    #         # beta_hat = np.linalg.inv(V_train.values.T@V_train.values)@V_train.values.T@(Y_train-Kernel_Gram@alpha_hat)
    #         beta_hat = np.linalg.inv(V_train_one.T@V_train_one)@V_train_one.T@(Y_train-(eigvec_sorted[:,:nfactors]*eigval_sorted[:nfactors])@delta_hat)
            
    #         Y_hat = np.concatenate((np.ones((n_val,1)),V_val.values,),axis=1)@beta_hat + F_val.iloc[:,:nfactors]@delta_hat
    #         delta_hat_dict[cv_i,cv_j] = delta_hat
    #         beta_hat_dict[cv_i,cv_j] = beta_hat
    #         val_err[:, cv_i,cv_j] = Y_val.values-Y_hat.values


    # min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    # min_idx_i,min_idx_j = np.unravel_index(min_idx, np.mean(np.array(val_err)**2, axis=0).shape)
    # val_err_RKHS = np.mean(np.array(val_err)**2, axis=0)
    # val_err_RKHS[min_idx_i,min_idx_j]
    # np.min(val_err_RKHS)
    
    # Validation_Err['RKHS'] = {}
    # Validation_Err['RKHS']['nfactors_list'] = nfactors_list
    # Validation_Err['RKHS']['lambda_list'] = lambda_list
    # Validation_Err['RKHS']['val_err'] = val_err_RKHS
    # beta_tot['RKHS'] = beta_hat_dict[min_idx_i,min_idx_j][1:]

    

    # plt.plot(lambda_list, val_err_RKHS[min_idx_i,:])
    # plt.xlabel('alpha')
    # plt.title('Validation Error, RKHS, Dim=%i, argmin=%0.2f'%(nfactors_list[min_idx_i],lambda_list[min_idx_j]))
    # plt.show()
    
    # plt.plot(nfactors_list, val_err_RKHS[:,min_idx_j])
    # plt.xlabel('Number of principal components')
    # plt.title('Validation Error, RKHS, alpha=%0.2f, argmin=%i'%(lambda_list[min_idx_j],nfactors_list[min_idx_i]))
    # plt.show()
    
    # # temp = LinearRegression(fit_intercept=True)
    # # temp.fit(pd.concat((V_train,W_train_stzd),axis=1),Y_train)
    # # Y_hat = temp.predict(pd.concat((V_test,W_test_stzd),axis=1))

    # Y_hat = np.concatenate((np.ones((n_test,1)),V_test.values),axis=1)@beta_hat_dict[min_idx_i,min_idx_j] + F_test.iloc[:,:nfactors_list[min_idx_i]]@delta_hat_dict[min_idx_i,min_idx_j]
    # test_err_PCR = Y_test.values - Y_hat.values
    # RMSE_RKHS = np.sqrt(np.sum(test_err_PCR**2)/len(test_err_PCR))

    # RKHS_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
    #         'Target': Target,
    #         'Value': Y_test.values,
    #         'Prediction': Y_hat,
    #         'Model': 'PCR-PL-%ilags' %num_lags,
    #         'Seed': seed,
    #         'Parameter': 'alpha%f_nf%i'%(lambda_list[min_idx_j],nfactors_list[min_idx_i]),
    #         'Window_size': n_train,
    #         'Validation_size': n_val,
    #         'Transformation': Transformation
    #         }
    # RKHS_out = pd.DataFrame.from_dict(RKHS_out)
    
    
    #######################################################################################
    ###################################   Neural Net    ###################################
    #######################################################################################
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size = W_train_nnan.shape[0]
    epochs = 20
    n_node_list = np.linspace(1,37,19).astype(int)
    val_err = np.zeros((n_val, len(n_node_list)))
    model_NN_dict = {}
    beta_dict_NN = {}
    for cv_i, n_node in enumerate(n_node_list):

        model1 = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(W_train_nnan.shape[1],)),
                tf.keras.layers.Dense(n_node, activation="relu"),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_lags)
            ]
        )
        
        model1.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        
        model2 = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(W_train_nnan.shape[1],)),
                tf.keras.layers.Dense(n_node, activation="relu"),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1)
            ]
        )
        
        model2.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        
        model_NN = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(W_train_nnan.shape[1],)),
                tf.keras.layers.Dense(n_node, activation="relu"),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1)
            ]
        )
        # model_NN.summary()
        
        model_NN.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        
        model1.fit(W_train_nnan, V_train, batch_size=batch_size,
                epochs=epochs,verbose=0)
        
        resi_v = V_train.values - model1.predict(W_train_nnan,verbose=0)
        model2.fit(W_train_nnan, Y_train, batch_size=batch_size,
                epochs=epochs,verbose=0)
        resi_y = Y_train.values - model2.predict(W_train_nnan,verbose=0).reshape(-1,)
        beta = np.linalg.inv(resi_v.T @ resi_v) @ (resi_v.T @ resi_y)
        resi_train = Y_train.values - V_train.values @ beta
        
        model_NN_dict[cv_i] = model_NN
        model_NN_dict[cv_i].fit(W_train_nnan, resi_train, batch_size=batch_size,
                epochs=epochs,verbose=0)
        resi_hat = model_NN_dict[cv_i].predict(W_val_nnan,verbose=0)
        Y_hat = V_val.values @ beta + resi_hat.reshape(-1,)
        
        beta_dict_NN[cv_i] = beta
        val_err[:, cv_i] = Y_val.values-Y_hat
        
    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_NN = np.mean(np.array(val_err)**2, axis=0)

    Validation_Err['NN',seed] = pd.DataFrame()
    Validation_Err['NN',seed]['n_node_list'] = n_node_list
    Validation_Err['NN',seed]['val_err'] = val_err_NN
    beta_tot['NN',seed] = beta_dict_NN[min_idx]

    # plt.plot(n_node_list, val_err_NN)
    # plt.xlabel('Number of Nodes')
    # plt.title('Validation Error, NN, Minimum=%i'%n_node_list[min_idx])
    # # plt.show()
    # plt.savefig("Figures/NN_validation_seed%i.png"%seed)
    # plt.close()

    Y_hat = V_test.values @ beta_dict_NN[min_idx] + model_NN_dict[min_idx].predict(W_test_nnan,verbose=0).reshape(-1,)
    test_err_NN = Y_test.values - Y_hat
    RMSE_NN = np.sqrt(np.sum(test_err_NN**2)/len(test_err_NN))

    NN_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat.reshape((-1,)),
            'Model': 'NN-PL-%ilags' %num_lags,
            'Seed': seed,
            'Parameter': str(n_node_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    NN_out = pd.DataFrame.from_dict(NN_out)

    out = np.concatenate((RF_out.values, XGB_out.values, XGBs_out.values, PCR_out.values,PCRp_out.values, RKHS_out.values, NN_out.values), axis=0)

    query = ''' insert or replace into Results (Date,Target,Value,Prediction,Model,Seed,Parameter,Window_size,Validation_size,Transformation) values (?,?,?,?,?,?,?,?,?,?) '''
    cur.executemany(query, out)
    con.commit()
    con.close()

file_name1 ='Validation_Err_PL_%ilags_%s.pkl'%(num_lags, Transformation.replace(' ','_'))
file_name2 ='beta_tot_PL_%ilags_%s.pkl'%(num_lags,Transformation.replace(' ','_'))

with open(os.path.join('Results', file_name1), 'wb') as outp:
    pickle.dump(Validation_Err, outp)

with open(os.path.join('Results', file_name2), 'wb') as outp:
    pickle.dump(beta_tot, outp)
