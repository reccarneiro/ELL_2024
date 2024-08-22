import numpy as np
import pandas as pd

def get_data(seed):

    np.random.seed(seed) # For shuffling X_used and Y    
    data_used = pd.read_csv('Data/Data_used.csv')
    
    Y = data_used.happiness
    X = data_used.drop(['happiness'], axis=1)
    
    col_used = ['hh_inc', 'consumption_tot', 'savings_tot', 'hh_size', 'zerotofive_n','sixtotwenty_n','grp', 'home_own',
                'gender', 'HH_position', 'age', 'live_together', 'college_educ',
                'educ_years', 'work', 'get_social_benefit', 'religion','marriage', 'health', 'exercise', 'smoke', 'alcohol']
    
    X_used = X[col_used]
    
    X_used.loc[X_used.zerotofive_n!=0,'zerotofive_n'] = 1
    X_used.loc[X_used.sixtotwenty_n!=0,'sixtotwenty_n'] = 1
    X_used = X_used.rename(columns={'zerotofive_n':'zerotofive','sixtotwenty_n':'sixtotwenty'})
    X_used.loc[X_used.grp!=0,'grp'] = 1
    
    
    X_used.loc[X_used.home_own!=1,'home_own'] = 0
    X_used.loc[X_used.live_together==2,'live_together'] = 0
    
    get_social_benefit = X_used.get_social_benefit.replace({1:0, 3:0, 2:1})
    got_social_benefit = X_used.get_social_benefit.replace({1:1, 3:0, 2:0})
    
    X_used.loc[:,'get_social_benefit'] = get_social_benefit
    X_used = X_used.assign(got_social_benefit=got_social_benefit.values)
    
    X_used.loc[X_used.HH_position!=10,'HH_position'] = 0
    X_used.loc[X_used.HH_position==10,'HH_position'] = 1
    X_used = X_used.rename(columns={'HH_position':'isHHH'})
    
    religion = X_used.religion.replace({1:0, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1})
    X_used.loc[:,'religion'] = religion
    
    marriage = X_used.marriage.replace({1:0, 2:1, 3:0, 4:0, 5:0})
    X_used.loc[:,'marriage'] = marriage
    
    X_used.loc[np.isin(X_used.smoke,[2,3]),'smoke'] = 0
    X_used.loc[np.isin(X_used.alcohol,[2,3]),'alcohol'] = 0
    
    health = X_used['health']
    X_used.loc[:,'health'] = 6-health # 5 means very healthy
    
    exercise = X_used['exercise']
    X_used.loc[:,'exercise'] = 4-exercise # 3 means exercise regularly, 0 means no exercise
    
    w_low = np.quantile(X_used.hh_inc, 0.01)
    w_high = np.quantile(X_used.hh_inc, 0.99)
    idx = (X_used.hh_inc >= w_low) & (X_used.hh_inc <= w_high)
    
    Y = Y[idx].reset_index(drop=True)
    X_used = X_used[idx].reset_index(drop=True)
    
    X_used.hh_inc = np.log(X_used.hh_inc/np.sqrt(X_used.hh_size))
    X_used.savings_tot = X_used.savings_tot/np.sqrt(X_used.hh_size)

    Y = Y.drop(np.where(X_used.consumption_tot<1)[0],axis=0).reset_index(drop=True)
    X_used = X_used.drop(np.where(X_used.consumption_tot<1)[0],axis=0).reset_index(drop=True)
    X_used.consumption_tot = np.log(X_used.consumption_tot/np.sqrt(X_used.hh_size))
    
    idx = np.random.permutation(X_used.index) # Shuffle rows of X_used and Y in the same way in case rows are ordered in a specific way
    X_used = X_used.reindex(idx).reset_index(drop=True)
    Y = Y.reindex(idx).reset_index(drop=True)
    return X_used, Y

def Gaussian(u): 
    val = 1/(np.sqrt(2*np.pi))*np.exp(-u**2/2)
    return val
