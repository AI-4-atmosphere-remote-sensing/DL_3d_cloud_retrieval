
#train test split for 5-fold cross validation

from sklearn.model_selection import KFold
import numpy as np
def cross_val(r,c,ts,os,n_folds=5):
    
    kf = KFold(n_splits=n_folds,random_state=None, shuffle=False)
    ratio=int(r.shape[0]/n_folds)
    X_train=np.zeros((n_folds,ratio*(n_folds-1),ts))
    y_train=np.zeros((n_folds,ratio*(n_folds-1),os))
    X_test=np.zeros((n_folds,ratio,ts))
    y_test=np.zeros((n_folds,ratio,os))
    count=0
    for train_index, test_index in kf.split(r):
        X_train[count], X_test[count] = r[train_index], r[test_index]
        y_train[count], y_test[count] = c[train_index], c[test_index]
        count+=1
    return X_train,X_test,y_train,y_test







