'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''

'''
utility function for cross validation
'''
# Import libraries
import numpy as np
from sklearn.model_selection import KFold

def cross_val(r,c,n_folds=5):
    
    # total samples
    num      = np.shape(r)[0]
    fold_len = np.int16(num / n_folds)

    kf_outer = KFold(n_splits=n_folds,random_state=None, shuffle=False)
    X_train,X_valid,X_test,Y_train,Y_valid,Y_test = [],[],[],[],[],[]

    for train_index, test_index in kf_outer.split(r):

        p = (test_index[-1])+1
        if p ==num:
            valid_index = train_index[0:fold_len]
            for v in reversed(range(fold_len)):
                train_index = np.delete(train_index,v)
            # print("Train index: ",train_index)
            # print("valid index: ",valid_index)
            # print("Test index: ",test_index)
        else:
            valid_index = train_index[p-fold_len:p]
            a_list = list(range(p-fold_len,p))
            for v in reversed(a_list):
                train_index = np.delete(train_index,v)
            # print("Train index: ",train_index)            
            # print("valid index: ",valid_index)
            # print("Test index: ",test_index)
        X_train.append(r[train_index])
        X_valid.append(r[valid_index])
        X_test.append(r[test_index])
        Y_train.append(c[train_index])
        Y_valid.append(c[valid_index])
        Y_test.append(c[test_index])
    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test


if __name__=="__main__":
    np.random.seed(5)
    r = np.random.randint(1,5,size = (100,10,10,2))
    c = np.random.randint(1,5,size = (100,10,10,1))


    X_train,X_valid,X_test,Y_train,Y_valid,Y_test = cross_val(r,c)

