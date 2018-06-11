import numpy as np
from sklearn import preprocessing as pp
import os
import shutil

size = 512

def standardize(feat, scaler=None):
    if scaler is None:
        scaler = pp.StandardScaler().fit(feat)
    out = scaler.transform(feat)
    return out, scaler

if __name__ == '__main__':
    dump_path = "std/"
    if os.path.exists(dump_path):
        shutil.rmtree(dump_path)
    os.makedirs(dump_path)

    for size in [512, 1024,2048,4096,8192, 16384]:
        #path
        feat_te_fp = 'ex_data/feat_test_'+str(size)+'.npy'

        # standardize
        feat_te = np.load(feat_te_fp)
        k = feat_te.shape[-1]
        n = feat_te.shape[0]
        feat_te = feat_te.reshape((-1, k))
        feat_te_s, scaler = standardize(feat_te)
        feat_te_s = feat_te_s.reshape((n, 1, -1, k))

        np.save(dump_path+"feat_test_"+str(size)+".npy", feat_te_s)
