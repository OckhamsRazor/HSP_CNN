import os
import numpy as np
from clip2frame import utils
import json
import shutil

floatX = 'float32'
length = 911
win_size = 512

#gpu usage
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu1")

def _append_zero_row(array, n_total_row):
    r, c = array.shape
    if r >= n_total_row:
        return array
    else:
        temp = np.zeros((n_total_row-r, c))
        return np.vstack([array, temp])

def make_batch_feat(feat_fp_list, length=911):
    feat = []
    a_array = []
    feat_fp_list.sort()
    for idx, term in enumerate(feat_fp_list):
        # print(term)
        np.load(term)
        a_array = _append_zero_row(np.load(term), length)[None, None, :length, :].astype(floatX)
        feat.append(a_array)

    feat = np.vstack(feat)
    return feat


dump_path = "ex_data/"
if os.path.exists(dump_path):
    shutil.rmtree(dump_path)
os.makedirs(dump_path)
for win_size in [512, 1024,2048,4096,8192, 16384]:

    #path
    feat_dir = "jy_feat/out"+str(win_size)+"/"

    #get files
    fn_list = []
    for files in os.listdir(feat_dir):
        if files.endswith('.npy'):
        	fn_list.append(files)
    fn_list.sort()

    #make files
    feat_fp_list = [os.path.join(feat_dir, fn) for fn in fn_list]
    feat = make_batch_feat(feat_fp_list, length).astype(floatX)

    #save files
    np.save(dump_path + "feat_test_"+str(win_size)+".npy", feat)
    np.save(dump_path + 'name.npy', fn_list)
