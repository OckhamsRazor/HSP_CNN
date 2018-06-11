import os
import numpy as np
from clip2frame import utils
import json

floatX = 'float32'
length = 911

def _append_zero_row(array, n_total_row):
    r, c = array.shape
    if r >= n_total_row:
        return array
    else:
        temp = np.zeros((n_total_row-r, c))
        return np.vstack([array, temp])

def make_batch_feat(feat_fp_list, length=911):
    feat = [
        _append_zero_row(
            np.load(term), length)[None, None, :length, :].astype(floatX)
        for term in feat_fp_list]
 
    feat = np.vstack(feat)
    return feat

feat_dir = "feature/"

fn_list = []
for root, dirs, files in os.walk("feature/"):
    for in_fn in files:
    	if in_fn.endswith('.npy'):
    		fn_list.append(in_fn)
fn_list.sort()

with open('file_list.txt', 'w') as outfile:
    json.dump(fn_list, outfile)

feat_fp_list = [os.path.join(feat_dir, fn) for fn in fn_list]
feat = make_batch_feat(feat_fp_list, length).astype(floatX)

np.save("feat_test_512.npy", feat)