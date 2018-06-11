import theano
import theano.tensor as T
from lasagne import layers
import numpy as np
from clip2frame import utils, measure
import network_structure as ns
import os
import json
import ast

#gpu usage
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu3")

if __name__ == '__main__':
    # Test settings
    build_func = ns.build_fcn_gaussian_multiscale
    test_measure_type_list = ['mean_auc_y', 'mean_auc_x', 'map_y', 'map_x']
    n_top_tags_te =50  # Number of tags used for testing

    #threshold setting
    tag_tr_fp = 'data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = 'data/data.magnatagatune/tag_list.top50.txt'

    tag_idx_list = utils.get_test_tag_50(tag_tr_fp, tag_te_fp)

    model_dir = 'data/models'
    thres_fp = os.path.join(model_dir, 'threshold.20160309_111546.with_magnatagatune.npy')
    thresholds_raw = np.load(thres_fp)
    thresholds = thresholds_raw[tag_idx_list]

    # Test data directory
    # The complete MagnaATagATune training/testing data can be downloaded from
    # http://mac.citi.sinica.edu.tw/~liu/data/exp_data.MagnaTagATune.188tags.zip
    # After downloading, replace the data_dir with the new directory path
    use_real_data = False

    # data_dir = '../data/HL30/standardize/'
    data_dir = 'std'

    # Files
    param_fp = 'data/models/model.20160309_111546.npz'
    tag_tr_fp = 'data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = 'data/data.magnatagatune/tag_list.top{}.txt'.format(n_top_tags_te)

    # Load tag list
    tag_tr_list = utils.read_lines(tag_tr_fp)
    tag_te_list = utils.read_lines(tag_te_fp)

    label_idx_list = [tag_tr_list.index(tag) for tag in tag_te_list]

    X_te_list = []
    # Load data
    print("Loading data...")
    X_te_list.append(np.load(data_dir +'/feat_test_512.npy'))
    X_te_list.append(np.load(data_dir +'/feat_test_1024.npy'))
    X_te_list.append(np.load(data_dir +'/feat_test_2048.npy'))
    X_te_list.append(np.load(data_dir +'/feat_test_4096.npy'))
    X_te_list.append(np.load(data_dir +'/feat_test_8192.npy'))
    X_te_list.append(np.load(data_dir +'/feat_test_16384.npy'))

    print(len(X_te_list), len(X_te_list[0]), len(X_te_list[0][0]), len(X_te_list[0][0][0]))

    # Building Network
    print("Building network...")
    num_scales = 6
    network, input_var_list, _, _ = build_func(num_scales)

    # Computing loss
    target_var = T.matrix('targets')
    epsilon = np.float32(1e-6)
    one = np.float32(1)

    output_va_var = layers.get_output(network, deterministic=True)
    output_va_var = T.clip(output_va_var, epsilon, one-epsilon)

    func_pr = theano.function(input_var_list, output_va_var)

    # Load params
    utils.load_model(param_fp, network)

    # Predict
    print('Predicting...')

    pred_list_raw = utils.predict_multiscale(X_te_list, func_pr)
    pred_all_raw = np.vstack(pred_list_raw)

    pred_all = pred_all_raw[:, label_idx_list]
    pred_binary = ((pred_all-thresholds) > 0).astype(int)

    np.save('test_data/tgte.jynet.npy', pred_all)
    # with open('tags/tag_list.txt', 'w') as f:
    #     for pred in pred_binary:
    #         f.write(str(list(pred)) + '\n')
