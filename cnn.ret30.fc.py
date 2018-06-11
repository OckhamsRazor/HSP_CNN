'''
Fully CNN approach
'''
import argparse
from math import floor, log2
from os import fsync, mkdir, path
from shutil import rmtree
from sys import stdout

import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau, spearmanr
from tensorflow.contrib import learn

# from trend_clstr import DTWDistance


MEL_BIN = 128
FRAME_NUM = 323
TAG_SIZE = 50
A2V_SIZE = 40
DR = 0.25
LR = 5e-3
LR_DECAY = 0.9
TRAIN_SIZE = 0.8  # 80% data for training
VAL_SIZE = 0.1
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE
W = 5
TIMESPAN = 60
RET_TIMESPAN = 29


GPU_USAGE = 0.4


def leaky_relu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


NON_LINEAR = leaky_relu


def norm(x):
    return -1 / np.log10(x)


def denorm(x):
    x[x < 0.0] = 0.0
    return 10 ** (-1 / x)


def make_tag_batch(tg, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_tg = np.take(tg, ids, axis=0)

    return b_tg


def make_a2v_batch(a2v, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_a2v = np.take(a2v, ids, axis=0)

    return b_a2v


def make_batch(feat, pt, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_f = np.take(feat, ids, axis=0)
    b_pt = np.take(pt, ids, axis=0) if pt is not None else None

    return b_f, b_pt


def make_mt_batch(feat, pt, trd, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_trd = np.take(trd, ids, axis=0)
    b_f, b_pt = make_batch(feat, pt, trix, pos, b_size)
    return b_f, b_pt, b_trd


def make_mr_batch(feat, pt, ret, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_ret = np.take(ret, ids, axis=0)
    b_f, b_pt = make_batch(feat, pt, trix, pos, b_size)
    return b_f, b_pt, b_ret


def make_mtr_batch(feat, pt, trd, ret, trix, pos, b_size):
    ids = trix[pos:pos + b_size]
    b_ret = np.take(ret, ids, axis=0)

    b_f, b_pt, b_trd = make_mt_batch(feat, pt, trd, trix, pos, b_size)
    return b_f, b_pt, b_trd, b_ret


def conv1_128_4_gen(x):
    return tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[128, 4],
        activation=NON_LINEAR)


def cnn(x, dr, mode, stt):
    # 128*323*1

    # 1*320*32
    return cnn_body(conv1_128_4_gen(x), dr, mode, stt)


def inception_cnn(x, dr, mode, stt):
    # 128*323*1

    # 1*320*32
    conv1_128_4 = conv1_128_4_gen(x)

    # 132*327*1
    pad1_132_8 = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])
    # 1*320*16
    conv1_132_8 = tf.layers.conv2d(
        inputs=pad1_132_8,
        filters=16,
        kernel_size=[132, 8],
        activation=NON_LINEAR)

    # 140*335*1
    pad1_140_16 = tf.pad(x, [[0, 0], [6, 6], [6, 6], [0, 0]])
    # 1*320*16
    conv1_140_16 = tf.layers.conv2d(
        inputs=pad1_140_16,
        filters=16,
        kernel_size=[140, 16],
        activation=NON_LINEAR)

    # 1*320*64
    concat = tf.concat([conv1_128_4, conv1_132_8, conv1_140_16], axis=3)
    # 1*320*32
    conv1 = tf.layers.conv2d(
        inputs=concat,
        filters=32,
        kernel_size=[1, 1],
        activation=NON_LINEAR)

    return cnn_body(conv1, dr, mode, stt)


def cnn_body(head, dr, mode, stt):
    # 1*320*32

    logits_all = {}

    # 1*160*32
    pool1 = tf.layers.max_pooling2d(
        inputs=head, pool_size=[1, 2], strides=[1, 2])

    # 1*157*64
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[1, 4],
        activation=NON_LINEAR)

    # 1*78*64
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 2])
    drop1 = tf.layers.dropout(
        inputs=pool2, rate=dr, training=(mode == learn.ModeKeys.TRAIN))

    # 1*78*256
    conv3 = tf.layers.conv2d(
        inputs=drop1,
        filters=256,
        kernel_size=[1, 1],
        activation=NON_LINEAR)
    drop2 = tf.layers.dropout(
        inputs=conv3, rate=dr, training=(mode == learn.ModeKeys.TRAIN))
    conv4 = tf.layers.conv2d(
        inputs=drop2,
        filters=256,
        kernel_size=[1, 1],
        activation=NON_LINEAR)
    drop3 = tf.layers.dropout(
        inputs=conv4, rate=dr, training=(mode == learn.ModeKeys.TRAIN))

    # 1*78*1
    conv5 = tf.layers.conv2d(
        inputs=drop3,
        filters=1,
        kernel_size=[1, 1],
        activation=NON_LINEAR)

    # 1
    logits_pt = tf.reduce_mean(
        conv5, axis=[1, 2], name='GlobalAveragePooling')

    if 't' in stt:
        # 1*78*TIMESPAN
        conv5_trd = tf.layers.conv2d(
            inputs=drop3,
            filters=TIMESPAN,
            kernel_size=[1, 1],
            activation=NON_LINEAR)
        # TIMESPAN
        logits_trd = tf.reduce_mean(
            conv5_trd, axis=[1, 2], name='GlobalAveragePooling')
        logits_all['trd'] = logits_trd

    if 'r' in stt:
        # 1*78*RET_TIMESPAN
        conv5_ret = tf.layers.conv2d(
            inputs=drop3,
            filters=RET_TIMESPAN,
            kernel_size=[1, 1],
            activation=NON_LINEAR)
        # RET_TIMESPAN
        logits_ret = tf.reduce_mean(
            conv5_ret, axis=[1, 2], name='GlobalAveragePooling')
        logits_all['ret'] = logits_ret

    logits_all['pt'] = logits_pt
    return logits_all


def tag_regression_clsf(tags, dr, mode):
    dense1 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=tags,
            units=100,
            activation=NON_LINEAR,
            name='dense1'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    dense2 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=dense1,
            units=100,
            activation=NON_LINEAR,
            name='dense2'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    dense3 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=dense2,
            units=30,
            activation=NON_LINEAR,
            name='dense3'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    logits_tag = tf.layers.dense(
        inputs=dense3,
        units=1,
        activation=NON_LINEAR,
        name='logits_tag')

    return logits_tag


def a2v_regression_clsf(a2v, dr, mode):
    dense1 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=a2v,
            units=100,
            activation=NON_LINEAR,
            name='a2v_dense1'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    dense2 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=dense1,
            units=100,
            activation=NON_LINEAR,
            name='a2v_dense2'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    dense3 = tf.layers.dropout(
        inputs=tf.layers.dense(
            inputs=dense2,
            units=30,
            activation=NON_LINEAR,
            name='a2v_dense3'),
        rate=dr,
        training=(mode == learn.ModeKeys.TRAIN))
    logits_a2v = tf.layers.dense(
        inputs=dense3,
        units=1,
        activation=NON_LINEAR,
        name='logits_a2v')

    return logits_a2v


def test_only(args):
    print ('Test only.')
    stdout.flush()

    train_set = args.train_set
    pred_f = '{}.test.pred'.format(args.output)
    log_f = '{}.test.log'.format(args.output)
    batch_size = int(args.batch_size)
    model_type = args.model_type
    stt = args.secondary_target_type
    use_tag = args.use_tag
    tag_type = args.tag_type
    use_a2v = args.use_a2v
    dr_rate = float(args.dropout_rate)
    model_dir = '{}.mdl/'.format(args.output)
    loss_t_w = float(args.tagging_loss_weight)
    loss_a2v_w = float(args.a2v_loss_weight)

    flog = open(log_f, 'w')
    feat = np.load(
        path.join(train_set, 'feat.npy')).reshape(-1, MEL_BIN, FRAME_NUM, 1)
    ids = np.load(path.join(train_set, 'id.feat.npy'))
    # trd = np.load(path.join(train_set, 'trend.feat.npy'))
    ret = np.load(path.join(train_set, 'ret.feat.npy'))

    # trd = -1 / np.log10(trd)
    # trd = norm(trd)
    ret = norm(ret)

    data_num = ids.size

    tr_val = floor(data_num * (TRAIN_SIZE + VAL_SIZE))
    teix = np.arange(tr_val, data_num)
    test_feat = np.take(feat, teix, axis=0)
    test_ids = np.take(ids, teix)
    # test_trd = np.take(trd, teix, axis=0)
    test_ret = np.take(ret, teix, axis=0)

    # test_trd_last = test_trd[:, -1].reshape(-1, 1)
    test_ret_30 = np.mean(test_ret, axis=1).reshape(-1, 1)

    # timespan = test_trd.shape[1]
    # ret_timespan = None
    test_pt, y_size = None, 1
    # test_pt = test_trd_last
    test_pt = test_ret_30

    test_tags, test_a2v = None, None
    if use_tag:
        test_tags = np.load(
            path.join(train_set, 'tgte.{}.npy'.format(tag_type)))
    if use_a2v:
        test_a2v = np.load(
            path.join(train_set, 'a2vte.npy'))

    x_f = tf.placeholder(tf.float32, [None, MEL_BIN, FRAME_NUM, 1])
    y_t = tf.placeholder(tf.float32, [None, y_size])
    mode = tf.placeholder(tf.string)  # TRAIN, EVAL, INFER
    tags, a2v = None, None
    if use_tag:
        tags = tf.placeholder(tf.float32, [None, TAG_SIZE])
    if use_a2v:
        a2v = tf.placeholder(tf.float32, [None, A2V_SIZE])
    # y_trd, y_trd_input, y_ret, y_ret_input = None, None, None, None
    # if 't' in stt:
    #     y_trd = tf.placeholder(tf.float32, [None, TIMESPAN])
    # if 'r' in stt:
    #     y_ret = tf.placeholder(tf.float32, [None, RET_TIMESPAN])

    logits_all = None
    if model_type == 'incept':
        flog.write('Model type: Inception CNN.\n')
        logits_all = inception_cnn(
            x_f, dr_rate, mode, stt)
    else:
        flog.write('Model type: Plain CNN.\n')
        logits_all = cnn(
            x_f, dr_rate, mode, stt)

    logits_pt = logits_all['pt']  # primary target(s)
    if use_tag:
        logits_tag = tag_regression_clsf(
            tags, dr_rate, mode)
        # logits_pt += loss_t_w * logits_tag
    if use_a2v:
        logits_a2v = a2v_regression_clsf(
            a2v, dr_rate, mode)
        # logits_pt += loss_a2v_w * logits_a2v

    if use_tag and use_a2v:
        logits_pt = loss_t_w * logits_tag + loss_a2v_w * logits_a2v + \
            (1 - loss_t_w - loss_a2v_w) * logits_pt
    elif use_tag:
        logits_pt = (1 - loss_t_w) * logits_pt + loss_t_w * logits_tag
    elif use_a2v:
        logits_pt = (1 - loss_a2v_w) * logits_pt + loss_a2v_w * logits_a2v

    loss_pt = tf.losses.mean_squared_error(
        labels=y_t,
        predictions=logits_pt)

    test(model_dir, test_feat, test_pt, test_ids, test_tags, test_a2v, teix,
         batch_size, loss_pt, logits_pt, x_f, y_t, tags, a2v, mode, flog,
         pred_f, stt, use_tag, use_a2v)
    flog.close()


def score_pred_only(args):
    print ('Pred only.')
    stdout.flush()

    train_set = args.train_set
    batch_size = int(args.batch_size)
    model_type = args.model_type
    stt = args.secondary_target_type
    use_tag = args.use_tag
    tag_type = args.tag_type
    dr_rate = float(args.dropout_rate)
    model_dir = '{}.mdl/'.format(args.output)
    loss_t_w = float(args.tagging_loss_weight)
    pred_f = 'ret30.npy'

    test_feat = np.load(
        path.join(train_set, 'xte.npy')).reshape(-1, MEL_BIN, FRAME_NUM, 1)
    y_num = test_feat.shape[0]

    test_pt, y_size = None, 1
    test_tags = None
    if use_tag:
        test_tags = np.load(
            path.join(train_set, 'tgte.{}.npy'.format(tag_type)))

    x_f = tf.placeholder(tf.float32, [None, MEL_BIN, FRAME_NUM, 1])
    y_t = tf.placeholder(tf.float32, [None, y_size])
    mode = tf.placeholder(tf.string)  # TRAIN, EVAL, INFER
    tags = None
    if use_tag:
        tags = tf.placeholder(tf.float32, [None, TAG_SIZE])

    logits_all = None
    if model_type == 'incept':
        print ('Model type: Inception CNN.\n')
        logits_all = inception_cnn(
            x_f, dr_rate, mode, stt)
    else:
        print ('Model type: Plain CNN.\n')
        logits_all = cnn(
            x_f, dr_rate, mode, stt)

    logits_pt = logits_all['pt']  # primary target(s)
    if use_tag:
        logits_tag = tag_regression_clsf(
            tags, dr_rate, mode)
        logits_pt += loss_t_w * logits_tag

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path.join(model_dir, 'model.ckpt'))
        test_logitss = None
        for test_pos in range(0, test_feat.shape[0], batch_size):
            b_f, b_y = make_batch(
                test_feat, test_pt, np.arange(y_num), test_pos, batch_size)
            if use_tag:
                b_tg = make_tag_batch(
                    test_tags, np.arange(y_num), test_pos, batch_size)
                test_logits = sess.run(
                    logits_pt,
                    feed_dict={
                        x_f: b_f, tags: b_tg,
                        mode: learn.ModeKeys.INFER})
            else:
                test_logits = sess.run(
                    logits_pt,
                    feed_dict={
                        x_f: b_f,
                        mode: learn.ModeKeys.INFER})
            if test_logitss is None:
                test_logitss = test_logits  # * NORM_FACTOR
            else:
                test_logitss = np.concatenate(
                    (test_logitss, test_logits), axis=0)
    # print (test_logitss)
    print (denorm(test_logitss))
    np.save(pred_f, denorm(test_logitss))
    # np.save('{}.')


def get_ndcg(pred_rank, ref_rank, overall_ref):
    dcg, idcg = 0.0, 0.0
    for i, r in enumerate(ref_rank):
        idcg += overall_ref[r] / log2(i + 2)
    for i, r in enumerate(pred_rank):
        dcg += overall_ref[r] / log2(i + 2)

    return dcg / idcg


def test(model_dir, test_feat, test_pt, test_ids, test_tags, test_a2v,
         teix, batch_size, loss_pt, logits, x_f, y_t, tags, a2v,
         mode, flog, pred_f, stt, use_tag, use_a2v, prev_val_loss=None):

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_USAGE)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, path.join(model_dir, 'model.ckpt'))
        test_losses = []
        test_logitss = None
        # test_dtw = []
        test_mse = []
        for test_pos in range(0, test_feat.shape[0], batch_size):
            b_f, b_y = make_batch(
                test_feat, test_pt, np.arange(teix.size), test_pos, batch_size)
            if use_tag:
                b_tg = make_tag_batch(
                    test_tags, np.arange(teix.size), test_pos, batch_size)
            if use_a2v:
                b_a2v = make_a2v_batch(
                    test_a2v, np.arange(teix.size), test_pos, batch_size)

            if use_tag:
                if use_a2v:
                    test_loss, test_logits, = sess.run(
                        [loss_pt, logits],
                        feed_dict={
                            x_f: b_f, y_t: b_y, tags: b_tg, a2v: b_a2v,
                            mode: learn.ModeKeys.INFER})
                else:
                    test_loss, test_logits, = sess.run(
                        [loss_pt, logits],
                        feed_dict={
                            x_f: b_f, y_t: b_y, tags: b_tg,
                            mode: learn.ModeKeys.INFER})
            else:
                if use_a2v:
                    test_loss, test_logits, = sess.run(
                        [loss_pt, logits],
                        feed_dict={
                            x_f: b_f, y_t: b_y, a2v: b_a2v,
                            mode: learn.ModeKeys.INFER})
                else:
                    test_loss, test_logits, = sess.run(
                        [loss_pt, logits],
                        feed_dict={
                            x_f: b_f, y_t: b_y, mode: learn.ModeKeys.INFER})

            test_losses.append(test_loss)
            if test_logitss is None:
                test_logitss = test_logits  # * NORM_FACTOR
            else:
                test_logitss = np.concatenate(
                    (test_logitss, test_logits), axis=0)

            trd_pred = test_logits  # * NORM_FACTOR
            trd_ref = b_y  # * NORM_FACTOR
            for ref, pred in zip(trd_ref, trd_pred):
                # test_dtw.append(DTWDistance(ref, pred, W))
                test_mse.append(np.sum((ref - pred) ** 2) ** 0.5)

        # true_dtw = sum(test_dtw) / len(test_dtw)
        true_mse = sum(test_mse) / len(test_mse)
        test_loss_overall = sum(test_losses) / len(test_losses)
        # test_ref *= NORM_FACTOR

        overall_ref = denorm(test_pt)
        overall_pred = denorm(test_logitss)
        overall_ref = overall_ref.reshape(-1)
        overall_pred = overall_pred.reshape(-1)

        overall_mse = np.sqrt(
            np.mean(
                (overall_ref - overall_pred) ** 2))

        ref_rank_100 = np.flipud(np.argsort(overall_ref))[:100]
        pred_rank_100 = np.flipud(np.argsort(overall_pred))[:100]
        ref_rank_150 = np.flipud(np.argsort(overall_ref))[:150]
        pred_rank_150 = np.flipud(np.argsort(overall_pred))[:150]
        rec100 = len(set(ref_rank_100) & set(pred_rank_100))
        rec150 = len(set(ref_rank_150) & set(pred_rank_150))

        ndcg = get_ndcg(pred_rank_150, ref_rank_150, overall_ref)

        ref_rank_150_set = set(ref_rank_150)
        pred_150, ref_150 = [], []
        for pid, pred in enumerate(overall_pred):
            if pid in ref_rank_150_set:
                pred_150.append(pred)
                ref_150.append(overall_ref[pid])
        pred_150 = np.array(pred_150)
        ref_150 = np.array(ref_150)

        tau150, _ = kendalltau(pred_150, ref_150)
        rho150, _ = spearmanr(pred_150, ref_150)
        tau, _ = kendalltau(overall_pred, overall_ref)
        rho, _ = spearmanr(overall_pred, overall_ref)

        if prev_val_loss is not None:
            flog.write(
                'Minimum Validation Loss: {:.6e}\n'.format(prev_val_loss))
        flog.write('Testing MSE: {:.6e}\n'.format(test_loss_overall))
        # flog.write('True MSE/DTW: {:.6e}\t{:.6e}\n'.format(true_mse, true_dtw))
        flog.write('Overall RMSE: {:.6e}\n\n'.format(overall_mse))
        flog.write('Recall@100: {:d}\n'.format(rec100))
        flog.write('Recall@150: {:d}\n'.format(rec150))
        flog.write('nDCG@150: {:.6f}\n'.format(ndcg))
        flog.write('Kendall tau@150: {:.6f}\n'.format(tau150))
        flog.write('Spearman rho@150: {:.6f}\n'.format(rho150))
        flog.write('Kendall tau: {:.6f}\n'.format(tau))
        flog.write('Spearman rho: {:.6f}\n\n'.format(rho))

        np.save(pred_f, overall_pred)
        np.save('{}.id'.format(pred_f), test_ids)


def main():
    parser = argparse.ArgumentParser(
        description='STP (Song Trend Prediction)' +
        ' baseline (ICASSP 17, Richard) with CNN regression')
    parser.add_argument(
        '-i', '--train_set', help='training data dir', required=True
    )
    parser.add_argument(
        '-o', '--output', help='output name', required=True)
    parser.add_argument(
        '-ep', '--ep_num', help='episode num for training', default=150
    )
    parser.add_argument(
        '-b', '--batch_size', help='mini batch size', default=10
    )
    parser.add_argument(
        '-t', '--model_type', help='[plain]|incept', default='plain'
    )
    parser.add_argument(
        '-stt', '--secondary_target_type',
        help='[m]|mt|mtr: main, main+trend60, main+trend60+retention30',
        default='m'
    )
    parser.add_argument(
        '-op', '--optimizer_type',
        help='[adagrad]|adam|sgd', default='adagrad')
    parser.add_argument(
        '-sh', '--to_shuffle', action='store_true', default=False)
    parser.add_argument(
        '-tg', '--use_tag', action='store_true',
        help='Use music tags (from JYnet) for training', default=False
    )
    parser.add_argument(
        '-tgt', '--tag_type',
        help='[jynet]|densenet|resnet', default='jynet')
    parser.add_argument(
        '-wt', '--tagging_loss_weight', default=0.5)
    parser.add_argument(
        '-a2v', '--use_a2v', action='store_true',
        help='Use Audio2Vec (by SY) for training', default=False
    )
    parser.add_argument(
        '-wa2v', '--a2v_loss_weight', default=0.5)
    parser.add_argument(
        '-lr', '--learning_rate', default=LR)
    parser.add_argument(
        '-dr', '--dropout_rate', default=DR)
    parser.add_argument(
        '-test', '--to_test', action='store_true', default=False)
    parser.add_argument(
        '-ps', '--to_pred_score', action='store_true', default=False)

    args = parser.parse_args()

    to_pred_score = args.to_pred_score
    if to_pred_score:
        score_pred_only(args)
        return

    to_test = args.to_test
    if to_test:
        test_only(args)
        return

    train_set = args.train_set
    pred_f = '{}.pred'.format(args.output)
    log_f = '{}.log'.format(args.output)
    ep_num = int(args.ep_num)
    batch_size = int(args.batch_size)
    model_type = args.model_type
    optimizer_type = args.optimizer_type
    to_shuffle = args.to_shuffle
    stt = args.secondary_target_type
    use_tag = args.use_tag
    tag_type = args.tag_type
    use_a2v = args.use_a2v
    init_lrt = float(args.learning_rate)
    dr_rate = float(args.dropout_rate)
    loss_t_w = float(args.tagging_loss_weight)
    loss_a2v_w = float(args.a2v_loss_weight)

    flog = open(log_f, 'w')
    model_dir = '{}.mdl/'.format(args.output)
    if path.exists(model_dir):
        rmtree(model_dir)
    mkdir(model_dir)

    # feat:    3D, data_num * MEL_BIN * FRAME_NUM
    # clss:    1D, data_num (int; 1:clsn)
    # ids:     1D, data_num (int)
    # trd_max: 1D, data_num (float)
    # trd:     2D, data_num * time_len (float)
    feat = np.load(
        path.join(train_set, 'feat.npy')).reshape(-1, MEL_BIN, FRAME_NUM, 1)
    ids = np.load(path.join(train_set, 'id.feat.npy'))
    trd = np.load(path.join(train_set, 'trend.feat.npy'))
    ret = np.load(path.join(train_set, 'ret.feat.npy'))

    # trd = -1 / np.log10(trd)
    # trd = norm(trd)
    ret = norm(ret)

    data_num = ids.size

    tr_val = floor(data_num * (TRAIN_SIZE + VAL_SIZE))
    trix = np.arange(tr_val)
    teix = np.arange(tr_val, data_num)
    train_feat = np.take(feat, trix, axis=0)
    test_feat = np.take(feat, teix, axis=0)
    test_ids = np.take(ids, teix)
    train_trd = np.take(trd, trix, axis=0)
    # test_trd = np.take(trd, teix, axis=0)
    train_ret = np.take(ret, trix, axis=0)
    test_ret = np.take(ret, teix, axis=0)

    train_ret_30 = np.mean(train_ret, axis=1).reshape(-1, 1)
    test_ret_30 = np.mean(test_ret, axis=1).reshape(-1, 1)

    # timespan = train_trd.shape[1]
    # ret_timespan = None
    train_pt, test_pt, y_size = None, None, 1
    train_pt = train_ret_30
    test_pt = test_ret_30

    train_tags, test_tags = None, None
    if use_tag:
        flog.write('Use tag from {}.\n'.format(tag_type))
        # all_tags = np.load(path.join(train_set, 'tg.npy'))
        train_tags = np.load(
            path.join(train_set, 'tgtr.{}.npy'.format(tag_type)))
        # valid_tags = np.take(all_tags, vaix, axis=0)
        valid_tags = np.load(
            path.join(train_set, 'tgva.{}.npy'.format(tag_type)))
        train_tags = np.concatenate((train_tags, valid_tags), axis=0)
        test_tags = np.load(
            path.join(train_set, 'tgte.{}.npy'.format(tag_type)))

    train_a2v, test_a2v = None, None
    if use_a2v:
        flog.write('Use Audio2Vec.\n')
        train_a2v = np.load(
            path.join(train_set, 'a2vtr.npy'))
        valid_a2v = np.load(
            path.join(train_set, 'a2vva.npy'))
        train_a2v = np.concatenate((train_a2v, valid_a2v), axis=0)
        test_a2v = np.load(
            path.join(train_set, 'a2vte.npy'))

    x_f = tf.placeholder(tf.float32, [None, MEL_BIN, FRAME_NUM, 1])
    y_t = tf.placeholder(tf.float32, [None, y_size])
    mode = tf.placeholder(tf.string)  # TRAIN, EVAL, INFER
    lrt = tf.placeholder(tf.float32)
    tags, a2v = None, None
    if use_tag:
        tags = tf.placeholder(tf.float32, [None, TAG_SIZE])
    if use_a2v:
        a2v = tf.placeholder(tf.float32, [None, A2V_SIZE])
    y_trd, y_ret = None, None
    if 't' in stt:
        y_trd = tf.placeholder(tf.float32, [None, TIMESPAN])
    if 'r' in stt:
        y_ret = tf.placeholder(tf.float32, [None, RET_TIMESPAN])

    logits_all = None
    if model_type == 'incept':
        flog.write('Model type: Inception CNN.\n')
        logits_all = inception_cnn(
            x_f, dr_rate, mode, stt)
    else:
        flog.write('Model type: Plain CNN.\n')
        logits_all = cnn(
            x_f, dr_rate, mode, stt)

    loss_trd = None
    if 't' in stt:
        loss_trd = tf.losses.mean_squared_error(
            labels=y_trd,
            predictions=logits_all['trd'])

    loss_ret = None
    if 'r' in stt:
        loss_ret = tf.losses.mean_squared_error(
            labels=y_ret,
            predictions=logits_all['ret'])

    logits_tag, logits_a2v = 0, 0
    logits_pt = logits_all['pt']  # primary target(s)
    if use_tag:
        logits_tag = tag_regression_clsf(
            tags, dr_rate, mode)
    if use_a2v:
        logits_a2v = a2v_regression_clsf(
            a2v, dr_rate, mode)

    if use_tag and use_a2v:
        logits_pt = loss_t_w * logits_tag + loss_a2v_w * logits_a2v + \
            (1 - loss_t_w - loss_a2v_w) * logits_pt
    elif use_tag:
        logits_pt = (1 - loss_t_w) * logits_pt + loss_t_w * logits_tag
    elif use_a2v:
        logits_pt = (1 - loss_a2v_w) * logits_pt + loss_a2v_w * logits_a2v

    loss_pt = tf.losses.mean_squared_error(
        labels=y_t,
        predictions=logits_pt)
    loss = loss_pt
    if 't' in stt:
        loss += loss_trd
    if 'r' in stt:
        loss += loss_ret

    optimizer = None
    use_lr_decay = True
    if optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lrt).minimize(loss)
        use_lr_decay = False
    elif optimizer_type == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lrt).minimize(loss)
    else:
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=lrt).minimize(loss)

    tr_size = floor(data_num * TRAIN_SIZE)
    ep_size = floor(data_num * TRAIN_SIZE / batch_size)  # steps per episode
    val_start = tr_size + 1
    val_end = floor(data_num * (TRAIN_SIZE + VAL_SIZE))
    total_steps = ep_num * ep_size
    flog.write('total_steps: {:d}\n'.format(total_steps))
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_USAGE)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        current_lr = init_lrt
        losses = []
        prev_val_loss = None
        if to_shuffle:
            np.random.shuffle(trix)
        for step in range(total_steps):
            stdout.flush()
            pos = (step * batch_size) % tr_size
            if pos + batch_size > tr_size or (pos == 0 and losses):
                flog.write('ep {:.6f} passed. LR: {:.6f}\n'.format(
                    (step / ep_size), current_lr))
                val_losses = []
                for val_pos in range(val_start, val_end, batch_size):
                    b_f, b_y = make_batch(
                        train_feat, train_pt,
                        trix, val_pos, batch_size)
                    val_loss = None
                    if use_tag:
                        b_tg = make_tag_batch(
                            train_tags, trix, val_pos, batch_size)
                    if use_a2v:
                        b_a2v = make_a2v_batch(
                            train_a2v, trix, val_pos, batch_size)

                    if use_tag:
                        if use_a2v:
                            val_loss = sess.run(
                                loss_pt, feed_dict={
                                    x_f: b_f, y_t: b_y, tags: b_tg, a2v: b_a2v,
                                    mode: learn.ModeKeys.EVAL})
                        else:
                            val_loss = sess.run(
                                loss_pt, feed_dict={
                                    x_f: b_f, y_t: b_y, tags: b_tg,
                                    mode: learn.ModeKeys.EVAL})
                    else:
                        if use_a2v:
                            val_loss = sess.run(
                                loss_pt, feed_dict={
                                    x_f: b_f, y_t: b_y, a2v: b_a2v,
                                    mode: learn.ModeKeys.EVAL})
                        else:
                            val_loss = sess.run(
                                loss_pt, feed_dict={
                                    x_f: b_f, y_t: b_y,
                                    mode: learn.ModeKeys.EVAL})

                    val_losses.append(val_loss)
                ave_train_loss = sum(losses) / len(losses)
                ave_val_loss = sum(val_losses) / len(val_losses)
                flog.write('  training loss: {:.6f}\n'.format(ave_train_loss))
                flog.write('  validation loss: {:.6f}\n'.format(ave_val_loss))
                losses = []
                if prev_val_loss is None or ave_val_loss < prev_val_loss:
                    prev_val_loss = ave_val_loss
                    saver.save(sess, path.join(model_dir, 'model.ckpt'))

                elif use_lr_decay:
                    current_lr *= LR_DECAY

                if floor(step / ep_size) % 10 == 0:
                    test(model_dir, test_feat, test_pt, test_ids, test_tags,
                         test_a2v, teix, batch_size, loss_pt, logits_pt, x_f,
                         y_t, tags, a2v, mode, flog, pred_f, stt, use_tag,
                         use_a2v, prev_val_loss)
                    if to_shuffle:
                        flog.write('Shuffling...\n')
                        np.random.shuffle(trix)

            train_loss = None
            if stt == 'mt':
                b_f, b_y, b_t = make_mt_batch(
                    train_feat, train_pt, train_trd, trix, pos, batch_size)
                _, train_loss = sess.run(
                    [optimizer, loss_pt],
                    feed_dict={
                        x_f: b_f, y_t: b_y, y_trd: b_t,
                        mode: learn.ModeKeys.TRAIN, lrt: current_lr})
            elif stt == 'mr':
                b_f, b_y, b_r = make_mr_batch(
                    train_feat, train_pt, train_ret, trix, pos, batch_size)
                _, train_loss = sess.run(
                    [optimizer, loss_pt],
                    feed_dict={
                        x_f: b_f, y_t: b_y, y_ret: b_r,
                        mode: learn.ModeKeys.TRAIN, lrt: current_lr})
            elif stt == 'mtr':
                b_f, b_y, b_t, b_r = make_mtr_batch(
                    train_feat, train_pt, train_trd,
                    train_ret, trix, pos, batch_size)
                _, train_loss = sess.run(
                    [optimizer, loss_pt],
                    feed_dict={
                        x_f: b_f, y_t: b_y, y_trd: b_t, y_ret: b_r,
                        mode: learn.ModeKeys.TRAIN, lrt: current_lr})
            else:
                train_loss = None
                b_f, b_y = make_batch(
                    train_feat, train_pt, trix, pos, batch_size)
                if use_tag:
                    b_tg = make_tag_batch(train_tags, trix, pos, batch_size)
                if use_a2v:
                    b_a2v = make_a2v_batch(train_a2v, trix, pos, batch_size)

                if use_tag:
                    if use_a2v:
                        _, train_loss = sess.run(
                            [optimizer, loss_pt],
                            feed_dict={
                                x_f: b_f, y_t: b_y, tags: b_tg, a2v: b_a2v,
                                mode: learn.ModeKeys.TRAIN, lrt: current_lr})
                    else:
                        _, train_loss = sess.run(
                            [optimizer, loss_pt],
                            feed_dict={
                                x_f: b_f, y_t: b_y, tags: b_tg,
                                mode: learn.ModeKeys.TRAIN, lrt: current_lr})
                else:
                    if use_a2v:
                        _, train_loss = sess.run(
                            [optimizer, loss_pt],
                            feed_dict={
                                x_f: b_f, y_t: b_y, a2v: b_a2v,
                                mode: learn.ModeKeys.TRAIN, lrt: current_lr})
                    else:
                        _, train_loss = sess.run(
                            [optimizer, loss_pt],
                            feed_dict={
                                x_f: b_f, y_t: b_y,
                                mode: learn.ModeKeys.TRAIN, lrt: current_lr})

            losses.append(train_loss)
            flog.flush()
            fsync(flog.fileno())

    flog.write('---------------\n')
    test(model_dir, test_feat, test_pt, test_ids, test_tags, test_a2v,
         teix, batch_size, loss_pt, logits_pt, x_f, y_t, tags, a2v,
         mode, flog, pred_f, stt, use_tag, use_a2v, prev_val_loss)
    flog.close()


if __name__ == '__main__':
    main()
