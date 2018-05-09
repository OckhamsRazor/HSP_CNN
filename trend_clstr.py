import argparse
import random
from math import sqrt
from os import mkdir, path
from shutil import rmtree

import numpy as np
from tqdm import tqdm, trange


def DTWDistance(s1, s2, w):
    DTW = {}
    w = max(w, abs(len(s1) - len(s2)))
    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(
                DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    lb_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            lb_sum = lb_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            lb_sum = lb_sum + (i - lower_bound) ** 2

    return sqrt(lb_sum)


def k_means_clust(data, num_clust, num_iter, output_dir, w=5):
    """
    data: np array (#data * time length)
    num_clust: K
    num_iter: num_iter
    cntrd_f: centroids output file(s)
    w: window size
    """
    centroids = random.sample(data, num_clust)
    for itr in trange(num_iter, desc='Iteration', leave=True):
        assignments = {}
        # assign data points to clusters
        icls = [None] * data.shape[0]
        for ind, i in enumerate(tqdm(data, desc='Data idx')):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
            icls[ind] = closest_clust

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

        np.save(
            path.join(output_dir, '{:d}.npy'.format(itr)),
            np.array(centroids))
        np.save(
            path.join(output_dir, '{:d}.cls.npy'.format(itr)),
            np.array(icls, dtype=np.int))


def main():
    parser = argparse.ArgumentParser(
        description='Trend curve clustering with DTW and K-means')
    parser.add_argument(
        '-i', '--trend_curves', help='training data dir', required=True)
    parser.add_argument(
        '-o', '--output_dir',
        help='centroid positions output dir', required=True)
    parser.add_argument(
        '-k', '--cntrd_num', help='centroid number', required=True)
    parser.add_argument(
        '-it', '--iterations', help='training iteration number', required=True)
    args = parser.parse_args()
    trend_curve_f = args.trend_curves
    k = int(args.cntrd_num)
    its = int(args.iterations)

    output_dir = args.output_dir
    if path.exists(output_dir):
        rmtree(output_dir)
    mkdir(output_dir)

    trend_curves = np.load(trend_curve_f)
    trend_max = trend_curves.max(axis=1)
    trend_curves_norm = trend_curves / trend_max[:, np.newaxis]

    k_means_clust(trend_curves_norm, k, its, output_dir)


if __name__ == '__main__':
    main()
