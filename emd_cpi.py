from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import numpy as np
import os
from wemd import computeWEMD
import math

scene_path = 'path_to_a_scene_in_cpi_dataset'  # location of the ground-truth
predition_path = 'path_to_prediction_folder'  # location of the prediction output
history = 3
future_dist = 20

def read_gt(obj_index):  # obj_index: 0 for pedestrian, 1 for vehicle
    gts = np.zeros([1000, 2])
    for j in range(1000):
        gt_array = readFloat(os.path.join(scene_path, '%03d-%06d-%03d-objects.float3' % (history - 1, j, future_dist)))  # shape (1, 1, 6)
        if obj_index == 0:
            gts[j, :] = gt_array[0, 0, 0:2]
        else:
            gts[j, :] = gt_array[0, 0, 3:5]
    return gts

def compute_histogram_points(samples):
    Z, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=np.linspace(0, 512, 512))
    return Z

def read_pred():
    means = readFloat('%s-mixture_distribution_means.float3' % predition_path)  # shape (4, 2)
    sigmas = readFloat('%s-mixture_distribution_sigmas.float3' % predition_path)  # shape (4, 2)
    weights = readFloat('%s-mixture_distribution_weights.float3' % predition_path)  # shape (4)

    sigmas = 2 * sigmas * sigmas

    gmm = GaussianMixture(n_components=4, covariance_type='diag')

    precisions_cholesky = _compute_precision_cholesky(sigmas, 'diag')
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.precisions_cholesky_ = precisions_cholesky
    gmm.covariances_ = sigmas

    return gmm

def compute_histogram_gmm(clf):
    samples = clf.sample(1000)[0]
    Z, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=np.linspace(0, 512, 512))
    return Z


gts = read_gt(obj_index)
Z_gt = compute_histogram_points(gts)
clf = read_pred()
Z_lmm = compute_histogram_gmm(clf)
wemd_distance = computeWEMD(Z_lmm, Z_gt)