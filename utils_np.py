import numpy as np
import os
import math
import cv2
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import matplotlib.pyplot as plt
from wemd import computeWEMD

# read a file with extension .float and return a numpy array
def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))
    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim == 2:
        data = np.transpose(data, (0, 1))
    elif dim == 3:
        data = np.transpose(data, (1, 2, 0))
    elif dim == 4:
        data = np.transpose(data, (2, 3, 1, 0))
    else:
        raise Exception('bad float file dimension: %d' % dim)

    return data

# read an image amd resize when needed
def decode_img(file_path, width=None, height=None):
    img = cv2.imread(file_path)
    img = img / 255.0
    img = np.subtract(img, 0.4)
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img

# read the float file containing the object information
def decode_obj(file_path, id, coeff_x=1.0, coeff_y=1.0):
    object = np.expand_dims(np.expand_dims(np.expand_dims(readFloat(file_path)[id], 0), 0), 3).astype(np.float32)
    x_tl = object[:, :, 0:1, :] / coeff_x
    y_tl = object[:, :, 1:2, :] / coeff_y
    x_br = object[:, :, 2:3, :] / coeff_x
    y_br = object[:, :, 3:4, :] / coeff_y
    object = np.concatenate((x_tl, y_tl, x_br, y_br, object[:, :, 4:6, :]), axis=2)
    return object

# draw a set of hypotheses (bounding boxes), object history, and the ground truth on an image
def draw_hyps(img_path, hyps, gt_object, objects):
    img = cv2.imread(img_path)
    # draw object history
    tranparency = {0:0.2, 1:0.5, 2:1.0}
    for i in range(3):
        overlay = img.copy()
        cv2.rectangle(overlay, (int(objects[i, 0, 0, 0, 0]), int(objects[i, 0, 0, 1, 0])),
                      (int(objects[i, 0, 0, 2, 0]), int(objects[i, 0, 0, 3, 0])), (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, tranparency[i], img, 1 - tranparency[i], 0)

    # draw the ground truth future
    cv2.rectangle(img, (int(gt_object[0, 0, 0, 0]), int(gt_object[0, 0, 1, 0])), (int(gt_object[0, 0, 2, 0]), int(gt_object[0, 0, 3, 0])), (255, 0, 255), -1)
    gt_width = gt_object[0, 0, 2, 0] - gt_object[0, 0, 0, 0]
    gt_height = gt_object[0, 0, 3, 0] - gt_object[0, 0, 1, 0]
    
    # draw hypotheses (different futures)
    for h in hyps:
        x1 = int(h[0, 0, 0, 0])
        y1 = int(h[0, 1, 0, 0])
        x2 = int(x1 + gt_width)
        y2 = int(y1 + gt_height)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img

# draw a heatmap on an image and save it to a file
def draw_heatmap(img_path, means, sigmas, weights, objects, width, height, output_path, gt=None):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
        return mycmap

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # draw history
    tranparency = {0: 0.2, 1: 0.5, 2: 1.0}
    for i in range(3):
        overlay = img.copy()
        cv2.rectangle(overlay, (int(objects[i, 0, 0, 0, 0]), int(objects[i, 0, 0, 1, 0])),
                      (int(objects[i, 0, 0, 2, 0]), int(objects[i, 0, 0, 3, 0])), (255, 0, 0), -1)
        img = cv2.addWeighted(overlay, tranparency[i], img, 1 - tranparency[i], 0)

    gt_width = gt[0, 0, 2, 0] - gt[0, 0, 0, 0]
    gt_height = gt[0, 0, 3, 0] - gt[0, 0, 1, 0]
    mapped_means = []
    mapped_sigmas = []
    for i in range(len(means)):
        center_mean = means[i][:, 0:2, :, :] + [gt_width/2, gt_height/2]
        center_sigma = sigmas[i][:, 0:2, :, :]
        mapped_means.append(center_mean)
        mapped_sigmas.append(center_sigma)

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    if gt is not None:
        cv2.rectangle(img, (int(gt[0, 0, 0, 0]), int(gt[0, 0, 1, 0])),
                      (int(gt[0, 0, 2, 0]), int(gt[0, 0, 3, 0])), (255, 0, 255), -1)

    # construct the GMM
    c_means = np.stack([mapped_means[i][0,0:2,0,0] for i in range(len(mapped_means))], axis=0)  # (4,2)
    c_sigmas = np.stack([mapped_sigmas[i][0,0:2,0,0] for i in range(len(mapped_sigmas))], axis=0)  # (4,2)
    c_weights = np.concatenate(weights, axis=0)[:,0,0,0]  # (4)

    clf = mixture.GaussianMixture(n_components=4, covariance_type='diag')
    var = c_sigmas * c_sigmas * 2
    precisions_cholesky = _compute_precision_cholesky(var, 'diag')
    clf.weights_ = c_weights
    clf.means_ = c_means
    clf.precisions_cholesky_ = precisions_cholesky
    clf.covariances_ = var

    Z = np.exp(clf.score_samples(XX))

    Z = Z.reshape(X.shape)

    vmax = np.max(Z)
    vmin = np.min(Z)
    plt.imshow(img)
    plt.contourf(X, Y, Z, cmap=transparent_cmap(plt.cm.jet), vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()

# get the NLL score for a sample (GT) given the parameters of the mixture model
def compute_nll(pred_means, pred_sigmas, pred_weights, gt):
    means = np.concatenate(pred_means, axis=0)[:, :, 0, 0]
    sigmas = np.concatenate(pred_sigmas, axis=0)[:, :, 0, 0]
    weights = np.concatenate(pred_weights, axis=0)[:, 0, 0, 0]
    gt_loc = gt[:, 0, 0:2, 0]
    sum = 0
    for i in range(means.shape[0]):
        diff = means[i] - gt_loc
        diff2 = diff * diff
        diff5 = math.sqrt(diff2[0, 0])
        diff6 = math.sqrt(diff2[0, 1])

        sxe = sigmas[i, 0]
        sye = sigmas[i, 1]
        sxe_sq_inv = 1.0 / (sxe)
        sye_sq_inv = 1.0 / (sye)
        c1 = diff5 * sxe_sq_inv
        c2 = diff6 * sye_sq_inv
        c = c1 + c2
        c_exp = math.exp(-c)
        sxsy = 4.0 * sxe * sye
        sxsy_scaled = 1.0 / (sxsy)
        final = c_exp * sxsy_scaled
        final_weighted = final * weights[i]
        sum += final_weighted
    sum = sum if sum > 0 else 1e-10
    return math.log(sum) * -1

# returns the closest hypothesis to the ground truth (oracle selection)
def get_best_hyp(hyps, gt):
    num_hyps = len(hyps)
    gts = np.stack([gt for i in range(0, num_hyps)], axis=1)  # n,num,c,1,1
    hyps = np.stack(hyps, axis=1)  # n,num,c,1,1

    def spatial_error(hyps, gts):
        diff = np.square(hyps - gts) # n,num,c,1,1
        channels_sum = np.sum(diff, axis=2) # n,num,1,1
        spatial_epes = np.sqrt(channels_sum) # n,num,1,1
        return np.expand_dims(spatial_epes, axis=2) # n,num,1,1,1

    def get_best(hypotheses, errors, num_hyps):
        indices = np.argmin(errors, axis=1) # n,1,1,1
        shape = indices.shape
        # compute one-hot encoding
        encoding = np.zeros((shape[0],num_hyps,shape[1],shape[2],shape[3]))
        encoding[np.arange(shape[0]),indices,np.arange(shape[1]),np.arange(shape[2]),np.arange(shape[3])] = 1 # n,num,1,1,1

        hyps_channels = hypotheses.shape[2]
        encoding = np.concatenate([encoding for i in range(hyps_channels)], axis=2) # n,num,c,1,1
        reduced = hypotheses * encoding # n,num,c,1,1
        reduced = np.sum(reduced, axis=1) # n,c,1,1
        return reduced

    errors = spatial_error(hyps, gts) # n,num,1,1,1
    best = get_best(hyps, errors, num_hyps) #n,c,1,1
    return best

# compute the final displacement error between a hypothesis and ground truth
def get_FDE(hyp, gt):
    diff = np.square(hyp[:, 0:2, :, :] - gt[:, 0:2, :, :])
    channels_sum = np.sum(diff, axis=1)
    spatial_epe = np.sqrt(channels_sum)
    fde = np.mean(spatial_epe)
    return fde

# compute the final displacement error between the best hypothesis and the ground truth
def compute_oracle_FDE(hyps, gt):
    gt_loc = np.transpose(gt[:,:,0:2,:], [0,2,1,3])
    best_hyp = get_best_hyp(hyps, gt_loc)
    return get_FDE(best_hyp, gt_loc)


def compute_histogram_gmm(clf):
    samples = clf.sample(1000)[0]
    Z, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=[np.linspace(0, 319, 320), np.linspace(0, 575, 576)])
    return Z

# compute the SEMD metric which evaluate the degree of multimodality of a mixture model
def get_multimodality_score(means, sigmas, weights):
    means_stacked = np.concatenate(means, axis=0)[:, :, 0, 0]
    sigmas_stacked = np.concatenate(sigmas, axis=0)[:, :, 0, 0]
    weights_stacked = np.concatenate(weights, axis=0)[:, 0, 0, 0]
    gmm = GaussianMixture(n_components=4, covariance_type='diag')
    vars = 2 * sigmas_stacked * sigmas_stacked
    precisions_cholesky = _compute_precision_cholesky(vars, 'diag')
    gmm.weights_ = weights_stacked
    gmm.means_ = means_stacked
    gmm.precisions_cholesky_ = precisions_cholesky
    gmm.covariances_ = vars
    gmm_uni = GaussianMixture(n_components=1, covariance_type='diag')
    argmax = np.argmax(gmm.weights_)
    gmm_uni.means_ = gmm.means_[argmax, :].reshape([1, 2])
    gmm_uni.covariances_ = gmm.covariances_[argmax, :].reshape([1, 2])
    gmm_uni.precisions_cholesky_ = gmm.precisions_cholesky_[argmax, :].reshape([1, 2])
    gmm_uni.weights_ = np.array([1]).reshape([1])
    Z_uni = compute_histogram_gmm(gmm_uni)
    Z = compute_histogram_gmm(gmm)
    ratio = computeWEMD(Z, Z_uni)
    return ratio

