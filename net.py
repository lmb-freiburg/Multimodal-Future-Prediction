import os
import math
import argparse
import tensorflow as tf
from encoder import FutureEncoderArchitecture
from utils_tf import *

class EWTA_MDF():
    def __init__(self, imgs, objects):
        self.imgs = imgs
        self.objects = objects

    def disassembling(self, data):  # input has shape (1, 80, 1, 1)
        splits = tf.split(data, [2 for i in range(40)], 1)  # set of (1, 2, 1, 1)
        hyps = splits[0:20]
        log_sigmas = splits[20:40]
        bounded_log_sigmas = [tf_adjusted_sigmoid(log_sigmas[i], -6, 6) for i in range(len(log_sigmas))]
        return hyps, bounded_log_sigmas

    def disassembling_fitting(self, data):  # input has shape (1, 80, 1, 1)
        hyps = tf.split(data, [4 for i in range(20)], 1)  # set of (1, 4, 1, 1)
        return hyps

    def prepare_input(self):
        [n, c, h, w] = self.imgs[0].get_shape().as_list()
        input_list = []
        for i in range(2, -1, -1):
            object = self.objects[i]
            mask = tf_get_mask(object[0, 0, :, 0], w, h, fill_value=object[0, 0, 4, 0])
            input_list.insert(0, self.imgs[i])
            input_list.insert(0, mask)
        input = tf.concat(input_list, axis=1)
        return input

    def make_graph(self):
        arch = FutureEncoderArchitecture(num_outputs=2 * 2 * 20, num_objects=1, channel_factor=1.0, learn_encoder=False)
        input = self.prepare_input()
        output = arch.make_graph(input)
        out_hyps, out_log_sigmas = self.disassembling(output)
        hyps_concat = tf.concat(out_hyps, axis=1)
        log_scales_concat = tf.concat(out_log_sigmas, axis=1)  # (batch, 20*2, 1, 1)
        input_2 = tf.concat([hyps_concat, log_scales_concat], axis=1)  # (batch, 20*4, 1, 1)

        # fitting
        with tf.variable_scope("net2"):
            intermediate = tf.tanh(tf_full_conn(input_2, name='predict_fc0', num_output=500))
            intermediate_drop = intermediate
            predicted = tf_full_conn(intermediate_drop, name='predict_fc1', num_output=20 * 4)
        out_soft_assignments = self.disassembling_fitting(predicted)
        means, bounded_log_sigmas, mixture_weights = tf_assemble_lmm_parameters_independent_dists(samples_means=out_hyps,
                                                                        samples_log_scales=out_log_sigmas,
                                                                        assignments=out_soft_assignments)
        sigmas = [tf.exp(x) for x in bounded_log_sigmas]

        return means, sigmas, mixture_weights, bounded_log_sigmas, out_hyps, out_log_sigmas, input, output, input_2

    # The functions below are used for training the sampling network and fitting network.
    # As explained in our paper, you need first to train the sampling network, and then fix it while training the fitting.
    # Training the sampling network is done via EWTA (optionally EWTAD), this is acheived by training the sampling network in multiple steps,
    # where at each step you call the make_sampling_loss with different parameters (mode, top_n).
    # For example, we predict 20 hypotheses. Then you start training the network by penalizing all: make_sampling_loss(....,mode='epe-all').
    # then after some time (e.g, 50k), you continue training with: make_sampling_loss(....,mode='epe-top-n', top_n=10),
    # then at each step you reduce the top_n to top_n/2, and so on.
    # Finally you invoke the function: make_sampling_loss(....,mode='epe').
    # Additionally, if you need the variant of EWTAD, then you add one more step by calling make_sampling_loss(.....,mode='iul')
    def make_sampling_loss(self, hyps, bounded_log_sigmas, gt, mode='epe', top_n=1):
        # gt has the shape (batch,2,1,1) which corresponds to the ground-truth future location (x,y)
        # hyps list of 20 hypotheses each has the shape (batch,2,1,1), this corresponds to the mean of the hypothesis
        # bounded_log_sigmas list of 20 hypotheses each has the shape (batch,2,1,1), this corresponds to the log sigma of the hypothesis
        num_hyps = len(hyps)
        gts = tf.stack([gt for i in range(0, num_hyps)], axis=1)  # (batch,20,2,1,1)
        hyps_stacked = tf.stack([h for h in hyps], axis=1)

        epsillon = 0.05
        eps = 0.001
        diff = tf.square(hyps_stacked - gts)
        channels_sum = tf.reduce_sum(diff, axis=2)
        spatial_epes = tf.sqrt(channels_sum + eps)  # (batch,20,1,1)
        sum_losses = tf.constant(0.0)

        if mode == 'epe':
            spatial_epe = tf.reduce_min(spatial_epes, axis=1)
            loss = tf.multiply(tf.reduce_mean(spatial_epe), 1.0)
            sum_losses = tf.add(loss, sum_losses)
            tf.add_to_collection('losses', loss)

        elif mode == 'epe-relaxed':
            spatial_epe = tf.reduce_min(spatial_epes, axis=1)
            loss0 = tf.multiply(tf.reduce_mean(spatial_epe), 1 - 2 * epsillon)
            tf.add_to_collection('losses', loss0)

            for i in range(num_hyps):
                loss = tf.multiply(tf.reduce_mean(spatial_epes[:, i, :, :]), epsillon / (num_hyps - 1))
                sum_losses = tf.add(loss, sum_losses)
                tf.add_to_collection('losses', loss)
            sum_losses = tf.add(loss0, sum_losses)

        elif mode == 'epe-top-n' and top_n > 1:
            spatial_epes_transposed = tf.multiply(tf.transpose(spatial_epes, perm=[0, 2, 3, 1]), -1)
            top_k, ignores = tf.nn.top_k(spatial_epes_transposed, k=top_n)
            spatial_epes_min = tf.multiply(tf.transpose(top_k, perm=[0, 3, 1, 2]), -1)
            for i in range(top_n):
                loss = tf.multiply(tf.reduce_mean(spatial_epes_min[:, i, :, :]), 1.0)
                sum_losses = tf.add(loss, sum_losses)
                tf.add_to_collection('losses', loss)

        elif mode == 'epe-all':
            for i in range(num_hyps):
                loss = tf.multiply(tf.reduce_mean(spatial_epes[:, i, :, :]), 1.0)
                sum_losses = tf.add(loss, sum_losses)
                tf.add_to_collection('losses', loss)

        elif mode == 'iul':
            eps = 1e-2 / 2.0
            total_loss = []
            for i in range(num_hyps):
                diff2 = tf.square(gt - hyps[i])  # (batch,2,1,1)
                diff2 = tf.add(diff2, tf.fill(diff2, eps))
                diff2 = tf.pow(diff2, 0.5)
                se = tf.exp(-1 * bounded_log_sigmas[i])
                e = tf.multiply(diff2, se)
                e_sum = tf.math.reduce_sum(e, axis=1, keepdims=True)  # (batch,1,1,1)
                sxsy = tf.math.reduce_sum(bounded_log_sigmas[i], axis=1, keepdims=True)
                total_loss.append(tf.add(sxsy, e_sum))

            total = tf.concat(total_loss, axis=1)  # (batch,20,1,1)
            errors_inv = -1 * spatial_epes  # (batch,20,1,1)
            best_index = tf.stop_gradient(tf.argmax(errors_inv, axis=1))  # (batch,1,1)
            indices = tf.one_hot(best_index, num_hyps, axis=1)  # (batch,20,1,1)
            merged = total * indices
            best_loss = tf.reduce_sum(merged, axis=1)  # (batch,1,1)
            loss = tf.reduce_mean(best_loss)
            sum_losses = tf.add(loss, sum_losses)
            tf.add_to_collection('losses', loss)

        return sum_losses

    def make_fitting_loss(self, means, bounded_log_sigmas, mixture_weights, gt):
        # means, sigmas list of 4, each has the shape of (batch,2,1,1)
        # mixture_weights list of 4, each has the shape of (batch,1,1,1)
        # gt has the shape (batch,2,1,1)
        num_modes = len(means)
        total_loss = None
        eps = 1e-5 / 2.0
        for i in range(num_modes):
            diff2 = tf.square(gt - means[i])  # (batch,2,1,1)
            diff2 = tf.add(diff2, tf.fill(diff2, eps))
            diff2 = tf.pow(diff2, 0.5)
            sigma = tf.exp(bounded_log_sigmas[i])  # (batch,2,1,1)
            sigma = tf.add(sigma, tf.fill(sigma, eps))
            sigma_inv = tf.pow(sigma, -1)
            c = tf.multiply(diff2, sigma_inv)  # (batch,2,1,1)
            c = tf.math.reduce_sum(c, axis=1, keepdims=True)  # (batch,1,1,1)
            c_exp = tf.exp(-1 * c)
            sxsy = tf.multiply(sigma[:, 0:1, :, :], sigma[:, 1:2, :, :])
            sxsy_inv = tf.pow(sxsy + eps, -1)  # (batch,1,1,1)
            likelihood = tf.multiply(c_exp, sxsy_inv)
            likelihood_weighted = tf.multiply(likelihood, mixture_weights[i])
            if i == 0:
                total_loss = likelihood_weighted
            else:
                total_loss = tf.add(total_loss, likelihood_weighted)

        total = total_loss + eps
        nll = -1 * tf.log(total)
        loss = tf.reduce_sum(nll)
        tf.add_to_collection('losses', loss)
        return loss
