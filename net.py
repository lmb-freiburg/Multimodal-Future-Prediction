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

        return means, sigmas, mixture_weights, out_hyps, out_log_sigmas, input, output, input_2