import numpy as np
import os
import math
import tensorflow as tf
from functools import partial

# resample each hypothesis given the scaling coefficients
def tf_resample_hyps(hyps, coeff_x, coeff_y):
    resampled_hyps = []
    for h in hyps:
        x_center = h[:, 0:1, :, :] / coeff_x
        y_center = h[:, 1:2, :, :] / coeff_y
        width = h[:, 2:3, :, :] / coeff_x
        height = h[:, 3:4, :, :] / coeff_y
        resampled_hyps.append(tf.concat([x_center, y_center, width, height], axis=1))
    return resampled_hyps

# create a tensorflow session and initialize the global variables
def create_session():
    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    return session

# restore the variables from a snapshot
def optimistic_restore(session, save_file, ignore_vars=None, verbose=False, ignore_incompatible_shapes=False):
    def vprint(*args, **kwargs):
        if verbose: print(*args, flush=True, ** kwargs)

    if ignore_vars is None:
        ignore_vars = []

    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.dtype, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes and not var in ignore_vars])
    restore_vars = []

    nonfinite_values = False

    with tf.variable_scope('', reuse=True):
        for var_name, var_dtype, saved_var_name in var_names:
            curr_var = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if saved_var_name in var.name][0]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                tmp = reader.get_tensor(saved_var_name)

                # check if there are nonfinite values in the tensor
                if not np.all(np.isfinite(tmp)):
                    nonfinite_values = True
                    print('{0} contains nonfinite values!'.format(saved_var_name), flush=True)

                if isinstance(tmp, np.ndarray):
                    saved_dtype = tf.as_dtype(tmp.dtype)
                else:
                    saved_dtype = tf.as_dtype(type(tmp))
                if not saved_dtype.is_compatible_with(var_dtype):
                    raise TypeError('types are not compatible for {0}: saved type {1}, variable type {2}.'.format(
                        saved_var_name, saved_dtype.name, var_dtype.name))

                print('restoring    ', saved_var_name)
                restore_vars.append(curr_var)
            else:
                vprint('not restoring', saved_var_name, 'incompatible shape:', var_shape, 'vs',
                       saved_shapes[saved_var_name])
                if not ignore_incompatible_shapes:
                    raise RuntimeError(
                        'failed to restore "{0}" because of incompatible shapes: var: {1} vs saved: {2} '.format(
                            saved_var_name, var_shape, saved_shapes[saved_var_name]))
    if nonfinite_values:
        raise RuntimeError('"{0}" contains nonfinite values!'.format(save_file))
    saver = tf.train.Saver(var_list=restore_vars, restore_sequentially=True)
    saver.restore(session, save_file)

# simple wrapper for a fully connected layer with activation
def tf_full_conn(input, activation=None, **kwargs):

    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    b_initializer = tf.zeros_initializer

    num_output = kwargs.pop('num_output', False)
    name = kwargs.pop('name', 'conv_no_name')

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(input)

    dense_out = tf.layers.dense(fc1,
                                int(num_output),
                                activation=activation,
                                kernel_initializer=k_initializer,
                                bias_initializer=b_initializer,
                                kernel_regularizer=k_regularizer,
                                trainable=True,
                                name=name)
    output = tf.reshape(dense_out, [dense_out.shape[0], dense_out.shape[1], 1, 1])
    return output

# padding a tensor
def tf_pad_input(input, pad):
    padded = tf.pad(input, [[0,0],[0,0],[pad,pad],[pad,pad]])
    return padded

# simple wrapper for a convolution layer with a default leaky_relu as activation
def tf_conv(input, activation=partial(tf.nn.leaky_relu, alpha=0.1), **kwargs):

    k_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
    k_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    b_initializer = tf.zeros_initializer

    # for shared params
    dropout = kwargs.pop("dropout", False)
    if dropout: raise NotImplementedError

    kernel_size = kwargs.pop('kernel_size', False)
    num_output = kwargs.pop('num_output', False)
    stride = kwargs.pop('stride', 1)
    pad = kwargs.pop('pad', 0)
    name = kwargs.pop('name', 'conv_no_name')

    if not kernel_size:
        raise KeyError('Missing kernel size')
    if not num_output:
        raise KeyError('Missing output size')


    # layer
    # note: input might be a tuple, in which case weights are shared
    if not isinstance(input, tuple):
        conv_out = tf.layers.conv2d(tf_pad_input(input, pad),
                                num_output,
                                kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                trainable=False,
                                activation=activation,
                                kernel_regularizer = k_regularizer,
                                kernel_initializer = k_initializer,
                                bias_initializer = b_initializer,
                                name=name)
        return conv_out
    else:
        outputs = []
        for i in input:
            outputs.append(tf.layers.conv2d(pad_input(i, pad),
                                num_output,
                                kernel_size,
                                strides=stride,
                                data_format='channels_first',
                                trainable=nd.scope.learn(),
                                reuse=tf.AUTO_REUSE,
                                activation=activation,
                                kernel_regularizer = k_regularizer,
                                kernel_initializer = k_initializer,
                                bias_initializer = b_initializer,
                                name=name,
                                ))

        return outputs

# Post-processing function applied on the log sigmas. This should constraint the log_sigma given (min, max).
# We use this technique to ensure stability during training (e.g, we start training by small range and we increase over time). 
def tf_adjusted_sigmoid(X, min, max):
    tf.add_to_collection('log_scale_bound', max)
    const = lambda z: tf.fill(X.get_shape(), z)
    min = tf.to_float(min)
    max = tf.to_float(max)
    range = max - min
    x_scaled = tf.multiply(X, const(4.0 / range))
    sig = tf.sigmoid(x_scaled)
    sig_scaled = tf.multiply(sig, const(range))
    if min != 0:
        sig_scaled_shifted = tf.add(sig_scaled, const(min))
    else:
        sig_scaled_shifted = sig_scaled
    return sig_scaled_shifted

# wrapper function for sum(w_i * x_i)/sum(w_i)
def tf_average_weighted_norm(x, w):
    sum_w = tf.reduce_sum(w, axis=1)
    sum_w_inv = tf.pow(tf.add(sum_w, tf.fill(sum_w.get_shape(), 1e-6 / 2.0)), -1)

    x_weighted = tf.multiply(x, w) 
    x_weighted_sum = tf.reduce_sum(x_weighted, axis=1) 
    result = tf.multiply(x_weighted_sum, sum_w_inv)
    result = tf.expand_dims(result, axis=1)
    return result

# generates a laplacian mixture model parameters given a set of independent laplacian distributions
# this corresponds to the equations 6,7,8,9 in the paper.
def tf_get_laplace_mixture_model_from_independent_dists(samples_means, samples_log_scales, assignments):
    num_of_modes = assignments[0].shape[1]
    num_samples = len(samples_means)

    expanded_means = [tf.expand_dims(samples_means[i], axis=1) for i in range(num_samples)]
    samples_means_concat = tf.concat(expanded_means, axis=1)

    expanded_scales = [tf.expand_dims(tf.exp(samples_log_scales[i]), axis=1) for i in range(num_samples)]
    samples_b_concat = tf.concat(expanded_scales, axis=1)
    # map b (scale) to sigma^2
    samples_var_concat = tf.scalar_mul(2.0, tf.pow(samples_b_concat, 2))

    expanded_assignments = [tf.expand_dims(assignments[i], axis=1) for i in range(len(assignments))]
    assignments_adjusted = tf.nn.softmax(tf.concat(expanded_assignments, axis=1), dim=2)

    mixture_weights = []
    means = []
    log_scales = []
    for k in range(num_of_modes):
        y_ik = assignments_adjusted[:,:,k,:,:]

        w_k = tf.expand_dims(tf.reduce_mean(y_ik, axis=1), axis=1)

        mu_k_x = tf_average_weighted_norm(samples_means_concat[:,:,0,:,:], y_ik)
        mu_k_y = tf_average_weighted_norm(samples_means_concat[:,:,1,:,:], y_ik)
        mu_k = tf.concat([mu_k_x, mu_k_y], axis=1)

        # Var =E[Var()] + Var(E[])
        mu_k_repeated = tf.concat([tf.expand_dims(mu_k, axis=1) for i in range(num_samples)], axis=1)
        diff = tf.subtract(samples_means_concat, mu_k_repeated)
        diff2 = tf.pow(diff, 2)
        var_E_k_x = tf_average_weighted_norm(diff2[:,:,0,:,:], y_ik)
        var_E_k_y = tf_average_weighted_norm(diff2[:,:,1,:,:], y_ik)
        var_E_k = tf.concat([var_E_k_x, var_E_k_y], axis=1)
        E_var_k_x = tf_average_weighted_norm(samples_var_concat[:, :, 0, :, :], y_ik)
        E_var_k_y = tf_average_weighted_norm(samples_var_concat[:, :, 1, :, :], y_ik)
        E_var_k = tf.concat([E_var_k_x, E_var_k_y], axis=1)
        var_k = tf.add(E_var_k, var_E_k)
        # map sigma^2 to b
        b_k = tf.pow(tf.scalar_mul(0.5, var_k), 0.5)
        log_scale_k = tf.log(b_k)

        mixture_weights.append(w_k)
        means.append(mu_k)
        log_scales.append(log_scale_k)

    return means, log_scales, mixture_weights

def tf_assemble_lmm_parameters_independent_dists(samples_means, samples_log_scales, assignments):
    means, log_sigmas, mixture_weights = tf_get_laplace_mixture_model_from_independent_dists(samples_means, samples_log_scales, assignments)
    bounded_log_sigmas = [tf_adjusted_sigmoid(log_sigmas[i], -6, 6) for i in range(len(log_sigmas))]
    return means, bounded_log_sigmas, mixture_weights

# get a binary mask (B_i) for the object bounding box
def tf_get_mask(indices, width, height, fill_value=1.0):# shape of indices is 5
    indices = tf.to_int32(indices)
    tl_x = indices[0]
    tl_y = indices[1]
    bbox_width = indices[2]-indices[0]
    bbox_height = indices[3]-indices[1]
    ind_row = [tl_y, height - tl_y - bbox_height]
    ind_col = [tl_x, width - tl_x - bbox_width]
    padding = tf.stack([ind_row, ind_col])
    input = tf.ones([bbox_height, bbox_width]) * (fill_value + 1)
    padded = tf.expand_dims(tf.expand_dims(tf.pad(input, padding, "CONSTANT"), axis=0), axis=0)
    return padded