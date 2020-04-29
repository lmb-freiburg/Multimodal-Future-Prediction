import tensorflow as tf
from utils_tf import *


default_encoder_channels = {
    'conv1': 64,
    'conv2': 128,
    'conv3': 256,
    'conv3_1': 256,
    'conv4': 512,
    'conv4_1': 512,
    'conv5': 512,
    'conv5_1': 512,
    'conv6': 1024,
    'conv6_1': 1024,
    'conv7': 1024,
    'conv8': 1024
}


default_loss_weights = {
    'object0': 1/1,
    'object1': 1/1
}

class FutureEncoderArchitecture:

    def __init__(self,
                 loss_function=None,
                 disassembling_function=None,
                 channel_factor=1.0,
                 num_objects = 2, # for toyset pedestrian and vehicle and for real dataset 1 object
                 num_outputs=4, # [object_tl.x, object_tl.y] * num_objects
                 learn_encoder=True,
                 dropout=False,
                 target_id=None):

        self._channel_factor = channel_factor
        self._loss_function = loss_function
        self._disassembling_function = disassembling_function
        self._num_outputs = num_outputs
        self._num_objects = num_objects
        self._loss_weights = default_loss_weights
        self._encoder_channels = {}
        self._learn_encoder = learn_encoder
        self._dropout = dropout
        self._target_id = target_id
        for name, channels in default_encoder_channels.items():
            self._encoder_channels[name] = int(self._channel_factor*channels)

    def predict(self, input):
        with tf.variable_scope("predict"):
            predicted = []
            for i in range(self._num_objects):
                predicted.append(tf_full_conn(input, name='predict_fc_%d'%i,
                                            num_output=self._num_outputs / self._num_objects))

            predicted = tf.concat(predicted, axis=1)

            return predicted

    def make_graph(self, input):

        with tf.variable_scope('encoder'):
            conv1 = tf_conv(input, name="conv1", kernel_size=7, stride=2, pad=3,
                                       num_output=self._encoder_channels['conv1'])
            conv2 = tf_conv(conv1, name="conv2", kernel_size=5, stride=2, pad=2,
                                       num_output=self._encoder_channels['conv2'])
            conv3 = tf_conv(conv2, name="conv3", kernel_size=5, stride=2, pad=2,
                                       num_output=self._encoder_channels['conv3'])
            conv3_1 = tf_conv(conv3, name="conv3_1", kernel_size=3, stride=1, pad=1,
                                       num_output=self._encoder_channels['conv3_1'])
            conv4 = tf_conv(conv3_1, name="conv4", kernel_size=3, stride=2, pad=1,
                                       num_output=self._encoder_channels['conv4'])
            conv4_1 = tf_conv(conv4, name="conv4_1", kernel_size=3, stride=1, pad=1,
                                       num_output=self._encoder_channels['conv4_1'])
            conv5 = tf_conv(conv4_1, name="conv5", kernel_size=3, stride=2, pad=1,
                                       num_output=self._encoder_channels['conv5'])
            conv5_1 = tf_conv(conv5, name="conv5_1", kernel_size=3, stride=1, pad=1,
                                       num_output=self._encoder_channels['conv5_1'])
            conv6 = tf_conv(conv5_1, name="conv6", kernel_size=3, stride=2, pad=1,
                                       num_output=self._encoder_channels['conv6'])
            conv6_1 = tf_conv(conv6, name="conv6_1", kernel_size=3, stride=1, pad=1,
                                       num_output=self._encoder_channels['conv6_1'])
            if self._target_id is not None:
                list_vectors = []
                for i in range(conv6_1.shape[0]):
                    list_vectors.append(tf.fill((1, 1, conv6_1.shape[2], conv6_1.shape[3]), self._target_id[i]))
                target_id_vec = tf.concat(list_vectors, axis=0)
                conv6_1 = tf.concat([target_id_vec, conv6_1], axis=1)

        with tf.variable_scope('mapper'):

            conv7 = tf_conv(conv6_1, name="conv7", kernel_size=1, stride=1, pad=0,
                                       num_output=self._encoder_channels['conv7'])
            conv8 = tf_conv(conv7, name="conv8", kernel_size=1, stride=1, pad=0,
                                     num_output=self._encoder_channels['conv8'])

        out = self.predict(conv8)
        return out
