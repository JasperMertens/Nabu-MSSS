'''@file encoder_decoder_cnn.py
contains de EncoderDecoderCNN class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer, ops
import numpy as np
import pdb

class SegCapsNet(model.Model):
    '''A capsule encoder decoder model adapted from
        @article{lalonde2018capsules,
          title={Capsules for Object Segmentation},
          author={LaLonde, Rodney and Bagci, Ulas},
          journal={arXiv preprint arXiv:1804.04241},
          year={2018}
        }
    https://github.com/lalonderodney/SegCaps/blob/master/capsnet.py
    '''

    def  _get_outputs(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            - output, which is a [batch_size x time x ...] tensors
        '''
        use_bias = self.conf['use_bias'] == 'True'
        leaky_softmax = self.conf['leaky_softmax'] == 'True'
        probability_fn = None
        if leaky_softmax:
            probability_fn = ops.leaky_softmax

        # code not available for multiple inputs!!
        if len(inputs) > 1:
            raise 'The implementation of DCNN expects 1 input and not %d' % len(inputs)
        else:
            inputs = inputs[0]

        with tf.variable_scope(self.scope):
            if is_training and float(self.conf['input_noise']) > 0:
                inputs = inputs + tf.random_normal(
                    tf.shape(inputs),
                    stddev=float(self.conf['input_noise']))


            logits = tf.identity(inputs, 'inputs')
            input_seq_length = tf.identity(input_seq_length, 'input_seq_length')

            # Convolution
            batch_size = logits.shape[0].value
            num_freq = logits.shape[2].value
            logits = tf.expand_dims(logits, -1)

            # Layer 1: Just a conventional Conv2D layer
            conv1 = tf.layers.conv2d(logits, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='conv1')

            # Reshape layer to be 1 capsule x [filters] atoms
            conv1_reshaped = tf.reshape(conv1, [batch_size, -1, num_freq, 1, 16])

            # Layer 1: Primary Capsule: Conv cap with routing 1
            primary_caps = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=2, capsule_dim=16, strides=(2,2), padding='SAME',
                                            routing_iters=0, use_bias=True, name='primarycaps')(conv1_reshaped)

            # Layer 2: Convolutional Capsule
            conv_cap_2_1 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=4, capsule_dim=16, strides=(1,1), padding='SAME',
                                               routing_iters=3, use_bias=True, name='conv_cap_2_1')(primary_caps)

            # Layer 2: Convolutional Capsule
            conv_cap_2_2 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=4, capsule_dim=32, strides=(2,2), padding='SAME',
                                               routing_iters=3, use_bias=True, name='conv_cap_2_2')(conv_cap_2_1)

            # Layer 3: Convolutional Capsule
            conv_cap_3_1 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=8, capsule_dim=32, strides=(1,1), padding='SAME',
                                               routing_iters=3, use_bias=True, name='conv_cap_3_1')(conv_cap_2_2)

            # Layer 3: Convolutional Capsule
            conv_cap_3_2 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=8, capsule_dim=64, strides=(2,2), padding='SAME',
                                               routing_iters=3, use_bias=True, name='conv_cap_3_2')(conv_cap_3_1)

            # Layer 4: Convolutional Capsule
            conv_cap_4_1 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=8, capsule_dim=32, strides=(1,1), padding='SAME',
                                               routing_iters=3, use_bias=True, name='conv_cap_4_1')(conv_cap_3_2)

            # Layer 1 Up: Deconvolutional Capsule
            t_out = tf.shape(conv_cap_3_1)[1]
            freq_out = conv_cap_3_1.shape[2]
            deconv_cap_1_1 = layer.EncDecCapsule(kernel_size=(4,4), num_capsules=8, capsule_dim=32, transpose=True,
                                                 strides=(2, 2), padding='SAME', routing_iters=3, use_bias=True,
                                                name='deconv_cap_1_1')(conv_cap_4_1, t_out, freq_out)
            # Skip connection
            up_1 = tf.concat([deconv_cap_1_1, conv_cap_3_1], axis=-2, name='up_1')

            # Layer 1 Up: Deconvolutional Capsule
            deconv_cap_1_2 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=4, capsule_dim=32, strides=(1,1),
                                              padding='SAME', routing_iters=3, use_bias=True, name='deconv_cap_1_2')(up_1)

            # Layer 2 Up: Deconvolutional Capsule
            t_out = tf.shape(conv_cap_2_1)[1]
            freq_out = conv_cap_2_1.shape[2]
            deconv_cap_2_1 = layer.EncDecCapsule(kernel_size=(4,4), num_capsules=4, capsule_dim=16, transpose=True,
                                                 strides=(2, 2), padding='SAME', routing_iters=3, use_bias=True,
                                                name='deconv_cap_2_1')(deconv_cap_1_2, t_out, freq_out)

            # Skip connection
            up_2 = tf.concat([deconv_cap_2_1, conv_cap_2_1], axis=-2, name='up_2')

            # Layer 2 Up: Deconvolutional Capsule
            deconv_cap_2_2 = layer.EncDecCapsule(kernel_size=(5,5), num_capsules=4, capsule_dim=16, strides=(1,1),
                                              padding='SAME', routing_iters=3, use_bias=True, name='deconv_cap_2_2')(up_2)

            # Layer 3 Up: Deconvolutional Capsule
            t_out = tf.shape(conv1_reshaped)[1]
            freq_out = conv1_reshaped.shape[2]
            deconv_cap_3_1 = layer.EncDecCapsule(kernel_size=(4,4), num_capsules=2, capsule_dim=16, transpose=True,
                                                 strides=(2, 2), padding='SAME', routing_iters=3, use_bias=True,
                                                name='deconv_cap_3_1')(deconv_cap_2_2, t_out, freq_out)

            # Skip connection
            up_3 = tf.concat([deconv_cap_3_1, conv1_reshaped], axis=-2, name='up_3')

            # Layer 4: Convolutional Capsule: 1x1
            seg_caps = layer.EncDecCapsule(kernel_size=(1,1), num_capsules=1, capsule_dim=16, strides=(1,1), padding='SAME',
                                           routing_iters=3, use_bias=True, name='seg_caps')(up_3)


            output = seg_caps
            # Include frequency dimension
            output = tf.reshape(
                output,
                [batch_size,
                 -1,
                 num_freq, 16]
            )

        return output
