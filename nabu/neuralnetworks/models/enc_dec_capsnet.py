'''@file encoder_decoder_cnn.py
contains de EncoderDecoderCNN class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer, ops
import numpy as np
import pdb

class EncDecCapsNet(model.Model):
    '''A CNN classifier with encoder-decoder shape 
    (https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)
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

        kernel_size = map(int, self.conf['filters'].split(' '))
        # num_capsules_1st_layer = int(self.conf['num_capsules_1st_layer'])
        # capsule_dim = int(self.conf['capsule_dim'])
        routing_iters = int(self.conf['routing_iters'])
        f_pool_rate = int(self.conf['f_pool_rate'])
        t_pool_rate = int(self.conf['t_pool_rate'])
        num_encoder_layers = int(self.conf['num_encoder_layers'])
        num_decoder_layers = num_encoder_layers
        num_centre_layers = int(self.conf['num_centre_layers'])
        num_capsules_lst = map(int, self.conf['num_capsules_lst'].split(' '))
        capsule_dim_lst = map(int, self.conf['capsule_dim_lst'].split(' '))
        use_bias = self.conf['use_bias'] == 'True'
        # leaky_softmax = self.conf['leaky_softmax'] == 'True'
        leaky_softmax = False
        probability_fn = None
        if leaky_softmax:
            probability_fn = ops.leaky_softmax

        # the encoder layers
        encoder_layers = []
        for l in range(0, num_encoder_layers):
            # num_capsules_l = num_capsules_1st_layer * 2**(l+1)
            num_capsules_l = num_capsules_lst[l+1]
            capsule_dim_l = capsule_dim_lst[l+1]
            strides = [1, 1]
            if (t_pool_rate !=0) & (np.mod(l, t_pool_rate) == 0):
                strides[0] = 2
            if (f_pool_rate !=0) & (np.mod(l, f_pool_rate) == 0):
                strides[1] = 2

            encoder_layers.append(layer.EncDecCapsule(num_capsules=num_capsules_l,
                                                      capsule_dim=capsule_dim_l,
                                                      kernel_size=kernel_size,
                                                      strides=strides,
                                                      padding='SAME',
                                                      routing_iters=routing_iters,
                                                      use_bias=use_bias,
                                                      probability_fn=probability_fn))

        # the centre layers
        centre_layers = []
        for l in range(num_centre_layers):
            # num_capsules_l = num_capsules_1st_layer * 2**num_encoder_layers
            num_capsules_l = num_capsules_lst[l+1+num_encoder_layers]
            capsule_dim_l = capsule_dim_lst[l+1+num_encoder_layers]

            centre_layers.append(layer.EncDecCapsule(num_capsules=num_capsules_l,
                                                     capsule_dim=capsule_dim_l,
                                                     kernel_size=kernel_size,
                                                     strides=(1,1),
                                                     padding='SAME',
                                                     routing_iters=routing_iters,
                                                     use_bias=use_bias,
                                                     probability_fn=probability_fn))

        # the decoder layers
        decoder_layers = []
        for l in range(num_encoder_layers):
            corresponding_encoder_l = num_encoder_layers-1-l
            # if corresponding_encoder_l == 0:
            #     num_capsules_l = num_capsules_lst[corresponding_encoder_l]
            # else:
            #     num_capsules_l = encoder_layers[corresponding_encoder_l - 1].num_capsules
            num_capsules_l = num_capsules_lst[l + 1 + num_encoder_layers + num_centre_layers]
            capsule_dim_l = capsule_dim_lst[l + 1 + num_encoder_layers + num_centre_layers]
            strides = encoder_layers[corresponding_encoder_l].strides
            decoder_layers.append(layer.EncDecCapsule(num_capsules=num_capsules_l,
                                                      capsule_dim=capsule_dim_l,
                                                      kernel_size=kernel_size,
                                                      strides=strides,
                                                      padding='SAME',
                                                      transpose=True,
                                                      routing_iters=routing_iters,
                                                      use_bias=use_bias,
                                                      probability_fn=probability_fn))

        #code not available for multiple inputs!!
        if len(inputs) > 1:
            raise 'The implementation of DCNN expects 1 input and not %d' %len(inputs)
        else:
            inputs=inputs[0]

        with tf.variable_scope(self.scope):
            if is_training and float(self.conf['input_noise']) > 0:
                inputs = inputs + tf.random_normal(
                    tf.shape(inputs),
                    stddev=float(self.conf['input_noise']))

            # Primary capsule
            with tf.variable_scope('primary_capsule'):
                logits = tf.identity(inputs, 'inputs')
                input_seq_length = tf.identity(input_seq_length, 'input_seq_length')

                # Convolution
                batch_size = logits.shape[0].value
                num_freq = logits.shape[2].value
                output_dim = num_capsules_lst[0] * capsule_dim_lst[0]
                logits = tf.expand_dims(logits, -1)

                primary_capsules = tf.layers.conv2d(
                    logits,
                    output_dim,
                    kernel_size,
                    strides = (1,1),
                    padding='SAME'
                )

                primary_capsules = tf.reshape(primary_capsules,
                                              [batch_size,
                                               -1,
                                               num_freq,
                                               num_capsules_lst[0],
                                               capsule_dim_lst[0]]
                                              )

                primary_capsules = ops.squash(primary_capsules)
                logits = tf.identity(primary_capsules, 'primary_capsules')

            with tf.variable_scope('encoder'):
                encoder_outputs = []
                for l in range(num_encoder_layers):
                    with tf.variable_scope('layer_%s'%l):
                        logits = encoder_layers[l](logits)

                        encoder_outputs.append(logits)

                        if is_training and float(self.conf['dropout']) < 1:
                            raise 'have to check wheter dropout is implemented correctly'
                            logits = tf.nn.dropout(logits, float(self.conf['dropout']))

            with tf.variable_scope('centre'):
                for l in range(num_centre_layers):
                    with tf.variable_scope('layer_%s'%l):

                        logits = centre_layers[l](logits)

                        if is_training and float(self.conf['dropout']) < 1:
                            raise 'have to check wheter dropout is implemented correctly'
                            logits = tf.nn.dropout(logits, float(self.conf['dropout']))

            with tf.variable_scope('decoder'):
                for l in range(num_decoder_layers):
                    with tf.variable_scope('layer_%s'%l):
                        corresponding_encoder_l = num_encoder_layers-1-l
                        corresponding_encoder_output = encoder_outputs[corresponding_encoder_l]
                        # if l == 0:
                        #     decoder_input = logits
                        # else:
                        #     decoder_input = tf.concat([logits, corresponding_encoder_output], -2)
                        decoder_input = logits


                        if is_training and float(self.conf['dropout']) < 1:
                            raise 'have to check wheter dropout is implemented correctly'
                            logits = tf.nn.dropout(logits, float(self.conf['dropout']))

                        #get wanted output size
                        if corresponding_encoder_l==0:
                            wanted_size_tensor = tf.shape(primary_capsules)
                            wanted_size = primary_capsules.shape
                        else:
                            wanted_size_tensor = tf.shape(
                                encoder_outputs[corresponding_encoder_l-1])
                            wanted_size = encoder_outputs[corresponding_encoder_l-1].shape

                        wanted_t_size = wanted_size_tensor[1]
                        freq_out = wanted_size[2]

                        logits = decoder_layers[l](decoder_input, wanted_t_size, freq_out)

                output = logits
                # Include frequency dimension
                output = tf.reshape(
                    output,
                    [batch_size,
                     -1,
                     num_freq, num_capsules_lst[0] * capsule_dim_lst[0]]
                )

        return output
