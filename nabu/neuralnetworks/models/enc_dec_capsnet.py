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
        num_capsules_1st_layer = int(self.conf['num_capsules_1st_layer'])
        capsule_dim = int(self.conf['capsule_dim'])
        routing_iters = int(self.conf['routing_iters'])
        f_pool_rate = int(self.conf['f_pool_rate'])
        t_pool_rate = int(self.conf['t_pool_rate'])
        num_encoder_layers = int(self.conf['num_encoder_layers'])
        num_decoder_layers = num_encoder_layers
        num_centre_layers = int(self.conf['num_centre_layers'])

        # the encoder layers
        encoder_layers = []
        for l in range(0, num_encoder_layers):
            num_capsules_l = num_capsules_1st_layer * 2**l
            #stride = min(l*2,4)
            # strides = (2, 1)
            strides = [1, 1]
            if (t_pool_rate !=0) & (np.mod(l, t_pool_rate) == 0):
                strides[0] = 2
            if (f_pool_rate !=0) & (np.mod(l, f_pool_rate) == 0):
                strides[1] = 2

            encoder_layers.append(layer.Conv2DCapsule(num_capsules=num_capsules_l,
                                                    capsule_dim=capsule_dim,
                                                    kernel_size=kernel_size,
                                                    strides=strides,
                                                    padding='SAME',
                                                    routing_iters=routing_iters))
	    
        # the centre layers
        centre_layers = []
        for l in range(num_centre_layers):
            num_capsules_l = num_capsules_1st_layer * 2**num_encoder_layers

            centre_layers.append(layer.Conv2DCapsule(num_capsules=num_capsules_l,
                                              capsule_dim=capsule_dim,
                                              kernel_size=kernel_size,
                                              strides=(1,1),
                                              padding='SAME',
                                              routing_iters=routing_iters))

        # the decoder layers
        decoder_layers = []
        for l in range(num_encoder_layers):
            corresponding_encoder_l = num_encoder_layers-1-l
            num_capsules_l = encoder_layers[corresponding_encoder_l].num_capsules
            strides = encoder_layers[corresponding_encoder_l].strides

            decoder_layers.append(layer.Conv2DCapsule(num_capsules=num_capsules_l,
                                              capsule_dim=capsule_dim,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding='SAME',
                                              transpose=True,
                                              routing_iters=routing_iters))

        #last encoder
        # strides = [1, 1]
        # # if np.mod(1, t_pool_rate) == 0:
        # #     strides[0] = 2
        # # if np.mod(1, f_pool_rate) == 0:
        # #     strides[1] = 2
        # decoder_layers.append(layer.Conv2DCapsule(num_capsules=num_capsules_1st_layer,
        #                                           capsule_dim=capsule_dim,
        #                                           kernel_size=kernel_size,
        #                                           strides=strides,
        #                                           padding='SAME',
        #                                           max_pool_filter=(1, 1),
        #                                           transpose=True,
        #                                           routing_iters=routing_iters))

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
            with tf.variable_scope('encoder'):
                encoder_outputs = []

                # Primary capsule
                with tf.variable_scope('primary_capsule'):
                    logits = tf.identity(inputs, 'inputs')
                    input_seq_length = tf.identity(input_seq_length, 'input_seq_length')

                    # Convolution
                    batch_size = logits.shape[0].value
                    num_freq = logits.shape[2].value
                    output_dim = num_capsules_1st_layer * capsule_dim
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
                                                   num_capsules_1st_layer,
                                                   capsule_dim]
                                                  )

                    primary_capsules = ops.squash(primary_capsules)
                    logits = tf.identity(primary_capsules, 'primary_capsules')

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
                        decoder_input = tf.concat([logits, corresponding_encoder_output], -2)
                        # decoder_input = logits


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
                        #if corresponding_encoder_l==0:
                            #wanted_size = inputs.get_shape()
                        #else:
                            #wanted_size = encoder_outputs[corresponding_encoder_l-1].get_shape()

                        wanted_t_size = wanted_size_tensor[1]
                        wanted_f_size = wanted_size_tensor[2]
			freq_out = wanted_size[2]

                        logits = decoder_layers[l](decoder_input, wanted_t_size, freq_out)
                       # #get actual output size
                       # output_size = tf.shape(logits)
                       # #output_size = logits.get_shape()
                       # output_t_size = output_size[1]
                       # output_f_size = output_size[2]

                       # #compensate for potential mismatch, by adding duplicates
                       # missing_t_size = wanted_t_size-output_t_size
                       # missing_f_size = wanted_f_size-output_f_size

                       # def t_tiling(logits_to_tile):
                       #     last_t_slice = tf.expand_dims(logits_to_tile[:, -1, :, :, :], 1)
                       #     duplicate_logits = tf.tile(last_t_slice, [1, missing_t_size, 1, 1, 1])
                       #     tiled_logits = tf.concat([logits_to_tile, duplicate_logits], 1)
                       #     return tiled_logits

                       # def f_tiling(logits_to_tile):
                       #     last_f_slice = tf.expand_dims(logits_to_tile[:, :, -1, :, :], 2)
                       #     duplicate_logits = tf.tile(last_f_slice, [1, 1, missing_f_size, 1, 1])
                       #     tiled_logits = tf.concat([logits_to_tile, duplicate_logits], 2)
                       #     return tiled_logits


                       # logits = tf.cond(missing_t_size > 0, lambda: t_tiling(logits),
                       #                  lambda: tf.slice(logits, [0,0,0,0,0], [-1, wanted_t_size, -1, -1, -1]))
                       # logits = tf.cond(missing_f_size > 0, lambda: f_tiling(logits),
                       #                  lambda: tf.slice(logits, [0, 0, 0, 0, 0], [-1, -1, wanted_f_size, -1, -1]))
                       # logits.set_shape(wanted_size)

                output = logits
                # Include frequency dimension
                output = tf.reshape(
                    output,
                    [batch_size,
                     -1,
                     num_freq, output_dim]
                )

        return output
