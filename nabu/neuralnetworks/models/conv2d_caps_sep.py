'''@file conv_capsnet.py
contains the Convolutional CapsNet class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer, ops
import pdb

class Conv2DCapsSep(model.Model):
    '''A convolutional capsule network'''

    def _get_outputs(self, inputs, input_seq_length, is_training):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a list of
                [batch_size x time x ...] tensors
                Currently, the expected input is [batch_size x time x frequency]
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            - output, which is a [batch_size x time x ...] tensors
        '''
        # Inputs is presumed to be a list of [batch_size x time x frequency] tensors

        num_capsules = int(self.conf['num_capsules'])
        capsule_dim = int(self.conf['capsule_dim'])
        routing_iters = int(self.conf['routing_iters'])
        # TODO: Make kernel sizes a list of sizes for every layer, similarly for strides
        kernel_size = map(int, self.conf['conv_kernel_size'].split(' '))
        stride = int(self.conf['stride'])

        # code not available for multiple inputs!!
        if len(inputs) > 1:
            raise 'The implementation of CapsNet expects 1 input and not %d' % len(inputs)
        else:
            inputs = inputs[0]

        with tf.variable_scope(self.scope):
            if is_training and float(self.conf['input_noise']) > 0:
                inputs = inputs + tf.random_normal(
                    tf.shape(inputs),
                    stddev=float(self.conf['input_noise']))

            output = tf.identity(inputs, 'inputs')
            input_seq_length = tf.identity(input_seq_length, 'input_seq_length')
            # Include channel dimension
            batch_size = output.shape[0].value
            num_freq = output.shape[2].value
            output_dim = num_capsules * capsule_dim
            output = tf.expand_dims(output, -1)

            # Conv1 ReLU
            #conv1 = tf.layers.conv2d(
            #    output,
            #    output_dim,
            #    kernel_size,
            #    stride,
            #    padding='SAME',
            #    activation=tf.nn.relu
            #)

            # Primary capsule
            with tf.variable_scope('primary_capsule'):

                primary_capsules = tf.layers.conv2d(
                    output,
                    output_dim,
                    kernel_size,
                    stride,
                    padding='SAME'
                )

                primary_capsules = tf.reshape(primary_capsules,
                                              [batch_size,
                                               -1,
                                               num_freq,
                                               num_capsules,
                                               capsule_dim]
                                              )

                primary_capsules = ops.squash(primary_capsules)
                output = tf.identity(primary_capsules, 'primary_capsules')

            # non-primary capsules
            for l in range(1, int(self.conf['num_layers'])):
                with tf.variable_scope('layer%d' % l):
                    # a capsule layer
                    conv2d_caps_layer = layer.Conv2DCapsGridRouting(num_capsules=num_capsules,
                                               capsule_dim=capsule_dim,
                                               kernel_size=kernel_size,
                                                strides=(stride,stride),
                                               routing_iters=routing_iters)

                    output = conv2d_caps_layer(output)

                    if is_training and float(self.conf['dropout']) < 1:
                        output = tf.nn.dropout(output, float(self.conf['dropout']))

            # Include frequency dimension
            output = tf.reshape(
                output,
                [batch_size,
                 -1,
                 num_freq, output_dim]
            )

        return output
