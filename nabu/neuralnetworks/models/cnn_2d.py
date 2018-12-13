'''@file capsnet.py
contains the CapsNet class'''

import tensorflow as tf
import model
from nabu.neuralnetworks.components import layer, ops
import pdb

class CNN2D(model.Model):
    '''A capsule network'''

    def _get_outputs(self, inputs, input_seq_length, is_training):
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

        num_filters = int(self.conf['num_filters'])
        kernel_size = int(self.conf['conv_kernel_size'])
        stride = int(self.conf['stride'])

        #code not available for multiple inputs!!
        if len(inputs) > 1:
            raise 'The implementation of f_CNN expects 1 input and not %d' %len(inputs)
        else:
            inputs=inputs[0]

        with tf.variable_scope(self.scope):
            if is_training and float(self.conf['input_noise']) > 0:
                inputs = inputs + tf.random_normal(
                    tf.shape(inputs),
                    stddev=float(self.conf['input_noise']))

            #First layer
            with tf.variable_scope('first_layer'):
                output = tf.identity(inputs, 'inputs')
                input_seq_length = tf.identity(input_seq_length, 'input_seq_length')

                # Include frequency dimension
                output = tf.expand_dims(output,-1)

                first_layer = tf.layers.conv2d(
                    output,
                    num_filters,
                    kernel_size,
                    stride,
                    padding='SAME',
                    activation=tf.nn.relu
                )

            #tf.add_to_collection('image', tf.expand_dims(prim_norm, 3))
            output = tf.identity(first_layer, 'first_layer')

            # subsequent layers
            for l in range(1, int(self.conf['num_layers'])):
                with tf.variable_scope('layer%d' % l):
                    #a conv1d layer
                    output =  tf.layers.conv2d(
                        output,
                        num_filters,
                        kernel_size,
                        stride,
                        padding='SAME',
                        activation=tf.nn.relu
                    )

                    if is_training and float(self.conf['dropout']) < 1:
                        output = tf.nn.dropout(output, float(self.conf['dropout']))


            return output
