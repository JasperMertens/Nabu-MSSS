'''@file layer.py
Neural network layers '''

import string

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from nabu.neuralnetworks.components import ops, rnn_cell, rnn
from ops import capsule_initializer
import numpy as np
import pdb

_alphabet_str=string.ascii_lowercase
class Capsule(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_initializer=None,
            logits_initializer=None,
            routing_iters=3,
            activation_fn=None,
            probability_fn=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            **kwargs):

        '''Capsule layer constructor
        args:
            num_capsules: number of output capsules
            capsule_dim: output capsule dimsension
            kernel_initializer: an initializer for the prediction kernel
            logits_initializer: the initializer for the initial logits
            routing_iters: the number of routing iterations (default: 5)
            activation_fn: a callable activation function (default: squash)
            probability_fn: a callable that takes in logits and returns weights
                (default: tf.nn.softmax)
            activity_regularizer: Regularizer instance for the output (callable)
            trainable: wether layer is trainable
            name: the name of the layer
        '''

        super(Capsule, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.routing_iters = routing_iters
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        '''creates the variables of this layer
        args:
            input_shape: the shape of the input
        '''

        #pylint: disable=W0201

        #input dimensions
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        self.kernel = self.add_variable(
            name='kernel',
            dtype=self.dtype,
            shape=[num_capsules_in, capsule_dim_in,
                   self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        super(Capsule, self).build(input_shape)

    #pylint: disable=W0221
    def call(self, inputs):
        '''
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        #compute the predictions
        predictions, logits = self.predict(inputs)

        #cluster the predictions
        outputs = self.cluster(predictions, logits)

        return outputs

    def predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2

            #put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank-1])

            #compute the predictins
            predictions = tf.map_fn(
                fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            #transpose back
            predictions = tf.transpose(
                predictions, range(1, shared+1)+[0]+[rank-1, rank])

            logits = self.logits
            for i in range(shared):
                if predictions.shape[shared-i-1].value is None:
                    shape = tf.shape(predictions)[shared-i-1]
                else:
                    shape = predictions.shape[shared-i-1].value
                tile = [shape] + [1]*len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def predict_slow(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2
	  
	    if shared > 26-4:
	      raise 'Not enough letters in the alphabet to use Einstein notation'
	    #input_shape = [shared (typicaly batch_size,time),Nin,Din], kernel_shape = [Nin, Din, Nout, Dout],
	    #predictions_shape = [shared,Nin,Nout,Dout]
	    shared_shape_str=_alphabet_str[0:shared]
	    input_shape_str=shared_shape_str+'wx'
	    kernel_shape_str='wxyz'
	    output_shape_str=shared_shape_str+'wyz'
	    ein_not='%s,%s->%s'%(input_shape_str, kernel_shape_str, output_shape_str)
	    
	    predictions = tf.einsum(ein_not, inputs, self.kernel)

            logits = self.logits
            for i in range(shared):
                if predictions.shape[shared-i-1].value is None:
                    shape = tf.shape(predictions)[shared-i-1]
                else:
                    shape = predictions.shape[shared-i-1].value
                tile = [shape] + [1]*len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def cluster(self, predictions, logits):
        '''cluster the predictions into output capsules
        args:
            predictions: the predicted output capsules
            logits: the initial routing logits
        returns:
            the output capsules
        '''

        with tf.name_scope('cluster'):

            #define m-step
            def m_step(l):
                '''m step'''
                with tf.name_scope('m_step'):
                    #compute the capsule contents
                    w = self.probability_fn(l)
                    caps = tf.reduce_sum(
                        tf.expand_dims(w, -1)*predictions, -3)

                return caps, w

            #define body of the while loop
            def body(l):
                '''body'''

                caps, _ = m_step(l)
                caps = self.activation_fn(caps)

                #compare the capsule contents with the predictions
                similarity = tf.reduce_sum(
                    predictions*tf.expand_dims(caps, -3), -1)

                return l + similarity

            #get the final logits with the while loop
            lo = tf.while_loop(
                lambda l: True,
                body, [logits],
                maximum_iterations=self.routing_iters)

            #get the final output capsules
            capsules, _ = m_step(lo)
            capsules = self.activation_fn(capsules)

        return capsules

    def compute_output_shape(self, input_shape):
        '''compute the output shape'''

        if input_shape[-2].value is None:
            raise ValueError(
                'The number of capsules must be defined, but saw: %s'
                % input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The capsule dimension must be defined, but saw: %s'
                % input_shape)

        return input_shape[:-2].concatenate(
            [self.num_capsules, self.capsule_dim])


class FConvCapsule(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_size=3, stride=1,
            routing_iters=3,
            activation_fn=None,
            probability_fn=None,
            activity_regularizer=None,
            kernel_initializer=None,
            logits_initializer=None,
            trainable=True,
            name=None,
            **kwargs):

        '''Capsule layer constructor
        args:
            num_capsules: number of output capsules
            capsule_dim: output capsule dimsension
            kernel_size: size of convolutional kernel
            kernel_initializer: an initializer for the prediction kernel
            logits_initializer: the initializer for the initial logits
            routing_iters: the number of routing iterations (default: 5)
            activation_fn: a callable activation function (default: squash)
            probability_fn: a callable that takes in logits and returns weights
                (default: tf.nn.softmax)
            activity_regularizer: Regularizer instance for the output (callable)
            trainable: wether layer is trainable
            name: the name of the layer
        '''

        super(FConvCapsule, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.routing_iters = routing_iters
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        '''creates the variables of this layer
        args:
            input_shape: the shape of the input
        '''

        #pylint: disable=W0201

        #input dimensions
        num_freq_in = input_shape[-3].value
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value
        num_freq_out = num_freq_in
        #without padding:
        #num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        self.kernel = self.add_variable(
            name='kernel',
            dtype=self.dtype,
            shape=[self.kernel_size,
                    num_capsules_in, capsule_dim_in,
                    self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_freq_out,
                   num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        super(FConvCapsule, self).build(input_shape)

    #pylint: disable=W0221
    def call(self, inputs):
        '''
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        #compute the predictions
        predictions, logits = self.depthwise_predict(inputs)

        #cluster the predictions
        outputs = self.cluster(predictions, logits)

        return outputs

    def predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value
            num_filters = num_capsule_in*self.num_capsules*self.capsule_dim

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-3

            #reshape to [B*T, F, N_in*D_in]
            inputs = tf.reshape(inputs, shape=[-1, num_freq_in, num_capsule_in*capsule_dim_in])

            predictions = tf.layers.conv1d(inputs, num_filters,
                                    self.kernel_size, self.stride,
                                    padding='SAME', use_bias=False)
            predictions = tf.reshape(predictions, shape=[batch_size, -1, num_freq_out, num_capsule_in, self.num_capsules, self.capsule_dim])

            # tile the logits for each shared dimesion (batch, time)
            with tf.name_scope('tile_logits'):
                logits = self.logits
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def depthwise_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('depthwise_predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-3

            # pad the inputs with zeros along the freq dimension
            with tf.name_scope('zero_padding'):
                half_kernel = self.kernel_size // 2
                paddings = [[0, 0]] * shared + [[half_kernel, half_kernel], [0, 0], [0, 0]]
                inputs = tf.pad(inputs, paddings, "CONSTANT")

            # reshape to [B*T, F, D_in, N_in]
            inputs = tf.reshape(inputs, shape=[-1, num_freq_in+2*half_kernel,
                                               capsule_dim_in, num_capsule_in])

            # inputs = tf.reshape(inputs, shape=[-1, num_freq_in,
            #                                    capsule_dim_in, num_capsule_in])

            # reshape the kernel to [W, D_in, N_in, N_out*D_out]
            kernel = tf.transpose(self.kernel, [0, 2, 1, 3, 4])
            kernel = tf.reshape(kernel, shape=[self.kernel_size, capsule_dim_in, num_capsule_in,
                                               self.num_capsules*self.capsule_dim])

            # tensorflow does not pad along the D_in dimension with stride=D_in for that axis
            # but no support for different strides yet, so need manual padding
            predictions = tf.nn.depthwise_conv2d(inputs, kernel,
                                    strides=[1, self.stride, self.stride, 1],
                                    padding='VALID')
            predictions = tf.reshape(predictions, shape=[batch_size, -1, num_freq_out, num_capsule_in, self.num_capsules, self.capsule_dim])

            # tile the logits for each shared dimesion (batch, time)
            with tf.name_scope('tile_logits'):
                logits = self.logits
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def matmul_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('matmul_predict'):

            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            # without padding:
            #num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-3

            #put the frequencies as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank-2, rank-1])

            #pad the inputs with zeros
            with tf.name_scope('zero_padding'):
                half_kernel = self.kernel_size // 2
                zeros = tf.zeros(shape=tf.shape(inputs)[1:])
                zeros = tf.expand_dims(zeros, 0)
                for i in range(half_kernel):
                    inputs = tf.concat([zeros, inputs, zeros], 0)

            #create the indices list
            indices = []
            offset_lst = range(self.kernel_size)
            for i in range(0, num_freq_out, self.stride):
                indices.append([x+i for x in offset_lst])
            indices = tf.convert_to_tensor(indices)

            #split the inputs up for every frequency convolution
            #so the first two dimensions are [num_freq_out x kernel_size x ...]
            #TODO: solve the problem that it thorws a sparse to dense gradients warning
            inputs_split = tf.gather(inputs, indices)

            #transpose back so the last four dimensions are
            #  [... x num_freq_out x num_capsule_in x kernel_size x capsule_dim_in]
            inputs_split = tf.transpose(
                inputs_split, range(2, shared+2)+[0, rank-1, 1, rank])

            #tile the inputs over the num_capsules output dimension
            # so the last five dimensions are
            # [... x num_freq_out x num_capsule_in x num_capsule_out x kernel_size x capsule_dim_in]
            inputs_split = tf.expand_dims(inputs_split, -3)
            multiples = [1]*shared + [1, 1, self.num_capsules, 1, 1]
            inputs_tiled = tf.tile(inputs_split, multiples)

            #change the capsule dimensions into column vectors
            inputs_tiled = tf.expand_dims(inputs_tiled,-1)

            #transpose the kernel so the dimensions are
            # [num_capsules_in x num_capsules_out x kernel_size x capsule_dim_out x capsule_dim_in]
            kernel = tf.transpose(self.kernel, [1, 3, 0, 4, 2])

            #tile the kernel for every shared dimension (batch, time) and num_freq_out
            with tf.name_scope('tile_kernel'):
                kernel_tiled = kernel
                for i in range(shared+1):
                    if inputs_tiled.shape[shared-i].value is None:
                        shape = tf.shape(inputs_tiled)[shared-i]
                    else:
                        shape = inputs_tiled.shape[shared-i].value
                    tile = [shape] + [1]*len(kernel_tiled.shape)
                    kernel_tiled = tf.tile(tf.expand_dims(kernel_tiled, 0), tile)

            #compute the predictions

            #perform matrix multiplications so the last five dimensions are
            # [... x num_freq_out x num_capsule_in x num_capsule_out  x kernel_size x capsule_dim_out]
            predictions = tf.squeeze(tf.matmul(kernel_tiled, inputs_tiled), -1)

            #sum over the kernel_size dimension
            predictions = tf.reduce_sum(predictions, -2)

            # tile the logits for each shared dimesion (batch, time)
            with tf.name_scope('tile_logits'):
                logits = self.logits
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def roll_matmul_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('roll_matmul_predict'):

            num_freq_in = inputs.shape[-3].value
            num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-3

            #expand inputs for the kernel dimension and replicate it
            inputs = tf.expand_dims(inputs, shared+1)
            copy = tf.identity(inputs)

            #formulate the output shape of the concatenation of the frequency shifted copies
            input_shape = tf.shape(inputs)
            shape = tf.concat([input_shape[:shared],
                               [num_freq_out, self.kernel_size],
                               input_shape[shared+2:]], 0)

            #concatenate the frequency shifted copies
            half_kernel = self.kernel_size//2
            for i in range(self.kernel_size-1):
                inputs = tf.manip.roll(inputs, 1, shared)
                inputs = tf.concat([inputs, copy], shared+1)
            inputs = tf.manip.roll(inputs, -half_kernel, shared)

            #because the convolution is unpadded, values at the edges are dropped
            inputs = tf.slice(inputs, [0]*shared +
                              [half_kernel, 0, 0, 0],
                              shape)
            #transpose so the dimensions are
            # [... x num_freq_out, num_capsule_in x kernel_size x capsule_dim_in]
            inputs_split = tf.transpose(inputs, range(shared+1) + [shared+2, shared+1, shared+3])

           #tile the inputs over the num_capsules output dimension
            # so the last five dimensions are
            # [... x num_freq_out x num_capsule_in x num_capsule_out x kernel_size x capsule_dim_in]
            inputs_split = tf.expand_dims(inputs_split, -3)
            multiples = [1]*shared + [1, 1, self.num_capsules, 1, 1]
            inputs_tiled = tf.tile(inputs_split, multiples)

            #change the capsule dimensions into column vectors
            inputs_tiled = tf.expand_dims(inputs_tiled,-1)

            #transpose the kernel so the dimensions are
            # [num_capsules_in x num_capsules_out x kernel_size x capsule_dim_out x capsule_dim_in]
            kernel = tf.transpose(self.kernel, [1, 3, 0, 4, 2])

            #tile the kernel for every shared dimension (batch, time) and num_freq_out
            kernel_tiled = kernel
            for i in range(shared+1):
                if inputs_tiled.shape[shared-i].value is None:
                    shape = tf.shape(inputs_tiled)[shared-i]
                else:
                    shape = inputs_tiled.shape[shared-i].value
                tile = [shape] + [1]*len(kernel_tiled.shape)
                kernel_tiled = tf.tile(tf.expand_dims(kernel_tiled, 0), tile)

            #compute the predictions

            #perform matrix multiplications so the last four dimensions are
            # [... x num_freq_out x num_capsule_in x num_capsule_out  x kernel_size x capsule_dim_out]
            predictions = tf.squeeze(tf.matmul(kernel_tiled, inputs_tiled), -1)

            #sum over the kernel_size dimension
            predictions = tf.reduce_sum(predictions, -2)

            # tile the logits for each shared dimesion (batch, time)
            logits = self.logits
            for i in range(shared):
                if predictions.shape[shared-i-1].value is None:
                    shape = tf.shape(predictions)[shared-i-1]
                else:
                    shape = predictions.shape[shared-i-1].value
                tile = [shape] + [1]*len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def conv2d_matmul_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('conv2d_matmul_predict'):

        # code based on https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/ (18-11-'18)

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            #without padding:
            #num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride
            n_in = inputs.shape[-2].value
            d_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-3

            #reshape the inputs to [B*T, 1, F, N_in*D_in]
            input_shape = tf.shape(inputs)
            shared_size = 1
            for i in range(shared):
                shared_size = shared_size*input_shape[i]
            inputs = tf.reshape(inputs, shape=[shared_size, 1,
                                            num_freq_in, n_in*d_in])

            with tf.name_scope('group_freqs'):
                #create a filter that selects the values of the frequencies
                # within the convolution kernel
                tile_filter = np.zeros(shape=[1, self.kernel_size,
                                              n_in*d_in, self.kernel_size], dtype=self.dtype)
                for i in range(self.kernel_size):
                    tile_filter[0, i, :, i] = 1
                tile_filter_op = tf.constant(tile_filter)

                #Convolution with padding
                inputs = tf.nn.depthwise_conv2d(inputs, tile_filter_op,
                                                strides=[1, 1, self.stride, 1],
                                                padding='SAME')
                inputs = tf.squeeze(inputs, shared-1)
                # output_shape = tf.shape(inputs)
                # output_shape[1] should equal num_freq_out if no padding
                # reshape back to [B, T, F_out, N_in, D_in, W_f]
                inputs = tf.reshape(inputs, shape=[batch_size, -1, num_freq_out,
                                                   n_in, d_in, self.kernel_size])
                #transpose back so the last four dimensions are
                #  [... x num_freq_out x num_capsule_in x kernel_size x capsule_dim_in]
                inputs_split = tf.transpose(inputs, range(shared+2) + [shared+3, shared+2])

            #tile the inputs over the num_capsules output dimension
            # so the last five dimensions are
            # [... x num_freq_out x num_capsule_in x num_capsule_out x kernel_size x capsule_dim_in]
            inputs_split = tf.expand_dims(inputs_split, -3)
            multiples = [1]*shared + [1, 1, self.num_capsules, 1, 1]
            inputs_tiled = tf.tile(inputs_split, multiples)

            #change the capsule dimensions into column vectors
            inputs_tiled = tf.expand_dims(inputs_tiled,-1)

            #transpose the kernel so the dimensions are
            # [num_capsules_in x num_capsules_out x kernel_size x capsule_dim_out x capsule_dim_in]
            kernel = tf.transpose(self.kernel, [1, 3, 0, 4, 2])

            #tile the kernel for every shared dimension (batch, time) and num_freq_out
            with tf.name_scope('tile_kernel'):
                kernel_tiled = kernel
                for i in range(shared+1):
                    if inputs_tiled.shape[shared-i].value is None:
                        shape = tf.shape(inputs_tiled)[shared-i]
                    else:
                        shape = inputs_tiled.shape[shared-i].value
                    tile = [shape] + [1]*len(kernel_tiled.shape)
                    kernel_tiled = tf.tile(tf.expand_dims(kernel_tiled, 0), tile)

            #compute the predictions
            with tf.name_scope('compute_predictions'):
                #perform matrix multiplications so the last four dimensions are
                # [... x num_freq_out x num_capsule_in x num_capsule_out  x kernel_size x capsule_dim_out]
                predictions = tf.squeeze(tf.matmul(kernel_tiled, inputs_tiled), -1)

                #sum over the kernel_size dimension
                predictions = tf.reduce_sum(predictions, -2)

            # tile the logits for each shared dimesion (batch, time)
            with tf.name_scope('tile_logits'):
                logits = self.logits
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def cluster(self, predictions, logits):
        '''cluster the predictions into output capsules
        args:
            predictions: the predicted output capsules
            logits: the initial routing logits
        returns:
            the output capsules
        '''

        with tf.name_scope('cluster'):

            #define m-step
            def m_step(l):
                '''m step'''
                with tf.name_scope('m_step'):
                    #compute the capsule contents
                    w = self.probability_fn(l)
                    caps = tf.reduce_sum(
                        tf.expand_dims(w, -1)*predictions, -3)

                return caps, w

            #define body of the while loop
            def body(l):
                '''body'''

                caps, _ = m_step(l)
                caps = self.activation_fn(caps)

                #compare the capsule contents with the predictions
                similarity = tf.reduce_sum(
                    predictions*tf.expand_dims(caps, -3), -1)

                return l + similarity

            #get the final logits with the while loop
            lo = tf.while_loop(
                lambda l: True,
                body, [logits],
                maximum_iterations=self.routing_iters)

            #get the final output capsules
            capsules, _ = m_step(lo)
            capsules = self.activation_fn(capsules)

        return capsules

    def compute_output_shape(self, input_shape):
        '''compute the output shape'''

        if input_shape[-3].value is None:
            raise ValueError(
                'The number of frequencies must be defined, but saw: %s'
                % input_shape)

        if input_shape[-2].value is None:
            raise ValueError(
                'The number of capsules must be defined, but saw: %s'
                % input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The capsule dimension must be defined, but saw: %s'
                % input_shape)

        num_freq_in = input_shape[-3].value
        num_freq_out = num_freq_in
        #if no padding:
        #num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride

        return input_shape[:-3].concatenate(
            [num_freq_out, self.num_capsules, self.capsule_dim])

class BRCapsuleLayer(object):
    '''a Bidirectional recurrent capsule layer'''

    def __init__(self, num_capsules, capsule_dim, routing_iters=3, 
		 activation=None, input_probability_fn=None, 
		 recurrent_probability_fn=None, rec_only_vote=False):
        '''
        BRCapsuleLayer constructor

        Args:
            TODO
        '''

	self.num_capsules = num_capsules
	self.capsule_dim = capsule_dim
	self.routing_iters = routing_iters
	self._activation = activation
	self.input_probability_fn = input_probability_fn
	self.recurrent_probability_fn = recurrent_probability_fn
	self.rec_only_vote = rec_only_vote

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the rnn cell that will be used for the forward and backward
            #pass
            
            if self.rec_only_vote:
		CapsuleCellType = rnn_cell.RecCapsuleCell_RecOnlyVote
	    else:
		CapsuleCellType = rnn_cell.RecCapsuleCell
            
            rnn_cell_fw = CapsuleCellType(
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim,
                routing_iters=self.routing_iters,
                activation=self._activation,
                input_probability_fn=self.input_probability_fn,
                recurrent_probability_fn=self.recurrent_probability_fn,
                reuse=tf.get_variable_scope().reuse)
	    
            rnn_cell_bw = CapsuleCellType(
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim,
                routing_iters=self.routing_iters,
                activation=self._activation,
                input_probability_fn=self.input_probability_fn,
                recurrent_probability_fn=self.recurrent_probability_fn,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BRNNLayer(object):
    '''a BRNN layer'''

    def __init__(self, num_units, activation_fn=tf.nn.tanh, linear_out_flag=False):
        '''
        BRNNLayer constructor

        Args:
            num_units: The number of units in the one directon
            activation_fn: activation function
            linear_out_flag: if set to True, activation function will only be applied
            to the recurrent output.
        '''

        self.num_units = num_units
        self.activation_fn = activation_fn
        self.linear_out_flag = linear_out_flag

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the rnn cell that will be used for the forward and backward
            #pass
            if self.linear_out_flag:
		rnn_cell_type = rnn_cell.RNNCellLinearOut
	    else:
		rnn_cell_type = tf.contrib.rnn.BasicRNNCell
		
            rnn_cell_fw = rnn_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            rnn_cell_bw = rnn_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs
	  
class LSTMLayer(object):
    '''a LSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        '''
        LSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            activation_fn: activation function
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs, _ = dynamic_rnn(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs

class BLSTMLayer(object):
    '''a BLSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        '''
        BLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class LeakyLSTMLayer(object):
    '''a leaky LSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        '''
        LeakyLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell = rnn_cell.LayerNormBasicLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs, _ = dynamic_rnn(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs


class LeakyBLSTMLayer(object):
    '''a leaky BLSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        '''
        LeakyBLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell_fw = rnn_cell.LayerNormBasicLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormBasicLeakLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  

class LeakyBLSTMIZNotRecLayer(object):
    '''a leaky BLSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        '''
        LeakyBLSTMIZNotRecLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell_fw = rnn_cell.LayerNormIZNotRecLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormIZNotRecLeakLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  

class LeakyBLSTMNotRecLayer(object):
    '''a leaky BLSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        '''
        LeakyBLSTMNotRecLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell_fw = rnn_cell.LayerNormNotRecLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormNotRecLeakLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakychBLSTMLayer(object):
    '''a leaky ch BLSTM layer'''

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        '''
        LeakyBLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward and backward
            #pass
            lstm_cell_fw = rnn_cell.LayerNormBasicLeakchLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw = rnn_cell.LayerNormBasicLeakchLSTMCell(
                self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs
	  
class ResetLSTMLayer(object):
    '''a ResetLSTM layer'''

    def __init__(self, num_units, t_reset=1, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        '''
        ResetLSTM constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
        '''

        self.num_units = num_units
        self.t_reset = t_reset
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the lstm cell that will be used for the forward
            lstm_cell = rnn_cell.LayerNormResetLSTMCell(
                num_units=self.num_units,
                t_reset = self.t_reset,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            #do the forward computation
            outputs, _ = rnn.dynamic_rnn_time_input(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs


class BResetLSTMLayer(object):
    '''a BResetLSTM layer'''

    def __init__(self, num_units, t_reset=1, group_size=1, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        '''
        BResetLSTM constructor

        Args:
            num_units: The number of units in the one directon
            group_size: units in the same group share a state replicate
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
        '''

        self.num_units = num_units
        self.t_reset = t_reset
        self.group_size = group_size
        self.num_replicates = float(self.t_reset)/float(self.group_size)
        if int(self.num_replicates) != self.num_replicates:
	    raise ValueError('t_reset should be a multiple of group_size')
	self.num_replicates = int(self.num_replicates)
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn
        

    def __call__(self, inputs_for_forward, inputs_for_backward, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''
        
        if inputs_for_backward is None:
	    inputs_for_backward = inputs_for_forward
	
	batch_size = inputs_for_forward.get_shape()[0]

        with tf.variable_scope(scope or type(self).__name__):
            #create the lstm cell that will be used for the forward and backward
            #pass
            if self.group_size == 1:
		lstm_cell_fw = rnn_cell.LayerNormResetLSTMCell(
		    num_units=self.num_units,
		    t_reset = self.t_reset,
		    activation=self.activation_fn,
		    layer_norm=self.layer_norm,
		    dropout_keep_prob=self.recurrent_dropout,
		    reuse=tf.get_variable_scope().reuse)
		lstm_cell_bw = rnn_cell.LayerNormResetLSTMCell(
		    num_units=self.num_units,
		    t_reset = self.t_reset,
		    activation=self.activation_fn,
		    layer_norm=self.layer_norm,
		    dropout_keep_prob=self.recurrent_dropout,
		    reuse=tf.get_variable_scope().reuse)
	    else:
		lstm_cell_fw = rnn_cell.LayerNormGroupResetLSTMCell(
		    num_units=self.num_units,
		    t_reset = self.t_reset,
		    group_size = self.group_size,
		    activation=self.activation_fn,
		    layer_norm=self.layer_norm,
		    dropout_keep_prob=self.recurrent_dropout,
		    reuse=tf.get_variable_scope().reuse)
		lstm_cell_bw = rnn_cell.LayerNormGroupResetLSTMCell(
		    num_units=self.num_units,
		    t_reset = self.t_reset,
		    group_size = self.group_size,
		    activation=self.activation_fn,
		    layer_norm=self.layer_norm,
		    dropout_keep_prob=self.recurrent_dropout,
		    reuse=tf.get_variable_scope().reuse)
		

            #do the forward computation
            outputs_tupple, _ = rnn.bidirectional_dynamic_rnn_2inputs_time_input(
                lstm_cell_fw, lstm_cell_bw, inputs_for_forward, inputs_for_backward, 
                dtype=tf.float32, sequence_length=sequence_length)
	    
	    actual_outputs = tf.concat((outputs_tupple[0][0], outputs_tupple[1][0]), -1)

	    #the output replicas need to be permutated correclty such that the next layer receives
	    #the replicas in the correct order
	    T = tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size))) - 1 
	    T = tf.expand_dims(T, -1)
	    t = tf.expand_dims(range(0, self.num_replicates), 0)
	    backward_indices = tf.mod(T-t, self.num_replicates)
	    permuters = [tf.contrib.distributions.bijectors.Permute(permutation=backward_indices[utt_ind])
		  for utt_ind in range(batch_size)]
	    forward_replicas = outputs_tupple[0][1]
	    backward_replicas = outputs_tupple[1][1]
	    
	    forward_replicas_permute = tf.transpose(forward_replicas, perm=[0,1,3,2])
	    forward_replicas_for_backward_permute = [permuters[utt_ind].forward(forward_replicas_permute[utt_ind])
					      for utt_ind in range(batch_size)]
	    forward_replicas_for_backward_permute = tf.stack(forward_replicas_for_backward_permute, 0)
	    forward_replicas_for_backward = tf.transpose(forward_replicas_for_backward_permute, perm=[0,1,3,2])
	    
	    backward_replicas_permute = tf.transpose(backward_replicas, perm=[0,1,3,2])
	    backward_replicas_for_forward_permute = [permuters[utt_ind].forward(backward_replicas_permute[utt_ind])
					      for utt_ind in range(batch_size)]
	    backward_replicas_for_forward_permute = tf.stack(backward_replicas_for_forward_permute, 0)
	    backward_replicas_for_forward = tf.transpose(backward_replicas_for_forward_permute, perm=[0,1,3,2])
	    	    
	    #old and wrong implementation (did not concider correctly the array_ops.reverse which is called in 
	    #rnn.bidirectional_dynamic_rnn_2inputs_time_input)
	    #T = tf.shape(inputs_for_forward)[1]-1
	    #backward_indices = tf.mod(T - range(0, self.t_reset), self.t_reset)
	    #permuter = tf.contrib.distributions.bijectors.Permute(permutation=backward_indices)
	    #forward_replicas = outputs_tupple[0][1]
	    #backward_replicas = outputs_tupple[1][1]
	    
	    #forward_replicas_permute = tf.transpose(forward_replicas, perm=[0,1,3,2])
	    #forward_replicas_for_backward_permute = permuter.forward(forward_replicas_permute)
	    #forward_replicas_for_backward = tf.transpose(forward_replicas_for_backward_permute, perm=[0,1,3,2])
	    #backward_replicas_permute = tf.transpose(backward_replicas, perm=[0,1,3,2])
	    #backward_replicas_for_forward_permute = permuter.forward(backward_replicas_permute)
	    #backward_replicas_for_forward = tf.transpose(backward_replicas_for_forward_permute, perm=[0,1,3,2])
	    
	    outputs_for_forward = tf.concat((forward_replicas, backward_replicas_for_forward), -1)
	    outputs_for_backward = tf.concat((forward_replicas_for_backward, backward_replicas), -1)

            outputs = (actual_outputs,
		       outputs_for_forward,
		       outputs_for_backward)

            return outputs

	  
	  
class BGRULayer(object):
    '''a BGRU layer'''

    def __init__(self, num_units, activation_fn=tf.nn.tanh):
        '''
        BGRULayer constructor

        Args:
            num_units: The number of units in the one directon
        '''

        self.num_units = num_units
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the gru cell that will be used for the forward and backward
            #pass
            gru_cell_fw = tf.contrib.rnn.GRUCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            gru_cell_bw = tf.contrib.rnn.GRUCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                gru_cell_fw, gru_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs
	  	  

class LeakyBGRULayer(object):
    '''a leaky BGRU layer'''

    def __init__(self, num_units, activation_fn=tf.nn.tanh, leak_factor=1.0):
        '''
        LeakyBGRULayer constructor

        Args:
            num_units: The number of units in the one directon
            leak_factor: the leak factor (if 1, there is no leakage)
        '''

        self.num_units = num_units
        self.activation_fn = activation_fn
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences as a
                [batch_size] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):

            #create the gru cell that will be used for the forward and backward
            #pass
            gru_cell_fw = rnn_cell.LeakGRUCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            gru_cell_bw = rnn_cell.LeakGRUCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
                	    
            #do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                gru_cell_fw, gru_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs
	  

class Conv2D(object):
    '''a Conv2D layer, with max_pool and layer norm options'''

    def __init__(self, num_filters, kernel_size, strides=(1,1), padding='same', 
		 activation_fn=tf.nn.relu, layer_norm=False, max_pool_filter=(1,1),
		 transpose=False):
        '''
        BLSTMLayer constructor

        Args:
            num_filters: The number of filters
            kernel_size: kernel filter size
            strides: stride size
            padding: padding algorithm
            activation_fn: hidden unit activation
            layer_norm: whether layer normalization should be applied
            max_pool_filter: pooling filter size
            transpose: if true use tf.layers.conv2d_transpose
        '''

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.max_pool_filter = max_pool_filter
        self.transpose = transpose

    def __call__(self, inputs, scope=None):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim, in_channel] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        '''

        with tf.variable_scope(scope or type(self).__name__):
            
            if not self.transpose:
		outputs = tf.layers.conv2d(
				inputs=inputs,
				filters=self.num_filters,
				kernel_size=self.kernel_size,
				strides=self.strides,
				padding=self.padding,
				activation=self.activation_fn)
	    else:
		outputs = tf.layers.conv2d_transpose(
				inputs=inputs,
				filters=self.num_filters,
				kernel_size=self.kernel_size,
				strides=self.strides,
				padding=self.padding,
				activation=self.activation_fn)
			
	    if self.layer_norm:
		outputs = tf.layers.batch_normalization(outputs)
		
	    if self.max_pool_filter != (1,1):
		outputs = tf.layers.max_pooling2d(outputs, self.max_pool_filter, 
				   strides=self.max_pool_filter, padding='valid')
		

            return outputs