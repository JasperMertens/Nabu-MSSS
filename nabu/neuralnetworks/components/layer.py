'''@file layer.py
Neural network layers '''

import string

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from nabu.neuralnetworks.components import ops, rnn_cell, rnn
from ops import capsule_initializer
# import tensorflow.keras.constraints as constraint
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


class Conv2DCapsule(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_size=(9,9), strides=(1,1),
            padding='SAME',
            max_pool_filter=(1, 1),
            transpose=False,
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

        super(Conv2DCapsule, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
	self.conv_w = []
	self.conv_trans_w = []
        self.max_pool_filter = max_pool_filter
        self.transpose = transpose
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
        #num_freq_out = num_freq_in
        #without padding:
        #num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride
        k = self.kernel_size[0]/2

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')
	
	for i in range(0, num_capsules_in):
		
		
		self.conv_w.append(self.add_variable(
        	        name='conv_weights_%s'%i,
            		dtype=self.dtype,
            		shape=[self.kernel_size[0], self.kernel_size[1],
			capsule_dim_in, self.num_capsules*self.capsule_dim],
           		initializer=self.kernel_initializer)
		)

		self.conv_trans_w.append(self.add_variable(
        	        name='conv_trans_weights_%s'%i,
            		dtype=self.dtype,
            		shape=[self.kernel_size[0], self.kernel_size[1],
			self.num_capsules*self.capsule_dim, capsule_dim_in],
           		initializer=self.kernel_initializer)
		)

        # self.kernel = self.add_variable(
        #     name='kernel',
        #     dtype=self.dtype,
        #     shape=[num_capsules_in, capsule_dim_in,
        #             self.num_capsules, self.capsule_dim],
        #     initializer=self.kernel_initializer)
        # #
        # self.tf_kernel = self.add_variable(
        #     name='tf_kernel',
        #     dtype=self.dtype,
        #     shape=[self.kernel_size[0], self.kernel_size[1],
        #         num_capsules_in*self.num_capsules, 1],
        #     initializer=self.kernel_initializer,
        #     constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2]))

        # self.full_kernel = self.add_variable(
        #     name='full_kernel',
        #     dtype=self.dtype,
        #     shape=[self.kernel_size[0], self.kernel_size[1],
        #            num_capsules_in * capsule_dim_in,
        #            self.num_capsules * self.capsule_dim],
        #     initializer=self.kernel_initializer)

        #[num_capsules_in x num_capsules_out x kernel_size * kernel_size x capsule_dim_out x capsule_dim_in]
        # self.matmul_kernel = self.add_variable(
        #     name='matmul_kernel',
        #     dtype=self.dtype,
        #     shape=[num_capsules_in, self.num_capsules,
        #            self.kernel_size[0]*self.kernel_size[1],
        #            self.capsule_dim, capsule_dim_in],
        #     initializer=self.kernel_initializer)

        # [num_capsules_in x num_capsules_out x kernel_size * kernel_size x capsule_dim_out x capsule_dim_in]
        # self.tensordot_kernel = self.add_variable(
        #     name='tensordot_kernel',
        #     dtype=self.dtype,
        #     shape=[self.kernel_size[0]*self.kernel_size[1]*num_capsules_in,
        #            capsule_dim_in,
        #             self.num_capsules, self.capsule_dim],
        #     initializer=self.kernel_initializer)
        #
        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        super(Conv2DCapsule, self).build(input_shape)

    #pylint: disable=W0221
    def call(self, inputs, t_out_tensor=None, freq_out=None):
        '''
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        #compute the predictions
        predictions, logits = self.loop_predict(inputs, t_out_tensor, freq_out)

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
                expects inputs of dimension [B, T, F, N_in, D_in]
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            # if shared > 26 - 4:
            #     raise 'Not enough letters in the alphabet to use Einstein notation'
            # # input_shape = [shared (typicaly batch_size,time,freq),Nin,Din], kernel_shape = [Nin, Din, Nout, Dout],
            # # predictions_shape = [shared,Nin,Nout,Dout]
            # shared_shape_str = _alphabet_str[0:shared]
            # input_shape_str = shared_shape_str + 'wx'
            # kernel_shape_str = 'wxyz'
            # output_shape_str = shared_shape_str + 'wyz'
            # ein_not = '%s,%s->%s' % (input_shape_str, kernel_shape_str, output_shape_str)
            #
            # predictions = tf.einsum(ein_not, inputs, self.kernel)

            # put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank - 1])

            # compute the predictions
            predictions = tf.map_fn(
                fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            # transpose back
            predictions = tf.transpose(
                predictions, range(1, shared + 1) + [0] + [rank - 1, rank])

            # reshape to [B*D_out, T, F, N_in*N_out]
            #predictions = tf.transpose(predictions, range(shared-1) + [shared+1, shared+2, shared-1, shared])
            predictions = tf.transpose(predictions, range(shared-2)+[rank, shared-2, shared-1, rank-2, rank-1])
            predictions = tf.reshape(predictions,
                                    [batch_size*self.capsule_dim,
                                    -1, num_freq_in, num_capsule_in*self.num_capsules])

            if not self.transpose :
                # convolution over time and frequency
                # predictions = tf.layers.conv2d(predictions,
                #                                filters=1,
                #                                kernel_size=[1,self.kernel_size],
                #                                strides=self.stride,
                #                                padding="SAME",
                #                                use_bias=False)
                predictions = tf.nn.depthwise_conv2d(predictions, self.tf_kernel,
                                                     strides=[1, self.strides[0], self.strides[1], 1],
                                                     padding=self.padding)
            else :
                predictions = tf.layers.conv2d_transpose(
                    inputs=predictions,
                    filters=self.num_capsules*num_capsule_in,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    use_bias=False)
                # predictions = tf.nn.conv2d_transpose(predictions, self.tf_kernel,
                #                                      strides=[1, self.strides[0], self.strides[1], 1],
                #                                      output_shape=output_shape,
                #                                      padding=self.padding)


            # reshape back to [B, T, F, N_in, N_out, D_out]
            # predictions = tf.reshape(predictions, [batch_size, num_capsule_in, self.num_capsules, self.capsule_dim, -1, num_freq_out])
            # predictions = tf.transpose(predictions, range(shared-2)+[rank-1, rank, shared-2, shared-1, shared])
            predictions = tf.reshape(predictions, [batch_size, self.capsule_dim, -1,
                                                   predictions.shape[2], num_capsule_in, self.num_capsules])
            predictions = tf.transpose(predictions, range(shared-2)+[shared-1,rank-2,rank-1,rank,shared-2])

            # tile the logits for each shared dimesion (batch, time, frequency)
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

    def full_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
                expects inputs of dimension [B, T, F, N_in, D_in]
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            # reshape to [B, T, F, N_in*D_in]
            inputs = tf.reshape(inputs,
                            [batch_size, -1, num_freq_in,
                             num_capsule_in*capsule_dim_in])

            if not self.transpose :
                # convolution over time and frequency
                # predictions = tf.layers.conv2d(predictions,
                #                                filters=1,
                #                                kernel_size=[1,self.kernel_size],
                #                                strides=self.stride,
                #                                padding="SAME",
                #                                use_bias=False)
                predictions = tf.nn.depthwise_conv2d(inputs, self.full_kernel,
                                                     strides=[1, self.strides[0], self.strides[1], 1],
                                                     padding=self.padding)
                # reshape to [B, T, F, N_in, D_in, N_out, D_out]
                predictions = tf.reshape(predictions,
                                         [batch_size, -1, num_freq_out,
                                          num_capsule_in, capsule_dim_in,
                                          self.num_capsules, self.capsule_dim])
                # sum over D_in dimension
                predictions = tf.reduce_sum(predictions, -3)


            else :
                predictions = tf.layers.conv2d_transpose(
                    inputs=inputs,
                    filters=self.num_capsules*num_capsule_in,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    use_bias=False)


            # tile the logits for each shared dimesion (batch, time, frequency)
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


    def simple_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
                expects inputs of dimension [B, T, F, N_in, D_in]
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            # reshape to [B, T, F, N_in*D_in]
            inputs = tf.reshape(inputs,
                            [batch_size, -1, num_freq_in,
                             num_capsule_in*capsule_dim_in])


            # convolution over time and frequency
            predictions = tf.layers.conv2d(inputs,
                                           filters=num_capsule_in*self.num_capsules*self.capsule_dim,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding="SAME",
                                           use_bias=True)

            # reshape to [B, T, F, N_in, N_out, D_out]
            predictions = tf.reshape(predictions,
                                     [batch_size, -1, num_freq_out,
                                      num_capsule_in,
                                      self.num_capsules, self.capsule_dim])


            # tile the logits for each shared dimesion (batch, time, frequency)
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


    def loop_predict(self, inputs, t_out_tensor, freq_out):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
                expects inputs of dimension [B, T, F, N_in, D_in]
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
	    if not self.transpose:
            	freq_out = int(np.ceil(float(num_freq_in)/float(self.strides[1])))
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            convs = []
            for i in range(0, num_capsule_in):
                with tf.name_scope('conv_2d%d' % i):
                    slice = inputs[:, :, :, i, :]
                    if self.transpose:
                        conv = tf.nn.conv2d_transpose(slice, self.conv_trans_w[i],
						output_shape = [batch_size, t_out_tensor, freq_out,
							 self.capsule_dim*self.num_capsules],
                                                strides=[1, self.strides[0], self.strides[1], 1],
                                                padding=self.padding)
                    else:
                        conv = tf.nn.conv2d(slice, self.conv_w[i],
                                                strides=[1, self.strides[0], self.strides[1], 1],
                                                        padding=self.padding)
		    expanded = tf.expand_dims(conv, 3)
                    convs.append(expanded)

            prev_slice = tf.concat(convs, 3)

            # reshape to [B, T, F, N_in, N_out, D_out]
            predictions = tf.reshape(prev_slice,
                                     [batch_size, -1, freq_out,
                                      num_capsule_in,
                                      self.num_capsules, self.capsule_dim])

            # tile the logits for each shared dimension (batch, time, frequency)
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


    def loop_predict_v2(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
                expects inputs of dimension [B, T, F, N_in, D_in]
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            #num_freq_out = int(np.ceil(float(num_freq_in)/float(self.strides[1])))
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            convs = []
            for i in range(0, num_capsule_in):
                with tf.name_scope('conv_2d%d' % i):
                    slice = inputs[:, :, :, i, :]
                    if self.transpose:
                        conv = tf.layers.conv2d_transpose(slice, self.num_capsules * self.capsule_dim,
                                                kernel_size=self.kernel_size,
                                                strides=self.strides,
                                                padding=self.padding,
                                                use_bias=False)
                    else:
                        conv = tf.layers.conv2d(slice, self.num_capsules*self.capsule_dim,
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        padding=self.padding,
                                                        use_bias=False)
                    expanded = tf.expand_dims(conv, 3)
                    convs.append(expanded)

            prev_slice = tf.concat(convs, 3)
            num_freq_out = tf.shape(prev_slice)[2]

            # reshape to [B, T, F, N_in, N_out, D_out]
            predictions = tf.reshape(prev_slice,
                                     [batch_size, -1, num_freq_out,
                                      num_capsule_in,
                                      self.num_capsules, self.capsule_dim])

            # tile the logits for each shared dimension (batch, time, frequency)
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
            rank = len(inputs.shape) # rank=5
            shared = rank-2 # shared=3

            #reshape the inputs to [B, T, F, N_in*D_in]
            inputs = tf.reshape(inputs, shape=[batch_size, -1,
                                               num_freq_in, n_in*d_in])

            with tf.name_scope('group_freqs'):
                #create a filter that selects the values of the frequencies
                # within the convolution kernel
                tile_filter = np.zeros(shape=[self.kernel_size[0], self.kernel_size[1],
                                        n_in*d_in, self.kernel_size[0]*self.kernel_size[1]],
                                        dtype=self.dtype)
                for i in range(self.kernel_size[1]):
                    for j in range(self.kernel_size[0]):
                        tile_filter[i, j, :, i*self.kernel_size[1]+j] = 1
                tile_filter_op = tf.constant(tile_filter)

                #Convolution with padding
                inputs = tf.nn.depthwise_conv2d(inputs, tile_filter_op,
                                                strides=[1, self.strides[0], self.strides[0], 1],
                                                padding='SAME')
                # output_shape = tf.shape(inputs)
                # output_shape[1] should equal num_freq_out if no padding
                # reshape back to [B, T, F_out, N_in, D_in, W_f]
                inputs = tf.reshape(inputs, shape=[batch_size, -1, num_freq_out,
                                                   n_in, d_in, self.kernel_size[0]*self.kernel_size[1]])
                #transpose back so the last four dimensions are
                #  [... x num_freq_out x num_capsule_in x kernel_size*kernel_size x capsule_dim_in]
                inputs_split = tf.transpose(inputs, range(shared+1) + [shared+2, shared+1])

                # test if group_freqs takes a long time, it does not
                # inputs_split = tf.expand_dims(inputs, -2)
                # mult = [1]*(shared+1) + [self.kernel_size[0]*self.kernel_size[1], 1]
                # inputs_split = tf.tile(inputs_split, mult)

            #tile the inputs over the num_capsules output dimension
            # so the last five dimensions are
            # [... x num_freq_out x num_capsule_in x num_capsule_out x kernel_size*kernel_size x capsule_dim_in]
            inputs_split = tf.expand_dims(inputs_split, -3)
            multiples = [1]*(shared+1) + [self.num_capsules, 1, 1]
            inputs_tiled = tf.tile(inputs_split, multiples)

            #change the capsule dimensions into column vectors
            inputs_tiled = tf.expand_dims(inputs_tiled,-1)

            #tile the kernel for every shared dimension (batch, time) and num_freq_out
            with tf.name_scope('tile_kernel'):
                kernel_tiled = self.matmul_kernel
                for i in range(shared):
                    if inputs_tiled.shape[shared-i-1].value is None:
                        shape = tf.shape(inputs_tiled)[shared-i-1]
                    else:
                        shape = inputs_tiled.shape[shared-i-1].value
                    tile = [shape] + [1]*len(kernel_tiled.shape)
                    kernel_tiled = tf.tile(tf.expand_dims(kernel_tiled, 0), tile)

            #compute the predictions
            with tf.name_scope('compute_predictions'):
                #perform matrix multiplications so the last five dimensions are
                # [... x num_freq_out x num_capsule_in x num_capsule_out  x kernel_size*kernel_size x capsule_dim_out]
                predictions = tf.squeeze(tf.matmul(kernel_tiled, inputs_tiled), -1)
                #predictions = tf.squeeze(inputs_tiled, -1)

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

    def tensordot_predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        '''

        with tf.name_scope('tensordot_predict'):

        # code based on https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/ (18-11-'18)

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            #without padding:
            #num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride
            n_in = inputs.shape[-2].value
            d_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) # rank=5
            shared = rank-2 # shared=3

            #reshape the inputs to [B, T, F, N_in*D_in]
            inputs = tf.reshape(inputs, shape=[batch_size, -1,
                                               num_freq_in, n_in*d_in])

            with tf.name_scope('group_freqs'):
                #create a filter that selects the values of the frequencies
                # within the convolution kernel
                tile_filter = np.zeros(shape=[self.kernel_size[0], self.kernel_size[1],
                                        n_in*d_in, self.kernel_size[0]*self.kernel_size[1]],
                                        dtype=self.dtype)
                for i in range(self.kernel_size[1]):
                    for j in range(self.kernel_size[0]):
                        tile_filter[i, j, :, i*self.kernel_size[0]+j] = 1
                tile_filter_op = tf.constant(tile_filter)

                #Convolution with padding
                inputs = tf.nn.depthwise_conv2d(inputs, tile_filter_op,
                                                strides=[1, self.strides[0], self.strides[0], 1],
                                                padding='SAME')
                # output_shape = tf.shape(inputs)
                # output_shape[1] should equal num_freq_out if no padding
                # reshape back to [B, T, F_out, N_in, D_in, W*W]
                inputs = tf.reshape(inputs, shape=[batch_size, -1, num_freq_out,
                                                   n_in, d_in, self.kernel_size[0]*self.kernel_size[1]])
                #transpose back so the last four dimensions are
                #  [... x num_freq_out x num_capsule_in x kernel_size*kernel_size x capsule_dim_in]
                inputs = tf.transpose(inputs, range(shared) + [shared+2, shared, shared+1])

                #reshape to [B, T, F, W*W*N_in, D_in]
                inputs = tf.reshape(inputs, shape=[batch_size, -1, num_freq_out,
                                                   self.kernel_size[0]*self.kernel_size[1]*n_in, d_in])

            #compute the predictions
            with tf.name_scope('compute_predictions'):
                #inputs: [B, T, F, W*W*N_in, D_in]
                #kernel: [W*W*N_in, D_in, N_out, D_out]

                # put the input capsules within the kernel as the first dimension
                inputs = tf.transpose(inputs, [shared] + range(shared) + [rank - 1])

                # compute the predictions
                predictions = tf.map_fn(
                    fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                    elems=(inputs, self.tensordot_kernel),
                    dtype=self.dtype or tf.float32)

                # transpose back
                predictions = tf.transpose(
                    predictions, range(1, shared + 1) + [0] + [rank - 1, rank])

                #reshape back to [B, T, F, W*W, N_in, N_out, D_out]
                predictions = tf.reshape(predictions, shape=[batch_size, -1, num_freq_out,
                                                   self.kernel_size[0]*self.kernel_size[1],
                                                             n_in, self.num_capsules, self.capsule_dim])

                #sum over the kernel_size dimension
                predictions = tf.reduce_sum(predictions, shared)

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

class Conv2DCapsSep(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_size=(9,9), strides=(1,1),
            padding='SAME',
            max_pool_filter=(1, 1),
            transpose=False,
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

        super(Conv2DCapsSep, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.max_pool_filter = max_pool_filter
        self.transpose = transpose
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
        k = self.kernel_size[0]/2

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

        self.tf_kernel = self.add_variable(
            name='tf_kernel',
            dtype=self.dtype,
            shape=[self.kernel_size[0], self.kernel_size[1],
                self.num_capsules, 1],
            initializer=self.kernel_initializer,
            constraint= tf.keras.constraints.UnitNorm(axis=[0,1,2]))


        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        super(Conv2DCapsSep, self).build(input_shape)

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
        routed= self.cluster(predictions, logits)

        outputs = self.convolution(routed)

        return outputs

    def predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
                expects inputs of dimension [B, T, F, N_in, D_in]
        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[-3].value
            num_freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            # if shared > 26 - 4:
            #     raise 'Not enough letters in the alphabet to use Einstein notation'
            # # input_shape = [shared (typicaly batch_size,time,freq),Nin,Din], kernel_shape = [Nin, Din, Nout, Dout],
            # # predictions_shape = [shared,Nin,Nout,Dout]
            # shared_shape_str = _alphabet_str[0:shared]
            # input_shape_str = shared_shape_str + 'wx'
            # kernel_shape_str = 'wxyz'
            # output_shape_str = shared_shape_str + 'wyz'
            # ein_not = '%s,%s->%s' % (input_shape_str, kernel_shape_str, output_shape_str)
            #
            # predictions = tf.einsum(ein_not, inputs, self.kernel)

            # put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank - 1])

            # compute the predictions
            predictions = tf.map_fn(
                fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            # transpose back
            predictions = tf.transpose(
                predictions, range(1, shared + 1) + [0] + [rank - 1, rank])

            # tile the logits for each shared dimesion (batch, time, frequency)
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

    def convolution(self, routed):

        batch_size = routed.shape[0].value
        num_freq_in = routed.shape[2].value
        num_freq_out = num_freq_in

        # number of shared dimensions
        rank = len(routed.shape)  # normally rank=5
        shared = rank - 2  # normally shared=3

        # reshape to [B*D_out, T, F, N_out]
        routed = tf.transpose(routed, range(shared - 2) + [rank-1, shared - 2, shared - 1, rank - 2])
        routed = tf.reshape(routed,
                                 [batch_size * self.capsule_dim,
                                  -1, num_freq_in, self.num_capsules])

        outputs = tf.nn.depthwise_conv2d(routed, self.tf_kernel,
                                             strides=[1, self.strides[0], self.strides[1], 1],
                                             padding=self.padding)

        # reshape back to [B, T, F, N_out, D_out]
        outputs = tf.reshape(outputs, [batch_size, self.capsule_dim, -1,
                                               num_freq_out, self.num_capsules])
        outputs = tf.transpose(outputs, range(shared - 2) + [shared - 1, rank - 2, rank - 1, shared - 2])
        # Perform squash
        outputs = self.activation_fn(outputs)

        return outputs


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
		 recurrent_probability_fn=None, rec_only_vote=False,
		 accumulate_input_logits=True, accumulate_state_logits=True):
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
	self.accumulate_input_logits = accumulate_input_logits
	self.accumulate_state_logits = accumulate_state_logits
	

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
		rnn_cell_fw = rnn_cell.RecCapsuleCell_RecOnlyVote(
		    num_capsules=self.num_capsules,
		    capsule_dim=self.capsule_dim,
		    routing_iters=self.routing_iters,
		    activation=self._activation,
		    input_probability_fn=self.input_probability_fn,
		    recurrent_probability_fn=self.recurrent_probability_fn,
		    accumulate_input_logits=self.accumulate_input_logits,
		    accumulate_state_logits=self.accumulate_state_logits,
		    reuse=tf.get_variable_scope().reuse)
		
		rnn_cell_bw = rnn_cell.RecCapsuleCell_RecOnlyVote(
		    num_capsules=self.num_capsules,
		    capsule_dim=self.capsule_dim,
		    routing_iters=self.routing_iters,
		    activation=self._activation,
		    input_probability_fn=self.input_probability_fn,
		    recurrent_probability_fn=self.recurrent_probability_fn,
		    accumulate_input_logits=self.accumulate_input_logits,
		    accumulate_state_logits=self.accumulate_state_logits,
		    reuse=tf.get_variable_scope().reuse)
	    else:
		rnn_cell_fw = rnn_cell.RecCapsuleCell(
		    num_capsules=self.num_capsules,
		    capsule_dim=self.capsule_dim,
		    routing_iters=self.routing_iters,
		    activation=self._activation,
		    input_probability_fn=self.input_probability_fn,
		    recurrent_probability_fn=self.recurrent_probability_fn,
		    reuse=tf.get_variable_scope().reuse)
		
		rnn_cell_bw = rnn_cell.RecCapsuleCell(
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

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh,
		 separate_directions=False, fast_version=False):
        '''
        BLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            separate_directions: wether the forward and backward directions should
            be separated for deep networks.
            fast_version: wheter a fast version of the LSTM cell should be used. Not compatible 
            with layer normalization or recurrent_dropout.
        '''

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn
        self.separate_directions = separate_directions
        self.fast_version = fast_version
        if self.fast_version and (self.recurrent_dropout<1.0 or self.layer_norm):
	    raise 'Fast version of lstm cell is not compatible with layer normalization or recurrent_dropout'

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
            #if self.fast_version:
		## make input time major
		#inputs = tf.transpose(inputs, [1, 0] + range(2,len(inputs.get_shape())))

		#forward_rnn = tf.contrib.rnn.LSTMBlockFusedCell(
		    #num_units=self.num_units,
		    #reuse=tf.get_variable_scope().reuse)
		#backward_rnn = tf.contrib.rnn.LSTMBlockFusedCell(
		    #num_units=self.num_units,
		    #reuse=tf.get_variable_scope().reuse)
		
		#forward_output, _ = forward_rnn(inputs, dtype=tf.float32,
		    #sequence_length=sequence_length)
		#reverse_input=array_ops.reverse_sequence(
		    #input=inputs, seq_lengths=sequence_length,
		    #seq_dim=0, batch_dim=1)
		#backward_output, _ = forward_rnn(reverse_input, dtype=tf.float32,
		    #sequence_length=sequence_length)
		#reverse_backward_output=array_ops.reverse_sequence(
		    #input=backward_output, seq_lengths=sequence_length,
		    #seq_dim=0, batch_dim=1)
		
		#outputs = tf.concat((forward_output, backward_output), 2)
		#outputs = tf.transpose(outputs, [1, 0] + range(2,len(outputs.get_shape())))
	    #else:
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
            if not self.separate_directions:
		outputs_tupple, _ = bidirectional_dynamic_rnn(
		    lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
		    sequence_length=sequence_length)

		outputs = tf.concat(outputs_tupple, 2)
            else:
		outputs, _ = rnn.bidirectional_dynamic_rnn_2inputs(
		    lstm_cell_fw, lstm_cell_bw, inputs[0], inputs[1], dtype=tf.float32,
		    sequence_length=sequence_length)

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

    def __init__(self, num_units, t_reset=1, group_size=1, symmetric_context=False, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        '''
        BResetLSTM constructor

        Args:
            num_units: The number of units in the one directon
            group_size: units in the same group share a state replicate
            symmetric_context: if True, input to next layer should have same amount of context
            in both directions. If False, reversed input to next layers has full (t_reset) context.
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
        '''

        self.num_units = num_units
        self.t_reset = t_reset
        self.group_size = group_size
        self.num_replicates = float(self.t_reset)/float(self.group_size)
        if int(self.num_replicates) != self.num_replicates:
	    raise ValueError('t_reset should be a multiple of group_size')
	self.symmetric_context = symmetric_context
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
	max_length = tf.shape(inputs_for_forward)[1]

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
		
		tile_shape = [1,1,self.t_reset,1]
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
		
		tile_shape = [1,1,lstm_cell_fw._num_replicates,1]

            #do the forward computation
            outputs_tupple, _ = rnn.bidirectional_dynamic_rnn_2inputs_time_input(
                lstm_cell_fw, lstm_cell_bw, inputs_for_forward, inputs_for_backward, 
                dtype=tf.float32, sequence_length=sequence_length)
	    
	    actual_outputs_forward = outputs_tupple[0][0]
	    actual_outputs_backward = outputs_tupple[1][0]
	    actual_outputs = tf.concat((actual_outputs_forward,actual_outputs_backward), -1)
	    
	    forward_replicas = outputs_tupple[0][1]
	    backward_replicas = outputs_tupple[1][1]

	    if not self.symmetric_context:
		forward_for_backward = tf.expand_dims(actual_outputs_forward,-2)
		forward_for_backward = tf.tile(forward_for_backward, tile_shape)
		
		backward_for_forward = tf.expand_dims(actual_outputs_backward,-2)
		backward_for_forward = tf.tile(backward_for_forward, tile_shape)
		
		outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
		outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

	    else:
		##the output replicas need to be permutated correclty such that the next layer receives
		##the replicas in the correct order
		T = tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size))) 
	    	T_min_1 = tf.expand_dims(T - 1, -1)
	    	
	    	numbers_to_maxT = tf.range(0, max_length)
	    	numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT,0),-1)
	    	
	    	numbers_to_k = tf.expand_dims(range(0, self.num_replicates), 0)
	    	
	    	backward_indices_for_forward_t_0 = numbers_to_k+T_min_1
	    	#backward_indices_for_forward_t_0 = tf.mod(backward_indices_for_forward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
	    	backward_indices_for_forward_t_0 = tf.expand_dims(backward_indices_for_forward_t_0,1)
	    	backward_indices_for_forward_t = tf.mod(backward_indices_for_forward_t_0 + numbers_to_maxT, self.num_replicates)
	    	
	    	forward_indices_for_backward_t_0 = numbers_to_k-T_min_1
	    	#forward_indices_for_backward_t_0 = tf.mod(forward_indices_for_backward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
	    	forward_indices_for_backward_t_0 = tf.expand_dims(forward_indices_for_backward_t_0,1)
	    	forward_indices_for_backward_t = tf.mod(forward_indices_for_backward_t_0 - numbers_to_maxT, self.num_replicates)
	    	
	    	ra1=tf.range(batch_size)
	    	ra1=tf.expand_dims(tf.expand_dims(ra1,-1),-1)
	    	ra1=tf.tile(ra1,[1,max_length,self.num_replicates])
	    	ra2=tf.range(max_length)
	    	ra2=tf.expand_dims(tf.expand_dims(ra2,0),-1)
	    	ra2=tf.tile(ra2,[batch_size,1,self.num_replicates])
	    	stacked_backward_indices_for_forward_t = tf.stack([ra1,ra2,backward_indices_for_forward_t],axis=-1)
	    	backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward_t)
	    	stacked_forward_indices_for_backward_t = tf.stack([ra1,ra2,forward_indices_for_backward_t],axis=-1)
	    	forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward_t)
	    	
		outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
		outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)
	    	
	    #permuters = [tf.contrib.distributions.bijectors.Permute(permutation=backward_indices[utt_ind])
		  #for utt_ind in range(batch_size)]
	    #forward_replicas = outputs_tupple[0][1]
	    #backward_replicas = outputs_tupple[1][1]
	    
	    #forward_replicas_permute = tf.transpose(forward_replicas, perm=[0,1,3,2])
	    #forward_replicas_for_backward_permute = [permuters[utt_ind].forward(forward_replicas_permute[utt_ind])
					      #for utt_ind in range(batch_size)]
	    #forward_replicas_for_backward_permute = tf.stack(forward_replicas_for_backward_permute, 0)
	    #forward_replicas_for_backward = tf.transpose(forward_replicas_for_backward_permute, perm=[0,1,3,2])
	    
	    #backward_replicas_permute = tf.transpose(backward_replicas, perm=[0,1,3,2])
	    #backward_replicas_for_forward_permute = [permuters[utt_ind].forward(backward_replicas_permute[utt_ind])
					      #for utt_ind in range(batch_size)]
	    #backward_replicas_for_forward_permute = tf.stack(backward_replicas_for_forward_permute, 0)
	    #backward_replicas_for_forward = tf.transpose(backward_replicas_for_forward_permute, perm=[0,1,3,2])
	    
	    #outputs_for_forward = tf.concat((forward_replicas, backward_replicas_for_forward), -1)
	    #outputs_for_backward = tf.concat((forward_replicas_for_backward, backward_replicas), -1)

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
