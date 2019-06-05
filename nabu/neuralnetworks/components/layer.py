"""@file layer.py
Neural network layers """

import string

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from nabu.neuralnetworks.components import ops, rnn_cell, rnn, rnn_cell_impl


from ops import capsule_initializer

# import tensorflow.keras.constraints as constraint
import numpy as np
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
import pdb

_alphabet_str=string.ascii_lowercase


class Capsule(tf.layers.Layer):
    """a capsule layer"""

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_initializer=None,
            logits_initializer=None,
            logits_prior=False,
            routing_iters=3,
            activation_fn=None,
            probability_fn=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            **kwargs):

        """Capsule layer constructor
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
        """

        super(Capsule, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.logits_prior = logits_prior
        self.routing_iters = routing_iters
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        """creates the variables of this layer
        args:
            input_shape: the shape of the input
        """

        # pylint: disable=W0201

        # input dimensions
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
            trainable=self.logits_prior
        )

        super(Capsule, self).build(input_shape)

    # pylint: disable=W0221
    def call(self, inputs):
        """
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        """

        # compute the predictions
        predictions, logits = self.predict(inputs)

        # cluster the predictions
        outputs = self.cluster(predictions, logits)

        return outputs

    def predict(self, inputs):
        """
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        """

        with tf.name_scope('predict'):

            # number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2

            # put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank-1])

            # compute the predictins
            predictions = tf.map_fn(
                fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            # transpose back
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
        """
        compute the predictions for the output capsules and initialize the
        routing logits
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns: the output capsule predictions
        """

        with tf.name_scope('predict'):

            # number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2

            if shared > 26-4:
                raise ValueError('Not enough letters in the alphabet to use Einstein notation')
            # input_shape = [shared (typicaly batch_size,time),Nin,Din], kernel_shape = [Nin, Din, Nout, Dout],
            # predictions_shape = [shared,Nin,Nout,Dout]
            shared_shape_str = _alphabet_str[0:shared]
            input_shape_str = shared_shape_str+'wx'
            kernel_shape_str = 'wxyz'
            output_shape_str = shared_shape_str+'wyz'
            ein_not = '%s,%s->%s' % (input_shape_str, kernel_shape_str, output_shape_str)

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
        """cluster the predictions into output capsules
        args:
            predictions: the predicted output capsules
            logits: the initial routing logits
        returns:
            the output capsules
        """

        with tf.name_scope('cluster'):

            # define m-step
            def m_step(l):
                """m step"""
                with tf.name_scope('m_step'):
                    # compute the capsule contents
                    w = self.probability_fn(l)
                    caps = tf.reduce_sum(
                        tf.expand_dims(w, -1)*predictions, -3)

                return caps, w

            # define body of the while loop
            def body(l):
                """body"""

                caps, _ = m_step(l)
                caps = self.activation_fn(caps)

                # compare the capsule contents with the predictions
                similarity = tf.reduce_sum(
                    predictions*tf.expand_dims(caps, -3), -1)

                return l + similarity

            # get the final logits with the while loop
            lo = tf.while_loop(
                lambda l: True,
                body, [logits],
                maximum_iterations=self.routing_iters)

            # get the final output capsules
            capsules, _ = m_step(lo)
            capsules = self.activation_fn(capsules)

        return capsules

    def compute_output_shape(self, input_shape):
        """compute the output shape"""

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

        # self.conv_weights = self.add_variable(
        #     name='conv_weights',
        #     dtype=self.dtype,
        #     shape=[num_capsules_in, self.kernel_size[0], self.kernel_size[1],
        #            capsule_dim_in, self.num_capsules * self.capsule_dim],
        #     initializer=self.kernel_initializer)

        self.bias = self.add_variable(
            name='bias',
            dtype=self.dtype,
            shape=[self.num_capsules, self.capsule_dim],
            initializer=None)

        # self.shared_weights = self.add_variable(
        #     name='shared_weights',
        #     dtype=self.dtype,
        #     shape=[self.kernel_size[0], self.kernel_size[1],
        #            capsule_dim_in, self.num_capsules * self.capsule_dim],
        #     initializer=self.kernel_initializer)

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
        self.tensordot_kernel = self.add_variable(
            name='tensordot_kernel',
            dtype=self.dtype,
            shape=[self.kernel_size[0]*self.kernel_size[1]*num_capsules_in,
                   capsule_dim_in,
                    self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)
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
        predictions, logits, bias = self.tensordot_predict(inputs)

        #cluster the predictions
        outputs = self.cluster(predictions, logits, bias)

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


            predictions = tf.nn.depthwise_conv2d(predictions, self.tf_kernel,
                                                     strides=[1, self.strides[0], self.strides[1], 1],
                                                     padding=self.padding)


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

    def shared_predict(self, inputs):
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

            # reshape to [N_in*B, T, F, D_in]
            inputs = tf.transpose(inputs, [3,0,1,2,4])
            inputs = tf.reshape(inputs,
                            [batch_size*num_capsule_in, -1, num_freq_in,
                             capsule_dim_in])


            # convolution over time and frequency
            predictions = tf.nn.conv2d(inputs, self.shared_weights,
                                        strides=[1, self.strides[0], self.strides[1], 1],
                                        padding=self.padding)

            # reshape to [B, T, F, N_in, N_out, D_out]
            predictions = tf.reshape(predictions,
                                     [num_capsule_in, batch_size, -1, num_freq_out,
                                      self.num_capsules, self.capsule_dim])
            predictions = tf.transpose(predictions, [1, 2, 3, 0, 4, 5])


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


    def loop_predict(self, inputs):
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
            freq_out = num_freq_in
            num_capsule_in = inputs.shape[-2].value
            capsule_dim_in = inputs.shape[-1].value

            #number of shared dimensions
            rank = len(inputs.shape) #normally rank=5
            shared = rank-2 #normally shared=3

            convs = []
            for i in range(0, num_capsule_in):
                with tf.name_scope('conv_2d%d' % i):
                    slice = inputs[:, :, :, i, :]
                    conv = tf.nn.conv2d(slice, self.conv_weights[i,:,:,:,:],
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
            # tile the logits and bias for each shared dimension (batch, time, frequency)
            with tf.name_scope('tile_logits_bias'):
                logits = self.logits
                bias = self.bias
                for i in range(shared):
                    if predictions.shape[shared - i - 1].value is None:
                        shape = tf.shape(predictions)[shared - i - 1]
                    else:
                        shape = predictions.shape[shared - i - 1].value
                    tile = [shape] + [1] * len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)

            return predictions, logits, bias

        #     # tile the logits for each shared dimension (batch, time, frequency)
        #     with tf.name_scope('tile_logits'):
        #         logits = self.logits
        #         for i in range(shared):
        #             if predictions.shape[shared-i-1].value is None:
        #                 shape = tf.shape(predictions)[shared-i-1]
        #             else:
        #                 shape = predictions.shape[shared-i-1].value
        #             tile = [shape] + [1]*len(logits.shape)
        #             logits = tf.tile(tf.expand_dims(logits, 0), tile)
        #
        # return predictions, logits


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
                bias = self.bias
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)

        return predictions, logits, bias

    def tensordot_predict(self, inputs):
        #Faster than matmul predict
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
                bias = self.bias
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)

        return predictions, logits, bias

    def cluster(self, predictions, logits, bias):
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
                    caps = caps + bias

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
                   num_capsules_in*self.num_capsules, 1],
            initializer=self.kernel_initializer,
            constraint= tf.keras.constraints.UnitNorm(axis=[0,1]))


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
        outputs= self.cluster(predictions, logits)

        # outputs = self.convolution(routed)

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

            # reshape to [B*D_out, T, F, N_in*N_out]
            # predictions = tf.transpose(predictions, range(shared-1) + [shared+1, shared+2, shared-1, shared])
            predictions = tf.transpose(predictions,
                                       range(shared - 2) + [rank, shared - 2, shared - 1, rank - 2, rank - 1])
            predictions = tf.reshape(predictions,
                                     [batch_size * self.capsule_dim,
                                      -1, num_freq_in, num_capsule_in * self.num_capsules])

            predictions = tf.nn.depthwise_conv2d(predictions, self.tf_kernel,
                                                 strides=[1, self.strides[0], self.strides[1], 1],
                                                 padding=self.padding)

            # reshape back to [B, T, F, N_in, N_out, D_out]
            # predictions = tf.reshape(predictions, [batch_size, num_capsule_in, self.num_capsules, self.capsule_dim, -1, num_freq_out])
            # predictions = tf.transpose(predictions, range(shared-2)+[rank-1, rank, shared-2, shared-1, shared])
            predictions = tf.reshape(predictions, [batch_size, self.capsule_dim, -1,
                                                   predictions.shape[2], num_capsule_in, self.num_capsules])
            predictions = tf.transpose(predictions,
                                       range(shared - 2) + [shared - 1, rank - 2, rank - 1, rank, shared - 2])

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

class Conv2DCapsGridRouting(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_size=(9, 9), strides=(1, 1),
            padding='SAME',
            transpose=False,
            routing_iters=3,
            use_bias=False,
            shared=True,
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

        super(Conv2DCapsGridRouting, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.transpose = transpose
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.routing_iters = routing_iters
        self.use_bias = use_bias
        self.sharing_weights = shared
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        '''creates the variables of this layer
        args:
            input_shape: the shape of the input
        '''

        # pylint: disable=W0201

        # input dimensions
        num_freq_in = input_shape[-3].value
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value
        # num_freq_out = num_freq_in
        # without padding:
        # num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        kernel_area = self.kernel_size[0]*self.kernel_size[1]
        # self.grid_weights = self.add_variable(
        #     name='weights',
        #     dtype=self.dtype,
        #     shape=[kernel_area * num_capsules_in, self.num_capsules,
        #            self.capsule_dim, capsule_dim_in],
        #     initializer=self.kernel_initializer
        # )

        # [num_capsules_in x num_capsules_out x kernel_size * kernel_size x capsule_dim_out x capsule_dim_in]
        self.tensordot_kernel = self.add_variable(
            name='tensordot_kernel',
            dtype=self.dtype,
            shape=[self.kernel_size[0] * self.kernel_size[1] * num_capsules_in,
                   capsule_dim_in,
                   self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[kernel_area*num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        # self.constrained_logits = self.add_variable(
        #     name='constr_logits',
        #     dtype=self.dtype,
        #     shape=[num_capsules_in, self.num_capsules],
        #     initializer=self.logits_initializer,
        #     trainable=False
        # )

        self.bias = tf.zeros(shape=[self.num_capsules, self.capsule_dim])
        if self.use_bias:
            self.bias = self.add_variable(
            name='bias',
            dtype=self.dtype,
            shape=[self.num_capsules, self.capsule_dim],
            initializer=None)

        super(Conv2DCapsGridRouting, self).build(input_shape)

    # pylint: disable=W0221
    def call(self, inputs):
        '''
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        # compute the predictions
        predictions, logits, bias = self.tensordot_predict(inputs)

        # cluster the predictions
        outputs = self.cluster(predictions, logits, bias)

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
        if not self.transpose:
            freq_out = int(np.ceil(float(num_freq_in) / float(self.strides[1])))
        num_capsule_in = inputs.shape[-2].value
        capsule_dim_in = inputs.shape[-1].value

        # number of shared dimensions
        rank = len(inputs.shape)  # normally rank=5
        shared = rank-2

        # reshape to [B, T, F, N_in*D_in]
        inputs = tf.reshape(inputs,
                                 [batch_size, -1, freq_out,
                                  num_capsule_in*capsule_dim_in])


        with tf.name_scope('get_patches'):
            # code based on https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/ (18-11-'18)
            # create a filter that selects the values of the T-F bins
            # within the convolution kernel
            tile_filter = np.zeros(shape=[self.kernel_size[0], self.kernel_size[1],
                                          num_capsule_in*capsule_dim_in,
                                          self.kernel_size[0] * self.kernel_size[1]],
                                   dtype=self.dtype)
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[1]):
                    tile_filter[i, j, :, i * self.kernel_size[1] + j] = 1
            tile_filter_op = tf.constant(tile_filter)

            # Convolution with padding
            patches = tf.nn.depthwise_conv2d(inputs, tile_filter_op,
                                            strides=[1, self.strides[0], self.strides[1], 1],
                                            padding=self.padding)

            # reshape to [B, T_out, F_out, N_in, D_in, W]
            f_out = patches.shape[2].value
            patches = tf.reshape(patches, shape=[batch_size, -1, f_out,
                                                num_capsule_in, capsule_dim_in,
                                                 self.kernel_size[0] * self.kernel_size[1]])
            # transpose and reshape to [B, T_out, F_out, W*N_in, 1, D_in, 1]
            patches = tf.transpose(patches, [0, 1, 2, 5, 3, 4])
            kernel_area = self.kernel_size[0]*self.kernel_size[1]
            patches = tf.reshape(patches, shape=[batch_size, -1, f_out,
                                                 kernel_area*num_capsule_in,
                                                 1, capsule_dim_in, 1])
            # tile to [B, T_out, F_out, W*N_in, N_out, D_in, 1]
            patches = tf.tile(patches, [1]*shared + [1, self.num_capsules, 1, 1])

        with tf.name_scope('tiling'):
            logits = self.logits
            weights = self.grid_weights
            bias = self.bias
            for i in range(shared):
                if patches.shape[shared - i - 1].value is None:
                    shape = tf.shape(patches)[shared - i - 1]
                else:
                    shape = patches.shape[shared - i - 1].value
                tile = [shape] + [1] * len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)
                tile_w = [shape] + [1] * len(weights.shape)
                weights = tf.tile(tf.expand_dims(weights, 0), tile_w)
                if self.use_bias:
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)

        with tf.name_scope('predictions'):
            predictions = tf.reshape(tf.matmul(weights, patches), [batch_size, -1, f_out,
                                                                   kernel_area*num_capsule_in,
                                                                   self.num_capsules, self.capsule_dim])


        return predictions, logits, bias

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
                for i in range(self.kernel_size[0]):
                    for j in range(self.kernel_size[1]):
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

                # #reshape back to [B, T, F, W*W, N_in, N_out, D_out]
                # predictions = tf.reshape(predictions, shape=[batch_size, -1, num_freq_out,
                #                                    self.kernel_size[0]*self.kernel_size[1],
                #                                              n_in, self.num_capsules, self.capsule_dim])
                #
                # #sum over the kernel_size dimension
                # predictions = tf.reduce_sum(predictions, shared)

            # tile the logits for each shared dimesion (batch, time)
            with tf.name_scope('tile_logits'):
                logits = self.logits
                bias = self.bias
                for i in range(shared):
                    if predictions.shape[shared-i-1].value is None:
                        shape = tf.shape(predictions)[shared-i-1]
                    else:
                        shape = predictions.shape[shared-i-1].value
                    tile = [shape] + [1]*len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)

        return predictions, logits, bias

    def cluster(self, predictions, logits, bias):
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
                        tf.expand_dims(w, -1)*predictions, -3) # shape [B, N_out, D_out]
                    # caps = tf.cond(tf.cast(self.use_bias, 'bool'),
                    #                lambda: caps + bias,
                    #                lambda: caps)

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

    def constrained_cluster(self, predictions, logits, bias):
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
                    # caps = caps + bias

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

class EncDecCapsule(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_size=(9, 9), strides=(1, 1),
            padding='SAME',
            max_pool_filter=(1, 1),
            transpose=False,
            routing_iters=3,
            use_bias=False,
            shared=True,
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

        super(EncDecCapsule, self).__init__(
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
        self.use_bias = use_bias
        self.sharing_weights = shared
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        '''creates the variables of this layer
        args:
            input_shape: the shape of the input
        '''

        # pylint: disable=W0201

        # input dimensions
        num_freq_in = input_shape[-3].value
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value
        # num_freq_out = num_freq_in
        # without padding:
        # num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride
        k = self.kernel_size[0] / 2

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        if self.transpose:
            if self.sharing_weights:
                weights_shape = [self.kernel_size[0], self.kernel_size[1],
                               self.num_capsules * self.capsule_dim, capsule_dim_in]
            else:
                weights_shape = [num_capsules_in, self.kernel_size[0], self.kernel_size[1],
                                 self.num_capsules * self.capsule_dim, capsule_dim_in]
            self.conv_weights = self.add_variable(
                        name='conv_trans_weights',
                        dtype=self.dtype,
                        shape=weights_shape,
                        initializer=self.kernel_initializer)
        else:
            if self.sharing_weights:
                weights_shape = [self.kernel_size[0], self.kernel_size[1],
                                 capsule_dim_in, self.num_capsules * self.capsule_dim]
            else:
                weights_shape = [num_capsules_in, self.kernel_size[0], self.kernel_size[1],
                           capsule_dim_in, self.num_capsules * self.capsule_dim]
            self.conv_weights = self.add_variable(
                    name='conv_weights',
                    dtype=self.dtype,
                    shape=weights_shape,
                    initializer=self.kernel_initializer)

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        self.bias = tf.zeros(shape=[self.num_capsules, self.capsule_dim])
        if self.use_bias:
            self.bias = self.add_variable(
            name='bias',
            dtype=self.dtype,
            shape=[self.num_capsules, self.capsule_dim],
            initializer=None)

        super(EncDecCapsule, self).build(input_shape)

    # pylint: disable=W0221
    def call(self, inputs, t_out_tensor=None, freq_out=None):
        '''
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        # compute the predictions
        predictions, logits, bias = self.loop_predict(inputs, t_out_tensor, freq_out)

        # cluster the predictions
        outputs = self.cluster(predictions, logits, bias)

        return outputs

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
            freq_out = int(np.ceil(float(num_freq_in) / float(self.strides[1])))
        num_capsule_in = inputs.shape[-2].value
        capsule_dim_in = inputs.shape[-1].value

        # number of shared dimensions
        rank = len(inputs.shape)  # normally rank=5
        shared = rank - 2  # normally shared=3

        convs = []
        for i in range(0, num_capsule_in):
            with tf.name_scope('conv_2d%d' % i):
                slice = inputs[:, :, :, i, :]
                if self.sharing_weights:
                    conv_weights = self.conv_weights
                else:
                    conv_weights = self.conv_weights[i,:,:,:,:]
                if self.transpose:
                    conv = tf.nn.conv2d_transpose(slice, conv_weights,
                                                  output_shape=[batch_size, t_out_tensor, freq_out,
                                                                self.capsule_dim * self.num_capsules],
                                                  strides=[1, self.strides[0], self.strides[1], 1],
                                                  padding=self.padding)
                else:
                    conv = tf.nn.conv2d(slice, conv_weights,
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

        # tile the logits and bias for each shared dimension (batch, time, frequency)
        with tf.name_scope('tile_logits_bias'):
            logits = self.logits
            bias = self.bias
            for i in range(shared):
                if predictions.shape[shared - i - 1].value is None:
                    shape = tf.shape(predictions)[shared - i - 1]
                else:
                    shape = predictions.shape[shared - i - 1].value
                tile = [shape] + [1] * len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)
                if self.use_bias:
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)

        return predictions, logits, bias

    # Seems slower
    def loop_predict2(self, inputs, t_out_tensor, freq_out):
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
            freq_out = int(np.ceil(float(num_freq_in) / float(self.strides[1])))
        num_capsule_in = inputs.shape[-2].value
        capsule_dim_in = inputs.shape[-1].value

        # number of shared dimensions
        rank = len(inputs.shape)  # normally rank=5
        shared = rank - 2  # normally shared=3

        # put the input capsules as the first dimension
        inputs = tf.transpose(inputs, [3, 0, 1, 2, 4])

        # compute the predictins
        if self.transpose:
            predictions = tf.map_fn(
                fn=lambda x: tf.nn.conv2d_transpose(x[0], x[1],
                          output_shape=[batch_size, t_out_tensor, freq_out,
                                        self.capsule_dim * self.num_capsules],
                          strides=[1, self.strides[0], self.strides[1], 1],
                          padding=self.padding),
                elems=(inputs, self.conv_weights),
                dtype=self.dtype or tf.float32)
        else:
            predictions = tf.map_fn(
                fn=lambda x: tf.nn.conv2d(x[0], x[1],
                                strides=[1, self.strides[0], self.strides[1], 1],
                                padding=self.padding),
                elems=(inputs, self.conv_weights),
                dtype=self.dtype or tf.float32)

        # transpose back
        predictions = tf.transpose(predictions, [1, 2, 3, 0, 4])

        # reshape to [B, T, F, N_in, N_out, D_out]
        predictions = tf.reshape(predictions,
                                 [batch_size, -1, freq_out,
                                  num_capsule_in,
                                  self.num_capsules, self.capsule_dim])

        # tile the logits and bias for each shared dimension (batch, time, frequency)
        with tf.name_scope('tile_logits_bias'):
            logits = self.logits
            bias = self.bias
            for i in range(shared):
                if predictions.shape[shared - i - 1].value is None:
                    shape = tf.shape(predictions)[shared - i - 1]
                else:
                    shape = predictions.shape[shared - i - 1].value
                tile = [shape] + [1] * len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)
                if self.use_bias:
                    bias = tf.tile(tf.expand_dims(bias, 0), tile)


        return predictions, logits, bias


    def cluster(self, predictions, logits, bias):
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
                    caps = tf.cond(tf.cast(self.use_bias, 'bool'),
                                   lambda: caps + bias,
                                   lambda: caps)

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

class EncDecCapsulePooled(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_size=(9, 9), strides=(1, 1),
            padding='SAME',
            pool_filter=(1, 1),
            transpose=False,
            routing_iters=3,
            use_bias=False,
            shared=True,
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

        super(EncDecCapsulePooled, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pool_filter = pool_filter
        self.transpose = transpose
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.routing_iters = routing_iters
        self.use_bias = use_bias
        self.sharing_weights = shared
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        '''creates the variables of this layer
        args:
            input_shape: the shape of the input
        '''

        # pylint: disable=W0201

        # input dimensions
        num_freq_in = input_shape[-3].value
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value
        # num_freq_out = num_freq_in
        # without padding:
        # num_freq_out = (num_freq_in-self.kernel_size+1)/self.stride
        k = self.kernel_size[0] / 2

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        if self.transpose:
            if self.sharing_weights:
                weights_shape = [self.kernel_size[0], self.kernel_size[1],
                               self.num_capsules * self.capsule_dim, capsule_dim_in]
            else:
                weights_shape = [num_capsules_in, self.kernel_size[0], self.kernel_size[1],
                                 self.num_capsules * self.capsule_dim, capsule_dim_in]
            self.conv_weights = self.add_variable(
                        name='conv_trans_weights',
                        dtype=self.dtype,
                        shape=weights_shape,
                        initializer=self.kernel_initializer)
        else:
            if self.sharing_weights:
                weights_shape = [self.kernel_size[0], self.kernel_size[1],
                                 capsule_dim_in, self.num_capsules * self.capsule_dim]
            else:
                weights_shape = [num_capsules_in, self.kernel_size[0], self.kernel_size[1],
                           capsule_dim_in, self.num_capsules * self.capsule_dim]
            self.conv_weights = self.add_variable(
                    name='conv_weights',
                    dtype=self.dtype,
                    shape=weights_shape,
                    initializer=self.kernel_initializer)

        pool_size = self.pool_filter[0]*self.pool_filter[1]
        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[pool_size*num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer,
            trainable=False
        )

        self.bias = tf.zeros(shape=[self.num_capsules, self.capsule_dim])
        if self.use_bias:
            self.bias = self.add_variable(
            name='bias',
            dtype=self.dtype,
            shape=[self.num_capsules, self.capsule_dim],
            initializer=None)

        super(EncDecCapsulePooled, self).build(input_shape)

    # pylint: disable=W0221
    def call(self, inputs, t_out_tensor=None, freq_out=None):
        '''
        apply the layer
        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in
        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        # compute the predictions
        predictions, logits, bias = self.loop_predict(inputs, t_out_tensor, freq_out)

        # cluster the predictions
        outputs = self.cluster(predictions, logits, bias)

        return outputs

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
            freq_out = int(np.ceil(float(num_freq_in) / float(self.strides[1])))
        num_capsule_in = inputs.shape[-2].value
        capsule_dim_in = inputs.shape[-1].value

        # number of shared dimensions
        rank = len(inputs.shape)  # normally rank=5
        shared = rank - 2  # normally shared=3

        convs = []
        for i in range(0, num_capsule_in):
            with tf.name_scope('conv_2d%d' % i):
                slice = inputs[:, :, :, i, :]
                if self.sharing_weights:
                    conv_weights = self.conv_weights
                else:
                    conv_weights = self.conv_weights[i,:,:,:,:]
                if self.transpose:
                    conv = tf.nn.conv2d_transpose(slice, conv_weights,
                                                  output_shape=[batch_size, t_out_tensor, freq_out,
                                                                self.capsule_dim * self.num_capsules],
                                                  strides=[1, self.strides[0], self.strides[1], 1],
                                                  padding=self.padding)
                else:
                    conv = tf.nn.conv2d(slice, conv_weights,
                                        strides=[1, self.strides[0], self.strides[1], 1],
                                        padding=self.padding)
            expanded = tf.expand_dims(conv, 3)
            convs.append(expanded)

        prev_slice = tf.concat(convs, 3)

        # reshape to [B, T, F, N_in*N_out*D_out]
        predictions = tf.reshape(prev_slice,
                                 [batch_size, -1, freq_out,
                                  num_capsule_in*self.num_capsules*self.capsule_dim])

        with tf.name_scope('get_patches'):
            # code based on https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/ (18-11-'18)
            # create a filter that selects the values of the T-F bins
            # within the convolution kernel
            tile_filter = np.zeros(shape=[self.pool_filter[0], self.pool_filter[1],
                                          num_capsule_in*self.num_capsules*self.capsule_dim,
                                          self.pool_filter[0] * self.pool_filter[1]],
                                   dtype=self.dtype)
            for i in range(self.pool_filter[0]):
                for j in range(self.pool_filter[1]):
                    tile_filter[i, j, :, i * self.pool_filter[1] + j] = 1
            tile_filter_op = tf.constant(tile_filter)

            # Convolution with padding, non-overlapping kernel
            patches = tf.nn.depthwise_conv2d(predictions, tile_filter_op,
                                            strides=[1, self.pool_filter[0], self.pool_filter[0], 1],
                                            padding=self.padding)
            # output_shape = tf.shape(inputs)
            # output_shape[1] should equal num_freq_out if no padding
            # reshape back to [B, T_out, F_out, N_in, N_out, D_out, W]
            f_out = patches.shape[2].value
            patches = tf.reshape(patches, shape=[batch_size, -1, f_out,
                                                num_capsule_in, self.num_capsules, self.capsule_dim,
                                                 self.pool_filter[0] * self.pool_filter[1]])
            # transpose and reshape to [B, T_out, F_out, W*N_in, N_out, D_out]
            patches = tf.transpose(patches, [0,1,2,6,3,4,5])
            pool_size = self.pool_filter[0]*self.pool_filter[1]
            patches = tf.reshape(patches, shape=[batch_size, -1, f_out,
                                                 pool_size*num_capsule_in,
                                                 self.num_capsules, self.capsule_dim])
            # tile the logits and bias for each shared dimension (batch, time, frequency)
            with tf.name_scope('tile_logits_bias'):
                logits = self.logits
                bias = self.bias
                for i in range(shared):
                    if patches.shape[shared - i - 1].value is None:
                        shape = tf.shape(patches)[shared - i - 1]
                    else:
                        shape = patches.shape[shared - i - 1].value
                    tile = [shape] + [1] * len(logits.shape)
                    logits = tf.tile(tf.expand_dims(logits, 0), tile)
                    if self.use_bias:
                        bias = tf.tile(tf.expand_dims(bias, 0), tile)



        # # code from https://github.com/canpeng78/ConvCapsule-tensorflow/blob/master/capsLayers.py
        # # Do routing convolutionally
        # pool_strides = self.pool_filter
        # inputshape = inputs.shape.as_list()
        # patches, gridsz = ops.patches2d(inputshape[1:], self.pool_filter, pool_strides, self.padding)
        # out_caplayer = []
        # for patch in patches:
        #     capsules = []
        #     for i in range(0, num_capsule_in):
        #         cap = predictions[:,:,:,i,:,:]
        #         capptch = cap[:, patch[0]:patch[2], patch[1]:patch[3], :, :]
        #         capshape = capptch.get_shape().as_list()
        #         capptch = tf.reshape(capptch, shape=(capshape[0], -1, capshape[-2], capshape[-1]))
        #         capsules.append(capptch)
        #     capsules = tf.concat(capsules, 1) # shape [B, capsules_in_patch, N_out, D_out]
        #     out_capsule = tf.expand_dims(self.cluster(capsules, logits, bias),
        #                                  1)  # shape [B, 1, N_out, D_out]
        #     out_caplayer.append(out_capsule)
        # out_caplayer = tf.concat(out_caplayer, 1)
        # out_caplayer = tf.reshape(out_caplayer, shape=(
        #         tf.shape(out_caplayer)[0], gridsz[0], gridsz[1], self.num_capsules, self.capsule_dim))

        return patches, logits, bias


    def cluster(self, predictions, logits, bias):
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
                        tf.expand_dims(w, -1)*predictions, -3) # shape [B, N_out, D_out]
                    caps = tf.cond(tf.cast(self.use_bias, 'bool'),
                                   lambda: caps + bias,
                                   lambda: caps)

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
    """a Bidirectional recurrent capsule layer"""

    def __init__(self, num_capsules, capsule_dim, routing_iters=3, activation=None, input_probability_fn=None,
                 recurrent_probability_fn=None, rec_only_vote=False, logits_prior=False, accumulate_input_logits=True,
                 accumulate_state_logits=True):
        """
        BRCapsuleLayer constructor

        Args:
            TODO
        """

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        self._activation = activation
        self.input_probability_fn = input_probability_fn
        self.recurrent_probability_fn = recurrent_probability_fn
        self.rec_only_vote = rec_only_vote
        self.logits_prior = logits_prior
        self.accumulate_input_logits = accumulate_input_logits
        self.accumulate_state_logits = accumulate_state_logits

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the rnn cell that will be used for the forward and backward
            # pass
            
            if self.rec_only_vote:
                rnn_cell_fw = rnn_cell.RecCapsuleCellRecOnlyVote(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
                    accumulate_input_logits=self.accumulate_input_logits,
                    accumulate_state_logits=self.accumulate_state_logits,
                    reuse=tf.get_variable_scope().reuse)

                rnn_cell_bw = rnn_cell.RecCapsuleCellRecOnlyVote(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
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
                    logits_prior=self.logits_prior,
                    reuse=tf.get_variable_scope().reuse)

                rnn_cell_bw = rnn_cell.RecCapsuleCell(
                    num_capsules=self.num_capsules,
                    capsule_dim=self.capsule_dim,
                    routing_iters=self.routing_iters,
                    activation=self._activation,
                    input_probability_fn=self.input_probability_fn,
                    recurrent_probability_fn=self.recurrent_probability_fn,
                    logits_prior=self.logits_prior,
                    reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BLSTMCapsuleLayer(object):
    """a Bidirectional lstm capsule layer"""

    def __init__(self, num_capsules, capsule_dim, routing_iters=3, activation=None, input_probability_fn=None,
                 recurrent_probability_fn=None,  logits_prior=False, accumulate_input_logits=True,
                 accumulate_state_logits=True, gates_fc = False, use_output_matrix=False):
        """
        BRCapsuleLayer constructor

        Args:
            TODO
        """

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        self._activation = activation
        self.input_probability_fn = input_probability_fn
        self.recurrent_probability_fn = recurrent_probability_fn
        self.logits_prior = logits_prior
        self.accumulate_input_logits = accumulate_input_logits
        self.accumulate_state_logits = accumulate_state_logits
        self.gates_fc = gates_fc
        self.use_output_matrix = use_output_matrix

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the rnn cell that will be used for the forward and backward
            # pass

            if self.use_output_matrix:
                lstm_cell_type = rnn_cell.LSTMCapsuleCellRecOnlyVoteOutputMatrix
            else:
                lstm_cell_type = rnn_cell.LSTMCapsuleCellRecOnlyVote

            lstm_cell_fw = lstm_cell_type(
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim,
                routing_iters=self.routing_iters,
                activation=self._activation,
                input_probability_fn=self.input_probability_fn,
                recurrent_probability_fn=self.recurrent_probability_fn,
                logits_prior=self.logits_prior,
                accumulate_input_logits=self.accumulate_input_logits,
                accumulate_state_logits=self.accumulate_state_logits,
                gates_fc=self.gates_fc,
                reuse=tf.get_variable_scope().reuse)

            lstm_cell_bw = lstm_cell_type(
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim,
                routing_iters=self.routing_iters,
                activation=self._activation,
                input_probability_fn=self.input_probability_fn,
                recurrent_probability_fn=self.recurrent_probability_fn,
                logits_prior=self.logits_prior,
                accumulate_input_logits=self.accumulate_input_logits,
                accumulate_state_logits=self.accumulate_state_logits,
                gates_fc=self.gates_fc,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BRNNLayer(object):
    """a BRNN layer"""

    def __init__(self, num_units, activation_fn=tf.nn.tanh, linear_out_flag=False):
        """
        BRNNLayer constructor

        Args:
            num_units: The number of units in the one directon
            activation_fn: activation function
            linear_out_flag: if set to True, activation function will only be applied
            to the recurrent output.
        """

        self.num_units = num_units
        self.activation_fn = activation_fn
        self.linear_out_flag = linear_out_flag

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the rnn cell that will be used for the forward and backward
            # pass
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

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class LSTMLayer(object):
    """a LSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
        """
        LSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            activation_fn: activation function
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs, _ = dynamic_rnn(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs


class BLSTMLayer(object):
    """a BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, activation_fn=tf.nn.tanh,
                 separate_directions=False, linear_out_flag=False, fast_version=False):
        """
        BLSTMLayer constructor
        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            separate_directions: wether the forward and backward directions should
            be separated for deep networks.
            fast_version: deprecated
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn
        self.separate_directions = separate_directions
        self.linear_out_flag = linear_out_flag
        self.fast_version = fast_version

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass

            if self.linear_out_flag:
                lstm_cell_type = rnn_cell.LayerNormBasicLSTMCellLineairOut
            else:
                lstm_cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell

            lstm_cell_fw = lstm_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)
            lstm_cell_bw =lstm_cell_type(
                num_units=self.num_units,
                activation=self.activation_fn,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            if not self.separate_directions:
                outputs_tupple, _ = bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                                                              sequence_length=sequence_length)

                outputs = tf.concat(outputs_tupple, 2)
            else:
                outputs, _ = rnn.bidirectional_dynamic_rnn_2inputs(
                    lstm_cell_fw, lstm_cell_bw, inputs[0], inputs[1], dtype=tf.float32,
                    sequence_length=sequence_length)

        return outputs


class LeakyLSTMLayer(object):
    """a leaky LSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
            lstm_cell = rnn_cell.LayerNormBasicLeakLSTMCell(
                num_units=self.num_units,
                leak_factor=self.leak_factor,
                layer_norm=self.layer_norm,
                dropout_keep_prob=self.recurrent_dropout,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs, _ = dynamic_rnn(
                lstm_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            return outputs


class LeakyBLSTMLayer(object):
    """a leaky BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
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

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakyBLSTMIZNotRecLayer(object):
    """a leaky BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMIZNotRecLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
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

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakyBLSTMNotRecLayer(object):
    """a leaky BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMNotRecLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
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

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs	  


class LeakychBLSTMLayer(object):
    """a leaky ch BLSTM layer"""

    def __init__(self, num_units, layer_norm=False, recurrent_dropout=1.0, leak_factor=1.0):
        """
        LeakyBLSTMLayer constructor

        Args:
            num_units: The number of units in the one directon
            layer_norm: whether layer normalization should be applied
            recurrent_dropout: the recurrent dropout keep probability
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.layer_norm = layer_norm
        self.recurrent_dropout = recurrent_dropout
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the lstm cell that will be used for the forward and backward
            # pass
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

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class ResetLSTMLayer(object):
	"""a ResetLSTM layer"""

	def __init__(
			self, num_units, t_reset=1, next_t_reset=None, layer_norm=False, recurrent_dropout=1.0,
			activation_fn=tf.nn.tanh):
		"""
		ResetLSTM constructor

		Args:
			num_units: The number of units in the one directon
			layer_norm: whether layer normalization should be applied
			recurrent_dropout: the recurrent dropout keep probability
		"""

		self.num_units = num_units
		self.t_reset = t_reset
		if next_t_reset:
			if next_t_reset < t_reset:
				raise ValueError('T_reset in next layer must be equal to or bigger than T_reset in current layer')
			self.next_t_reset = next_t_reset
		else:
			self.next_t_reset = t_reset
		self.layer_norm = layer_norm
		self.recurrent_dropout = recurrent_dropout
		self.activation_fn = activation_fn

	def __call__(self, inputs, sequence_length, scope=None):
		"""
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
		"""
		batch_size = inputs.get_shape()[0]
		max_length = tf.shape(inputs)[1]

		with tf.variable_scope(scope or type(self).__name__):

			# create the lstm cell that will be used for the forward
			lstm_cell = rnn_cell.LayerNormResetLSTMCell(
				num_units=self.num_units,
				t_reset=self.t_reset,
				activation=self.activation_fn,
				layer_norm=self.layer_norm,
				dropout_keep_prob=self.recurrent_dropout,
				reuse=tf.get_variable_scope().reuse)

			# do the forward computation
			outputs_tupple, _ = rnn.dynamic_rnn_time_input(
				lstm_cell, inputs, dtype=tf.float32,
				sequence_length=sequence_length)

			if self.next_t_reset == self.t_reset:
				return outputs_tupple

			actual_outputs = outputs_tupple[0]
			replicas = outputs_tupple[1]

			# the output replicas need to be permuted correctly such that the next layer receives
			# the replicas in the correct order

			# numbers_to_maxT: [1, Tmax,1]
			numbers_to_maxT = tf.range(0, max_length)
			numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, -1), 0)

			# numbers_to_k: [1, 1,k]
			numbers_to_k = tf.expand_dims(tf.expand_dims(range(0, self.next_t_reset), 0), 0)

			mod1 = tf.mod(numbers_to_maxT - 1 - numbers_to_k, self.next_t_reset)
			mod2 = tf.mod(numbers_to_maxT - mod1 - 1, self.t_reset)
			mod3 = tf.tile(tf.mod(numbers_to_maxT, self.t_reset), [1, 1, self.next_t_reset])

			indices_for_next_layer = tf.where(
				mod1 < self.t_reset,
				x=mod2,
				y=mod3,
			)
			indices_for_next_layer = tf.tile(indices_for_next_layer, [batch_size, 1, 1])

			# ra1: [B,Tmax,k]
			ra1 = tf.range(batch_size)
			ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
			ra1 = tf.tile(ra1, [1, max_length, self.next_t_reset])
			ra2 = tf.range(max_length)
			ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
			ra2 = tf.tile(ra2, [batch_size, 1, self.next_t_reset])
			full_indices_for_next_layer = tf.stack([ra1, ra2, indices_for_next_layer], axis=-1)
			output_for_next_layer = tf.gather_nd(replicas, full_indices_for_next_layer)

			outputs = (actual_outputs, output_for_next_layer)

		return outputs


class BResetLSTMLayer(object):
	"""a BResetLSTM layer"""

	def __init__(
			self, num_units, t_reset=1, next_t_reset=None, group_size=1, symmetric_context=False, layer_norm=False,
			recurrent_dropout=1.0, activation_fn=tf.nn.tanh):
		"""
		BResetLSTM constructor

		Args:
			num_units: The number of units in the one directon
			group_size: units in the same group share a state replicate
			symmetric_context: if True, input to next layer should have same amount of context
			in both directions. If False, reversed input to next layers has full (t_reset) context.
			layer_norm: whether layer normalization should be applied
			recurrent_dropout: the recurrent dropout keep probability
		"""

		self.num_units = num_units
		self.t_reset = t_reset
		if next_t_reset:
			if next_t_reset < t_reset:
				raise ValueError('T_reset in next layer must be equal to or bigger than T_reset in current layer')
			self.next_t_reset = next_t_reset
		else:
			self.next_t_reset = t_reset
		self.group_size = group_size
		if self.group_size > 1 and self.next_t_reset != self.t_reset:
			raise NotImplementedError('Grouping is not yet implemented for different t_resets')
		self.num_replicates = float(self.t_reset)/float(self.group_size)
		if int(self.num_replicates) != self.num_replicates:
			raise ValueError('t_reset should be a multiple of group_size')
		self.symmetric_context = symmetric_context
		self.num_replicates = int(self.num_replicates)
		self.layer_norm = layer_norm
		self.recurrent_dropout = recurrent_dropout
		self.activation_fn = activation_fn

	def __call__(self, inputs_for_forward, inputs_for_backward, sequence_length, scope=None):
		"""
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
		"""

		if inputs_for_backward is None:
			inputs_for_backward = inputs_for_forward

		batch_size = inputs_for_forward.get_shape()[0]
		max_length = tf.shape(inputs_for_forward)[1]

		with tf.variable_scope(scope or type(self).__name__):
			# create the lstm cell that will be used for the forward and backward
			# pass
			if self.group_size == 1:
				lstm_cell_fw = rnn_cell.LayerNormResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				lstm_cell_bw = rnn_cell.LayerNormResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				tile_shape = [1, 1, self.t_reset, 1]
			else:
				lstm_cell_fw = rnn_cell.LayerNormGroupResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					group_size=self.group_size,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				lstm_cell_bw = rnn_cell.LayerNormGroupResetLSTMCell(
					num_units=self.num_units,
					t_reset=self.t_reset,
					group_size=self.group_size,
					activation=self.activation_fn,
					layer_norm=self.layer_norm,
					dropout_keep_prob=self.recurrent_dropout,
					reuse=tf.get_variable_scope().reuse)
				tile_shape = [1, 1, lstm_cell_fw._num_replicates, 1]

			# do the forward computation
			outputs_tupple, _ = rnn.bidirectional_dynamic_rnn_2inputs_time_input(
				lstm_cell_fw, lstm_cell_bw, inputs_for_forward, inputs_for_backward,
				dtype=tf.float32, sequence_length=sequence_length)

			# outputs are reordered for next layer.
			# TODO:This should be done in model.py and not in layer.py
			actual_outputs_forward = outputs_tupple[0][0]
			actual_outputs_backward = outputs_tupple[1][0]
			actual_outputs = tf.concat((actual_outputs_forward, actual_outputs_backward), -1)

			forward_replicas = outputs_tupple[0][1]
			backward_replicas = outputs_tupple[1][1]

			if not self.symmetric_context:
				forward_for_backward = tf.expand_dims(actual_outputs_forward, -2)
				forward_for_backward = tf.tile(forward_for_backward, tile_shape)

				backward_for_forward = tf.expand_dims(actual_outputs_backward, -2)
				backward_for_forward = tf.tile(backward_for_forward, tile_shape)

				outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
				outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

			elif False and self.t_reset == self.next_t_reset:
				# the output replicas need to be permuted correctly such that the next layer receives
				# the replicas in the correct order

				# T_min_1: [B,1]
				T = tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size)))
				T_min_1 = tf.expand_dims(T - 1, -1)

				# numbers_to_maxT: [1,Tmax,1]
				numbers_to_maxT = tf.range(0, max_length)
				numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, 0), -1)

				# numbers_to_k: [1,k]
				numbers_to_k = tf.expand_dims(range(0, self.num_replicates), 0)

				# backward_indices_for_forward_t_0: [B,1,k]
				backward_indices_for_forward_t_0 = numbers_to_k+T_min_1
				# backward_indices_for_forward_t_0 = tf.mod(backward_indices_for_forward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
				backward_indices_for_forward_t_0 = tf.expand_dims(backward_indices_for_forward_t_0, 1)
				# backward_indices_for_forward_t: [B,Tmax,k]
				backward_indices_for_forward_t = tf.mod(backward_indices_for_forward_t_0 - 2*numbers_to_maxT,
														self.num_replicates)

				forward_indices_for_backward_t_0 = numbers_to_k-T_min_1
				# forward_indices_for_backward_t_0 = tf.mod(forward_indices_for_backward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
				forward_indices_for_backward_t_0 = tf.expand_dims(forward_indices_for_backward_t_0, 1)
				forward_indices_for_backward_t = tf.mod(forward_indices_for_backward_t_0 + 2*numbers_to_maxT,
														self.num_replicates)

				# ra1: [B,Tmax,k]
				ra1 = tf.range(batch_size)
				ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
				ra1 = tf.tile(ra1, [1, max_length, self.num_replicates])
				ra2 = tf.range(max_length)
				ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
				ra2 = tf.tile(ra2, [batch_size, 1, self.num_replicates])
				stacked_backward_indices_for_forward_t = tf.stack([ra1, ra2, backward_indices_for_forward_t], axis=-1)
				backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward_t)
				stacked_forward_indices_for_backward_t = tf.stack([ra1, ra2, forward_indices_for_backward_t], axis=-1)
				forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward_t)

				outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
				outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

			else:
				# the output replicas need to be permuted correctly such that the next layer receives
				# the replicas in the correct order

				# T: [B,1, 1]
				T = tf.expand_dims(tf.expand_dims(
					tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size))), -1), -1)

				# numbers_to_maxT: [B,Tmax,k]
				numbers_to_maxT = tf.range(0, max_length)
				numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT, 0), -1)
				numbers_to_maxT = tf.tile(numbers_to_maxT, [batch_size, 1, self.next_t_reset])
				reversed_numbers_to_maxT = T - 1 - numbers_to_maxT

				# numbers_to_k: [B,Tmax,k]
				numbers_to_k = tf.expand_dims(tf.expand_dims(range(0, self.next_t_reset), 0), 0)
				numbers_to_k = tf.tile(numbers_to_k, [batch_size, max_length, 1])

				# next taus
				next_tau_forward = tf.mod(numbers_to_maxT-1-numbers_to_k, self.next_t_reset)
				next_tau_backward = tf.mod(reversed_numbers_to_maxT-1-numbers_to_k, self.next_t_reset)

				# max memory instances
				max_memory_forward = tf.mod(numbers_to_maxT, self.t_reset)
				max_memory_backward = tf.mod(reversed_numbers_to_maxT, self.t_reset)

				# forward for forward
				condition_forward = next_tau_forward < self.t_reset
				condition_true = tf.mod(numbers_to_maxT-1-next_tau_forward, self.t_reset)
				forward_indices_for_forward = tf.where(condition_forward, x=condition_true, y=max_memory_forward)

				# backward for forward
				condition_true = tf.mod(reversed_numbers_to_maxT-1-next_tau_forward, self.t_reset)
				backward_indices_for_forward = tf.where(condition_forward, x=condition_true, y=max_memory_backward)

				# backward for backward
				condition_backward = next_tau_backward < self.t_reset
				condition_true = tf.mod(reversed_numbers_to_maxT-1-next_tau_backward, self.t_reset)
				backward_indices_for_backward = tf.where(condition_backward, x=condition_true, y=max_memory_backward)

				# forward for backward
				condition_true = tf.mod(numbers_to_maxT-1-next_tau_backward, self.t_reset)
				forward_indices_for_backward = tf.where(condition_backward, x=condition_true, y=max_memory_forward)

				# ra1: [B,Tmax,k]
				ra1 = tf.range(batch_size)
				ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
				ra1 = tf.tile(ra1, [1, max_length, self.next_t_reset])
				ra2 = tf.range(max_length)
				ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
				ra2 = tf.tile(ra2, [batch_size, 1, self.next_t_reset])
				stacked_forward_indices_for_forward = tf.stack([ra1, ra2, forward_indices_for_forward], axis=-1)
				forward_for_forward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_forward)
				stacked_backward_indices_for_forward = tf.stack([ra1, ra2, backward_indices_for_forward], axis=-1)
				backward_for_forward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_forward)
				stacked_backward_indices_for_backward = tf.stack([ra1, ra2, backward_indices_for_backward], axis=-1)
				backward_for_backward = tf.gather_nd(backward_replicas, stacked_backward_indices_for_backward)
				stacked_forward_indices_for_backward = tf.stack([ra1, ra2, forward_indices_for_backward], axis=-1)
				forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward)

				outputs_for_forward = tf.concat((forward_for_forward, backward_for_forward), -1)
				outputs_for_backward = tf.concat((forward_for_backward, backward_for_backward), -1)

		outputs = (actual_outputs, outputs_for_forward, outputs_for_backward)

		return outputs


class BGRULayer(object):
    """a BGRU layer"""

    def __init__(self, num_units, activation_fn=tf.nn.tanh):
        """
        BGRULayer constructor

        Args:
            num_units: The number of units in the one directon
        """

        self.num_units = num_units
        self.activation_fn = activation_fn

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the gru cell that will be used for the forward and backward
            # pass
            gru_cell_fw = tf.contrib.rnn.GRUCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)
            gru_cell_bw = tf.contrib.rnn.GRUCell(
                num_units=self.num_units,
                activation=self.activation_fn,
                reuse=tf.get_variable_scope().reuse)

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                gru_cell_fw, gru_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class LeakyBGRULayer(object):
    """a leaky BGRU layer"""

    def __init__(self, num_units, activation_fn=tf.nn.tanh, leak_factor=1.0):
        """
        LeakyBGRULayer constructor

        Args:
            num_units: The number of units in the one directon
            leak_factor: the leak factor (if 1, there is no leakage)
        """

        self.num_units = num_units
        self.activation_fn = activation_fn
        self.leak_factor = leak_factor

    def __call__(self, inputs, sequence_length, scope=None):
        """
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
        """

        with tf.variable_scope(scope or type(self).__name__):

            # create the gru cell that will be used for the forward and backward
            # pass
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

            # do the forward computation
            outputs_tupple, _ = bidirectional_dynamic_rnn(
                gru_cell_fw, gru_cell_bw, inputs, dtype=tf.float32,
                sequence_length=sequence_length)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class BResetGRULayer(object):
    """a BResetGRU layer"""

    def __init__(self, num_units, t_reset=1, group_size=1, symmetric_context=False, activation_fn=tf.nn.tanh):
        """
        BResetLSTM constructor

        Args:
            num_units: The number of units in the one directon
            group_size: units in the same group share a state replicate
            symmetric_context: if True, input to next layer should have same amount of context
            in both directions. If False, reversed input to next layers has full (t_reset) context.
        """

        self.num_units = num_units
        self.t_reset = t_reset
        self.group_size = group_size
        self.num_replicates = float(self.t_reset)/float(self.group_size)
        if int(self.num_replicates) != self.num_replicates:
            raise ValueError('t_reset should be a multiple of group_size')
        self.symmetric_context = symmetric_context
        self.num_replicates = int(self.num_replicates)
        self.activation_fn = activation_fn

    def __call__(self, inputs_for_forward, inputs_for_backward, sequence_length, scope=None):
        """
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
        """
        
        if inputs_for_backward is None:
            inputs_for_backward = inputs_for_forward

        batch_size = inputs_for_forward.get_shape()[0]
        max_length = tf.shape(inputs_for_forward)[1]

        with tf.variable_scope(scope or type(self).__name__):
            # create the gru cell that will be used for the forward and backward
            # pass
            if self.group_size == 1:
                gru_cell_fw = rnn_cell_impl.ResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)
                gru_cell_bw = rnn_cell_impl.ResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)

                tile_shape = [1, 1 ,self.t_reset, 1]
            else:
                gru_cell_fw = rnn_cell_impl.GroupResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    group_size = self.group_size,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)
                gru_cell_bw = rnn_cell_impl.GroupResetGRUCell(
                    num_units=self.num_units,
                    t_reset = self.t_reset,
                    group_size = self.group_size,
                    activation=self.activation_fn,
                    reuse=tf.get_variable_scope().reuse)

                tile_shape = [1,1,gru_cell_fw._num_replicates,1]

            # do the forward computation
            outputs_tupple, _ = rnn.bidirectional_dynamic_rnn_2inputs_time_input(
                gru_cell_fw, gru_cell_bw, inputs_for_forward, inputs_for_backward, 
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
                # the output replicas need to be permutated correclty such that the next layer receives
                # the replicas in the correct order
                T = tf.to_int32(tf.ceil(tf.to_float(sequence_length)/tf.to_float(self.group_size)))
                T_min_1 = tf.expand_dims(T - 1, -1)

                numbers_to_maxT = tf.range(0, max_length)
                numbers_to_maxT = tf.expand_dims(tf.expand_dims(numbers_to_maxT,0),-1)

                numbers_to_k = tf.expand_dims(range(0, self.num_replicates), 0)

                backward_indices_for_forward_t_0 = numbers_to_k+T_min_1
                #backward_indices_for_forward_t_0 = tf.mod(backward_indices_for_forward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
                backward_indices_for_forward_t_0 = tf.expand_dims(backward_indices_for_forward_t_0, 1)
                backward_indices_for_forward_t = tf.mod(backward_indices_for_forward_t_0 - 2*numbers_to_maxT,
                                                        self.num_replicates)

                forward_indices_for_backward_t_0 = numbers_to_k-T_min_1
                #forward_indices_for_backward_t_0 = tf.mod(forward_indices_for_backward_t_0, self.num_replicates) #unnecessary since mod will be applied again further on
                forward_indices_for_backward_t_0 = tf.expand_dims(forward_indices_for_backward_t_0, 1)
                forward_indices_for_backward_t = tf.mod(forward_indices_for_backward_t_0 + 2*numbers_to_maxT,
                                                        self.num_replicates)

                ra1 = tf.range(batch_size)
                ra1 = tf.expand_dims(tf.expand_dims(ra1, -1), -1)
                ra1 = tf.tile(ra1, [1, max_length, self.num_replicates])
                ra2 = tf.range(max_length)
                ra2 = tf.expand_dims(tf.expand_dims(ra2, 0), -1)
                ra2 = tf.tile(ra2, [batch_size, 1, self.num_replicates])
                stacked_backward_indices_for_forward_t = tf.stack([ra1, ra2, backward_indices_for_forward_t], axis=-1)
                backward_for_forward = tf.gather_nd(backward_replicas,  stacked_backward_indices_for_forward_t)
                stacked_forward_indices_for_backward_t = tf.stack([ra1, ra2, forward_indices_for_backward_t], axis=-1)
                forward_for_backward = tf.gather_nd(forward_replicas, stacked_forward_indices_for_backward_t)

                outputs_for_forward = tf.concat((forward_replicas, backward_for_forward), -1)
                outputs_for_backward = tf.concat((forward_for_backward, backward_replicas), -1)

            outputs = (actual_outputs, outputs_for_forward, outputs_for_backward)

            return outputs


class Conv2D(object):
    """a Conv2D layer, with max_pool and layer norm options"""

    def __init__(self, num_filters, kernel_size, strides=(1, 1), padding='same', activation_fn=tf.nn.relu,
                 layer_norm=False, max_pool_filter=(1,1), transpose=False):
        """
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
        """

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.max_pool_filter = max_pool_filter
        self.transpose = transpose

    def __call__(self, inputs, scope=None):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim, in_channel] tensor
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

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


class EncDecCNN(tf.layers.Layer):
    '''a Conv2D layer, with max_pool and layer norm options'''

    def __init__(self, num_filters, kernel_size, strides=(1, 1), padding='same',
                 activation_fn=tf.nn.relu, layer_norm=False, max_pool_filter=(1, 1),
                 transpose=False, activity_regularizer=None, kernel_initializer=None,
                 trainable=True, name=None, **kwargs):


        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.max_pool_filter = max_pool_filter
        self.transpose = transpose
        self.kernel_initializer = kernel_initializer

        super(EncDecCNN, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

    def build(self, input_shape):
        '''creates the variables of this layer
        args:
            input_shape: the shape of the input
        '''

        # pylint: disable=W0201

        # input dimensions
        num_channels_in  = input_shape[-1].value

        if self.transpose:
            self.conv_weights = self.add_variable(
                        name='conv_trans_weights',
                        dtype=self.dtype,
                        shape=[self.kernel_size[0], self.kernel_size[1],
                               self.num_filters, num_channels_in],
                        initializer=self.kernel_initializer)
        else:
            self.conv_weights = self.add_variable(
                    name='conv_weights',
                    dtype=self.dtype,
                    shape=[self.kernel_size[0], self.kernel_size[1],
                           num_channels_in, self.num_filters],
                    initializer=self.kernel_initializer)

        self.bias = self.add_variable(
        name='bias',
        dtype=self.dtype,
        shape=[self.num_filters],
        initializer=None)

        super(EncDecCNN, self).build(input_shape)


    def call(self, inputs, t_out_tensor=None, freq_out=None, scope=None):
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
            batch_size = inputs.shape[0].value
            num_freq_in = inputs.shape[2].value
            if not self.transpose:
                freq_out = int(np.ceil(float(num_freq_in) / float(self.strides[1])))

            if not self.transpose:
                conv = tf.nn.conv2d(
                    inputs,
                    filter=self.conv_weights,
                    strides=self.strides,
                    padding=self.padding)
            else:
                conv = tf.nn.conv2d_transpose(
                    inputs,
                    filter=self.conv_weights,
                    output_shape=[batch_size, t_out_tensor, freq_out,
                                  self.num_filters],
                    strides=self.strides,
                    padding=self.padding)
            outputs = self.activation_fn(conv + self.bias)

            if self.max_pool_filter != (1, 1):
                outputs = tf.layers.max_pooling2d(outputs, self.max_pool_filter,
                                                  strides=self.max_pool_filter, padding=self.padding)

            return outputs



def unpool(pool_input, pool_output, unpool_input, pool_kernel_size, pool_stride, padding='VALID'):
    """ An unpooling layer as described in Adaptive Deconvolutional Networks for Mid and High Level Feature Learning
    from Matthew D. Zeiler, Graham W. Taylor and Rob Fergus,
    using the implementation of https://assiaben.github.io/posts/2018-06-tf-unpooling/
    """

    # Assuming pool_kernel_size and pool_stride are given in 'HW' format, converting them to 'NHWC' format
    if len(pool_kernel_size) != 2:
        raise ValueError('Expected kernel size to be in "HW" format.')
    pool_kernel_size = [1] + pool_kernel_size + [1]
    if len(pool_stride) != 2:
        raise ValueError('Expected stride size to be in "HW" format.')
    pool_stride = [1] + pool_stride + [1]

    unpool_op = gen_nn_ops.max_pool_grad(pool_input, pool_output, unpool_input, pool_kernel_size, pool_stride, padding)

    return unpool_op


# @ops.RegisterGradient("MaxPoolGradWithArgmax")
# def _MaxPoolGradGradWithArgmax(op, grad):
#     """Register max pooling gradient for the unpool operation. Copied from
#     https://assiaben.github.io/posts/2018-06-tf-unpooling/
#     """
#     print(len(op.outputs))
#     print(len(op.inputs))
#     print(op.name)
#     return (array_ops.zeros(
#       shape=array_ops.shape(op.inputs[0]),
#       dtype=op.inputs[0].dtype), array_ops.zeros(
#           shape=array_ops.shape(op.inputs[1]), dtype=op.inputs[1].dtype),
#           gen_nn_ops._max_pool_grad_grad_with_argmax(
#               op.inputs[0],
#               grad,
#               op.inputs[2],
#               op.get_attr("ksize"),
#               op.get_attr("strides"),
#               padding=op.get_attr("padding")))
