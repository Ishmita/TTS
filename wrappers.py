import numpy as np
import tensorflow as tf
from modules import prenet
from tensorflow.contrib.rnn import RNNCell

class DecoderPrenetWrapper(RNNCell):
    def __init__(self, cell, is_training, layer_sizes):
        super(SecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training
        self._layer_sizes = layer_sizes

    @property
    def state_size(self):
        return self._cell.state_size
    @property
    def output_size(self):
        return self._cell.output_size
    def call(self, inputs, state):
        prenet_out = prenet(inputs, self._is_training, self._layer_sizes, scope= 'decoder_prenet')
        # cell(...) calls the __call() method of RNNCell class
        # as _cell is a type of RNNCell
        return self._cell(prenet_out, state)
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

class ConcatOutputAndAttentionWrapper(RNNCell):
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init()
        self._cell = cell
        @property
        def state_size(self):
            return self._cell.state_size
        @property
        def output_size(self):
            return self._cell.output_size + self._cell.state_size.attention
        def call(self, inputs, state):
             # ._cell is an object of AttentionWrapper
             # So, ._cell(....) calls the __call() method of AttentionWrapper class
             # which returns the cell_output & next_state
             # next_state is an object of AttentionWrappreState class
             # which contains attention as an attribute/property.
            output, res_state = self._cell(inputs, state)
            return tf.concat([output, res_state.attention], axis = -1), res_state
        def zero_state(self, batch_size, dtype):
            return self._cell.zero_state(batch_size, dtype)
        
        
        
