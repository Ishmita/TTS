import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

def new_fc_layer(input,
                 num_cells,
                 is_training = True):
    layer = tf.layers.dense(input, units = num_cells, activation = tf.nn.relu)
    
    rate= 0.5 if is_training else 0.0
    layer = tf.layers.dropout(layer, rate = rate)
    return layer

def prenet(input,
           layer_sizes = [256,128],
           is_training):
    layer1 = new_fc_layer(input,
                          layer_sizes[0],
                          is_training)
    
    layer2 = new_fc_layer(layer1,
                          layer_sizes[1],
                          is_training)
    return layer2

def new_conv_layer(input,
                   num_filters,
                   filter_size,
                   activation,
                   is_training):
    layer = tf.layers.conv1d(inputs = input,
                             filters = num_filters,
                             kernel_size = filter_size,
                             activation = activation,
                             padding = 'SAME')
    
    return tf.layers.batch_normalization(layer, training = is_training)

def cbhg(input, K, is_training, input_lengths, projection_filter_sizes):

    # stack conv layers starting from the last one
    conv1d_bank_layers = tf.concat(
        [new_conv_layer(input, k, 128, tf.nn.relu, is_training) for k in range(1, K+1)],
        axis = -1)
    
    max_pooling_layer = tf.layers.max_pooling1d(conv1d_bank_layers,
                                    pool_size = 2,
                                    strides = 1,
                                    padding = 'SAME')
    
    conv1d_proj_layer1 = new_conv_layer(max_pooling_layer,
                                        3,
                                        projection_filter_sizes[0],
                                        tf.nn.relu,
                                        is_training)
    conv1d_proj_layer2 = new_conv_layer(conv1d_proj_layer1,
                                        3,
                                        projection_filter_sizes[1],
                                        None,           # by default linear activation
                                        is_training)
    
    # residual connection 
    highway_input = conv1d_proj_layer2 + input

    if highway_input.shape[2] != 128:
        highway_input = tf.layers.dense(highway_input, 128)
        
    for i in range(4):
        highway_layers = highwaynet(highway_input)
        
    gru_outputs, states = tf.nn.bidirectional_dynamic_rnn(GRUCell(128),
                                                          GRUCell(128),
                                                          highway_layers,
                                                          sequence_length = input_lengths,
                                                          dtype = tf.float32)
    
    # concatinating the forward and backward gru layers
    # gru_outputs -> A tuple (output_fw, output_bw)
    return tf.concat(gru_outputs, axis = 2) 

def highwaynet(input):
    g = tf.layers.dense(inputs = input,
                        units = 128,
                        activation = tf.nn.sigmoid
                        bias_initializer = tf.constant_intializer(-1.0))
    
    r = tf.layers.dense(inputs = input,
                        units = 128,
                        activation = tf.nn.relu)
    
    return g * r + input * (1.0 - g)

def encoder_cbhg(input, is_training, input_lengths):
    projection_filter_sizes = [128, 128]
    K = 16
    return cbhg(input, K, is_training, input_lengths, projection_filter_sizes)

def decoder_cbhg(input, is_training, input_lengths):
    projection_filter_sizes = [256, 80]
    K = 8
    return cbhg(input, K, is_training, input_lengths, projection_filter_sizes)
