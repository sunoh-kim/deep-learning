import numpy as np
import tensorflow as tf
import argparse

###########################################################################
# TO DO (ablation study)
# 1. Analyze the model performance according to 13 hyperparameters in class Hparams. (DEFINE 1-13)
#    All changeable hyperparameters are set as "-1".
# 2. Report at least 12 BLEU score results according to each different hyperparameters set.
#    You need to analyze at least 3 different hyperparameters.
#    (for example, (3 types of bath sizes) x (2 types of learning rates) x (2 types of num_heads) )
# 3. Report the hyperparameters set resulting the best performance of your test set
#     and submit the corresponding checkpoint file in ./transformer_checkpoint.
#
# IMPORTANT:
# DO NOT CHANGE other fixed hyperparameters (except DEFINE 1-13), databases, or functions, ...
# Performance (BLEU) score value is NOT considered for grading.
# You also need to define your own log, eval, test directory in Hparams.
#
# In summary, you need to submit 
# 1) report file(in Assignment3.pptx)
# 2) checkpoint file of the best performance
# use the last page of "Assignment3.pptx" file provided
############################################################################





class Hparams:
    parser = argparse.ArgumentParser()
    
    #flags.DEFINE_string('f', '', 'kernel')
    parser.add_argument('-f', default='', help='kernel')

    # prepro
    # DEFINE 1
    parser.add_argument('--vocab_size', default=-1, type=int) 

    # train
    ## files
    parser.add_argument('--train1', default='./iwslt2016/segmented/train.de.bpe',
                             help="german training segmented data")
    parser.add_argument('--train2', default='./iwslt2016/segmented/train.en.bpe',
                             help="english training segmented data")
    parser.add_argument('--eval1', default='./iwslt2016/segmented/eval.de.bpe',
                             help="german evaluation segmented data")
    parser.add_argument('--eval2', default='./iwslt2016/segmented/eval.en.bpe',
                             help="english evaluation segmented data")
    parser.add_argument('--eval3', default='./iwslt2016/prepro/eval.en',
                             help="english evaluation unsegmented data")

    ## vocabulary
    parser.add_argument('--vocab', default='./iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    # DEFINE 2
    parser.add_argument('--batch_size', default=-1, type=int) 
    # DEFINE 3
    parser.add_argument('--eval_batch_size', default=-1 , type=int) 
    # DEFINE 4
    parser.add_argument('--lr', default=-1, type=float, help="learning rate") 
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="", help="log directory")
    # DEFINE 5
    parser.add_argument('--num_epochs', default=-1, type=int) 
    parser.add_argument('--evaldir', default="", help="evaluation dir")

    # model
    # DEFINE 6
    parser.add_argument('--d_model', default=-1, type=int,
                        help="hidden dimension of encoder/decoder")
    # DEFINE 7
    parser.add_argument('--d_ff', default=-1, type=int,
                        help="hidden dimension of feedforward layer")
    # DEFINE 8
    parser.add_argument('--num_blocks', default=-1, type=int,
                        help="number of encoder/decoder blocks")
    # DEFINE 9
    parser.add_argument('--num_heads', default=-1, type=int,
                        help="number of attention heads")
    # DEFINE 10
    parser.add_argument('--maxlen1', default=-1, type=int,
                        help="maximum length of a source sequence")
    # DEFINE 11
    parser.add_argument('--maxlen2', default=-1, type=int,
                        help="maximum length of a target sequence")
    # DEFINE 12
    parser.add_argument('--dropout_rate', default=-1, type=float)
    # DEFINE 13
    parser.add_argument('--smoothing', default=-1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='./iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='./iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', default='./transformer_checkpoints', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="", help="test result dir")

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)




def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.
    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs