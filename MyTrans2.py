import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
#import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Embedding


from tensorflow.keras.utils import to_categorical

class MultiheadAttention(tf.keras.models.Model):
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(
            self,
            _input: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:

        q = self.q_dense_layer(_input)  # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(_input)  # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(_input)

        q = self._split_head(q)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        k = self._split_head(k)  # [batch_size, head_num, m_length, hidden_dim/head_num]
        v = self._split_head(v)  # [batch_size, head_num, m_length, hidden_dim/head_num]

        depth = self.hidden_dim // self.head_num
        q *= depth ** -0.5

        logit = tf.matmul(q, k, transpose_b=True)  # [batch_size, head_num, q_length, k_length]
        logit += tf.compat.v1.to_float(attention_mask) * _input.dtype.min  # mask は pad 部分などが1, 他は0


        attention_weight = tf.nn.softmax(logit, name='attention_weight')
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)


        attention_output = tf.matmul(attention_weight, v)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        attention_output = self._combine_head(attention_output)  # [batch_size, q_length, hidden_dim]
        return self.output_dense_layer(attention_output)

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:

        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:

        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])


class NomaskMultiheadAttention(tf.keras.models.Model):


    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(
            self,
            _input: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:

        q = self.q_dense_layer(_input)  # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(_input)  # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(_input)

        q = self._split_head(q)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        k = self._split_head(k)  # [batch_size, head_num, m_length, hidden_dim/head_num]
        v = self._split_head(v)  # [batch_size, head_num, m_length, hidden_dim/head_num]

        depth = self.hidden_dim // self.head_num
        q *= depth ** -0.5  # for scaled dot production

        logit = tf.matmul(q, k, transpose_b=True)  # [batch_size, head_num, q_length, k_length]

        attention_weight = tf.nn.softmax(logit, name='attention_weight')
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)

        attention_output = tf.matmul(attention_weight, v)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        attention_output = self._combine_head(attention_output)  # [batch_size, q_length, hidden_dim]
        return self.output_dense_layer(attention_output)

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
    
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:

        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])

class AddPositionalEncoding(tf.keras.layers.Layer):
    
    def __init__(self, vocab_size, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        inputs = self.token_emb(inputs)
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))
        depth_counter = tf.range(depth) // 2 * 2  # 0, 0, 2, 2, 4, ...
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length, 1])  # [max_length, depth]
        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type))  # [max_length, depth]

        # cos(x) == sin(x + π/2)
        phase = tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2  # 0, π/2, 0, π/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1])  # [max_length, depth]

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, depth]), fl_type)  # [max_length, depth]

        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)
        # [batch_size, max_length, depth]
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1])
        #print(positional_encoding)
        return inputs + positional_encoding


class LayerNormalization(tf.keras.layers.Layer):
    def build(self, input_shape: tf.TensorShape) -> None:
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],
                                     initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim],
                                    initializer=tf.zeros_initializer())
        super().build(input_shape)

    def call(self, x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)

        return norm_x * self.scale + self.bias

class FeedForwardNetwork(tf.keras.models.Model):

    def __init__(self, hidden_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(hidden_dim * 4, use_bias=True,
                                                        activation=tf.nn.relu, name='filter_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, _input: tf.Tensor, training: bool) -> tf.Tensor:

        tensor = self.filter_dense_layer(_input)
        tensor = self.dropout_layer(tensor, training=training)
        return self.output_dense_layer(tensor)
class ResidualNormalizationWrapper(tf.keras.models.Model):
    def __init__(self, layer: tf.keras.layers.Layer, dropout_rate: float,enbbvec:int,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.enbbvec=enbbvec
        self.shape_dense_layer=tf.keras.layers.Dense(self.enbbvec, use_bias=False, name='vec_dense_layer')
    def call(self, _input: tf.Tensor, training: bool, *args, **kwargs) -> tf.Tensor:
        tensor = self.layer_normalization(_input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)
        tensor =self.shape_dense_layer(tensor)
        return _input + tensor


PAD_ID=8515
def _create_dec_self_attention_mask(decoder_input: tf.Tensor):
    with tf.name_scope('dec_self_attention_mask'):
        batch_size, length = tf.unstack(tf.shape(decoder_input))
        pad_array = tf.equal(decoder_input, PAD_ID)  # [batch_size, m_length]
        pad_array = tf.reshape(pad_array, [batch_size, 1, 1, length])

        autoregression_array = tf.logical_not(tf.compat.v1.matrix_band_part(tf.ones([length, length], dtype=tf.bool), -1, 0)) 
        autoregression_array = tf.reshape(autoregression_array, [1, 1, length, length])

        return tf.logical_or(pad_array, autoregression_array)

def create_model(hidden_vec_L=32,enbd=256,training=True):
    input_dropout_layer = tf.keras.layers.Dropout(0.3)
    inputs_ = tf.keras.layers.Input(shape=(89,))
    enc_dec_attention_mask=_create_dec_self_attention_mask(inputs_)
    output_Flat =tf.keras.layers.Flatten()
    output_normalization = LayerNormalization()
    output_liner =tf.keras.layers.Dense(8516,name='out_linear',use_bias=False)
    #output_dense_layer = tf.keras.layers.Dense(16196, activation='softmax',name='out_softmax',use_bias=False)
    embedded_input=AddPositionalEncoding(8516,enbd)(inputs_)
    query = input_dropout_layer(embedded_input, training=training)
    attention_block_list: [[tf.keras.models.Model]] = []
    for i in range(1):
        self_attention_layer = MultiheadAttention(hidden_vec_L, 8, 0.1, name='enc_dec_attention')
        enc_dec_attention_layer = NomaskMultiheadAttention(hidden_vec_L,8, 0.1, name='self_attention')
        ffn_layer = FeedForwardNetwork(hidden_vec_L, 0.1, name='ffn')
        attention_block_list.append([
            ResidualNormalizationWrapper(self_attention_layer, 0.1,enbd,name='self_attention_wrapper'+str(i)),
            ResidualNormalizationWrapper(enc_dec_attention_layer,0.1,enbd,name='enc_dec_attention_wrapper'+str(i)),
            ResidualNormalizationWrapper(ffn_layer, 0.1,enbd,name='ffn_wrapper'+str(i)),
        ])
    for i, layers in enumerate(attention_block_list):
        self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
        with tf.name_scope(f'hopping_{i}'):
            query = self_attention_layer(query,attention_mask=enc_dec_attention_mask,  training=training)
            query = enc_dec_attention_layer(query, training=training)
            query = ffn_layer(query, training=training)
    queryf=output_normalization(query)
    outputs = output_liner(queryf)
    #outputs=output_dense_layer(querys)
    model = keras.Model(inputs=inputs_, outputs=outputs)
    return model
