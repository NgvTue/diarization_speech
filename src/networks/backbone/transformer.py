import tensorflow as tf 
import numpy as np


from src.networks.blocks.transformer_encoder_layer import TransformerEncoderLayer, TransformerEncoder


class TransformerBackbone(tf.keras.Model):
    """
    ARGS:
        in_size: size of features inputs, mel, mfcc,...
        n_heads: number of head in self attention
        n_units: number of units embedding features
        n_layers: number of layers encoder transformer 
        dim_feedfoward: size of botneck layer in  transformer
        dropout: droprate ratio
    """
    def __init__(
        self,
        *args,
        in_size: int =64,
        n_heads: int =4,
        n_units: int = 128,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        has_position: bool=False,
        **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units=n_units
        self.n_layers = n_layers
        self.dim_feedforward=dim_feedforward
        self.has_position=has_position
        self.dropout = dropout

        self.norm_feature  = tf.keras.layers.LayerNormalization()
        self.laten_space_feature=tf.keras.layers.Dense(n_units, activation=None, use_bias=True)



        if self.has_position:
            self.position_encode = PositionEncoding(n_units,)
        
        encoder_layer = TransformerEncoderLayer(n_units, n_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu', )
        self.encoder_block = TransformerEncoder(encoder_layer, n_layers)


    def call(self, inputs, training=None, mask=None ):
        inputs = self.norm_feature(inputs, training=training)
        inputs = self.laten_space_feature(inputs)
        if self.has_position:
            inputs = inputs + self.position_encode(inputs)
        # bs, time, n_units
        return self.encoder_block(inputs, training=training, mask=mask) # bs, time, n_units




    def get_config(self,):
        cfg = super().get_config()
        cfg.update(
            in_size=self.in_size,
            n_heads=self.n_heads,
            n_units=self.n_units,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            has_position=self.has_position,
            dropout=self.dropout)
        return cfg 








@tf.keras.utils.register_keras_serializable()
class PositionEncoding(tf.keras.layers.Layer):
    """
    Args:
        d_model : the emb dimg
        max_length: max_length of seq inputs
    """
    def __init__(
        self,
        d_model : int,
        max_length:int = 5000,
        ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.position = self.positional_encoding(max_length, d_model)
    def get_angle(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    def call(self, inputs, mask=None, training=None):
        seq_len = tf.shape(inputs)[1]
        return self.position[:,:seq_len,:]
    def compute_shape(self, inputs_shape):
        return inputs_shape

    def positional_encoding(self, max_length, d_model):
        angle_rads = self.get_angles(np.arange(max_length)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
    def get_config(self):
        config  = super().get_config()
        config.update(
            d_model = self.d_model,
            max_length = self.max_length
        )
        return config