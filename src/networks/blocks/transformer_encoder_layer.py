import imp
import typing
import tensorflow as tf
import importlib
from typing import Optional, Union, Callable, List
from src.networks.blocks.multihead_att import MultiHeadAttention
@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder_layer: tf.keras.layers.Layer,
        num_layers: int,
        norm: Optional[Callable] = None) -> None:
        super().__init__()
        self.norm = norm
        if self.norm is not None:
            self.norm_layer = tf.keras.layers.LayerNormalization()
        self.num_layers=  num_layers
        config = encoder_layer.get_config()
        self.encoder_config=config
        self.encoder_import_module = encoder_layer.__class__.__module__ 
        self.encoder_import_name = encoder_layer.__class__.__name__
        self.layers = [type(encoder_layer).from_config(config) for i in range(num_layers)]
    def call(self, inputs, mask=None, training=None):
        x=inputs
        for mod in self.layers:
            x=mod(x, mask=mask, training=training)
        if self.norm is not None: 
            x=self.norm_layer(x, training=training)
        return x
    def compute_shape(self, inputs_shape):
        return inputs_shape
    @classmethod
    def from_config(cls, config):
        cfg_encoder = config.pop("encoder_config")
        encoder_layer = getattr(importlib.import_module(config.pop("encoder_import_module")), config.pop("encoder_import_name")).from_config(cfg_encoder)
        config.update(
            encoder_layer=encoder_layer
        )
        [config.pop(i,None) for i in ['name','trainable','dtype']]
        return cls(**config)  
    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update(
            encoder_config = self.encoder_config,
            encoder_import_module=self.encoder_import_module,
            encoder_import_name=self.encoder_import_name,
            num_layers=self.num_layers,
            norm = self.norm 
        )
        return config
@tf.keras.utils.register_keras_serializable()
class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        kwargs: **key_word multihead_attention

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    __constants__=['norm_first', 'nhead', 'dim_feedforward','dropout','activation','layer_norm_eps']
    def __init__(self,
        d_model: int,
        nhead : int, 
        dim_feedforward :int =2048,
        dropout :float =0.1,
        activation: Union[None, str, Callable]="relu",
        layer_norm_eps: float =1e-5,
        norm_first: bool=False,
        **kwargs 
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_first=norm_first
        self.nhead= nhead
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.activation=activation
        self.layer_norm_eps =layer_norm_eps


        self.self_att = MultiHeadAttention(head_size=d_model,num_heads=nhead, dropout=dropout,**kwargs)
        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation=None, use_bias=True)
        # self.dropout_all = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(d_model, activation=None, use_bias=True)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout1  = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        if isinstance(activation,str):
            self.activation_layer  = tf.keras.layers.Activation(activation)
        else:
            self.activation_layer = activation
    def call(self, inputs, mask=None, training=None):
        """
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src = inputs
        
        
        
        x = src 
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x, training=training), mask=mask, training=training)
            x= x + self._ff_block(self.norm2(x, training=training))
        else:
            x= self.norm1(x + self._sa_block(x, mask=mask, training=training), training=training)
            x = self.norm2(x + self._ff_block(x, training=training), training=training )
        return x
    def _sa_block(self, x:tf.Tensor, mask: Optional[tf.Tensor], training=None)-> tf.Tensor:
        x  = self.self_att([x,x,x], mask=mask, training=training)
        return self.dropout1(x, training=training)
    def _ff_block(self, x: tf.Tensor, training=None,):
        x=self.linear2(self.dropout2( self.activation_layer(self.linear1(x)) , training=training))
        return x

    def get_config(self):
        config = super(TransformerEncoderLayer,self).get_config()
        config['norm_first'] = self.norm_first
        config['nhead']=self.nhead
        config['dim_feedforward']=self.dim_feedforward
        config['dropout'] = self.dropout
        config['activation']=self.activation
        config['layer_norm_eps'] = self.layer_norm_eps
        config['d_model']  = self.d_model
        return config
     
    def compute_output_shape(self, inputs_shape):
        return inputs_shape
