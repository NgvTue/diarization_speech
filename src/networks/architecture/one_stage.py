import tensorflow as tf

from src.networks.backbone.transformer import TransformerBackbone
from src.networks.backbone.conformer import ConformerBackbone
from src.networks.heads.fixed_speaker_head import FixedSpeakerHead 


class OneStage(tf.keras.Model):
    def __init__(self, backbone, heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.heads = heads 
    def call(self, inputs, training=None, mask=None):
        return self.heads(self.backbone(inputs, training=training, mask = mask),  training=training, mask=mask)


class CEEND(OneStage):
    def __init__(self, 
        subsampling,
        n_units=144,
        num_blocks=16,
        head_size=36,
        num_heads=4,
        kernel_size=32,
        depth_multiplier=1,
        fc_factor=0.5,
        dropout=0.1,
        max_speaker:int=4,
        **kwargs):
        backbone = ConformerBackbone(subsampling,n_units=n_units, num_blocks=num_blocks, head_size=head_size, num_heads=num_heads, kernel_size=kernel_size,depth_multiplier=depth_multiplier, fc_factor=fc_factor, dropout=dropout)
        heads = FixedSpeakerHead(n_speaker=max_speaker)
        super().__init__(backbone, heads, **kwargs)
class EEND(OneStage):
    """
    ARGS: See transformer args 
    max_speaker = : max_speaker in one_sample
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

        max_speaker: int = 4,
        **kwargs
        ):
        backbone = TransformerBackbone(in_size=in_size, n_heads=n_heads,n_units=n_units,n_layers=n_layers,dim_feedforward=dim_feedforward,dropout=dropout, has_position=has_position)
        head = FixedSpeakerHead(n_speaker=max_speaker)
        super().__init__(backbone, head,*args, **kwargs)

