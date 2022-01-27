import tensorflow as  tf 


class FixedSpeakerHead(tf.keras.Model):
    def __init__(self, *args, n_speaker=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_speaker = n_speaker
        self.pro_diarization_speaker = tf.keras.layers.Dense(n_speaker, activation=None , use_bias=True)
    def call(self, inputs, training=None, mask=None ):
        return self.pro_diarization_speaker(inputs)
    def get_config(self):
        cfg = super().get_config()
        cfg.update(n_speaker=self.n_speaker)
        return cfg 
    