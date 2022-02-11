import tensorflow as tf 
import numpy as np 
from src.networks.audio_processing.functional import db_scale



class Fbank(tf.keras.layers.Layer):
    def __init__(
        self,
        frame_length=25,
        frame_step=10,
        fft_length=400,
        sampling_rate=16000,
        num_mel_channels=80,
        freq_min=125,
        freq_max=7600,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_length = int(frame_length * sampling_rate / 1000)
        self.frame_step = int(frame_step * sampling_rate / 1000)
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        # Defining mel filter. This filter will be multiplied with the STFT output
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_channels,
            num_spectrogram_bins=self.fft_length // 2 + 1,
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.freq_min,
            upper_edge_hertz=self.freq_max,
        )

    def call(self, audio, training=True):
        # We will only perform the transformation during training.
        # if training:
        # Taking the Short Time Fourier Transform. Ensure that the audio is padded.
        # In the paper, the STFT output is padded using the 'REFLECT' strategy.
        stft = tf.signal.stft(
            audio,
            self.frame_length,
            self.frame_step,
            self.fft_length,
            pad_end=True,
        )

        # Taking the magnitude of the STFT output
        magnitude = tf.abs(stft)

        # Multiplying the Mel-filterbank with the magnitude and scaling it using the db scale
        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
        log_mel_spec = db_scale(mel, top_db=80)
        return log_mel_spec
       

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "frame_length": self.frame_length,
                "frame_step": self.frame_step,
                "fft_length": self.fft_length,
                "sampling_rate": self.sampling_rate,
                "num_mel_channels": self.num_mel_channels,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
            }
        )
        return config
class ContextWindow(tf.keras.layers.Layer):
    def __init__(self, left_context: int = 5, right_context: int = 5, ):
        super().__init__()
        self.left_context = left_context
        self.right_context = right_context
    def get_config(self):
        cfg= super().get_config()
        cfg.update(
            left_context=self.left_context,
            right_context = self.right_context
        )
        return cfg 
    def call(self, inputs):
        return tf.gather(inputs, self.kernel,axis=-1)
    def build(self,input_shape):

        feature = input_shape[-1] 
        
        array_select = []
        # DER=2%
        for i in range(feature):
            left = i- self.left_context
            right = i+self.right_context
            gather = list(range(left,right+1,1))
            gather = [max(k,0) for k in gather ]
            gather = [min(k, feature - 1) for k in gather ]
            array_select.extend(gather)
        array_select = np.array(array_select).reshape(-1,)
        kernel = tf.convert_to_tensor(array_select, dtype=tf.int32)
        self.kernel = tf.Variable(kernel,trainable=False)
        super().build(input_shape)
    def compute_output_shape(self, input_shape):
        context_window = self.left_context + self.right_context + 1 
        new_input_shape = [i for i in input_shape]
        new_input_shape[-1] = new_input_shape[-1] * context_window
        return new_input_shape
# context window feature size = 5                                  1,1,1,2,3,4,5,6,7,8,9,9,9
# 1,2,3,4,5,6,7,8,9, -> padding | 1. 1, 1,2,3,4,5,6,7,8,9,   9,9 |
# 1,1,1,2,3          -> 1,1, 1,2,3        
# 1,1,2,3,4          -> dp[i][j] = f[N/size + j]
# 1,2,3,4,5          -> dp_f[i*size + j] = f[N/size + j]
# 2,3,4,5,6
# 3,4,5,6,7
# 4,5,6,7,8
# 5,6,7,8,9
# 6,7,8,9,9
# 7,8,9,9,9

# -> dp[N,size]: Dp[i][j] = F[(N/size)*i+j]

# bs, time, feature 
# gather nd 
# (context_len ,1,feature*context_len)
