
import tensorflow as tf
from src.networks.losses.pit import pit_loss,faster_pit_loss_cross_entropy
from src.networks.architecture.one_stage import EEND
from src.networks.audio_processing.features import Fbank
from src.dataio.diarization_rttm import DiarizationRTTM
from src.networks.metrics.der import der, pit_per
import argparse

argparser = argparse.ArgumentParser("Training API")
# argparser.add_argument("path_config", type=str, help="path to file config")
args = argparser.parse_args()

@tf.function
def loss_fn(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_true,y_pred),-1),-1)


dataset = DiarizationRTTM("/home/tuenguyen/speech/fr_diarization/recipes/AMI/train.rttm","/home/tuenguyen/speech/fr_diarization/recipes/AMI/amicorpus/{}.wav",max_speaker=5,frame_per_sample=1000*15)
print(str(dataset))
# split train val
train,val = dataset.split_dataset(ratios=[0.9,0.1],option_kwargs=[{"shuffle":True,"name":"trainingset"},{"shuffle":False,"name":"validationset"}], seed=22)

# to tensor ds
train=train.to_tensor_dataset().batch(4, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(4)
val=val.to_tensor_dataset().batch(4, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).prefetch(4)

# define model
inputs = tf.keras.layers.Input((16000*15))
feature = Fbank(num_mel_channels=64)(inputs)
eend = EEND(max_speaker=5)(feature)
model = tf.keras.Model(inputs=[inputs], outputs=[eend])

# define pit loss
loss = pit_loss(loss_fn,k=5)
metric = pit_per(k=5)
model.compile(tf.optimizers.Adam(1e-4),loss=loss,metrics=[metric])
model.fit(train, epochs=5, validation_data=val)
