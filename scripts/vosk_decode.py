# wav="audio_Vi/reading/recording2k5/recording_silenceRemoved_16k_1channel-cleaned/news-reading-demo1k-0002-trannn3-angiang-1989-nu-news-reading-1.wav"
# wav = "/data1/T30/speechData/data_Vi/" + wav
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
SetLogLevel(0)
import argparse
wf = wave.open(wav,"rb")
model = Model("/data/speechDataTeam/tuenv1/task_diarization/model_asr/202005-STT-VoskAPI-GenericDomain-3k5VinData_2kYoutube_XSAMPAPhones_ConventionalTraining_Bigmodel")
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    help='data dir of single-speaker recordings')

args = parser.parse_args()
wav_scp=None
with open(os.path.join(args.data_dir,"wav.scp"),"r") as f:
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    wav_scp={x[0]: x[1] for x in lines}
def predict_one(wav):
    i=0
    while True:
        i = i+1
        data = wf.readframes(4000)
        if len(data)==0:break
        if rec.AcceptWaveform(data):
            print(rec.Result())
        else:
            print(rec.PartialResult())
    a = rec.FinalResult()
    return a
from tqdm import tqdm
import json
data=[]
for i in tqdm(wav_scp):
    a=predict_one(i)
    a=json.loads(a)
    data.append(a)
data.to_csv("db_kaldi.csv",index=False)

import pandas as pd 
import json 