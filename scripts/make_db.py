import json
import pandas as pd
import numpy as np
import argparse
import os
import soundfile as sf
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                            help='data dir of single-speaker recordings')

parser.add_argument('--file_mixture', type=str, default="",
                            help='file random mixture')

parser.add_argument("--folder_db",type=str,help="folder save db")
args = parser.parse_args()
os.makedirs(args.folder_db)
os.makedirs(os.path.join(args.folder_db,"wavs"))

with open(os.path.join(args.data_dir,"wav.scp"),"r") as  f:
    wav_scp = f.read().splitlines()
    wav_scp = {i.split()[1]:i.split()[0] for i in wav_scp}
with open(os.path.join(args.data_dir,"utt2spk"),"r") as  f:
    utt_spk = f.read().splitlines()
    utt_spk = {i.split()[0]:i.split()[1] for i in utt_spk}
    print(utt_spk)
#print(wav_scp)
datas = []
with open(args.file_mixture,"r") as  f:
    for line in f:
        mixture_id, json_str = line.strip().split(None, 1)
        json_str = json.loads(json_str)
        datas.append([mixture_id, json_str])


from tqdm import tqdm

print(datas[2])
#exit(0)
total_s=0.
total_sil=[]
total_overlap=[]
rttms=[]
for mix,d_mix in tqdm(datas):
    wave_file_mix = f"{mix}.wav"
    wave_file_mix = os.path.join(args.folder_db,"wavs", wave_file_mix)
    len_audio = max([i['end'] for i in d_mix])
    len_audio = int(len_audio * 16000) + 100
    audio_all = np.zeros((len_audio,))
    audio_ov = np.zeros((len_audio,))
    l_o=0.
    for d in d_mix:
        start = d['start']

        endx = d['end']
        path_wav,st,end = d['utt']
        spk=utt_spk[ wav_scp[path_wav]]
        path_wav = path_wav.replace("audio_Vi","/data/speechDataTeam/tuenv1/task_diarization/datasets/Testing_28092021/Testing-old-mobile-car-2021-12h/wavs_test_Vi")

        wav,sr = sf.read(
            
        )
        #print(path_wav, wav.shape[0]-(end-st)*sr)
        assert sr==16000
        assert wav.shape[0] >= int((end-st)*sr)
        wav = wav.reshape(-1,)[int(st*sr):int(end*sr)]

        start = int(start * 16000)
        endx  =int(endx * 16000)
        if wav.shape[0] - (endx-start) > 100:
            print(mix)
            exit(0)
        if (endx-start) - wav.shape[0]>100:
            print(mix,d,endx,start,wav.shape,endx-start-wav.shape[0])
            exit(0)
        if wav.shape[0] > len_audio:
            print(mix)
            print(wav.shape,len_audio)
            exit(0)


        l_o = l_o + start-endx
        #print(start,endx,len_audio, wav.shape[0],st,end) 
        audio = np.zeros((len_audio,))
        audio_ov[start:endx]+=1
        audio[start:endx] = wav[:endx-start]
        rttms.append(f"SPEAKER {mix} 1    {start/sr}    {endx/sr-start/sr} <NA> <NA> {spk} <NA>")
        audio_all = audio_all+audio

    sf.write(wave_file_mix, audio_all,sr)
    l_o = audio_ov>1
    l_o = np.mean(l_o)
    total_overlap.append(l_o)
    total_sil.append(np.sum(audio_ov==0)/sr)
    total_s = total_s+len_audio/sr
print(sum(total_overlap)/len(total_overlap))
print("total_duration = ",total_s)
print("total_sil = ",sum(total_sil))
print(sum(total_sil) / total_s)

with open(os.path.join(args.folder_db,"db.rttm"),"w+") as f:
    f.write("\n".join(rttms))
