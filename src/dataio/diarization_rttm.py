from chunk import Chunk
import numpy as np
import numpy as np
from pytorch_lightning import logging
from torch import dtype
from src.dataio.base import DataSet

from typing import List, Optional, Callable
import pandas as pd
from tqdm import tqdm
import soundfile as sf 

import tensorflow as tf 


class DiarizationRTTM(DataSet):
    """
    args:
        path_rttm: path file rttm db
        pattern_rttm: pattern file wav in rttm : example {record_id_in_rttm}.wav
    """
    def __init__(
        self,
        path_rttm: str = None,
        pattern_rttm: str="{record_id}.wav",
        max_speaker : int =  4,
        frame_per_sample: int = 1000* 15,
        sample_rate: int = 16000,
        frame_length : int = 25,
        frame_shift: int = 10,
        shuffle=True, 

    ):
        signature = {
            (
                tf.TensorSpec(shape=(int(frame_per_sample * sample_rate / 1000),), dtype=tf.float32 ), # audios


            ),
            (
                tf.TensorSpec(shape=(int(frame_per_sample / frame_shift),), dtype=tf.int32 ), # labels
            )
        } 
        super().__init__(signature=signature, shuffle=shuffle)
        self.max_speaker = max_speaker
        self.sample_rate =sample_rate
        self.frame_length=frame_length
        self.frame_shift = frame_shift
        self.frame_per_sample = frame_per_sample
        self.path_rttm=path_rttm
        self.pattern_rttm=pattern_rttm

        record_db={}
        record_pandas = []
        with open(self.pattern_rttm,"r") as f:
            for line in f:
                # SPEAKER record_id 1 start duration <NA> <NA> speaker_id <NA> <NA>
                line_split = line.split()
                record_id = line_split[1]
                start = float(line_split[3])
                duration = float(line_split[4])
                speaker_id = line_split[7]

                if record_id not in record_db:
                    record_db[record_id]={
                        'segment':[],
                        'length':0
                    }
                
                record_db[record_id]['segment'].append(
                    {
                        'start':start,
                        'duration':duration,
                        'speaker_id':speaker_id
                    }
                )
                record_pandas.append({
                    'record_id':record_id,
                    'start':start,
                    'duration':duration,
                    'speaker_id':speaker_id
                })

        self.record_db = record_db
        self.record_pandas = pd.DataFrame(record_pandas)

        print("Run evaluate rttm")
        for record_id in tqdm(record_db):
            wav = pattern_rttm.format(record_id)
            y,sr = sf.read(wav)
            assert sr  == sample_rate
            length = y.shape[0] / sr 
            self.record_db[record_id]['length'] = length
        
        self.record_pandas['length']=self.record_pandas.record_id.apply(lambda x:self.record_db[x]['length'])
        
        length_assurance = self.record_pandas['start'] + self.record_pandas['duration'] <= self.record_pandas['length'] 
        self.record_pandas['end'] = self.record_pandas['start'] + self.record_pandas['duration']
        if length_assurance.sum() != self.record_pandas.shape[0]:
            print("some segment rttm has length greater than length of record_id")
            print(self.record_pandas[~length_assurance])
        
        print("generate chunk per record_id")
        max_speaker_persample=0
        for record_id in tqdm(self.record_db):
            total_duration = self.record_db[record_id]['length']
            st = 0
            self.record_db['chunks']=[]
            chunk=[]
            while st < total_duration:
                end = st + (frame_per_sample/1000)
                end = min(end, total_duration)
                chunk_infor  = {
                    "start":st,
                    'end':end,
                    'segment_speaker':[]  
                }
                segment_filter = self.record_pandas[(self.record_pandas.record_id == record_id) & (self.record_pandas['start'] <=end) & (self.record_pandas['end'] >= st)]
                segment_filter['start'] =segment_filter['start'].apply(lambda x:min(0,x-st)) 
                segment_filter['end'] = segment_filter['end'] - st 
                max_speaker_persample=max(max_speaker_persample, segment_filter['speaker_id'].unique().shape[0])
                for row in range(len(segment_filter)):
                    chunk_infor['segment_speaker'].append(
                        {
                            'start':segment_filter.iloc[row]['start'],
                            'end':segment_filter.iloc[row]['end'],
                            'speaker_id':segment_filter.iloc['row']['speaker_id']
                        }
                    )
                chunk.append(chunk_infor)

                st = end 
            
            self.record_db['chunks']=chunk
        
        print("Max speaker overlap : ", max_speaker_persample)
        self.max_speaker_overlap = max_speaker_persample
        self.idxs = list(self.record_db.keys())
    def __str__(self) -> str:
        return str({
            "number record id":len(self.record_db),
            "number segment speaker": len(self.record_pandas),
            "max speaker overlap":self.max_speaker_overlap,
            "total_chunks":sum([len(self.record_db[i]['chunks']) for i in self.record_db]),
            "mean lenght speaker segment": self.record_pandas['duration'].mean()
        })
    def __len__(self):return len(self.record_db)
    def __getitem__(self, idx):
        record_id = self.idxs[idx] 
        # get random chunks 
        chunks  = self.record_db[record_id]['chunks']
        random_chunk = np.random.randint(0,len(chunks))
        chunk = chunks[random_chunk]
        fpath_wav = self.pattern_rttm.format(record_id)
        wav,sr = sf.read(fpath_wav,start=int(chunk['start'] * self.sample_rate), frames=int(self.sample_rate * self.frame_per_sample / 1000), fill_value=0)
        speaker = np.zeros((int(self.frame_per_sample / self.frame_shift), self.max_speaker))
        for idx,ordered_speaker in enumerate(chunk['segment_speaker']):
            if idx >= self.max_speaker:
                logging.warning(f"record_id {record_id} has chunk with more than {self.max_speaker} overlap")
                break
            start = int(ordered_speaker['start'] * self.sample_rate)
            end = int(ordered_speaker['end'] * self.sample_rate)
            speaker[start:end,idx] = 1
        return wav, speaker


