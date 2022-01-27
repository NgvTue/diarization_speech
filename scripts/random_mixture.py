
import argparse
import os
import os,sys
sys.path.append("../")
from src.dataio import kaldi_data
import random
import numpy as np
import json
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    help='data dir of single-speaker recordings')

parser.add_argument('--n_mixtures', type=int, default=10,
                    help='number of mixture recordings')

parser.add_argument('--n_speakers', type=int, default=5,
                    help='number of speakers in a mixture')

parser.add_argument('--min_utts', type=int, default=1,
                    help='minimum number of uttenraces per speaker')

parser.add_argument('--max_utts', type=int, default=3,
                    help='maximum number of utterances per speaker')


parser.add_argument('--sil_scale', type=float, default=2.0,
                    help='average silence time betwwen utt of one speaker IE: stop time, sil,..')

parser.add_argument('--sil_scale_with_two_spk', type=float, default=3.0,
                    help='average silence time between two record (different spk) : If two speaker seperate by sil')

parser.add_argument('--overlap_scale', type=float, default=5.0,
                    help='average overlap time  between two record (different spk): If two speaker say on time overlap')

parser.add_argument('--overlap_prob', type=float, default=0.1,
                    help='average overlap prob, Prob of overlap')


parser.add_argument('--n_speakers_overlap', type=int, default=4,
                    help='max number of speakers in one overlap interval')

parser.add_argument('--random_seed', type=int, default=777,
                    help='random seed')

args = parser.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)

# load list of wav files from kaldi-style data dirs
wavs = kaldi_data.load_wav_scp(
        os.path.join(args.data_dir, 'wav.scp'))


# spk2utt is used for counting number of utterances per speaker
spk2utt = kaldi_data.load_spk2utt(
        os.path.join(args.data_dir, 'spk2utt'))

segments = kaldi_data.load_segments_hash(
        os.path.join(args.data_dir, 'segments'))

# choice lists for random sampling
all_speakers = list(spk2utt.keys())



def mixture_to_time_format(mixture):
    flatten_mixture = []
    for spk in mixture['speakers']:
        spkid = spk['spkid']
        utts = spk['utts']
        intervals = spk['intervals']
        pos=0
        data  =[]
        for interval, utt in zip(intervals, utts):
            # data.append({"type":"sil","start":pos,"end":pos + interval})
            pos = pos + interval
            duration = utt[-1] - utt[-2]
            data.append({"start":pos,"end":pos+duration,"utt":utt})
            pos = pos + duration
        flatten_mixture.append({"data":data,"end":pos})

    flatten_mixture_final = [i for i in flatten_mixture[0]['data']]

    pos = flatten_mixture_final[-1]['end']
    for i in range(1,len(flatten_mixture)):
        distance = mixture['distance'][i-1]
        ov=False
        if distance <=0: # overlap
            distance = -1*distance
            distance = min(distance, pos) 
            start_real = pos - distance
            ov=True
            start_overlap = start_real
            end_overlap = pos
        else:
            start_real = distance + pos
        for k in range(len(flatten_mixture[i]['data'])):
            flatten_mixture[i]['data'][k]['start'] += start_real
            flatten_mixture[i]['data'][k]['end'] += start_real
        
        
        flatten_mixture_final.extend([k for k in flatten_mixture[i]['data']])
        pos = flatten_mixture_final[-1]['end']
        if ov ==True:
            num_overlap = args.n_speakers_overlap - 2
            if num_overlap >= 1:
                utt_overlaps = [a for a in mixture['get_utt_overlap']]
                utt_overlaps.pop(i)
                utt_overlaps.pop(i-1) 
                np.random.shuffle(utt_overlaps)
                utt_overlaps = utt_overlaps[:num_overlap]
                utt_overlaps = [(k[0], k[1], min(k[2],k[1] + distance))  for k in utt_overlaps]
                d = [{
                    'utt':utt_overlaps[k],
                    "start":start_overlap,
                    "end":start ,
                } for k in range(len(utt_overlaps))
                ]
                flatten_mixture_final.extend(d)

        
    return flatten_mixture_final
        
mixtures = []
for it in range(args.n_mixtures):
    recid = 'mix_{:07d}'.format(it + 1)
    # randomly select speakers, a background noise and a SNR
    speakers = random.sample(all_speakers, args.n_speakers)
    # max_spk = random.randint(1,args.n_speakers)
    # speakers=speakers[:max_spk]
    # random select utt
    mixture={"speakers":[], "get_utt_overlap":[]}
    for speaker in speakers:
        n_utts = np.random.randint(args.min_utts, args.max_utts + 1)
        n_utts= min(n_utts, len(spk2utt))
        cycle_utts = itertools.cycle(spk2utt[speaker])
        roll = np.random.randint(0, len(spk2utt[speaker]))
        for i in range(roll):
            next(cycle_utts)
        utts = [next(cycle_utts) for i in range(n_utts)]
        intervals = np.random.exponential(args.sil_scale, size=n_utts)
        
        # intervals = np.array([max(i,0.75) for i in intervals])
        intervals[0]  = 0
        # intervals = np.max(intervals, 0.75) # maximum vad >= 0.75

        if segments is not None:
            utts = [segments[utt] for utt in utts]
            utts = [(wavs[rec], st, et) for (rec, st, et) in utts]
            duration =0.
            for ix in range(len(utts)):
                duration = duration + (utts[ix][-1] - utts[ix][-2]) + intervals[ix]
            mixture['speakers'].append({
                'spkid': speaker,
                'utts': utts,
                'intervals': intervals.tolist(),
                'duration':duration
                })
        else:
            raise Exception("provided segments file")
            mixture['speakers'].append({
                'spkid': speaker,
                'utts': [wavs[utt] for utt in utts],
                'intervals': intervals.tolist()
                })
        utts_overlap = next(cycle_utts)
        utts_overlap = segments[utts_overlap]
        
        utts_overlap = (wavs[utts_overlap[0]], utts_overlap[1], utts_overlap[2])
        mixture['get_utt_overlap'].append(utts_overlap)
    
    # handle overlap spk
    
    mixture['distance'] = []
    
    for i in range(1,len(mixture['speakers'])):
        if np.random.random() <= args.overlap_prob: # overlap
            d_overlap = np.random.exponential(args.overlap_scale, size=1)[0]
            d_overlap = max(d_overlap, 1.)
            mixture['distance'].append(-1 * d_overlap) 
        else: # no overlap 
            d_sil = np.random.exponential(args.sil_scale_with_two_spk, size=1)[0]
            mixture['distance'].append(d_sil)

    
    # print(mixture)
    # print("-"*100)
    # mixtures.append(mixture_to_time_format(mixture))
    # break
    print(recid, json.dumps(mixture_to_time_format(mixture)))
    # break
    # mixtures

        




    




# for it in range(args.n_mixtures):
    
#     print(recid, json.dumps(mixture))