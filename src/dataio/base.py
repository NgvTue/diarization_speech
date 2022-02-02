


import imp
import numpy as np
from sys import implementation
from abc import ABC, abstractmethod 
import tensorflow as tf 
import logging
import copy
import pickle
class ShuffleSampler:
    def __call__(self, idxs):
        np.random.shuffle(idxs)
        return idxs

class DataSet(ABC):
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name","dataset")
        self._trace_mode = kwargs.pop("trace_mode",False)
        self.signature = kwargs.pop("signature", None)
        self.sampler = kwargs.pop("sampler",None)
        is_shuffle = kwargs.pop("shuffle", None)
        if is_shuffle is not None and self.sampler is  None:
            if is_shuffle:
                self.sampler = ShuffleSampler()
            else:
                self.sampler = lambda x:x
    def split_dataset(self, ratios: list=[0.9,0.1], option_kwargs=[{"shuffle":False,"name":f"split_index_{i}"} for i in range(2)],seed=44):
        logging.warning(f"when split dataset note that the splited dataset 'llnot change: eg sampler is default, not shuffle,...\nAffter call split wwe make dataset_splited.sampler = lambda x:x to make sure everything work correctly")
        logging.warning(f"We are copying {len(ratios)} of parrent dataset\nThis can make more memory!!\nIf you don't want it pls define split_dataset by self")
        index = list(range(len(self)))
        rd=np.random.RandomState(seed)
        rd.shuffle(index)
        ratios = [int(i * len(index)) for i in ratios]
        lag = len(index) - sum(ratios)
        ratios[0] = ratios[0] + lag
        index_splited = [index[:ratios[0]]]
        signature = self.signature
        for i in range(len(option_kwargs)):
            option_kwargs[i]['signature'] = signature
        for i in range(1,len(ratios)):
            index_splited.append(index[ratios[i-1]:ratios[i]+ratios[i-1]])
            ratios[i] = ratios[i-1] + ratios[i]
        ds_splited = [SplitedDataset(self, i, **k) for i,k in zip(index_splited, option_kwargs)]
        return ds_splited
    @abstractmethod
    def __len__(self):
        pass 
    @abstractmethod
    def __getitem__(self, idx):
        pass 
    def __sampler__(self):
        idx = list(range(len(self)))
        if hasattr(self,'sampler'):
            idx = getattr(self,'sampler')(idx)
        return idx 
    def __call__(self):
        if self._trace_mode:
            logging.info("Resampler dataset")
        idx = self.__sampler__()
        for index_any_type in idx:
            yield self[index_any_type]
    def to_tensor_dataset(self):
        if self.signature is None:
            raise ValueError(f"signature in {self.name} is not define")
        return tf.data.Dataset.from_generator(self, output_signature = self.signature)
    def save(self, fpath):
        with open(fpath, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def from_disk(fpath):
        with open(fpath, "rb") as f:
            return pickle.load(fpath) 
class SplitedDataset(DataSet):
    def __init__(self, parrent_ds:DataSet, index:list, *args, **kwargs):
        self.parrent_ds = copy.deepcopy(parrent_ds)
        self.parrent_ds.sampler = lambda x:x
        self.parrent_ds.shuffle=False
        self.index = copy.deepcopy(index)
        super().__init__(*args,**kwargs) 
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        return self.parrent_ds[self.index[idx]]
