


import imp
import numpy as np
from sys import implementation
from abc import ABC, abstractmethod 
import tensorflow as tf 
import logging

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