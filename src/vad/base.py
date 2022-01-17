from typing import List, Optional, Dict
from src.core import SlidingWindowFeature, SlidingWindow
import numpy as np
import os 

class BaseVAD():
    NAME = "VAD"
    def __init__(self, *args, **kwargs):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
    def predict(self, audio_raw, context : Dict = None ) -> SlidingWindowFeature:
        """audio_raw: audio dười dạng [n,] với n là chiều dài wave (bs=1)
           context: Các output của các step khác nếu cần 
        -----------------------------------------------------------------
        Return: slidingWindowFeature 
        """
        raise Exception("implement predict")
        pass 