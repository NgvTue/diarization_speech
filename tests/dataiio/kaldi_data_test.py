FOLDER_DB="/home/tuenguyen/Desktop/end2end_diarization/data/vin18k/"
import os,sys
sys.path.append("../../")
from src.core import Segment
from src.dataio import KaldiData


train_db = KaldiData(os.path.join(FOLDER_DB,"train_database"))
