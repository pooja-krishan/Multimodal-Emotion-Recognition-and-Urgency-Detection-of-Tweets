import librosa 
import os
import glob
import math
import pickle

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

labels_df = pd.read_csv('./iemocap_label.csv')

# build audio vectors
audio_vectors = {}
sr = 22050

folder_path = "./Session5/dialog/wav"
files = Path(folder_path).glob("*.wav")

for file in tqdm(files):
    path = os.path.join(file)
    wav_vector, _sr = librosa.load(file, sr=sr)
    wav_file_path, file_format = path.split('.')    
    wav_file = wav_file_path.split("/")[-1]
    
    for index, row in labels_df[labels_df['wav_file'].str.contains(wav_file)].iterrows():

        start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
        start_frame = math.floor(start_time * sr)
        end_frame = math.floor(end_time * sr)
        truncated_wav_vector = wav_vector[start_frame:end_frame + 1]
        audio_vectors[truncated_wav_file_name] = truncated_wav_vector

print(len(audio_vectors))

with open('audio_vectors_5.pkl', 'wb') as f:
    pickle.dump(audio_vectors, f)
