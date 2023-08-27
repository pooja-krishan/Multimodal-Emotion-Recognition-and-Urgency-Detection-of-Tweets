import librosa 
import os
import glob
import math
import pickle

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def extract_features(signal, sr):
    # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
    rms = librosa.feature.rms(signal + 0.0001)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # mel frequnecy cepstral coefficients
    # mfcc = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    stft = np.abs(librosa.stft(signal))

    pitches, magnitudes = librosa.piptrack(signal, sr=sr, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])
    # print('pitch: ', pitch)
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    flatness = np.mean(librosa.feature.spectral_flatness(y=signal))

    cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
    centroid = cent / np.sum(cent)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    ext_features = np.array([
        rms_mean, rms_std, pitch_tuning_offset, pitch_mean, pitch_std,
        flatness, centroid_mean, centroid_std])

    extract_features = np.concatenate((ext_features, mfccs, delta_mfccs, delta2_mfccs))

    return extract_features

columns = ['wav_file', 'label', 'rms_mean', 'rms_std', 'pitch_tuning_offset', 'pitch_mean', 'pitch_std', 'flatness', 'centroid_mean', 'centroid_std']
columns_plus = columns + [(lambda x: "mfcc_" + str(x))(x) for x in range(1, 14)] + [(lambda x: "delta_mfccs_" + str(x))(x) for x in range(1, 14)] + [(lambda x: "delta2_mfccs_" + str(x))(x) for x in range(1, 14)]
df_features = pd.DataFrame(columns=columns_plus)


labels_df = pd.read_csv('./iemocap_label.csv')
sr = 22050

with open('audio_vectors_5.pkl', 'rb') as f:
    audio_vectors = pickle.load(f)

for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses05')].iterrows()):
    try:
        wav_file_name = row['wav_file']
        label = row['emotion']
        y = audio_vectors[wav_file_name]
        features_all = list(extract_features(y, sr))
        feature_list = [wav_file_name, label] + features_all
        df_features = df_features.append(pd.DataFrame(feature_list, index=columns_plus).transpose(), ignore_index=True)
    except:
        print('An exception occured for {}'.format(wav_file_name))

print(df_features.shape)
print(df_features.head(5))

df_features.to_csv("df_features_session5.csv")



