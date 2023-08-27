import pandas as pd
import re
import os
import glob
from pathlib import Path


useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)


def get_file_content(path):
    with open(path) as f:
        file_content = f.read()
            
    info_lines = re.findall(useful_regex, file_content)
    # print(len(info_lines))
    return info_lines


def collect_data(info_lines):

    start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []
    
    for line in info_lines[1:]:  # the first line is a header
        start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
        start_time, end_time = start_end_time[1:-1].split('-')
        val, act, dom = val_act_dom[1:-1].split(',')
        val, act, dom = float(val), float(act), float(dom)
        start_time, end_time = float(start_time), float(end_time)
        start_times.append(start_time)
        end_times.append(end_time)
        wav_file_names.append(wav_file_name)
        emotions.append(emotion)
        vals.append(val)
        acts.append(act)
        doms.append(dom)   
       
    return  start_times, end_times, wav_file_names, emotions, vals, acts, doms


def make_df(start_times, end_times, wav_file_names, emotions, vals, acts, doms):
    df_temp = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

    df_temp['start_time'] = start_times
    df_temp['end_time'] = end_times
    df_temp['wav_file'] = wav_file_names
    df_temp['emotion'] = emotions
    df_temp['val'] = vals
    df_temp['act'] = acts
    df_temp['dom'] = doms

    return df_temp


folder_path = "./Session5/dialog/EmoEvaluation"
files = Path(folder_path).glob("*.txt")

df = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

for file in files:
    path = os.path.join(file)
    info_lines = get_file_content(path)
    start_times, end_times, wav_file_names, emotions, vals, acts, doms = collect_data(info_lines)
    df_temp = make_df(start_times, end_times, wav_file_names, emotions, vals, acts, doms)
    df = pd.concat([df, df_temp], ignore_index=True, sort=False)

# df.to_csv('session5.csv')

# Concatenate csv files of the five sessions
csv_files_path = os.path.join("./", "*.csv")
csv_files = glob.glob(csv_files_path)
iemocap = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
iemocap.to_csv('iemocap.csv')

print(iemocap['emotion'].value_counts())