import glob
import os
import pandas as pd

files = glob.glob("clear_threads/*_*.tsv")

frames = []
counter = 1

for file in files:
    counter += 1
    df = pd.read_csv(file, delimiter='\t', quoting=3, header=None, names=["timestamp", "id", "text"])
    df.drop(columns=['timestamp'], inplace=True)
    df['reply'] = df['text'].shift(-1)
    df = df.iloc[:-1]
    os.remove(file)
    frames.append(df)
    if counter % 1000 == 0:
        out = pd.concat(frames, axis=0, ignore_index=True)
        out.dropna(axis=0, inplace=True)
        frames = []
        out.to_parquet(file.replace('.tsv', '.parquet'))
        del out
