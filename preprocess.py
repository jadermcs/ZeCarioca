import os
import glob
import tqdm
import random
import pandas as pd


path = "clear_threads/"
files = glob.glob(path+"*_*.tsv")
random.seed(42)
random.shuffle(files)

frames = []
counter = 0

for filein in tqdm.tqdm(files):
    counter += 1
    df = pd.read_csv(filein, delimiter='\t', quoting=3, header=None,
                     names=["timestamp", "id", "text"])
    df.drop(columns=['timestamp'], inplace=True)
    df['reply'] = df['text'].shift(-1)
    df = df.iloc[:-1]
    os.remove(filein)
    frames.append(df)
    if counter % 1000 == 0:
        out = pd.concat(frames, axis=0, ignore_index=True)
        out.dropna(axis=0, inplace=True)
        frames = []
        out.to_parquet(path+str(counter)+'.parquet')
        del out
