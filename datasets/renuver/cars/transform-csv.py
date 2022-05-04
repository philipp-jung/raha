import os
import pandas as pd

for root, _, files in os.walk('./'):
    for file in files:
        end = file.split('.')[-1]
        if end == 'csv':
            pd.read_csv(root+file, sep=';').to_csv(root+file)
