from typing import Tuple, List, Dict
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from jenga.corruptions.generic import SwappedValues, CategoricalShift

openml_ids_binary = [725, 310, 1046, 823, 137, 42493, 4135, 251, 151, 40922]
openml_ids_multiclass = [40498, 30, 1459, 1481, 184, 375, 32, 41027, 6, 40685]

fractions = [0.01, 0.05, 0.1, 0.3, 0.5]
corruptions = {# 'swapped_values': SwappedValues,
               'categorical_shift': CategoricalShift}


def dataset_paths(data_id: int, corruption: str, error_fraction: int) -> Tuple[Path, Path]:
    directory = Path(f'openml/{data_id}')
    directory.mkdir(exist_ok=True)
    clean_path = directory / 'clean.csv'
    corrupt_path = directory / f'{corruption}_{int(100*error_fraction)}'
    return clean_path, corrupt_path


def fetch_corrupt_dataset(data_id: int) -> List[Dict]:
    """
    Goal of this exercise is to showcase that the imputer approach can do meaningful things
    with continuous features whereas baran and the likes just fail miserably.
    """
    res = fetch_openml(data_id=data_id, as_frame=True)

    df = res['frame']
    clean_path, _ = dataset_paths(data_id, '', 0)
    df.to_csv(clean_path, index=False)
    metadata = []

    for corruption_name, corruption in corruptions.items():
        for fraction in fractions:
            df_corrupted = df.copy()
            for column in df_corrupted.columns:
                corruption_instance = corruption(column=column, fraction=fraction)
                df_corrupted = corruption_instance.transform(df_corrupted)

            error_pct = int(round((df != df_corrupted).sum().sum() / df_corrupted.size, 2) * 100)
            metadata.append({'dataset_id': data_id,
                             'error_type': corruption_name,
                             'fraction':  fraction,
                             'error_pct': error_pct})

            clean_path, corrupt_path = dataset_paths(data_id, corruption_name, fraction)
            df_corrupted.to_parquet(str(corrupt_path) + '.parquet', index=False)
            df_corrupted.to_csv(str(corrupt_path) + '.csv', index=False)

    return metadata


if __name__ == '__main__':
    metadatas = []
    for dataset_id in openml_ids_binary:
        metadata = fetch_corrupt_dataset(dataset_id)
        metadata = [{**x, 'dataset_type': 'binary classification'} for x in metadata]
        metadatas.extend(metadata)
    for dataset_id in openml_ids_multiclass:
        metadata = fetch_corrupt_dataset(dataset_id)
        metadata = [{**x, 'dataset_type': 'multiclass classification'} for x in metadata]
        metadatas.extend(metadata)
    errors = pd.DataFrame(metadatas)
    errors.to_csv("error_stats.csv", index=False)
