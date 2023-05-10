import json
import numpy as np
from typing import Union, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict

import openai
from openai_key import key

openai.api_key = key

def get_data_dict(
    dataset_name: Union[str, int],
    error_fraction: Union[int, None] = None,
    version: Union[int, None] = None,
    error_class: Union[str, None] = None
) -> Dict:
    """
    I currently use four different sources of datasets: the original Baran paper, the RENUVER paper, datasets that
    I assemble from OpenML, and hand-selected datasets from the UCI website. Depending on the source, the datasets
    differ:
    - Datasets from the Baran paper are uniquely identified by their name.
    - Datasets from the RENUVER paper are uniquely identified by their name, error_fraction and version in [1, 5].
    - Datasets that I generate from OpenML are identified by their name and error_fraction.
    - Datasets that I generate from UCI are identified by their name and error_fraction.
    @param dataset_name: Name of the dataset.
    @param error_fraction: Baran datasets don't have this. The % of cells containing errors in RENUVER datasets and the
    Ruska-corrupted datasets. The % of values in a column in OpenML datasets. Value between 0 - 100.
    @param version: generating errors in Jenga is not deterministic. So it makes sense to create a couple of versions
    to avoid outlier corruptions.
    @return: A data dictionary as expected by Baran. I hacked this to make the imputer feature generator use a version
    of the dataset that comes with dtypes.
    """
    dataset_name = str(dataset_name)
    openml_dataset_ids = [
        725,
        310,
        1046,
        823,
        137,
        42493,
        4135,
        251,
        151,
        40922,
        40498,
        30,
        1459,
        1481,
        184,
        375,
        32,
        41027,
        6,
        40685,
    ]
    openml_dataset_ids = [str(x) for x in openml_dataset_ids]

    if dataset_name in ["bridges", "cars", "glass", "restaurant"]:  # renuver dataset
        data_dict = {
            "name": dataset_name,
            "path": f"../datasets/renuver/{dataset_name}/{dataset_name}_{error_fraction}_{version}.csv",
            "clean_path": f"../datasets/renuver/{dataset_name}/clean.csv",
        }

    elif dataset_name in [
        "beers",
        "flights",
        "hospital",
        "tax",
        "rayyan",
        "toy",
        "debug",
        "synth-debug"
    ]:  # Baran dataset
        data_dict = {
            "name": dataset_name,
            "path": f"../datasets/{dataset_name}/dirty.csv",
            "clean_path": f"../datasets/{dataset_name}/clean.csv",
        }

    elif dataset_name in openml_dataset_ids:  # OpenML dataset
        if error_class is None:
            raise ValueError('Please specify the error class with which the openml dataset has been corrupted.')
        data_dict = {
            "name": dataset_name,
            "path": f"../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.csv",
            "parquet_path": f"../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.parquet",
            "clean_path": f"../datasets/openml/{dataset_name}/clean.csv",
        }

    elif dataset_name in [
        "adult",
        "breast-cancer",
        "letter",
        "nursery",
    ]:  # ruska-corrupted datasets
        data_dict = {
            "name": dataset_name,
            "path": f"../datasets/{dataset_name}/MCAR/dirty_{error_fraction}.csv",
            "clean_path": f"../datasets/{dataset_name}/clean.csv",
        }

    else:
        raise ValueError("Dataset not supported.")
    return data_dict


def assemble_cleaning_suggestion(
    transformation_string: str, model_name: str, old_value: str
) -> Union[str, None]:
    """
    Use the operation encoded in transform_string and the model_name to identify the operation to transform old_value
    into a cleaning suggestion
    @param transformation_string: the encoded transformation.
    @param model_name: operation name, which is adder, remover, replacer or swapper.
    @param old_value: the erroneous value.
    @return: a cleaning suggestion.
    """
    index_character_dictionary = {i: c for i, c in enumerate(old_value)}
    transformation = json.loads(transformation_string)
    for change_range_string in transformation:
        change_range = json.loads(change_range_string)
        if model_name in ["remover", "replacer"]:
            for i in range(change_range[0], change_range[1]):
                index_character_dictionary[i] = ""
        if model_name in ["adder", "replacer"]:
            ov = (
                ""
                if change_range[0] not in index_character_dictionary
                else index_character_dictionary[change_range[0]]
            )
            index_character_dictionary[change_range[0]] = (
                transformation[change_range_string] + ov
            )
    new_value = ""
    try:
        for i in range(len(index_character_dictionary)):
            new_value += index_character_dictionary[i]
    except KeyError:  # not possible to transform old_value.
        new_value = None
    return new_value


@dataclass
class ErrorPositions:
    detected_cells: Dict[Tuple[int, int], Union[str, float, int]]
    table_shape: Tuple[int, int]
    corrected_cells: Dict[Tuple[int, int], Tuple[int, str]]

    def count_column_errors(self, updated: bool) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        if not updated:
            for (row, col), error_value in self.detected_cells.items():
                column_errors[col].append((row, col))
        elif updated:
            for (row, col), error_value in self.detected_cells.items():
                if (row, col) not in self.corrected_cells:
                    column_errors[col].append((row, col))
        return column_errors

    def count_row_errors(self, updated: bool) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        if not updated:
            for (row, col), error_value in self.detected_cells.items():
                row_errors[row].append((row, col))
        elif updated:
            for (row, col), error_value in self.detected_cells.items():
                if (row, col) not in self.corrected_cells:
                    row_errors[row].append((row, col))
        return row_errors

    @property
    def original_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        return self.count_column_errors(updated=False)

    @property
    def updated_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        return self.count_column_errors(updated=True)

    @property
    def original_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        return self.count_row_errors(updated=False)

    @property
    def updated_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        return self.count_row_errors(updated=True)


class Corrections:
    """
    Store correction suggestions provided by the correction models in correction_store. In _feature_generator_process
    it is guaranteed that all models return something for each error cell -- if there are no corrections made, that
    will be an empty list. If a correction or multiple corrections has/have been made, there will be a list of
    correction suggestions and feature vectors.
    """

    def __init__(self, model_names: List[str]):
        self.correction_store = {name: dict() for name in model_names}

    def flat_correction_store(self):
        flat_store = {}
        for model in self.correction_store:
            if not model.startswith('vicinity'):
                flat_store[model] = self.correction_store[model]
            else:
                for cell in self.correction_store[model]:
                    for n_fd, corrections_pr in enumerate(self.correction_store[model][cell]):
                        vicinity_model_name = f'{model}_{n_fd}'
                        if flat_store.get(vicinity_model_name) is None:
                            flat_store[vicinity_model_name] = {}
                        flat_store[vicinity_model_name][cell] = corrections_pr
        return flat_store

    @property
    def available_corrections(self) -> List[str]:
        return list(self.correction_store.keys())

    def features(self) -> List[str]:
        """Return a list describing the features the Corrections come from."""
        return list(self.flat_correction_store().keys())

    def get(self, model_name: str) -> Dict:
        """
        For each error-cell, there will be a list of {corrections_suggestion: probability} returned here. If there is no
        correction made for that cell, the list will be empty.
        """
        return self.correction_store[model_name]

    def assemble_pair_features(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """Return an object as d.pair_features has been in Baran."""
        pair_features = defaultdict(dict)
        flat_corrections = self.flat_correction_store()
        for mi, model in enumerate(flat_corrections):
            for cell in flat_corrections[model]:
                for correction, pr in flat_corrections[model][cell].items():
                    if correction not in pair_features[cell]:
                        features = list(flat_corrections.keys())
                        pair_features[cell][correction] = np.zeros(len(features))
                    pair_features[cell][correction][mi] = pr
        return pair_features


def fetch_llm_value_cleaning(prompt: str) -> Dict[str, float]:
    """
    Send request to openai to get get value cleaning resolved.
    @return: A dictionary {correction_suggestion, pr}.
    """
    if prompt is None:
        return {}
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    try:
        text = response['choices'][0]['text'].strip()
        correction_suggestion = json.loads(text)
    except json.decoder.JSONDecodeError:
        print(f'Was not able to decode GPT correction {text}.')
        return {}
    # if response['choices'][0]['logprobs'] is None:
    # Is always None: logprobs are not a thing anymore annoyingly.
    if correction_suggestion is None:
        return {}
    return {correction_suggestion: 1.0}
