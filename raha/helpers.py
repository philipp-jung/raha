import json
import time
import random
import sqlite3
import numpy as np
from typing import Union, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict

import openai
import pandas as pd
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

    def original_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        for (row, col), error_value in self.detected_cells.items():
            column_errors[col].append((row, col))
        return column_errors

    @property
    def updated_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        for (row, col), error_value in self.detected_cells.items():
            if (row, col) not in self.corrected_cells:
                column_errors[col].append((row, col))
        return column_errors

    def original_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        for (row, col), error_value in self.detected_cells.items():
            row_errors[row].append((row, col))
        return row_errors

    def updated_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        for (row, col), error_value in self.detected_cells.items():
            if (row, col) not in self.corrected_cells:
                row_errors[row].append((row, col))
        return row_errors


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
            flat_store[model] = self.correction_store[model]
        return flat_store

    @property
    def available_corrections(self) -> List[str]:
        return list(self.correction_store.keys())

    def features(self) -> List[str]:
        """Return a list describing the features the Corrections come from."""
        return list(self.correction_store.keys())

    def get(self, model_name: str) -> Dict:
        """
        For each error-cell, there will be a list of {corrections_suggestion: probability} returned here. If there is no
        correction made for that cell, the list will be empty.
        """
        return self.correction_store[model_name]

    def assemble_pair_features(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """Return an object as d.pair_features has been in Baran."""
        flat_corrections = self.flat_correction_store()
        pair_features = defaultdict(dict)
        for mi, model in enumerate(flat_corrections):
            for cell in flat_corrections[model]:
                for correction, pr in flat_corrections[model][cell].items():
                    if correction not in pair_features[cell]:
                        features = list(flat_corrections.keys())
                        pair_features[cell][correction] = np.zeros(len(features))
                    pair_features[cell][correction][mi] = pr
        return pair_features


def connect_to_cache() -> sqlite3.Connection:
    """
    Connect to the cache for LLM prompts.
    @return: a connection to the sqlite3 cache.
    """
    conn = sqlite3.connect('cache.db')
    return conn


def fetch_cached_llm(dataset: str, error_cell: Tuple[int, int], prompt: str, correction_model_name: str) -> Tuple[dict, dict, dict]:
    """
    Sending requests to LLMs is expensive (time & money). We use caching to mitigate that cost. As primary key for
    a correction serves (dataset_name, error_cell, correction_model_name). This is imperfect, but a reasonable
    approximation: Since the prompt-generation itself as well as its dependencies are non-deterministic, the prompt
    cannot serve as part of the primary key.

    If no correction is available in the cache, a request to the LLM is sent.

    @param dataset: name of the dataset that is cleaned.
    @param error_cell: (row, column) position of the error.
    @param prompt: prompt that is sent to the LLM.
    @param correction_model_name: "llm_vicinity" or "llm_value".
    @return: correction_tokens, token_logprobs, and top_logprobs.
    """
    conn = connect_to_cache()
    cursor = conn.cursor()
    cursor.execute("""SELECT
                        correction_tokens,
                        token_logprobs,
                        top_logprobs
                      FROM cache
                      WHERE dataset=? AND row=? AND column=? AND correction_model=?""",
                   (dataset, error_cell[0], error_cell[1], correction_model_name))
    result = cursor.fetchone()
    conn.close()
    if result is not None:
        return json.loads(result[0]), json.loads(result[1]), json.loads(result[2])  # access the correction
    else:
        return fetch_llm(prompt, dataset, error_cell, correction_model_name)


def fetch_llm(prompt: str, dataset: str, error_cell: Tuple[int, int], correction_model_name: str) -> Tuple[dict, dict, dict]:
    """
    Send request to openai to get a prompt resolved. Write result into cache.
    """
    if prompt is None:
        return {}, {}, {}

    retries = 0
    while True:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                logprobs=3
            )
            correction_tokens = response['choices'][0]['logprobs']['tokens']
            token_logprobs = response['choices'][0]['logprobs']['token_logprobs']
            top_logprobs = response['choices'][0]['logprobs']['top_logprobs']
            break
        except openai.error.RateLimitError as e:
            if retries > 5:
                print('Exceeded maximum number of retries. Skipping correction. The OpenAI API appears unreachable:')
                print(e)
                return {}, {}, {}
            delay = (2 ** retries) + random.random()
            print(f"Rate limit exceeded, retrying in {delay} seconds.")
            time.sleep(delay)
            retries += 1

    row, column = error_cell
    conn = connect_to_cache()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO cache
               (dataset, row, column, correction_model, correction_tokens, token_logprobs, top_logprobs)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (dataset, row, column, correction_model_name, json.dumps(correction_tokens), json.dumps(token_logprobs), json.dumps(top_logprobs))
    )
    conn.commit()
    conn.close()
    return correction_tokens, token_logprobs, top_logprobs


def construct_llm_corrections(dictionaries: List[Dict[str, float]], current_sentence: str = '',
                        current_probability: float = 0) -> List[Dict[str, float]]:
    """
    Construct all possible sentences from a OpenAI davinci-003 API response. Returns a list of {correction: log_pr}.
    @param dictionaries:
    @param current_sentence:
    @param current_probability:
    @return:
    """
    if not dictionaries:
        # Base case: reached the end of the list, return the constructed correction and its probability as a dictionary
        return [{'correction': current_sentence, 'logprob': current_probability}]
    else:
        # Get the first dictionary in the list
        current_dict = dictionaries[0]
        result = []

        for token, probability in current_dict.items():
            # Recursively call the function for the remaining dictionaries
            sentences = construct_llm_corrections(dictionaries[1:], current_sentence + token,
                                            current_probability + probability)
            result.extend(sentences)

        return result


def llm_response_to_corrections(correction_tokens: List[str], token_logprobs: dict, top_logprobs: List[Dict[str, float]]) -> Dict[str, float]:
    # if len(correction_tokens) <= 7:
    #     corrections = construct_llm_corrections(top_logprobs)
    #     # filter out all corrections with pr < 1% <=> logprob > -4.60517.
    #     top_corrections = {c['correction']: np.exp(c['logprob']) for c in corrections if c['logprob'] > -4.60517}
    #     return top_corrections
    correction = ''.join(correction_tokens)
    return {correction: np.exp(sum(token_logprobs))}


def error_free_row_to_prompt(df: pd.DataFrame, row: int, column: int) -> Tuple[str, str]:
    """
    Turn an error-free dataframe-row into a string, and replace the error-column with an <Error> token.
    Return a tuple of (stringified_row, correction). Be mindful that correction is only the correct value if
    the row does not contain an error to begin with.
    """
    values = df.iloc[row, :].values
    row_values = [f"{x}," if i != column else "<Error>," for i, x in enumerate(values)]
    row = ''.join(row_values)[:-1]
    correction = df.iloc[row, column]
    return row, correction
