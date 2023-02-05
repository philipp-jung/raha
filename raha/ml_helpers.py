import math
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def parity_train_test_data(x_train,
                           y_train,
                           possible_synthetic_cells):
    """
    Establish a 50/50 parity synth_data : user_input as suggested by Thorsten in 2022W45.
    """
    MIN_POSITIVE_SAMPLES = 8  # willkürlich gesetzt zzT

    n_synthetic_cells = min(MIN_POSITIVE_SAMPLES, math.ceil(len(possible_synthetic_cells) / 2))
    synthetic_error_cells = random.sample(possible_synthetic_cells, n_synthetic_cells)
    target_sum_y_train = n_synthetic_cells + sum(y_train)

    if sum(y_train) > 0:
        # if user input exists, duplicate it to keep 50/50 parity with synthetic data.
        i = 0
        while sum(y_train) + y_train[i] <= target_sum_y_train:  # get all 0s before the next 1 in y_train
            if i == len(y_train):
                i = 0
            x_train.append(x_train[i])
            y_train.append(y_train[i])
            i += 1
    return x_train, y_train, synthetic_error_cells


def factor_train_test_data(synth_error_factor: float, n_labelled_tuples: int, possible_synthetic_cells):
    """
    The first sampling algorithms as defined in 2022W44.
    """
    synthetic_error_cells = []
    n_synthetic_cells = math.floor(n_labelled_tuples * (synth_error_factor - 1))
    if n_synthetic_cells > 0:
        try:
            synthetic_error_cells = random.sample(possible_synthetic_cells, k=n_synthetic_cells)
        except ValueError:  # I want to sample more synth. training data than is available.
            synthetic_error_cells = possible_synthetic_cells
    return synthetic_error_cells


def generate_train_test_data(column_errors: Dict[int, List[Tuple[int, int]]],
                             labeled_cells: Dict[Tuple[int, int], List],
                             pair_features: Dict[Tuple[int, int], Dict[str, List]],
                             df_dirty: pd.DataFrame,
                             synth_pair_features: Dict[Tuple[int, int], Dict[str, List]],
                             column: int):
    """
    If the product of synth_error_factor * (number of error correction suggestions) is larger than the number of
    error correction suggestions, the difference gets filled with synthesized errors.
    """
    x_train = []  # train feature vectors
    y_train = []  # train labels
    x_test = []  # test features vectors
    all_error_correction_suggestions = []  # all cleaning suggestions for all errors flattened in a list
    corrected_cells = {}  # take user input as a cleaning result if available

    for error_cell in column_errors[column]:
        correction_suggestions = pair_features.get(error_cell, [])
        if error_cell in labeled_cells and labeled_cells[error_cell][0] == 1:
            # If an error-cell has been labeled by the user, use it to create the training dataset.
            # The second condition is always true if error detection and user labeling work without an error.
            for suggestion in correction_suggestions:
                x_train.append(pair_features[error_cell][suggestion])  # Puts features into x_train
                suggestion_is_correction = (suggestion == labeled_cells[error_cell][1])
                y_train.append(int(suggestion_is_correction))
                corrected_cells[error_cell] = labeled_cells[error_cell][1]  # user input as cleaning result
        else:  # put all cells that contain an error without user-correction in the "test" set.
            for suggestion in correction_suggestions:
                x_test.append(pair_features[error_cell][suggestion])
                all_error_correction_suggestions.append([error_cell, suggestion])

    if len(synth_pair_features) == 0:
        # for debugging
        # print('No features exist to synthesize training data with.')
        return x_train, y_train, x_test, corrected_cells, all_error_correction_suggestions

    for synth_cell in synth_pair_features:
        if synth_cell[1] == column:
            correction_suggestions = synth_pair_features.get(synth_cell, [])
            for suggestion in correction_suggestions:
                x_train.append(synth_pair_features[synth_cell][suggestion])
                suggestion_is_correction = (suggestion == df_dirty.iloc[synth_cell])
                y_train.append(int(suggestion_is_correction))

    return x_train, y_train, x_test, corrected_cells, all_error_correction_suggestions


def multi_generate_train_test_data(column_errors: Dict[int, List[Tuple[int, int]]],
                                   labeled_cells: Dict[Tuple[int, int], List],
                                   pair_features: Dict[Tuple[int, int], Dict[str, List]],
                                   synth_pair_features: Dict[Tuple[int, int], Dict[str, List]],
                                   df_dirty: pd.DataFrame,
                                   column: int):
    """
    Generate data for the machine-learning problem modeling the issue as a multiclass classification.
    We also leverage rows that do not contain errors to get more training data, called "synthetic" data.
    If the product of synth_error_factor * (number of error correction suggestions) is larger than the number of
    error correction suggestions, the difference gets filled with synthesized errors.
    """
    x_train = []  # train feature vectors
    y_train = []  # train labels
    x_test = []  # test features vectors
    corrected_cells = {}  # take user input as a cleaning result if available

    correction_order = []
    for error_cell in column_errors[column]:
        correction_suggestions = pair_features.get(error_cell, [])
        if correction_suggestions != []:
            correction_order = list(correction_suggestions.keys())
        if list(correction_suggestions.keys()) != [] and correction_order != list(correction_suggestions.keys()):
            raise ValueError('Should not be possible.')
        x = np.concatenate([pair_features[error_cell][suggestion] for suggestion in correction_suggestions])

        if error_cell in labeled_cells and labeled_cells[error_cell][0] == 1:
            # If an error-cell has been labeled by the user, use it to create the training dataset.
            y = labeled_cells[error_cell][1]  # user input as label
            corrected_cells[error_cell] = y
            x_train.append(x)
            y_train.append(y)

        else:  # put all cells that contain an error without user-correction in the "test" set.
            x_test.append(x)

    if len(synth_pair_features) == 0 or len(y_train) == 0:
        # abort if no synth features were created or no non-synth features were created.
        return x_train, y_train, x_test, corrected_cells

    for synth_cell in synth_pair_features:
        if synth_cell[1] == column:
            correction_suggestions = synth_pair_features.get(synth_cell, [])
            if list(correction_suggestions.keys()) != [] and list(correction_suggestions.keys()) != correction_order:
                raise ValueError('Should not be possible.')
            x = np.concatenate([pair_features[error_cell][suggestion] for suggestion in correction_suggestions])
            y = df_dirty.iloc[synth_cell]  # synth cell generation garantiert, dass die Zeile fehlerfrei ist.
            x_train.append(x)
            y_train.append(y)

    return x_train, y_train, x_test, corrected_cells
