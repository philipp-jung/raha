import random
import pandas as pd
from typing import List, Dict, Tuple


def generate_train_test_data(column_errors: Dict,
                             labeled_cells: Dict[Tuple[int, int], List],
                             pair_features: Dict[Tuple[int, int], Dict[str, List]],
                             df_dirty: pd.DataFrame,
                             synthesize_train_data: int = 0):
    # TODO continue debugging here

    x_train = []  # train feature vectors
    y_train = []  # train labels
    x_test = []  # test features vectors
    all_error_correction_suggestions = []  # all cleaning suggestions for all errors flattened in a list
    corrected_cells = {}  # take user input as a cleaning result if available


    synth_cell_population = [cell for cell in pair_features if cell not in column_errors]
    synth_column_errors = random.sample(synth_cell_population, k=synthesize_train_data)

    for error_cell in column_errors:
        correction_suggestions = pair_features.get(error_cell, [])
        if error_cell in labeled_cells and labeled_cells[error_cell][0] == 1:
            # If an error-cell has been labeled by the user, use it to create the training dataset.
            # The second condition is always true if error detection and user labeling work without an error.
            for suggestion in correction_suggestions:
                x_train.append(pair_features[error_cell][suggestion])  # puts features into x_train
                suggestion_is_correction = (suggestion == labeled_cells[error_cell][1])
                y_train.append(int(suggestion_is_correction))
                corrected_cells[error_cell] = labeled_cells[error_cell][1]  # user input as cleaning result
        else:  # put all cells that contain an error without user-correction in the "test" set.
            for suggestion in correction_suggestions:
                x_test.append(pair_features[error_cell][suggestion])
                all_error_correction_suggestions.append([error_cell, suggestion])
                suggestion_is_correction = (suggestion == labeled_cells[error_cell][1])
                y_train.append(int(suggestion_is_correction))

    for synth_cell in synth_column_errors:
        correction_suggestions = pair_features.get(synth_cell, [])
        for suggestion in correction_suggestions:
            x_train.append(pair_features[synth_cell][suggestion])
            all_error_correction_suggestions.append([synth_cell, suggestion])
            suggestion_is_correction = (suggestion == df_dirty.iloc[error_cell])
            y_train.append(int(suggestion_is_correction))

    return x_train, y_train, x_test, corrected_cells, all_error_correction_suggestions
