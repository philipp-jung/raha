import pandas as pd
import numpy as np
import sklearn
from typing import List, Dict, Tuple


class VotingClassifier(object):
    """
    Implements a voting classifier for pre-trained classifiers. Uses soft-voting to get the right label.
    Inspired by https://stackoverflow.com/a/50295289.
    """

    def __init__(self, estimators: List[Tuple[str, sklearn.pipeline.Pipeline]]):
        self.estimators = estimators

    def predict(self, X):
        # get values
        Y = np.zeros([X.shape[0], len(self.estimators)], dtype=int)
        for i, (name, clf) in enumerate(self.estimators):
            Y[:, i] = clf.predict(X)
        # voting
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.argmax(np.bincount(Y[i, :]))
        return y


def set_binary_cleaning_suggestions(predicted_labels, all_error_correction_suggestions, user_corrected_cells, d):
    """
    After the Classifier has done its work, take its outputs and set them as the correction in the global state d.
    If there is user input available for a cell, it always takes precedence over the ML output.
    """
    for index, predicted_label in enumerate(predicted_labels):
        if predicted_label:
            error_cell, predicted_correction = all_error_correction_suggestions[index]
            if error_cell in user_corrected_cells:  # use user input if available
                d.corrected_cells[error_cell] = user_corrected_cells[error_cell]
            else:
                d.corrected_cells[error_cell] = predicted_correction


def handle_edge_cases(x_train, x_test, y_train, d) -> Tuple[bool, List]:
    """
    Depending on the dataset and how much data has been labeled by the user, the data used to formulate the ML problem
    can be such that it leads to an invalid ML problem.
    To prevent the application from stopping unexpectedly, these edge-cases are handled here.

    @return: A tuple whose first position is a boolean, indicating if the ML problem should be started. The second
    position is a list of predicted labels. Which is used when the ML problem should not be started.
    """
    if sum(y_train) == 0:  # no correct suggestion was created for any of the error cells.
        return False, []  # nothing to do, need more user-input to work.
    elif sum(y_train) == len(y_train):  # no incorrect suggestion was created for any of the error cells.
        return False, np.ones(len(x_test))

    elif len(d.labeled_tuples) == 0:  # no training data is available because no user labels have been set.
        for cell in d.pair_features:
            correction_dict = d.pair_features[cell]
            if len(correction_dict) > 0:
                # select the correction with the highest sum of features.
                max_proba_feature = \
                    sorted([v for v in correction_dict.items()], key=lambda x: sum(x[1]), reverse=True)[0]
                d.corrected_cells[cell] = max_proba_feature[0]
        return False, []

    elif len(x_train) > 0 and len(x_test) == 0:  # len(x_test) == 0 because all rows have been labeled.
        return False, []  # trivial case - use manually corrected cells to correct all errors.

    elif len(x_train) == 0:
        return False, []  # nothing to learn because x_train is empty.

    elif len(x_train) > 0 and len(x_test) > 0:
        return True, []
    else:
        raise ValueError("Invalid state.")


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
        if len(correction_suggestions) > 0:
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
