from typing import Union
import datawig
from datawig.autogluon_imputer import TargetColumnException
import pandas as pd
import numpy as np

from IPython.core.debugger import set_trace


def train_cleaning_model(df_dirty: pd.DataFrame,
                         df_test: pd.DataFrame,
                         label: int,
                         verbosity: int = 0,
                         time_limit=90) -> Union[datawig.AutoGluonImputer, None]:
    """
    Train an autogluon model for the purpose of cleaning data.
    Optionally, you can set verbosity to control how much output AutoGluon
    produces during training.
    Returns a predictor object.
    """
    lhs = list(df_dirty.columns)
    del lhs[label - 1]
    rhs = df_dirty.columns[label]

    try:
        imputer = datawig.AutoGluonImputer(
            model_name=f'{label}-imputing-model',
            input_columns=lhs,
            output_column=rhs,
            force_multiclass=False,
            verbosity=verbosity,
            label_count_threshold=10
        )

        imputer.fit(train_df=df_dirty,
                    test_df=df_test,
                    time_limit=time_limit,
                    # hyperparameters=kwargs.get('hyperparameters', None),
                    # hyperparameter_tune_kwargs=kwargs.get('hyperparameter_tune_kwargs', None),
                    # preset='best_quality'
                    )
    except TargetColumnException:
        print(f"Failed to train an imputer model for column {label}")
        print(TargetColumnException)
        imputer = None

    return imputer


def make_cleaning_prediction(df_dirty,
                             probas,
                             target_col) -> pd.Series:
    """
    When cleaning, the task is defined such that the ids of the dirty rows
    are known. When cleaning, it is thus always incorrect to return the class
    that is known to be dirty.

    The strategy I use here is:
    1) Check the most probable class. If the class is not the dirty class,
    return it.
    2) If the returned class is the dirty class, return the second most likely
    class.
    """
    dirty_label_mask = pd.DataFrame()
    for c in probas.columns:
        dirty_label_mask[c] = df_dirty[target_col] == c
    probas[dirty_label_mask] = -1  # dirty labels are never chosen

    i_classes = np.argmax(probas.values, axis=1)
    cleaning_predictions = [probas.columns[i] for i in i_classes]
    return pd.Series(cleaning_predictions)


def imputation_based_corrector(d):
    df = d.repaired_dataframe.copy().astype(str)
    df_test = d.get_df_from_labeled_tuples()
    results = {c: [] for c in range(d.dataframe.shape[1])}

    for i_col, col in enumerate(df.columns):
        imputer = train_cleaning_model(df, df_test, label=i_col)
        if imputer is not None:
            error_rows = [row for (row, _) in list(d.get_actual_errors_dictionary().keys())]
            df_rhs_errors = df.iloc[error_rows, :]
            probas = imputer.predict(df_rhs_errors,
                                     precision_threshold=.01,
                                     return_probas=True)

            se_predicted = make_cleaning_prediction(df_rhs_errors,
                                                    probas,
                                                    col)

            n_error = 0
            for error_row, error_col in list(d.get_actual_errors_dictionary().keys()):
                if i_col == error_col:
                    results_dict = {}
                    results_dict[se_predicted.iloc[n_error]] = 1
                    results[i_col].append(results_dict)
                n_error = n_error+1
    return results
