########################################
# Dataset
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# October 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import re
import html
from typing import Union, Dict, Tuple

import pandas as pd
########################################


########################################
class Dataset:
    """
    The dataset class.
    """

    def __init__(self, dataset_dictionary, n_rows=None):
        """
        The constructor creates a dataset.

        If n_rows is specified, the dataset gets subsetted to the first
        n_rows rows.
        """
        self.name = dataset_dictionary["name"]
        self.path = dataset_dictionary["path"]
        self.dataframe = self.read_csv_dataset(dataset_dictionary["path"])
        if dataset_dictionary.get('parquet_path') is not None:
            self.typed_dataframe = self.read_parquet_dataset(dataset_dictionary.get('parquet_path'))
        else:  # no .parquet with typed data available. fall back to .csv file.
            self.typed_dataframe = self.dataframe
        if n_rows is not None:
            self.dataframe = self.dataframe.iloc[:n_rows, :]
        if "clean_path" in dataset_dictionary:
            self.has_ground_truth = True
            self.clean_path = dataset_dictionary["clean_path"]
            self.clean_dataframe = self.read_csv_dataset(dataset_dictionary["clean_path"])
            if n_rows is not None:
                self.clean_dataframe = self.clean_dataframe.iloc[:n_rows, :]
            typed_clean_path = os.path.splitext(dataset_dictionary['clean_path'])[0] + '.parquet'
            if os.path.exists(typed_clean_path):
                self.typed_clean_dataframe = self.read_parquet_dataset(typed_clean_path)
                if n_rows is not None:
                    self.typed_clean_dataframe = self.typed_clean_dataframe.iloc[:n_rows, :]
            else:  # no .parquet with typed dataframe available. fall back to .csv file.
                self.typed_clean_dataframe = self.clean_dataframe
                if n_rows is not None:
                    self.typed_clean_dataframe = self.typed_clean_dataframe.iloc[:n_rows, :]

        if "repaired_path" in dataset_dictionary:
            self.has_been_repaired = True
            self.repaired_path = dataset_dictionary["repaired_path"]
            self.repaired_dataframe = self.read_csv_dataset(dataset_dictionary["repaired_path"])
            if n_rows is not None:
                self.repaired_dataframe = self.repaired_dataframe.iloc[:n_rows, :]

    @staticmethod
    def value_normalizer(value):
        """
        This method takes a value and minimally normalizes it.
        """
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
        return value

    def read_parquet_dataset(self, dataset_path: Union[str, None]):
        """
        This method reads a dataset from a parquet file path. This is nice for the imputer because
        parquet preserves dtypes.

        I did not figure out how to apply the value_normalizer function to solely non-numerical
        columns. Hope this still works!
        """
        if dataset_path is None:
            return None
        dataframe = pd.read_parquet(dataset_path)
        return dataframe

    def read_csv_dataset(self, dataset_path):
        """
        This method reads a dataset from a csv file path.
        """
        dataframe = pd.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).applymap(self.value_normalizer)
        return dataframe

    @staticmethod
    def write_csv_dataset(dataset_path, dataframe):
        """
        This method writes a dataset to a csv file path.
        """
        dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

    @staticmethod
    def get_dataframes_difference(df_1: pd.DataFrame, df_2: pd.DataFrame) -> Dict:
        """
        This method compares two dataframes df_1 and df_2. It returns a dictionary whose keys are the coordinates of
        a cell. The corresponding value is the value in df_1 at the cell's position if the values of df_1 and df_2 are
        not the same at the given position.
        """
        if df_1.shape != df_2.shape:
            raise ValueError("Two compared datasets do not have equal sizes.")
        difference_dictionary = {}
        for row in range(df_1.shape[0]):
            for col in range(df_1.shape[1]):
                if df_1.iloc[row, col] != df_2.iloc[row, col]:
                    difference_dictionary[(row, col)] = df_1.iloc[row, col]
        return difference_dictionary

    def create_repaired_dataset(self, correction_dictionary):
        """
        This method takes the dictionary of corrected values and creates the repaired dataset.
        """
        self.repaired_dataframe = self.dataframe.copy()
        for cell in correction_dictionary:
            self.repaired_dataframe.iloc[cell] = self.value_normalizer(correction_dictionary[cell])

    def get_df_from_labeled_tuples(self):
        """
        Added by Philipp. Turns the labeled tuples into a dataframe.
        """
        return self.clean_dataframe.iloc[list(self.labeled_tuples.keys()), :]

    def _get_actual_errors_dictionary_ground_truth(self) -> Dict[Tuple[int, int], str]:
        """
        Returns a dictionary that resolves every error cell to the ground truth.
        """
        return self.get_dataframes_difference(self.clean_dataframe, self.dataframe)

    def get_errors_dictionary(self) -> Dict[Tuple[int, int], str]:
        """
        This method compares the clean and dirty versions of a dataset. The returned dictionary resolves to the error
        values in the dirty dataframe.
        """
        return self.get_dataframes_difference(self.dataframe, self.clean_dataframe)

    def get_correction_dictionary(self):
        """
        This method compares the repaired and dirty versions of a dataset.
        """
        return self.get_dataframes_difference(self.repaired_dataframe, self.dataframe)

    def get_data_quality(self):
        """
        This method calculates data quality of a dataset.
        """
        return 1.0 - float(len(self._get_actual_errors_dictionary_ground_truth())) / (self.dataframe.shape[0] * self.dataframe.shape[1])

    def get_data_cleaning_evaluation(self, correction_dictionary, sampled_rows_dictionary=False):
        """
        This method evaluates data cleaning process.
        """
        actual_errors = self._get_actual_errors_dictionary_ground_truth()
        if sampled_rows_dictionary:
            actual_errors = {(i, j): actual_errors[(i, j)] for (i, j) in actual_errors if i in sampled_rows_dictionary}
        ed_tp = 0.0
        ec_tp = 0.0
        output_size = 0.0
        for cell in correction_dictionary:
            if (not sampled_rows_dictionary) or (cell[0] in sampled_rows_dictionary):
                output_size += 1
                if cell in actual_errors:
                    ed_tp += 1.0
                    if correction_dictionary[cell] == actual_errors[cell]:
                        ec_tp += 1.0
        ed_p = 0.0 if output_size == 0 else ed_tp / output_size
        ed_r = 0.0 if len(actual_errors) == 0 else ed_tp / len(actual_errors)
        ed_f = 0.0 if (ed_p + ed_r) == 0.0 else (2 * ed_p * ed_r) / (ed_p + ed_r)
        ec_p = 0.0 if output_size == 0 else ec_tp / output_size
        ec_r = 0.0 if len(actual_errors) == 0 else ec_tp / len(actual_errors)
        ec_f = 0.0 if (ec_p + ec_r) == 0.0 else (2 * ec_p * ec_r) / (ec_p + ec_r)
        return [ed_p, ed_r, ed_f, ec_p, ec_r, ec_f]
########################################


########################################
if __name__ == "__main__":
    dataset_dict = {
        "name": "toy",
        "path": "datasets/dirty.csv",
        "clean_path": "datasets/clean.csv"
    }
    d = Dataset(dataset_dict)
    print(d.get_data_quality())
########################################
